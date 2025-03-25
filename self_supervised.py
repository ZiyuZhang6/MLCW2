import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from modified_tpc_sample_selection import tpc_rp_sampling, extract_features, SimCLRModel, IndexedDataset
from random_sample_selection import random_sampling

# Set environment variable to avoid duplicate library errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Speed up training with cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Set random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# Define device to use GPU if available otherwise "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 datasets
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
indexed_cifar10_dataset = IndexedDataset(cifar10_train)

# Create a DataLoader
def get_dataloader(dataset, batch_size=512, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

# Define a linear probe model
class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)

# Train a linear probe model
def train_supervised_incremental(selected_samples, encoder, classifier, optimizer, scheduler, epoch=200, verbose=True):
    # Subset the CIFAR-10 dataset with selected samples
    train_subset = Subset(cifar10_train, selected_samples)
    train_loader = get_dataloader(train_subset, batch_size=512)
    test_loader = get_dataloader(cifar10_test, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss() # Cross-entropy loss

    for e in range(epoch):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = encoder(images)
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        if verbose and (e % 10 == 0 or e == epoch - 1):
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    features = encoder(images)
                    outputs = classifier(features)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            test_acc = 100. * correct / total
            print(f"Epoch [{e+1}/{epoch}] Loss: {avg_loss:.4f} Train Acc: {train_acc:.2f}% Test Acc: {test_acc:.2f}%")

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    final_acc = 100. * correct / total
    return classifier, final_acc

# Active learning iterations
def active_learning_iterations(budgets, repeats=10):
    encoder = SimCLRModel("simclr_cifar-10.pth.tar").to(device)
    dataloader = get_dataloader(indexed_cifar10_dataset, batch_size=512)
    all_features, all_indices, all_labels = extract_features(encoder, dataloader, device)

    accuracies_tpc_all = []
    accuracies_random_all = []

    for repeat in range(repeats):
        accuracies_tpc = []
        accuracies_random = []
        print(f"\nRepeat {repeat + 1}/{repeats}")

        for budget in budgets:
            # Perform TPC-RP sampling
            all_selected_tpc = tpc_rp_sampling(all_features, all_indices, all_labels, budget, 0, max_clusters=500)
            # Perform random sampling
            all_selected_random = random_sampling(cifar10_train, budget, seed=seed_value + repeat)

            # Initialize linear probe models, optimizers and schedulers for TPC-RP and random sampling
            classifier_tpc = LinearProbe().to(device)
            optimizer_tpc = optim.SGD(classifier_tpc.parameters(), lr=2.5, momentum=0.9, weight_decay=0.0001, nesterov=True)
            scheduler_tpc = lr_scheduler.CosineAnnealingLR(optimizer_tpc, T_max=200)

            classifier_random = LinearProbe().to(device)
            optimizer_random = optim.SGD(classifier_random.parameters(), lr=2.5, momentum=0.9, weight_decay=0.0001, nesterov=True)
            scheduler_random = lr_scheduler.CosineAnnealingLR(optimizer_random, T_max=200)

            # Train linear probe models with TPC-RP and random sampling
            print(f"Budget: {budget}")
            classifier_tpc, acc_tpc = train_supervised_incremental(
                all_selected_tpc, encoder, classifier_tpc, optimizer_tpc, scheduler_tpc, epoch=200, verbose=False
            )
            accuracies_tpc.append(acc_tpc)
            print(f"Budget = {budget} TPC_acc: {acc_tpc:.2f}%")

            classifier_random, acc_random = train_supervised_incremental(
                all_selected_random, encoder, classifier_random, optimizer_random, scheduler_random, epoch=200, verbose=False
            )
            accuracies_random.append(acc_random)
            print(f"Budget = {budget} random_acc: {acc_random:.2f}%")

        # Append the results for each repeat
        accuracies_tpc_all.append(accuracies_tpc)
        accuracies_random_all.append(accuracies_random)

    # Calculate the mean and standard deviation of accuracies for TPC-RP and random sampling
    accuracies_tpc_all = np.array(accuracies_tpc_all)
    accuracies_random_all = np.array(accuracies_random_all)
    accuracies_tpc_mean = np.mean(accuracies_tpc_all, axis=0)
    accuracies_tpc_std = np.std(accuracies_tpc_all, axis=0)
    accuracies_random_mean = np.mean(accuracies_random_all, axis=0)
    accuracies_random_std = np.std(accuracies_random_all, axis=0)
    accuracies_tpc_se = accuracies_tpc_std / np.sqrt(repeats)
    accuracies_random_se = accuracies_random_std / np.sqrt(repeats)

    # Plot the accuracy curve with standard error(shaded region)
    plt.figure(figsize=(10, 6))
    plt.plot(budgets, accuracies_tpc_mean, 'o-', label="TPC-RP", color='blue')
    plt.fill_between(budgets, accuracies_tpc_mean - accuracies_tpc_se, accuracies_tpc_mean + accuracies_tpc_se,
                     color='blue', alpha=0.2, label="TPC-RP Std Error")
    plt.plot(budgets, accuracies_random_mean, 's-', label="Random", color='orange')
    plt.fill_between(budgets, accuracies_random_mean - accuracies_random_se, accuracies_random_mean + accuracies_random_se,
                     color='orange', alpha=0.2, label="Random Std Error")
    plt.xlabel("Labeling Budget")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Self-Supervised Embeddings with Linear Probe (AL Iterations): TPC-RP vs Random")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve_self_supervised_al_with_se.png", dpi=300)
    plt.show()

    for i, budget in enumerate(budgets):
        print(f"Budget {budget}: TPC-RP Mean: {accuracies_tpc_mean[i]:.2f}% (±{accuracies_tpc_se[i]:.2f}), "
              f"Random Mean: {accuracies_random_mean[i]:.2f}% (±{accuracies_random_se[i]:.2f})")

if __name__ == "__main__":
    budgets = [10, 20, 30, 40, 50, 60]
    active_learning_iterations(budgets, repeats=10)
