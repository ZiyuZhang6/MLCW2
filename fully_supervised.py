import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from tpc_sample_selection import tpc_rp_sampling, extract_features, SimCLRModel, IndexedDataset
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
device = torch.device("cuda")

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

# Define data transformation for testing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 datasets
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
indexed_cifar10_dataset = IndexedDataset(cifar10_dataset)

# Create a DataLoader
def get_dataloader(dataset, batch_size=512, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

# Reset weights of a model
def reset_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

# Define ResNet18
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

# Train the model incrementally with selected samples
def train_supervised_incremental(selected_samples, model, optimizer, scheduler, scaler, max_epochs=100):
    labeled_dataset = Subset(cifar10_dataset, selected_samples)
    labeled_loader = get_dataloader(labeled_dataset, batch_size=512)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()
        avg_loss = running_loss / total
        train_acc = 100. * correct / total
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Evaluate the model on the test set
    model.eval()
    test_loader = get_dataloader(cifar10_test, batch_size=2048, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    final_acc = 100. * correct / total
    return model, optimizer, scheduler, scaler, final_acc

# Perform active learning iterations with TPC and random sampling
def active_learning_iterations(budgets, repeats=10):
    simclr_model = SimCLRModel(pretrained_path="simclr_cifar-10.pth.tar").to(device)
    dataloader = get_dataloader(indexed_cifar10_dataset, batch_size=512)
    all_features, all_indices, all_labels = extract_features(simclr_model, dataloader, device)

    accuracies_tpc_all = []
    accuracies_random_all = []

    for repeat in range(repeats):
        print(f"\nRepeat {repeat + 1}/{repeats}")
        accuracies_tpc = []
        accuracies_random = []

        for budget in budgets:
            # Perform TPC-RP sampling and random sampling
            all_selected_tpc = tpc_rp_sampling(all_features, all_indices, all_labels, budget, 0, max_clusters=500)
            all_selected_random = random_sampling(cifar10_dataset, budget, seed=seed_value + repeat)

            # Initialize models, optimizers, schedulers, and scalers for TPC-RP and random sampling
            model_tpc = ResNet()
            model_tpc.model.apply(reset_weights)
            model_tpc.to(device)

            model_random = ResNet()
            model_random.model.apply(reset_weights)
            model_random.to(device)

            optimizer_tpc = optim.SGD(model_tpc.parameters(), lr=0.025, momentum=0.9, nesterov=True, weight_decay=0.005)
            scheduler_tpc = optim.lr_scheduler.CosineAnnealingLR(optimizer_tpc, T_max=100, eta_min=0)
            scaler_tpc = GradScaler('cuda')

            optimizer_random = optim.SGD(model_random.parameters(), lr=0.025, momentum=0.9, nesterov=True, weight_decay=0.005)
            scheduler_random = optim.lr_scheduler.CosineAnnealingLR(optimizer_random, T_max=100, eta_min=0)
            scaler_random = GradScaler('cuda')

            # Train models with selected samples
            print(f"Budget: {budget}")
            model_tpc, optimizer_tpc, scheduler_tpc, scaler_tpc, acc_tpc = train_supervised_incremental(
                all_selected_tpc, model_tpc, optimizer_tpc, scheduler_tpc, scaler_tpc
            )
            accuracies_tpc.append(acc_tpc)
            print(f"✅ TPC-RP Budget {budget} Test Accuracy: {acc_tpc:.2f}%")

            model_random, optimizer_random, scheduler_random, scaler_random, acc_random = train_supervised_incremental(
                all_selected_random, model_random, optimizer_random, scheduler_random, scaler_random
            )
            accuracies_random.append(acc_random)
            print(f"✅ Random Budget {budget} Test Accuracy: {acc_random:.2f}%")

        # Append the results for each repeat
        accuracies_tpc_all.append(accuracies_tpc)
        accuracies_random_all.append(accuracies_random)

    # Calculate mean and standard error
    accuracies_tpc_all = np.array(accuracies_tpc_all)
    accuracies_random_all = np.array(accuracies_random_all)
    accuracies_tpc_mean = np.mean(accuracies_tpc_all, axis=0)
    accuracies_tpc_std = np.std(accuracies_tpc_all, axis=0)
    accuracies_random_mean = np.mean(accuracies_random_all, axis=0)
    accuracies_random_std = np.std(accuracies_random_all, axis=0)
    accuracies_tpc_se = accuracies_tpc_std / np.sqrt(repeats)
    accuracies_random_se = accuracies_random_std / np.sqrt(repeats)

    # Plot accuracy curves with standard error(shaded region)
    plt.figure(figsize=(10, 6))
    plt.plot(budgets, accuracies_tpc_mean, 'o-', label="TPC-RP", color='blue')
    plt.fill_between(budgets, accuracies_tpc_mean - accuracies_tpc_se, accuracies_tpc_mean + accuracies_tpc_se,
                     color='blue', alpha=0.2, label="TPC-RP Std Error")
    plt.plot(budgets, accuracies_random_mean, 's-', label="Random", color='orange')
    plt.fill_between(budgets, accuracies_random_mean - accuracies_random_se, accuracies_random_mean + accuracies_random_se,
                     color='orange', alpha=0.2, label="Random Std Error")
    plt.xlabel("Labeling Budget")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Fully Supervised Learning with AL Iterations: TPC-RP vs Random")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve_fully_supervised_al_with_se.png", dpi=300)
    plt.show()

    # Print final results
    for i, budget in enumerate(budgets):
        print(f"Budget {budget}: TPC-RP Mean: {accuracies_tpc_mean[i]:.2f}% (±{accuracies_tpc_se[i]:.2f}), "
              f"Random Mean: {accuracies_random_mean[i]:.2f}% (±{accuracies_random_se[i]:.2f})")

if __name__ == "__main__":
    budgets = [10, 20, 30, 40, 50, 60]
    active_learning_iterations(budgets, repeats=10)