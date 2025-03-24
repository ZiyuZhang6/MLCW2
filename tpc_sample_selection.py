"""
This script implements Algorithm TPC-RP for sample selection on CIFAR-10 using SimCLR embeddings.
Key steps: feature extraction with SimCLR, K-means clustering, and typicality-based selection.
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans
import random
from sklearn.neighbors import NearestNeighbors

# Set environment variable to avoid duplicate library errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
strong_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Define transformation for visualization
visual_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10 datasets
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=strong_transform)
cifar10_visual = datasets.CIFAR10(root='./data', train=True, download=True, transform=visual_transform)

# Custom dataset class to include indices
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = np.arange(len(dataset))

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return self.indices[index], data, target

    def __len__(self):
        return len(self.dataset)

indexed_full_dataset = IndexedDataset(cifar10_dataset)

# Create a DataLoader
def get_dataloader(dataset, batch_size=512, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

# Define SimCLR model for feature extraction
class SimCLRModel(nn.Module):
    def __init__(self, pretrained_path="simclr_cifar-10.pth.tar"):# Use the pretrained model
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # Load and adjust pretrained weights
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = {}
        for k, v in checkpoint.items():
            if "backbone" in k:
                new_k = k.replace("backbone.", "").replace("shortcut", "downsample") # Adjust key names
                state_dict[new_k] = v
        self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.to(device).eval()

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, dim=1) # L2 normalize

# Extract features using SimCLR model
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    indices = []
    labels = []

    for batch in dataloader:
        idx, images, targets = batch
        with torch.no_grad():
            batch_features = model(images.to(device)).cpu().numpy()
        features.append(batch_features)
        indices.append(idx.numpy())
        labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    indices = np.concatenate(indices, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, indices, labels

# TPC-RP sampling algorithm
def tpc_rp_sampling(features, all_indices, all_labels, budget, num_labeled, max_clusters=500):
    remaining_features = features.copy()
    remaining_indices = all_indices.copy()
    remaining_labels = all_labels.copy()
    n_samples = len(remaining_features)

    if n_samples == 0:
        raise ValueError("No remaining samples to cluster.")

    index_to_position_all = {idx: pos for pos, idx in enumerate(all_indices)}
    index_to_position = {idx: pos for pos, idx in enumerate(remaining_indices)}

    selected_samples = []
    labeled_mask = np.zeros(len(all_indices), dtype=bool)

    K = min(num_labeled + budget, max_clusters)
    if n_samples < K:
        K = max(1, n_samples)

    if K <= 50: # Use KMeans for small number of clusters
        kmeans = KMeans(n_clusters=K, n_init=1, max_iter=20, random_state=seed_value)
    else:
        kmeans = MiniBatchKMeans(n_clusters=K, n_init=1, max_iter=20, random_state=seed_value, batch_size=6144) # Use MiniBatchKMeans for large number of clusters
    cluster_labels = kmeans.fit_predict(remaining_features)

    clusters = [[] for _ in range(K)]
    for i, label in enumerate(cluster_labels):
        clusters[label].append(remaining_indices[i])

    while len(selected_samples) < budget:
        valid_clusters = []
        for c in range(K):
            if len(clusters[c]) < 20:
                continue
            contains_labeled = any(labeled_mask[index_to_position_all[idx]] for idx in clusters[c])
            if not contains_labeled:
                valid_clusters.append(c)

        if not valid_clusters:
            break

        samples_to_select = min(budget - len(selected_samples), len(valid_clusters))
        cluster_sizes = [(c, len(clusters[c])) for c in valid_clusters]
        cluster_sizes.sort(key=lambda x: -x[1])
        selected_clusters = [c for c, _ in cluster_sizes[:samples_to_select]]

        for target_cluster in selected_clusters:
            cluster_indices = np.array(clusters[target_cluster])
            positions = [index_to_position[idx] for idx in cluster_indices]
            cluster_features = remaining_features[positions]

            nn = NearestNeighbors(n_neighbors=20, metric='euclidean')
            nn.fit(cluster_features)
            D, _ = nn.kneighbors(cluster_features)
            typicality = 1 / (np.mean(D, axis=1) + 1e-6) # typicality score

            unlabeled_mask = ~labeled_mask[[index_to_position_all[idx] for idx in cluster_indices]]
            if not unlabeled_mask.any():
                continue
            typicality_unlabeled = typicality[unlabeled_mask]
            top_idx = np.argmax(typicality_unlabeled)
            selected_idx = cluster_indices[unlabeled_mask][top_idx]

            selected_samples.append(selected_idx)
            labeled_mask[index_to_position_all[selected_idx]] = True

        remaining_mask = ~labeled_mask[[index_to_position_all[idx] for idx in remaining_indices]]
        remaining_features = remaining_features[remaining_mask]
        remaining_indices = remaining_indices[remaining_mask]
        remaining_labels = remaining_labels[remaining_mask]
        n_samples = len(remaining_features)
        index_to_position = {idx: pos for pos, idx in enumerate(remaining_indices)}

    if len(selected_samples) < budget: # Select randomly if not enough samples
        remaining_unlabeled_indices = remaining_indices[~labeled_mask[[index_to_position_all[idx] for idx in remaining_indices]]]
        if len(remaining_unlabeled_indices) > 0:
            remaining_to_select = budget - len(selected_samples)
            extra_indices = np.random.choice(remaining_unlabeled_indices, min(remaining_to_select, len(remaining_unlabeled_indices)), replace=False)
            selected_samples.extend(extra_indices)

    selected_samples = np.array(selected_samples[:budget], dtype=int)
    if len(selected_samples) == 0:
        raise ValueError("No samples selected.")

    positions = [index_to_position_all[idx] for idx in selected_samples]
    selected_labels = all_labels[positions]
    num_classes = 10
    print(f"Class distribution (Budget {num_labeled + budget}):")
    for cls in range(num_classes):
        count = np.sum(selected_labels == cls)
        print(f"  Class {cls}: {count} samples")

    return selected_samples

# Visualize selected samples
def visualize_samples(selected_samples, all_indices, all_labels, budget):
    num_classes = 10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    samples_by_class = [[] for _ in range(num_classes)]
    for idx in selected_samples:
        label = all_labels[np.where(all_indices == idx)[0][0]]
        samples_by_class[label].append(idx)

    samples_per_class = [len(samples) for samples in samples_by_class]
    max_samples = max(samples_per_class)

    fig, axes = plt.subplots(num_classes, max_samples, figsize=(max_samples * 2, num_classes * 2))
    if max_samples == 1:
        axes = axes.reshape(num_classes, 1)

    for cls in range(num_classes):
        cls_samples = samples_by_class[cls]
        for i in range(max_samples):
            ax = axes[cls, i]
            if i < len(cls_samples):
                img_idx = cls_samples[i]
                img = cifar10_visual[img_idx][0]
                img = img.permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(f"{cifar10_classes[cls]}")
            else:
                ax.axis('off')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"sampled_images_budget_{budget}.png", dpi=150)
    plt.show()

# Generate samples (10 samples per iteration)
def generate_samples(B=10, num_iterations=6, max_clusters=500):
    dataloader = get_dataloader(indexed_full_dataset, batch_size=512)
    all_features, all_indices, all_labels = extract_features(simclr_model, dataloader, device)

    selected_indices_dict = {}
    all_selected = np.array([], dtype=int)

    for iteration in range(num_iterations):
        num_labeled = len(all_selected)
        current_budget = B
        budget = num_labeled + current_budget

        remaining_mask = np.isin(all_indices, all_selected, invert=True)
        remaining_features = all_features[remaining_mask]
        remaining_indices = all_indices[remaining_mask]
        remaining_labels = all_labels[remaining_mask]

        new_selected = tpc_rp_sampling(remaining_features, remaining_indices, remaining_labels, current_budget, num_labeled, max_clusters)
        all_selected = np.unique(np.concatenate([all_selected, new_selected]))
        selected_indices_dict[budget] = all_selected.copy()

        if budget == 60:
            visualize_samples(all_selected, all_indices, all_labels, budget)

    for budget, selected in selected_indices_dict.items():
        np.save(f"selected_samples_typiclust_{budget}.npy", selected)
        print(f"Save samples of budget= {budget} as selected_samples_typiclust_{budget}.npy")

    return selected_indices_dict

# Visualize t-SNE of selected samples
def visualize_tsne(selected_samples, all_features, all_labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(all_features)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=all_labels, cmap='tab10', alpha=0.3, s=10)
    plt.scatter(reduced[np.isin(np.arange(len(all_features)), selected_samples),0],
                reduced[np.isin(np.arange(len(all_features)), selected_samples),1],
                c='black', marker='x', s=50, label='Selected')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE Visualization with Selected Samples")
    plt.show()

if __name__ == "__main__":
    model_path = "simclr_cifar-10.pth.tar"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained model {model_path} not found!")
    simclr_model = SimCLRModel(pretrained_path=model_path).to(device)

    print("Starting TPC sampling...")
    selected_indices_dict = generate_samples(B=10, num_iterations=6)

    all_features, all_indices, all_labels = extract_features(simclr_model, get_dataloader(indexed_full_dataset), device)
    visualize_tsne(selected_samples=selected_indices_dict[60], all_features=all_features, all_labels=all_labels)