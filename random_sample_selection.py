import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=strong_transform)

def random_sampling(dataset: Dataset, budget: int, seed: int=42) -> np.ndarray:

    np.random.seed(seed)

    total_samples = len(dataset)
    all_indices = np.arange(total_samples)
    selected_indices = np.random.choice(all_indices, size=budget, replace=False)

    return selected_indices

def generate_random_samples(dataset, budgets=[10,20,30,40,50,60], seed=42):
    for budget in budgets:
        indices = random_sampling(dataset, budget, seed)
        np.save(f"selected_samples_random_{budget}.npy", indices)
        print(f"Save budget={budget} samples as selected_samples_random_{budget}.npy")

if __name__ == "__main__":
    generate_random_samples(cifar10_train)