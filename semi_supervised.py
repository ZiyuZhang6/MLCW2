import logging
import time
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.datasets import CIFAR10
import numpy as np
from torchvision.transforms.autoaugment import RandAugment
import torch.nn.functional as F
import os
from collections import Counter
from train_utils import EMA, Bn_Controller, get_optimizer, get_cosine_schedule_with_warmup, ce_loss, AverageMeter
from flexmatch_utils import consistency_loss, Get_Scalar
from modified_tpc_sample_selection import tpc_rp_sampling, extract_features, SimCLRModel, IndexedDataset
from random_sample_selection import random_sampling

# Allow duplicate OpenMP runtime libraries to avoid conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("training_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Enable cuDNN benchmark for performance optimization
torch.backends.cudnn.benchmark = True

# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# Define device to use GPU if available otherwise "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define test data transformation (normalization only)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Define weak data augmentation for labeled data
weak_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Define strong data augmentation for unlabeled data
strong_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 training dataset
cifar10_dataset = CIFAR10(root='./data', train=True, download=True)

# Load CIFAR-10 indexed dataset for feature extraction
indexed_cifar10_dataset = IndexedDataset(CIFAR10(root='./data', train=True, download=True, transform=test_transform))

# Create a DataLoader
def get_dataloader(dataset, batch_size=512):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Define a dataset class for dual transformations (weak and strong) on unlabeled data
class DualTransformDataset(Dataset):
    def __init__(self, base_dataset, exclude_indices):
        self.base_dataset = base_dataset
        self.exclude_indices = set(exclude_indices)
        # Filter out excluded indices to create valid indices list
        self.valid_indices = [i for i in range(len(base_dataset)) if i not in self.exclude_indices]

    def __getitem__(self, index):
        # Get the actual index from valid_indices
        actual_idx = self.valid_indices[index]
        image, _ = self.base_dataset[actual_idx]
        # Apply weak and strong transformations
        weak_img = weak_transform(image)
        strong_img = strong_transform(image)
        return actual_idx, weak_img, strong_img

    def __len__(self):
        return len(self.valid_indices)

# Define WideResNet model
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, num_classes=10):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        # Three wide residual blocks
        self.layer1 = self._wide_layer(WideResNetBlock, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(WideResNetBlock, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(WideResNetBlock, nStages[3], n, stride=2)
        # Batch normalization and final linear layer
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.apply(self._init_weights)

    def _wide_layer(self, block, planes, num_blocks, stride):
        # Construct a sequence of residual blocks
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Forward pass through the network
        out = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.leaky_relu(self.bn1(out), negative_slope=0.1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Define a single WideResNet block
class WideResNetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(WideResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.9)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.9)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        # Add shortcut connection if dimensions or stride change
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes, momentum=0.9)
            )

    def forward(self, x):
        # Forward pass through the residual block
        out = F.leaky_relu(self.bn1(x), negative_slope=0.1)
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = F.leaky_relu(self.bn2(out), negative_slope=0.1)
        out = self.conv2(out)
        out += shortcut
        return out

# Define FlexMatchTrainer for semi-supervised learning
class FlexMatchTrainer:
    def __init__(self, selected_samples, model=None, num_classes=10, ema_m=0.999, T=1.0, p_cutoff=0.95, lambda_u=1.0, num_train_iter=1048576):
        # Initialize model, default to WideResNet if none provided
        self.model = model if model is not None else WideResNet(depth=28, widen_factor=2, num_classes=num_classes).to(device)
        self.num_classes = num_classes
        self.ema_m = ema_m
        self.t_fn = Get_Scalar(T)
        self.p_fn = Get_Scalar(p_cutoff)
        self.lambda_u = lambda_u
        self.it = 0
        self.num_train_iter = num_train_iter
        self.num_eval_iter = 10000  # Log every 10000 iterations
        self.print_fn = logger.info

        # Initialize EMA for model parameters
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()

        # Set up optimizer and scheduler
        self.optimizer = get_optimizer(self.model, optim_name='SGD', lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_training_steps=num_train_iter, num_warmup_steps=1000)

        # Create datasets and data loaders
        labeled_dataset = Subset(CIFAR10(root='./data', train=True, download=True, transform=weak_transform), selected_samples)
        unlabeled_base = CIFAR10(root='./data', train=True, download=True)
        self.loader_dict = {
            'train_lb': DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True),
            'train_ulb': DataLoader(DualTransformDataset(unlabeled_base, selected_samples), batch_size=64*7, shuffle=True, num_workers=0, pin_memory=True),
            'eval': DataLoader(CIFAR10(root='./data', train=False, transform=test_transform), batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
        }
        self.ulb_dset = unlabeled_base

        # Initialize iterators for labeled and unlabeled data
        self.lb_iterator = iter(self.loader_dict['train_lb'])
        self.ulb_iterator = iter(self.loader_dict['train_ulb'])

        # Initialize pseudo-labels and class-wise accuracy
        self.selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, device=device) * -1
        self.classwise_acc = torch.zeros((self.num_classes,), device=device)
        self.bn_controller = Bn_Controller()

        # Set uniform target distribution for CIFAR-10
        self.p_target = torch.full((num_classes,), 1.0 / num_classes, device=device)  # [0.1, 0.1, ..., 0.1]
        self.p_model = None

        # Initialize loss and mask meters
        self.sup_loss_meter = AverageMeter()
        self.unsup_loss_meter = AverageMeter()
        self.total_loss_meter = AverageMeter()
        self.mask_meter = AverageMeter()

    def get_batch(self, iterator, loader):
        # Fetch the next batch, reset iterator if exhausted
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        # Update the corresponding iterator
        self.lb_iterator = iterator if loader == self.loader_dict['train_lb'] else self.lb_iterator
        self.ulb_iterator = iterator if loader == self.loader_dict['train_ulb'] else self.ulb_iterator
        return batch

    def train(self):
        # Set model to training mode
        self.model.train()
        best_eval_acc, best_it = 0.0, 0

        start_time = time.time()
        # Training loop for specified number of iterations
        for it in range(self.num_train_iter):
            # Get batches of labeled and unlabeled data
            x_lb, y_lb = self.get_batch(self.lb_iterator, self.loader_dict['train_lb'])
            x_ulb_idx, x_ulb_w, x_ulb_s = self.get_batch(self.ulb_iterator, self.loader_dict['train_ulb'])

            # Move data to device
            x_lb, y_lb = x_lb.to(device), y_lb.to(device)
            x_ulb_w, x_ulb_s = x_ulb_w.to(device), x_ulb_s.to(device)
            x_ulb_idx = x_ulb_idx.to(device)

            # Compute class-wise accuracy for pseudo-labeling
            pseudo_counter = Counter(self.selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset):
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

            # Concatenate inputs for a single forward pass
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            logits = self.model(inputs)
            logits_x_lb = logits[:x_lb.shape[0]]
            logits_x_ulb_w, logits_x_ulb_s = logits[x_lb.shape[0]:].chunk(2)

            # Compute supervised loss
            sup_loss = ce_loss(logits_x_lb, y_lb, use_hard_labels=True, reduction='mean')

            # Compute unsupervised loss with consistency regularization
            T = self.t_fn(it)
            p_cutoff = self.p_fn(it)
            unsup_loss, mask, select, pseudo_lb, self.p_model = consistency_loss(
                logits_x_ulb_s, logits_x_ulb_w, self.classwise_acc, self.p_target, self.p_model,
                'ce', T, p_cutoff, use_hard_labels=True, use_DA=True
            )

            # Update pseudo-labels for selected samples
            if x_ulb_idx[select == 1].nelement() != 0:
                self.selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

            # Compute total loss
            total_loss = sup_loss + self.lambda_u * unsup_loss

            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update()
            self.it = it + 1

            # Update loss and mask meters
            self.sup_loss_meter.update(sup_loss.item())
            self.unsup_loss_meter.update(unsup_loss.item())
            self.total_loss_meter.update(total_loss.item())
            self.mask_meter.update(mask.item())

            # Evaluate and log periodically
            if it % self.num_eval_iter == 0:
                eval_dict = self.evaluate()
                logger.info(
                    f"{it} iteration, USE_EMA: {self.ema_m != 0}, "
                    f"Sup Loss: {self.sup_loss_meter.avg:.4f}, Unsup Loss: {self.unsup_loss_meter.avg:.4f}, "
                    f"Total Loss: {self.total_loss_meter.avg:.4f}, Mask: {self.mask_meter.avg:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, {eval_dict}, "
                    f"BEST_EVAL_ACC: {best_eval_acc:.2f}, at {best_it} iters"
                )
                if eval_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = eval_dict['eval/top-1-acc']
                    best_it = it
                # Reset meters after logging
                self.sup_loss_meter.reset()
                self.unsup_loss_meter.reset()
                self.total_loss_meter.reset()
                self.mask_meter.reset()

            # Adjust evaluation frequency in later stages
            if it > 0.8 * self.num_train_iter:
                self.num_eval_iter = 1000

        # Final evaluation
        eval_dict = self.evaluate()
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return self.model, eval_dict

    @torch.no_grad() # Disable gradient calculation for efficiency
    def evaluate(self):
        # Set model to evaluation mode and apply EMA weights
        self.model.eval()
        self.ema.apply_shadow()
        eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true, y_pred = [], []
        # Evaluate on test set
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            total_num += x.shape[0]
            total_loss += loss.detach() * x.shape[0]
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
        top1 = sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1}

# Define active learning iterations with TPC and random sampling
def active_learning_iterations(budgets, B=10, repeats=1):
    num_iterations = len(budgets)
    budgets = [B * i for i in range(1, num_iterations + 1)]
    logger.info(f"Generated budgets based on B={B}: {budgets}")

    # Initialize SimCLR model for TPC sampling
    simclr_model = SimCLRModel(pretrained_path="simclr_cifar-10.pth.tar").to(device)
    dataloader = get_dataloader(indexed_cifar10_dataset, batch_size=512)
    all_features, all_indices, all_labels = extract_features(simclr_model, dataloader, device)

    accuracies_tpc_all = []
    accuracies_random_all = []

    # Repeat experiments
    for repeat in range(repeats):
        logger.info(f"Repeat {repeat + 1}/{repeats}")
        accuracies_tpc = []
        accuracies_random = []

        for budget in budgets:
            start_time = time.time()
            # Perform TPC sampling
            all_selected_tpc = tpc_rp_sampling(all_features, all_indices, all_labels, budget, 0, max_clusters=500)
            # Perform random sampling
            all_selected_random = random_sampling(cifar10_dataset, budget, seed=seed_value + repeat)

            # Log class distribution of selected samples
            tpc_labels = [cifar10_dataset.targets[i] for i in all_selected_tpc]
            random_labels = [cifar10_dataset.targets[i] for i in all_selected_random]
            logger.info(f"Budget {budget}: Loaded {len(all_selected_tpc)} TPC samples, class distribution: {np.bincount(tpc_labels, minlength=10)}")
            logger.info(f"Budget {budget}: Loaded {len(all_selected_random)} Random samples, class distribution: {np.bincount(random_labels, minlength=10)}")

            # Train with TPC samples
            trainer_tpc = FlexMatchTrainer(selected_samples=all_selected_tpc)
            acc_tpc = trainer_tpc.train()[1]['eval/top-1-acc']
            accuracies_tpc.append(acc_tpc)
            logger.info(f"TPC-RP budget {budget} test accuracy: {acc_tpc:.2f}")

            # Train with random samples
            trainer_random = FlexMatchTrainer(selected_samples=all_selected_random)
            acc_random = trainer_random.train()[1]['eval/top-1-acc']
            accuracies_random.append(acc_random)
            logger.info(f"Random budget {budget} test accuracy: {acc_random:.2f}")

            iteration_time = time.time() - start_time
            logger.info(f"Budget {budget} completed in {iteration_time:.2f} seconds")

        accuracies_tpc_all.append(accuracies_tpc)
        accuracies_random_all.append(accuracies_random)

    # Compute mean accuracies
    accuracies_tpc_mean = np.mean(accuracies_tpc_all, axis=0)[0]
    accuracies_random_mean = np.mean(accuracies_random_all, axis=0)[0]

    # Plot results
    methods = ['TPC-RP', 'Random']
    accuracies = [accuracies_tpc_mean, accuracies_random_mean]
    plt.figure(figsize=(8, 6))
    plt.bar(methods, accuracies, color=['blue', 'orange'], width=0.5)
    plt.xlabel("Sampling Method")
    plt.ylabel("Test Accuracy (%)")
    plt.title("FlexMatch Accuracy Comparison (Budget=10)")
    plt.ylim(0, 100)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("accuracy_bar_flexmatch_budget10.png", dpi=300)
    plt.show()

    logger.info(f"Budget 10: TPC-RP Mean: {accuracies_tpc_mean:.2f}%, Random Mean: {accuracies_random_mean:.2f}%")

if __name__ == "__main__":
    budgets = [10]
    active_learning_iterations(budgets, B=10, repeats=1)