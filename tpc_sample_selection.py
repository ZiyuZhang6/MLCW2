import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import torchvision.models as models
import torch.nn as nn

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),           # 随机裁剪
    transforms.RandomHorizontalFlip(),               # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载 CIFAR-10
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 带索引的数据集包装类
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = np.arange(len(dataset))  # 保存原始索引

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return self.indices[index], data, target  # 返回三元组

    def __len__(self):
        return len(self.dataset)

# 创建带索引的完整数据集
indexed_full_dataset = IndexedDataset(cifar10_dataset)

# DataLoader
def get_dataloader(dataset, batch_size=256, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)

# 加载 SimCLR 预训练模型
class SimCLRModel(nn.Module):
    def __init__(self, model_path):
        super(SimCLRModel, self).__init__()

        # **直接加载 `state_dict`**
        checkpoint = torch.load(model_path, map_location=device)

        # **去掉 `contrastive_head` 层**
        state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint.items() if "contrastive_head" not in k}

        # **创建 ResNet18（去掉分类头）**
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # CIFAR-10 不需要 maxpool
        self.model.fc = nn.Identity()  # 去掉分类头

        # **加载权重**
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()

    def forward(self, x):
        return nn.functional.normalize(self.model(x), dim=1)  # 归一化嵌入

# 特征提取函数
def extract_features(model, dataloader, device):
    features = []
    global_indices = []

    with torch.no_grad():
        for batch in dataloader:
            indices, images, _ = batch  # 正确解包三元组
            images = images.to(device)
            batch_features = model(images).cpu()
            features.append(batch_features)
            global_indices.append(indices.numpy())

    return torch.cat(features), np.concatenate(global_indices)

# 运行 TPC-RP 选样本
def TPC_RP(dataloader, current_budget, total_budget, n_clusters=None, min_cluster_size=20):
    """新增参数传递和逻辑修复"""
    features, all_indices = extract_features(simclr_model, dataloader, device)

    # 动态调整聚类数（如果未指定）
    if n_clusters is None:
        n_clusters = max(10, current_budget // 5)
    n_clusters = min(n_clusters, len(features))  # 不能超过样本数

    # 进行 K-Means 聚类，限制聚类数量为 n_clusters
    kmeans = KMeans(n_clusters=min(len(features), n_clusters), random_state=42, n_init=50)  # 聚类数不超过 n_clusters
    cluster_labels = kmeans.fit_predict(features.numpy())

     # === 关键修复：筛选有效聚类（基于min_cluster_size） ===
    cluster_sizes = np.bincount(cluster_labels)
    valid_clusters = [i for i in range(n_clusters) if cluster_sizes[i] >= min_cluster_size]
    valid_clusters = valid_clusters[:n_clusters]  # 确保不超过原始聚类数

      # 保存聚类标签（包含有效聚类信息）
    np.save(
        f"cluster_labels_{total_budget}.npy",
        {
            "global_indices": all_indices,
            "cluster_labels": cluster_labels,
            "valid_clusters": valid_clusters  # 新增有效聚类标识
        },
        allow_pickle=True
    )

    # 计算每个聚类的大小
    cluster_sizes = np.bincount(cluster_labels)

    # 筛选出每个聚类中大于最小样本数的聚类
    valid_clusters = [i for i in range(n_clusters) if cluster_sizes[i] >= min_cluster_size]

    # 确保聚类数量不超过预算
    valid_clusters = valid_clusters[:n_clusters]  # 限制最多选择 n_clusters 个聚类

   # === 修改样本选择逻辑（仅在有效聚类中选择） ===
    selected_samples = []
    for cluster_id in valid_clusters:  # 仅遍历有效聚类
        cluster_mask = (cluster_labels == cluster_id)
        cluster_features = features[cluster_mask]

        # 计算典型性
        distances = torch.cdist(cluster_features, cluster_features)
        typicality = torch.mean(distances, dim=1)
        selected_idx = torch.argmin(typicality)

        selected_samples.append(all_indices[cluster_mask][selected_idx])
        if len(selected_samples) >= current_budget:
            break

    # 补充剩余样本（如果有效聚类不足）
    if len(selected_samples) < current_budget:
        remaining = current_budget - len(selected_samples)
        candidate_indices = np.setdiff1d(all_indices, selected_samples)

         # 优先从未使用的有效聚类中选择
        unused_clusters = [i for i in valid_clusters if i not in cluster_labels[selected_samples]]
        for cluster_id in unused_clusters:
            cluster_mask = (cluster_labels == cluster_id)
            candidates = all_indices[cluster_mask]
            num_to_pick = min(len(candidates), remaining)
            selected_samples.extend(np.random.choice(candidates, num_to_pick, replace=False))
            remaining -= num_to_pick
            if remaining <= 0:
                break
        # 随机补充剩余样本
        if remaining > 0:
            selected_samples.extend(np.random.choice(candidate_indices, remaining, replace=False))

    return np.array(selected_samples)


def generate_multicycle_samples(budgets):
    all_selected = np.array([], dtype=int)
    full_dataset = indexed_full_dataset  # 使用带索引的完整数据集

    for total_budget in budgets:
        print(f"\n=== 处理总预算 {total_budget} ===")

        # 计算剩余样本索引（关键修复点）==========================
        remaining_indices = np.setdiff1d(
            np.arange(len(full_dataset)),  # 原始数据集所有索引
            all_selected                   # 已选样本索引
        )
        # ===================================================

        # 检查是否需要选择新样本
        current_budget = total_budget - len(all_selected)
        if current_budget <= 0:
            print(f"⚠️ 预算 {total_budget} 已满足（已有 {len(all_selected)} 个样本），跳过选择")
            continue

        # 创建剩余数据集子集
        subset = Subset(full_dataset, remaining_indices)
        dataloader = get_dataloader(subset)

        # 动态参数计算（可根据需要调整公式）
        n_clusters = max(10, current_budget // 5)
        min_cluster_size = max(20, current_budget // 10)

        # 执行样本选择
        new_selected = TPC_RP(
            dataloader=dataloader,
            current_budget=current_budget,
            total_budget=total_budget,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size
        )

        # 合并结果（去重）
        all_selected = np.unique(np.concatenate([all_selected, new_selected]))

        # 保存结果
        np.save(f"selected_samples_{total_budget}.npy", all_selected)
        print(f"✅ 已保存总预算 {total_budget} 的样本（累计 {len(all_selected)} 个）")


# 主流程
if __name__ == "__main__":
    budgets = [10, 20, 30, 40, 50, 60]
    simclr_model = SimCLRModel("simclr_cifar-10.pth.tar").to(device)
    generate_multicycle_samples(budgets)

