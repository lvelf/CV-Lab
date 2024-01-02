import torch
from torchvision import datasets, transforms

# 定义数据预处理的transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 获取MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 获取训练集的样本和标签
train_samples = train_dataset.data
train_labels = train_dataset.targets

# 创建新的训练集
new_train_samples = []
new_train_labels = []

# 两两组合样本
for i in range(len(train_samples)):
    for j in range(i+1, len(train_samples)):
        # 判断标签是否相同
        if train_labels[i] == train_labels[j]:
            new_label = 1
        else:
            new_label = 0
        
        # 将两个样本组合起来
        combined_sample = torch.cat((train_samples[i], train_samples[j]), dim=1)
        
        # 添加到新的训练集中
        new_train_samples.append(combined_sample)
        new_train_labels.append(new_label)

# 将新的训练集转换为TensorDataset对象
new_train_dataset = torch.utils.data.TensorDataset(torch.stack(new_train_samples), torch.tensor(new_train_labels))

# 打印新的训练集的大小
print("新的训练集大小：", len(new_train_dataset))



