## Lab 3 Report

### 框架选择

​	选择的是pytorch框架。



### 实验

​	在具体实践中主要尝试了简单的Siamese网络，和应用了残差块的Siamese网络，以及应用了ResNet模块的CNN。在实验具体说明中，将会描述Dataset的设置，以及尝试过的各个神经网络框架的设计和效果。



#### Dataset的设置

​	对于Dataset的选取大体思路为：对于每一个sample，从MNIST数据集中随机选取两张图片，如果它们的label相同，则这个sample的label是1，否则为0，对于每个sample的数据格式，是由两张[28x28]横向拼接并增加channel的[1x28x56]的inputs。在选取sample的时候，需要创建set，避免train和test内部数据重复，并且train和test数据集之间有重复，导致数据泄露。

​	并且需要保存缓存数据集，因为在实际实验中，尝试了较多网络框架，需保证每次使用的数据集是相同的，不然没有对比性。

​	在数量级上，train和test数据集都选取了1e5的数据，其中train数据集有10229个sample的label为1，测试集中有9934个sample的label为1。batch_size设置为64。

​	这是Dataset的具体处理代码

```python
def get_transferred_data(train_dataset, batch_size, shuffle, set_name, repeatition = None):
    
    dataset_path = f'./Dataset/{set_name}_data_set'
    labels_path = f'./Dataset/{set_name}_labels'
    
    if os.path.exists(dataset_path) and os.path.exists(labels_path):
        new_train_samples = torch.load(dataset_path)
        new_train_labels = torch.load(labels_path)
        train_data_set = torch.utils.data.TensorDataset(torch.stack(new_train_samples), torch.tensor(new_train_labels))

        train_data_loader = DataLoader(train_data_set, batch_size = batch_size, shuffle = shuffle)
        return train_data_loader, None
    
    # randomly get 10% train data
    train_samples = train_dataset.data
    num_samples = int(len(train_samples) * 0.1)
    random_index = torch.randperm(train_samples.size(0))
    
    
    train_samples = train_samples[random_index[:num_samples] ]
    print(num_samples)
    print(len(train_samples))
    
    #get labels
    train_labels = train_dataset.targets[random_index[:num_samples]]
    
    new_train_samples = []
    new_train_labels = []
    
    
    train_number = 0
    train_set = set()
        
        
    while(train_number < 100000):
        i = random.randint(0, len(train_samples) - 1)
        j = random.randint(0, len(train_samples) - 1)
        
        if i == j or (i,j) in train_set:
            continue
        
        if repeatition != None:
            if (i,j) in repeatition:
                continue

        if train_labels[i] == train_labels[j]:
            new_label = 1
        else:
            new_label = 0
                
        combined_sample = torch.cat((train_samples[i], train_samples[j]), dim = 1)
            
        new_train_samples.append(combined_sample)
        new_train_labels.append(new_label)
        
        train_number += 1
        train_set.add((i, j))
        train_set.add((j, i))
            
            
    for i in range(len(new_train_samples)):
        new_train_samples[i] = new_train_samples[i].unsqueeze(0).float()
    
    new_train_labels = torch.tensor(new_train_labels, dtype=torch.float).unsqueeze(1)
    
    
    train_data_set = torch.utils.data.TensorDataset(torch.stack(new_train_samples), torch.tensor(new_train_labels))

    train_data_loader = DataLoader(train_data_set, batch_size = batch_size, shuffle = shuffle)
    
    
    os.makedirs('./Dataset', exist_ok=True) 
    torch.save(new_train_samples, dataset_path)
    torch.save(new_train_labels, labels_path)
            
    if repeatition != None:
        return train_data_loader, None
    
    if repeatition == None:
        return train_data_loader, train_set
```

​	代码思路如上所述，在此不过多赘述。



#### Simple Siamese Network



#### Complex Siamese Network



#### ResNet CNN

##### Net设计

​	在设计中，尝试将网络设计为通过深层卷积结构和残差连接来提取拼接图像中的特征，并通过全连接层和Sigmoid激活函数来执行最终的分类任务的CNN。

​	这是该CNN的具体设计代码。

​	首先是残差块的实现

```python
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

​	是基础的ResNet模块设计，前文已有描述，在此不过多赘述。

​	其次是Net的具体设计

```python
class Net(nn.Module):

    def __init__(self, dropout=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.resnet_block1 = self.make_layer(64, 64, 2, stride=1)
        self.resnet_block2 = self.make_layer(64, 128, 2, stride=2)
        self.resnet_block3 = self.make_layer(128, 256, 2, stride=2)
        self.resnet_block4 = self.make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p = dropout)
        self.fc = nn.Linear(512, 1)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(Residual(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, num_blocks):
            layers.append(Residual(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.resnet_block1(out)
        out = self.resnet_block2(out)
        out = self.resnet_block3(out)
        out = self.resnet_block4(out)

        out = self.avgpool(out)
        
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))

        return out
```

1. **深层特征提取**：设计中，网络使用多个残差块（`resnet_block1` 到 `resnet_block4`）来提取深层次的特征。这些块逐渐增加通道数（从64到512），同时逐步减少特征图的空间维度，这是通过stride为2的卷积层实现的。
2. **残差连接**：`Residual` 类定义了残差块，这些块通过在卷积层之间添加恒等映射（或通过下采样调整的恒等映射）来帮助解决深层网络中的梯度消失问题。这允许网络学习更复杂的特征，同时保持较低的训练难度。因此在实验中在较小的epoch即跑出了较好的效果。
3. **自适应平均池化**：`nn.AdaptiveAvgPool2d((1, 1))` 将特征图尺寸降为 1x1，这有助于减少参数数量，同时保留了通道中的重要信息。
4. **丢弃（Dropout）**：在全连接层之前应用丢弃操作，以减少过拟合的风险。因为使用了四个残差块进行连接，担心过拟合风险。
5. **全连接层和Sigmoid激活**：最后一个全连接层 `fc` 将特征映射到一个单一的输出，而 `torch.sigmoid` 函数将这个输出转换为一个介于0和1之间的值。这样的输出适合于二分类任务，即判断两张图片是否代表同一个数字。

##### 实验效果

在acc上表现优异



































