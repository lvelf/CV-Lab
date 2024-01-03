import torch
import torch.nn as nn

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
    
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1) 
        output = self.fc(output)
        return output

    def forward(self, x):
        
        x1 = x[:, :, :, :28]
        x2 = x[:, :, :, 28:]

        
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        
        return output1, output2
    
    def euclidean_distance(self, output1, output2):
        return torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
    
    def eval(self, output1, output2, threshold=0.01):
        distance = self.euclidean_distance(output1, output2)
        predictions = distance < threshold
        return predictions.int()



class SiameseNet_Residual(nn.Module):
    def __init__(self, dropout=0.5):
        super(SiameseNet_Residual, self).__init__()
        

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.make_layer(64, 64, 2, stride=1), 
            self.make_layer(64, 128, 2, stride=2),
            self.make_layer(128, 256, 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(p = dropout)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
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

    def forward_once(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1) 
        output = self.fc(output)
        return output

    def forward(self, x):
        
        x1 = x[:, :, :, :28]
        x2 = x[:, :, :, 28:]

        
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)

        
        distance = torch.abs(output1 - output2)
        return 1 - distance

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive



