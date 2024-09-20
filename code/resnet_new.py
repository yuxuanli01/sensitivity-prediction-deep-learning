import torch
import torch.nn as nn
import torch.optim
from torchinfo import summary
from torchvision.models import resnet18 as torchvision_resnet18

from torchvision.models import resnet


# class newResNet18(nn.Module):
#     def __init__(self, num_chan=1, dropout=0.39636414430926586): #0.2
#         super().__init__()
#         self.num_chan = num_chan

#         res = resnet.resnet18()
#         # if self.num_chan == 3:
#         #     self.conv = res.conv1
#         # else:
#         self.conv = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)

#         self.bn = res.bn1
#         self.relu = res.relu
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = res.layer1
#         self.layer2 = res.layer2
#         self.layer3 = res.layer3
#         self.layer4 = res.layer4
#         self.dropout = nn.Dropout(dropout)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, 1)

#     def forward(self, x):
#         x = self.conv(x)   # Adjusted Conv2d
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.maxpool(x)  # Adjusted MaxPool2d
#         l1 = self.layer1(x)
#         l1 = self.dropout(l1)
#         l2 = self.layer2(l1)
#         l2 = self.dropout(l2)
#         l3 = self.layer3(l2)
#         l3 = self.dropout(l3)
#         l4 = self.layer4(l3)

#         x = self.avgpool(l4)   # Final output size of 1x1
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

# # Example usage
# model2 = newResNet18(num_chan=1)
# print(summary(model2, input_size=(1, 1, 496, 32)))
# sample_input = torch.randn(1, 1, 496, 32)
# output = model2(sample_input)
# print(output.shape) # [1,1]


class newResNet18_Simplified(nn.Module):
    def __init__(self, num_chan=1, dropout=0.39636414430926586): 
        super().__init__()
        self.num_chan = num_chan

        res = resnet.resnet18()
        
        # Adjust initial convolution to handle single-channel input
        self.conv = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn = res.bn1
        self.relu = res.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Retain only layer1, layer2, and layer3
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        
        # Removed layer4
        self.dropout = nn.Dropout(dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Adjusted input size to 256 for the fully connected layer
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)  # Adjusted Conv2d
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)  # Adjusted MaxPool2d

        l1 = self.layer1(x)
        l1 = self.dropout(l1)
        l2 = self.layer2(l1)
        l2 = self.dropout(l2)
        l3 = self.layer3(l2)
        l3 = self.dropout(l3)

        # Removed layer4 and proceed directly to avgpool
        x = self.avgpool(l3)  # Final output size of 1x1
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Example usage:
from torchinfo import summary
# print(summary(newResNet18_Simplified(num_chan=1), input_size=(1, 1, 496, 32)))