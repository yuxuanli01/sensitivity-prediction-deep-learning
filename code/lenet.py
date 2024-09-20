import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torchinfo import summary

class LeNet5(nn.Module):

    def __init__(self, num_chan=3):
        super(LeNet5, self).__init__()
        self.num_chan = num_chan
        if self.num_chan==1:
            self.feature_extractor = nn.Sequential(            
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                nn.Tanh()
            )
        else:
            self.feature_extractor = nn.Sequential(            
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                nn.Tanh()
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
    
#     def __init__(self, num_chan=1):
#         super(LeNet5, self).__init__()
#         self.num_chan = num_chan
#         if self.num_chan == 1:
#             self.feature_extractor = nn.Sequential(            
#                 nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=2),  # Reduced filters, increased stride
#                 nn.Sigmoid(),  # Changed activation to Sigmoid
#                 nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2),  # Reduced filters
#                 # Removed Tanh and pooling layers
#                 nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),  # Reduced filters
#                 # Removed Tanh and pooling layers
#             )
#         else:
#             self.feature_extractor = nn.Sequential(            
#                 nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=2),  # Reduced filters, increased stride
#                 nn.Sigmoid(),  # Changed activation to Sigmoid
#                 nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2),  # Reduced filters
#                 # Removed Tanh and pooling layers
#                 nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),  # Reduced filters
#                 # Removed Tanh and pooling layers
#             )

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=16, out_features=10),  # Drastically reduced features
#             nn.Sigmoid(),  # Changed activation to Sigmoid
#             nn.Linear(in_features=10, out_features=1),
#         )


#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         logits = self.classifier(x)
#         return logits
    

# #summary(LeNet5(num_chan=1), input_size=(32, 1, 480, 32))

