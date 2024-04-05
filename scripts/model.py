import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAttention(nn.Module):
    def __init__(self,in_channels, kernel_size):
        super(SpectralAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        b, c, _, _ = x.size()

        # Average pool the input feature map
        y = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(b, c, 1)

        # Apply 1D convolutions and activation functions
        y = torch.sigmoid(self.conv1(y))
        y = torch.sigmoid(self.conv2(y)).view(b,c,1,1) # relu or sigmoid?
        # y = y.expand_as(x)
        # print(y)
        # Multiply the input feature map by the attention map
        return x * y.expand_as(x)
    

class CNNSAM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNNSAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.spectral_attention1 = SpectralAttention(32, kernel_size=3)
        self.spectral_attention2 = SpectralAttention(64, kernel_size=3)
        self.apply(self.init_weights)
        self.fc = nn.Linear(128*10*10, num_classes)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Apply the spectral attention module after each convolutional layer
        x = F.relu(self.conv1(x))
        # print('mean after conv1')
        # print(x.mean())
        x = self.spectral_attention1(x)
        x = F.relu(self.conv2(x))
        # print('mean after conv2')
        # print(x.mean())
        x = self.spectral_attention2(x)
        x = F.relu(self.conv3(x))
        # print('mean after conv3')
        # print(x.mean())
        # Globally average pool the feature map and apply a fully connected layer
        x = torch.flatten(x, 1)
        # Apply a linear transformation without activation function
        logits = self.fc(x)

        # out = torch.sigmoid(logits)
        return logits