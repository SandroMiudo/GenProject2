import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.ConvTranspose2d(inchannel, outchannel, kernel_size=3, padding=1, output_padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(inchannel, outchannel, kernel_size=1, output_padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, norm_shape, norm_layer='batch', stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel) if norm_layer == 'batch' else nn.LayerNorm(norm_shape),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel) if norm_layer == 'batch' else nn.LayerNorm(norm_shape)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel) if norm_layer == 'batch' else nn.LayerNorm(norm_shape)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out