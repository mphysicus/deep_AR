"""
CNN Model to generate 3 channel images for the modified SAM model to generate segmentation masks.
It receives 1024 x 1024 dimensional tensor arrays as input and outputs 3 channel RGB images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class IVT2RGB(nn.Module):
    """
    CNN model to convert IVT data to RGB images.
    The output is a 3 dimensional tensor with shape (3, 1024, 1024).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding='same')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        """
        Forward pass of the model.
        """
        y = F.relu(self.conv1(x), inplace=True)
        y = F.relu(self.conv2(y), inplace=True)
        y = F.relu(self.conv3(y), inplace=True)
        y = self.conv4(y)
        y = F.sigmoid(y + x)  # Ensure output is in range [0, 1)
        return y