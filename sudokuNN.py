import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

def randomprobs(matrix):
    return np.random.random_sample((9,9,9))

def fakeneuralnetwork(matrix):
    value = np.random.random_sample()
    return (randomprobs(matrix), value)

class SudokuNNold(nn.Module):
    def __init__(self):
        super(SudokuNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 9, 3, stride=3)
        self.conv2 = nn.Conv2d(9, 27, 2)
        self.conv3 = nn.Conv2d(27, 1, 1)
        self.conv4 = nn.ConvTranspose2d(1,1,8)
        self.conv5 = nn.ConvTranspose2d(1,9,1)

        self.bn1 = nn.BatchNorm2d(9)
        self.bn2 = nn.BatchNorm2d(27)

        self.fc1 = nn.Linear(4,4)
        self.fc2 = nn.Linear(4,1)

    def forward(self, state):
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.conv3(state))
        v = state.view(-1,4)
        v = self.fc2(self.fc1(v))
        pi = F.relu(self.conv4(state))
        pi = F.relu(self.conv5(pi))

        return F.log_softmax(pi), F.sigmoid(v)

class SudokuNN(nn.Module):
    def __init__(self):
        super(SudokuNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 32, 3, stride=3)
        self.conv4 = nn.Conv2d(32,1,2)
        self.conv5 = nn.ConvTranspose2d(1,1,8)
        self.conv6 = nn.ConvTranspose2d(1,9,1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(4,4)
        self.fc2 = nn.Linear(4,1)

    def forward(self, state, mask):
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = F.relu(self.conv4(state))
        v = state.view(-1,4)
        v = self.fc2(self.fc1(v))
        pi = F.relu(self.conv5(state))
        pi = mask*pi
        pi = F.relu(self.conv6(pi))

        return F.log_softmax(pi), F.sigmoid(v)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to downsample residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
