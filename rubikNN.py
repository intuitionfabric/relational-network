import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def randomprobs_sudoku(matrix):
    return np.random.random_sample((9,9,9))

def randomprobs(matrix):
    return np.random.random_sample(18)

def fakeneuralnetwork(matrix):
    value = np.random.random_sample()
    return (randomprobs(matrix), value)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class RubiksResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(RubiksResNet, self).__init__()

        # Initial input conv
        # NOTE: the number of input channels must match HISTO_MAX+1 in RubiksCube class
        self.conv1 = nn.Conv2d(6,64,3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.policyconv = nn.Conv2d(256,2,1)
        self.policybn = nn.BatchNorm2d(2)
        self.policyfc = nn.Linear(162, 18)

        self.valconv = nn.Conv2d(256,1,1)
        self.valbn = nn.BatchNorm2d(1)
        self.valfc1 = nn.Linear(81,256)
        self.valfc2 = nn.Linear(256,1)

        self.restower = nn.Sequential(
            ResidualBlock(64,128),
            ResidualBlock(128,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.restower(out)

        pi = self.policybn(self.policyconv(out))
        pi = self.policyfc(pi.view(-1,162))
        v = self.valbn(self.valconv(out))
        v = self.valfc2(self.valfc1(v.view(-1,81)))
        return F.log_softmax(pi), F.sigmoid(v)
