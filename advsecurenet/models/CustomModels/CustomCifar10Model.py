import torch.nn as nn


class CustomCifar10Model(nn.Module):
    def __init__(self, num_classes=10, num_input_channels=3, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 32,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*32*32, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
