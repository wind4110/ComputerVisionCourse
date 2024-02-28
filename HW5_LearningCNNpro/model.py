import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
from PIL import Image


class pokemonCNN(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels,
                               16,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 150, kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(150)
        self.maxpool5 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fullconnection1 = nn.Linear(150 * (120 // 32) * (120 // 32), 64)
        self.fullconnection2 = nn.Linear(64, self.out_channels)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(F.relu(self.bn1(self.conv2(x))))
        x = self.maxpool3(F.relu(self.bn1(self.conv3(x))))
        x = self.maxpool4(F.relu(self.bn1(self.conv4(x))))
        x = self.maxpool5(F.relu(self.bn1(self.conv5(x))))
        x = self.flatten(x)
        x = F.relu(self.fullconnection1(x))
        x = self.fullconnection2(x)
        return x


class pokemonDataset(Dataset):

    def __init__(self, X, y, transform=None) -> None:
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image_path = self.X[index]
        label = torch.tensor(self.y[index], dtype=torch.float32)

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# # 创建模型实例
# model = pokemonCNN(in_channels=3, out_channels=18)

# # 打印模型结构
# print(model)
