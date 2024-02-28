import numpy as np
# import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# from unet import UNet

# from datetime import datetime
# import os

# class AttentionLayer(nn.Module):

#     def __init__(self, hidden_size, input_size):
#         super(AttentionLayer, self).__init__()
#         self.hidden_size = hidden_size
#         self.W_q = nn.Linear(input_size, hidden_size)
#         self.W_k = nn.Linear(input_size, hidden_size)

#     def forward(self, query, key, value):
#         Q = self.W_q(query)
#         K = self.W_k(key)
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
#             torch.tensor(self.hidden_size, dtype=torch.float32))
#         attention_weights = F.softmax(attention_scores, dim=-1)
#         output = torch.matmul(attention_weights, value)
#         return output, attention_weights


# 网络结构定义
class pokemonCNN(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.attention_hidden_size = attention_hidden_size

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

        # # 注意力层
        # self.attention = AttentionLayer(
        #     input_size=64, hidden_size=self.attention_hidden_size)

        self.flatten = nn.Flatten()
        self.fullconnection1 = nn.Linear(150 * (120 // 32) * (120 // 32), 64)
        self.fullconnection2 = nn.Linear(64, self.out_channels)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(F.relu(self.bn2(self.conv2(x))))
        x = self.maxpool3(F.relu(self.bn3(self.conv3(x))))
        # x, attention_weights = self.attention(x, x, x)  # 应用注意力层
        x = self.maxpool4(F.relu(self.bn4(self.conv4(x))))
        x = self.maxpool5(F.relu(self.bn5(self.conv5(x))))
        x = self.flatten(x)
        x = F.relu(self.fullconnection1(x))
        x = self.fullconnection2(x)
        return x


# 定义数据集
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


X_all = np.loadtxt(
    'ComputerVision/HW5_LearningCNNpro/archive/prepared/X_all.csv', dtype=str)
X_train = np.loadtxt(
    'ComputerVision/HW5_LearningCNNpro/archive/prepared/X_train.csv',
    dtype=str)
X_test = np.loadtxt(
    'ComputerVision/HW5_LearningCNNpro/archive/prepared/X_test.csv', dtype=str)

y_all_bin = np.loadtxt(
    'ComputerVision/HW5_LearningCNNpro/archive/prepared/y_all.csv',
    delimiter=',')
y_train_bin = np.loadtxt(
    'ComputerVision/HW5_LearningCNNpro/archive/prepared/y_train.csv',
    delimiter=',')
y_test_bin = np.loadtxt(
    'ComputerVision/HW5_LearningCNNpro/archive/prepared/y_test.csv',
    delimiter=',')

# with open('ComputerVision/HW5_LearningCNNpro/archive/prepared/labels.json'
#           ) as json_file:
#     type_encoding = json.load(json_file)
# # convert string keys to ints
# type_encoding = {int(k): v for k, v in type_encoding.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理变换
transform_train = transforms.Compose([transforms.ToTensor()])
transform_train_aug = transforms.Compose([  # 数据增强
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(degrees=45),
    transforms.ToTensor()
])
transform_test = transforms.Compose([transforms.ToTensor()])

# 数据装载
# 泄露数据集
datasets_all = pokemonDataset(X_all, y_all_bin, transform=transform_train_aug)
DataLoader_all = DataLoader(datasets_all, batch_size=32, shuffle=True)

# 未泄露训练集
datasets_train = pokemonDataset(X_train,
                                y_train_bin,
                                transform=transform_train_aug)
DataLoader_train = DataLoader(datasets_train, batch_size=32, shuffle=True)

# 测试集
datasets_test = pokemonDataset(X_test, y_test_bin, transform=transform_test)
DataLoader_test = DataLoader(datasets_test, batch_size=8, shuffle=False)

# model = pokemonCNN(in_channels=3, out_channels=18, attention_hidden_size=64).to(device)

# criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 模型训练与测试函数
def trainandtestModel(num_epochs, DataLoader_train, DataLoader_test, lr,
                      title):
    trainloss = []
    trainacc = []
    testacc = []
    model = pokemonCNN(
        in_channels=3,
        out_channels=18,
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        Add_loss = 0.0  # 记录训练损失
        Add_accuracy = 0.0  # 训练识别率
        Add_accuracy_test = 0.0  # 测试识别率
        for data, targets in DataLoader_train:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  # 梯度清零
            y_pred = model(data)  # 正向传播
            loss = criterion(y_pred, targets)  # 计算损失
            loss.backward()  # 反向传播计算损失函数梯度
            optimizer.step()  # 更新参数

            Add_loss += loss.item()
            # 计算识别率
            pre = (F.sigmoid(y_pred) > 0.5).float()
            accuracy = 0.0
            for i in range(0, len(targets)):
                if (pre[i] == targets[i]).sum().item() == len(pre[i]):
                    accuracy += 1
            Add_accuracy += accuracy / len(targets)

        with torch.no_grad():
            for data, targets in DataLoader_test:
                # 数据装载
                data = data.to(device)
                targets = targets.to(device)
                # 正向传播
                test_y_pred = model(data)
                # 记录识别率
                pre_test = (F.sigmoid(test_y_pred) > 0.5).float()

                accuracy_test = 0
                for i in range(0, len(targets)):
                    if (pre_test[i] == targets[i]).sum().item() == len(
                            pre_test[i]):
                        accuracy_test += 1
                Add_accuracy_test += accuracy_test / len(targets)

        trainloss.append(Add_loss / len(DataLoader_train))
        trainacc.append(Add_accuracy / len(DataLoader_train))
        testacc.append(Add_accuracy_test / len(DataLoader_test))

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train_Loss: {Add_loss/len(DataLoader_train):.4f}, Train_Accuarcy: {Add_accuracy / len(DataLoader_train):.4f}, Test_Accuarcy: {Add_accuracy_test / len(DataLoader_test):.4f}"
        )

    fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
    n = range(0, num_epochs)
    ax.plot(n, trainloss, linestyle='-', label='train_loss')
    ax.plot(n, trainacc, linestyle='--', label='train_acc')
    ax.plot(n, testacc, linestyle=':', label='test_acc')

    ax.set_title(title, fontdict={'size': 20})
    ax.legend(fontsize=20)
    ax.set_xlabel('epoch', fontdict={'size': 20})
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.savefig('ComputerVision/HW5_LearningCNNpro/images/' + title + '.jpg')
    return trainloss, trainacc, testacc


# trainandtestModel(num_epochs=50,
#                   DataLoader_train=DataLoader_all,
#                   DataLoader_test=DataLoader_test,
#                   lr=0.01,
#                   title='lr=0.01,aug,leakage')

# trainandtestModel(num_epochs=50,
#                   DataLoader_train=DataLoader_train,
#                   DataLoader_test=DataLoader_test,
#                   lr=0.01,
#                   title='lr=0.01,aug,non-leakage')

# 数据装载、取消数据增强
# 泄露数据集
datasets_all = pokemonDataset(X_all, y_all_bin, transform=transform_train)
DataLoader_all = DataLoader(datasets_all, batch_size=32, shuffle=True)

# 未泄露训练集
datasets_train = pokemonDataset(X_train,
                                y_train_bin,
                                transform=transform_train)
DataLoader_train = DataLoader(datasets_train, batch_size=32, shuffle=True)

# 测试集
datasets_test = pokemonDataset(X_test, y_test_bin, transform=transform_test)
DataLoader_test = DataLoader(datasets_test, batch_size=32, shuffle=False)

# trainandtestModel(num_epochs=50,
#                   DataLoader_train=DataLoader_all,
#                   DataLoader_test=DataLoader_test,
#                   lr=0.001,
#                   title='lr=0.001,leakage')

# trainandtestModel(num_epochs=50,
#                   DataLoader_train=DataLoader_train,
#                   DataLoader_test=DataLoader_test,
#                   lr=0.001,
#                   title='lr=0.001,non-leakage')
