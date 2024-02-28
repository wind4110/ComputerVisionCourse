import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def xshow_save(n: list,
               x: list,
               title='title',
               xlabel='x',
               ylabel='y',
               plotStyle='plot',
               annotationEnable=False) -> [plt.figure, plt.axes]:
    ''' 描绘序列

    :x: 需要绘制的序列
    :n: 与x等长的自然数序列
    :title: 图标题
    :xlabel: x轴标签
    :ylabel: y轴标签
    :plotStyle: 绘制风格，'stem'柱状散点,'plot'折线
    :annotationEnable: 数值标注使能
    '''

    # plot the chart
    fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
    if plotStyle == 'stem':
        ax.stem(n, x)
    elif plotStyle == 'plot':
        ax.plot(n, x)
    else:
        print("No such style yet")
        return

    # Title, Lable, Ticks, and Ylim
    ax.set_title(title, fontdict={'size': 20})
    ax.set_ylabel(ylabel, fontdict={'size': 20})
    plt.tick_params(labelsize=20)
    ax.set_xticks(n)
    ax.set_xlabel(xlabel, fontdict={'size': 20})

    # Annotation
    if (annotationEnable):
        # if (round(np.min(x), 2) < 0):
        #     ax.set_ylim(np.min(x) - 0.5, np.max(x) + 0.5)
        # else:
        #     ax.set_ylim(0, np.max(x) + 0.5)

        for i, j in zip(n, x):
            ax.text(i,
                    j,
                    s=round(j, 3),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=14)

    plt.savefig(title + '.jpg')

    return fig, ax


# 定义 LeNet-5 神经网络结构模型


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()  # 利用参数初始化父类
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fullconnection1 = nn.Linear(16 * 4 * 4, 120)
        self.fullconnection2 = nn.Linear(120, 84)
        self.fullconnection3 = nn.Linear(84, 10)

    # 定义前向传播
    def forward(self, x):
        x = self.avgpool1(F.relu(self.conv1(x)))
        x = self.avgpool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fullconnection1(x))
        x = F.relu(self.fullconnection2(x))
        x = self.fullconnection3(x)
        return x


# 前向传播过程中的维数变化
# 输入x: torch.Size([64, 1, 28, 28])
# ->conv1->avgpoo1: torch.Size([64, 6, 12, 12])
# ->conv2->avgpoo2: torch.Size([64, 16, 4, 4])
# ->flatten: torch.Size([64, 256])
# ->fullconnection1: torch.Size([64, 120])
# ->fullconnection2: torch.Size([64, 84])
# ->fullconnection3: torch.Size([64, 10])

# 加载MNIST数据集
root = 'D:\\VS\\vscode-py310\\ComputerVision\\HW4_LearningCNN\\Data'
# 定义数据预处理
transform = transforms.ToTensor()
# 下载数据到指定位置
trainset = datasets.MNIST(root,
                          train=True,
                          transform=transform,
                          target_transform=None,
                          download=True)
testset = datasets.MNIST(root,
                         train=False,
                         transform=transform,
                         target_transform=None,
                         download=True)
# 提取数据
train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size=64,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=64,
                                          shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型
model = Model().to(device)

# 实例化损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，用于多分类问题

# 构建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 自适应矩估计

# 训练次数
num_epochs = 10
train_loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
    Add_loss = 0.0  # 记录总平均损失
    Add300_loss = 0.0  # 记录每300batch训练平均损失
    train_percents = []  # 记录每batch训练的识别率
    for batch_index, (data, targets) in enumerate(train_loader):
        # 数据装载
        data = data.to(device)
        targets = targets.to(device)

        # 训练
        optimizer.zero_grad()  # 梯度清零
        y_pred = model(data)  # 正向传播
        loss = criterion(y_pred, targets)  # 计算损失
        loss.backward()  # 反向传播计算损失函数梯度
        optimizer.step()  # 更新参数

        Add_loss += loss.item()  # 叠加损失结果
        Add300_loss += loss.item()  # 叠加损失结果

        # 每300batch训练，打印训练结果
        if batch_index % 300 == 299:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_index+1}/{len(train_loader)}], Loss: {Add300_loss/300:.4f}"
            )
            Add300_loss = 0.0

        # 记录当前的识别率
        _, prediction = torch.max(y_pred, 1)
        train_percents.append(
            torch.sum(prediction == targets).item() / len(targets))

    # 每次训练结束，保存最终的损失和识别率，用于作图
    train_loss.append(Add_loss / len(train_loader))
    train_accuracy.append(sum(train_percents) / len(train_percents))

    # 每次训练后进行测试
    test_percents = []  # 用于记录每batch测试的识别率
    with torch.no_grad():
        for data, targets in test_loader:
            # 数据装载
            data = data.to(device)
            targets = targets.to(device)
            # 正向传播
            test_y_pred = model(data)
            # 记录识别率
            _, test_prediction = torch.max(test_y_pred, 1)
            test_percents.append(
                torch.sum(test_prediction == targets).item() / len(targets))

    # 每次测试结束，保存最终的识别率，用于作图
    test_accuracy.append(sum(test_percents) / len(test_percents))

    # 跟进训练进度
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Percent: {sum(test_percents) / len(test_percents):.4f}"
    )

# 保存曲线
xshow_save(range(len(train_loss)),
           train_loss,
           'train_loss',
           'epoch',
           'loss',
           annotationEnable=True)
xshow_save(range(len(train_accuracy)),
           train_accuracy,
           'train_accuracy',
           'epoch',
           'accuracy',
           annotationEnable=True)
xshow_save(range(len(test_accuracy)),
           test_accuracy,
           'test_accuracy',
           'epoch',
           'accuracy',
           annotationEnable=True)

print("Program finished.")
