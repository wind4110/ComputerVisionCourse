# 数据预处理，主要任务是构建训练集与测试集，

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
import json


# data process
def merge_type_strings(row):
    '''合并两个类型为一栏

    :row: 是一个DataFrame变量
    :return: 返回合并后的新列
    '''
    t1 = row['Type1']
    t2 = row['Type2']

    if t2 is np.nan:
        return t1
    return t1 + ' ' + t2


# read data label
df = pd.read_csv('ComputerVision/HW5_LearningCNNpro/archive/pokemon.csv')
df_train = pd.read_csv('ComputerVision/HW5_LearningCNNpro/archive/train.csv',
                       header=None,
                       names=['Name'])
df_test = pd.read_csv('ComputerVision/HW5_LearningCNNpro/archive/test.csv',
                      header=None,
                      names=['Name'])

# 合并两个类型
df['Type'] = df.apply(lambda row: merge_type_strings(row), axis=1)
df['Type'] = df['Type'].apply(lambda s: [ll for ll in str(s).split(' ')])
# print(df.head())

# 将要求的训练集与测试集标签补全
df_train = pd.merge(df_train, df[['Name', 'Type']], on='Name', how='left')
df_test = pd.merge(df_test, df[['Name', 'Type']], on='Name', how='left')

# print(df_test.head())
# print(df_train.head())

# 提取图片名与标签名
X_all = df['Name']
y_all = df['Type']
X_train = df_train['Name']
y_train = df_train['Type']
X_test = df_test['Name']
y_test = df_test['Type']

# 将图片名转为路径
X_all = [
    os.path.join("ComputerVision/HW5_LearningCNNpro/archive/images/",
                 str(f) + '.png') for f in X_all
]
X_train = [
    os.path.join("ComputerVision/HW5_LearningCNNpro/archive/images/",
                 str(f) + '.png') for f in X_train
]
X_test = [
    os.path.join("ComputerVision/HW5_LearningCNNpro/archive/images/",
                 str(f) + '.png') for f in X_test
]

# print(X_all[:3])

# 将标签转为list类型
y_all = list(y_all)
y_train = list(y_train)
y_test = list(y_test)

# print(y_all[:3])

# 构建多元编码方式并保存编码信息
type_encoding = {}

mlb = MultiLabelBinarizer()
mlb.fit(y_all)

# print("Labels: ")
# Loop over all labels and show
# N_LABELS = len(mlb.classes_)
for i, label in enumerate(mlb.classes_):
    # print("{}. {}".format(i, label))
    type_encoding[i] = label

# 对标签进行编码
y_all_bin = mlb.transform(y_all)
y_train_bin = mlb.transform(y_train)
y_test_bin = mlb.transform(y_test)

# for i in range(5):
#     print(X_train[i], y_train_bin[i])

# 保存处理后的信息
np.savetxt('ComputerVision/HW5_LearningCNNpro/archive/prepared/X_all.csv',
           np.array(X_all),
           fmt='%s')
np.savetxt('ComputerVision/HW5_LearningCNNpro/archive/prepared/X_train.csv',
           np.array(X_train),
           fmt='%s')
np.savetxt('ComputerVision/HW5_LearningCNNpro/archive/prepared/X_test.csv',
           np.array(X_test),
           fmt='%s')

np.savetxt('ComputerVision/HW5_LearningCNNpro/archive/prepared/y_all.csv',
           y_all_bin,
           delimiter=',')
np.savetxt('ComputerVision/HW5_LearningCNNpro/archive/prepared/y_train.csv',
           y_train_bin,
           delimiter=',')
np.savetxt('ComputerVision/HW5_LearningCNNpro/archive/prepared/y_test.csv',
           y_test_bin,
           delimiter=',')

f = open("ComputerVision/HW5_LearningCNNpro/archive/prepared/labels.json", "w")
f.write(json.dumps(type_encoding))
f.close()
