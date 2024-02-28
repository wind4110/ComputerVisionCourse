#!/usr/bin/env python
# coding: utf-8
# writer: 许展风 email: zhanfeng_xu@outlook.com

# In[90]:

import numpy as np
import cv2
import matplotlib.pyplot as plt  # 绘图
import os  # 数据导入
from PIL import Image  # 图片变形
import json  # 读取josn文件
import math

# In[91]:
# 本栏函数用于图片变形
# Copyright (c) 2012, Philipp Wagner
# All rights reserved.


def Distance(p1: list, p2: list) -> float:
    '''求两点间距离

    :p1: 点1坐标
    :p2: 点2坐标
    :return: 距离
    '''
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def ScaleRotateTranslate(image,
                         angle,
                         center=None,
                         new_center=None,
                         scale=None,
                         resample=Image.BICUBIC):
    '''旋转图片'''
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size,
                           Image.AFFINE, (a, b, c, d, e, f),
                           resample=resample)


def CropFace(image,
             eye_left=(0, 0),
             eye_right=(0, 0),
             offset_pct=(0.2, 0.2),
             dest_sz=(70, 70)):
    '''根据左右眼距离缩放图片

    :image: 输入图像
    :eye_left: 左眼位置
    :eye_right: 右眼位置
    :offset_pct: 想要保持在眼睛旁边的图像的百分比(水平方向，垂直方向)
    :dest_sz: 输出图像的大小
    '''
    # 计算原图像便宜
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # 获得方向角度
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # 计算旋转角度
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # 左右眼距离
    dist = Distance(eye_left, eye_right)
    # 左眼相对边距离
    reference = dest_sz[0] - 2.0 * offset_h
    # 缩放常数
    scale = float(dist) / float(reference)
    # 以左眼为中心旋转
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # 裁剪已旋转的图片
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]),
         int(crop_xy[1] + crop_size[1])))
    # 设置大小
    image = image.resize(dest_sz, Image.LANCZOS)
    return image


# In[92]:

# 导入数据集


# 读取一个文件夹下的所有图片，输入参数是文件名，返回文件地址列表
def read_directory(directory_name):
    faces_addr = []
    for filename in os.listdir(directory_name):
        faces_addr.append(directory_name + "/" + filename)
    return faces_addr


# 读取所有人脸文件夹,保存图像地址在faces列表中
faces = []
eyes = []
for i in range(1, 41):
    faces_addr = read_directory(
        'D:/VS/vscode-py310/ComputerVision/HW3_Eigenface/att-face/s' + str(i))
    eyes_addr = read_directory(
        'D:/VS/vscode-py310/ComputerVision/HW3_Eigenface/ATT-eye-location/s' +
        str(i))
    for addr in faces_addr:
        faces.append(addr)
    for addr in eyes_addr:
        eyes.append(addr)

eyes_dests = []

for index, eye_dest in enumerate(eyes):
    f = open(eye_dest, 'r')
    eyes_dests.append(json.loads(f.read()))
    f.close()

# 读取图片数据,生成列表标签
total_images = []  # 总人脸图片集
train_images = []  # 一半图片集用于训练
test_images = []  # 一半图片集用于测试
nochange_images = []  # 未变形处理的一半测试集
labels = []
for index, face in enumerate(faces):

    tempIm = Image.open(face)
    tranIm = CropFace(tempIm,
                      eyes_dests[index]['centre_of_left_eye'],
                      eyes_dests[index]['centre_of_right_eye'],
                      offset_pct=(0.3, 0.43),
                      dest_sz=(92, 112))  # 根据眼睛位置变换图片

    initIm = np.array(tranIm)
    histIm = cv2.equalizeHist(initIm)  # 图片均衡化

    total_images.append(histIm)
    labels.append(int(index / 5 + 1))  # 标记一半图片的训练集和测试集
    # 分类为训练和测试集
    if (index % 10 < 5):
        train_images.append(histIm)
    else:
        test_images.append(histIm)
        nochange_images.append(cv2.equalizeHist(np.array(tempIm)))

# 读取个人人脸文件夹
selffaces = []
selfface_addr = read_directory(
    'D:/VS/vscode-py310/ComputerVision/HW3_Eigenface/self')
for addr in selfface_addr:
    selffaces.append(addr)

# 将个人人脸数据加入
for index, selfface in enumerate(selffaces):
    tempIm = cv2.imread(selfface, cv2.IMREAD_GRAYSCALE)
    histIm = cv2.equalizeHist(tempIm)

    total_images.append(histIm)
    labels.append(int((index + 400) / 5 + 1))
    # 分类为训练和测试集
    if (index % 10 < 5):
        train_images.append(histIm)
    else:
        test_images.append(histIm)
        nochange_images.append(np.array(histIm))

# 读取个人识别测试用图像
selftest_image = cv2.imread(
    'D:/VS/vscode-py310/ComputerVision/HW3_Eigenface/self_test.jpg',
    cv2.IMREAD_GRAYSCALE)


# In[93]:
# 三个主要过程函数


def mytrain(energyPercent: float, images: list, modelfile: str) -> None:
    '''训练过程, 对输入图像进行主成分分析, 得到平均脸和主特征脸PCs, 并将模型保存在指定文件夹中

    :energyPercent: 主成分能量百分比
    :images:  训练集
    :modelfile: 模型存储文件夹
    :return: 无返回值
    '''

    print('training...')

    # 样本数量
    K = len(images)

    # 图片像素维度
    SIZE = images[0].shape

    # 图像矩阵向量化
    flatten_faces = []
    for face in images:
        flatten_faces.append(face.flatten())

    # 对应维度上取平均得到平均脸
    average_face = np.sum(flatten_faces, axis=0) / K

    # 保存平均脸图像
    show_average = average_face.reshape(SIZE)
    # cv2.imwrite('show_average.jpg',show_average)

    # 求协方差矩阵
    diff_faces = flatten_faces - average_face
    C_faces = np.dot(diff_faces.T, diff_faces) / K

    print('EVD...')
    # EVD
    e_vals, e_vecs = np.linalg.eigh(C_faces)

    # 实数化
    e_vals = np.real(e_vals)
    e_vecs = np.real(e_vecs)

    # np.linalg.eigh求得的是升序排列的结果，现将其倒序
    e_vals_re = e_vals[::-1]
    e_vecs_re = (e_vecs.T)[::-1].T

    # 筛选主成分
    vecs_num = 0
    main_vals_Sum = 0
    vals_Sum = np.sum(e_vals_re)
    i = 0
    while (main_vals_Sum < vals_Sum * energyPercent):
        main_vals_Sum += e_vals_re[i]
        i += 1
    vecs_num = i
    main_vecs = []
    for i in range(0, vecs_num):
        main_vecs.append(e_vecs_re[:, i])
    main_vecs = np.array(main_vecs).T  # 得到按列排列的筛选后特征向量

    # 保存模型
    os.makedirs(modelfile, exist_ok=True)
    np.save(modelfile + '/main_vecs', main_vecs)
    np.save(modelfile + '/avg_face', show_average)

    print('finished.')

    return


# In[94]:


def mytest(test_image: np.ndarray,
           images: list,
           modelfile: str,
           PCN=-1,
           imagename='similar_image.jpg',
           savesimilar=False) -> int:
    '''识别过程, 对输入图片以及训练集做坐标变换, 变换到特征脸空间后比较欧氏距离, 筛选出最相似的标签

    :test_image: 被识别图片
    :images: 训练集
    :modelfile: 存储了模型的文件夹
    :PCN: 用于变换的特征脸数量
    :imagename: 保存图像名
    :savesimilar: 是否保存训练集最相似人脸
    :return: 返回训练集最相似人脸标签
    '''

    # 导入模型数据
    main_vecs = np.load(modelfile + '/main_vecs.npy')  # 提取特征脸
    main_num = main_vecs.shape[1]  # 特征脸个数
    if (PCN == -1):  # 取默认值
        PCN = main_num
    tran_vecs = main_vecs[:, 0:PCN]  # 提取PCN个特征脸用作变换
    avg_face = np.load(modelfile + '/avg_face.npy')  # 提取平均脸

    # 对输入被识别图片进行特征脸空间变换
    x_face = (test_image - avg_face).flatten()  # 与平均脸求差
    y_vecs = np.dot(tran_vecs.T, x_face)  # 变换

    # 对训练集图片进行特征脸空间变换
    # 二维图片展平
    flatten_faces = []
    for face in images:
        flatten_faces.append(face.flatten())
    flatten_faces = np.array(flatten_faces)

    # 变换
    diff_faces = flatten_faces - avg_face.flatten()  # 与平均脸求差
    train_vecs = np.dot(tran_vecs.T, diff_faces.T)  # 变换

    # 求最小的欧式距离的序号
    D = train_vecs - y_vecs.reshape((PCN, 1))  # 利用python广播性质求差
    D_2 = np.diag(np.dot(D.T, D))  # 对角线值即为欧式距离的平方
    index_similar = np.argmin(D_2)

    if (savesimilar):  # 保存最相似的训练集图片
        cv2.imwrite(imagename, images[index_similar])

    return labels[index_similar]  # 得到标签值


# In[95]:


def myrestruct(test_image: np.ndarray,
               modelfile: str,
               imagename='restruct_image') -> None:
    '''重构过程, 将输入图片进行特征脸空间变换后, 再将坐标变换为像素空间, 保存不同数量特征脸变换的结果图片

    :test_image: 输入图片
    :modelfile: 存储了模型的文件夹
    :imagename: 保存图像名
    :return: 无返回值
    '''
    # 提取模型数据
    main_vecs = np.load(modelfile + '/main_vecs.npy')
    main_num = main_vecs.shape[1]
    avg_face = np.load(modelfile + '/avg_face.npy')
    SIZE = avg_face.shape

    # 对输入图片预处理
    x_face = (test_image - avg_face).flatten()

    # 根据不同PCN求重构图片
    PCNs = [10, 25, 50, 100, main_num]
    for PCN in PCNs:
        tran_vecs = main_vecs[:, 0:PCN]  # 取对应数量
        y_vecs = np.dot(tran_vecs.T, x_face)  # 变换
        re_face = np.dot(tran_vecs, y_vecs)  # 重构

        # 绘制
        re_face = re_face.reshape(SIZE)
        re_face = cv2.normalize(re_face, None, 255, 0, cv2.NORM_MINMAX,
                                cv2.CV_8UC1)  # 映射到255像素
        re_face = np.uint8(re_face)
        cv2.imwrite(imagename + str(PCN) + '.jpg', re_face)
    return


# In[96]:


def plotmodel(modelfile: str, imagename: str) -> None:
    '''绘制模型数据, 包括平均脸, 前十个特征脸, 前十特征脸的合成

    :modelfile: 存储了模型的文件夹
    :imagename: 保存图片名
    :return: 无返回值
    '''
    # 提取模型数据
    main_vecs = np.load(modelfile + '/main_vecs.npy')
    # main_num = main_vecs.shape[1]
    avg_face = np.load(modelfile + '/avg_face.npy')
    SIZE = avg_face.shape

    # 保存平均脸图像
    cv2.imwrite(imagename + '_average.jpg', avg_face)

    # 绘制前10个特征脸
    main_image0 = main_vecs[:, 0].reshape(SIZE)
    # 将像素值正则化至255像素范围
    PCs = cv2.normalize(main_image0, None, 255, 0, cv2.NORM_MINMAX,
                        cv2.CV_8UC1)
    # 组合其他图片
    for i in range(1, 11):
        main_image = main_vecs[:, i].reshape(SIZE)
        main_image = cv2.normalize(main_image, None, 255, 0, cv2.NORM_MINMAX,
                                   cv2.CV_8UC1)
        PCs = np.hstack([PCs, main_image])
    PCs = np.uint8(PCs)
    cv2.imwrite(imagename + "_PCs_10.jpg", PCs)

    # 绘制展示特征脸的合成
    CC = np.sum(main_vecs[:, 0:10], axis=1)  # 按列合成
    CC = CC.reshape(SIZE)
    CC = cv2.normalize(CC, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    CC = np.uint8(CC)
    cv2.imwrite(imagename + "_PCs_sum.jpg", CC)

    return


# In[121]:
# 绘图函数


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
    fig, ax = plt.subplots(figsize=(15, 4), dpi=100)
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
    # ax.set_xticks(n)
    ax.set_xlabel(xlabel, fontdict={'size': 20})

    # Annotation
    if (annotationEnable):
        # if (round(np.min(x), 2) < 0):
        #     ax.set_ylim(np.min(x) - 0.5, np.max(x) + 0.5)
        # else:
        #     ax.set_ylim(0, np.max(x) + 0.5)

        for i, j in zip(n, x):
            ax.text(i + 5,
                    j - 0.002,
                    s=round(j, 2),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=14)

    plt.savefig(title + '.jpg')

    return fig, ax


# In[98]:


def PCN_test(test_images: list, train_images: list, modelfile: str) -> tuple:
    '''用于统一测试不同PCN值的识别率

    :test_images: 测试集
    :train_images: 训练集
    :modelfile: 保存图片名
    :return: PCN数量列表与对应识别率列表组成的元组
    '''
    # 导入模型数据
    main_vecs = np.load(modelfile + '/main_vecs.npy')
    main_num = main_vecs.shape[1]

    rates = []
    PCNs = np.arange(10, main_num, 20)
    for PCN in PCNs:
        test_results = []
        for index, test_image in enumerate(test_images):  # 遍历测试集
            test_results.append(
                mytest(test_image, train_images, modelfile,
                       PCN=PCN) == labels[index])  # 记录识别结果
        rates.append(np.sum(test_results) / len(test_images))  # 记录识别率
    return [PCNs, rates]


# In[128]:

# 将数据集全部用作训练集训练
modelfile = 'D:/VS/vscode-py310/ComputerVision/HW3_Eigenface/total_model'
energyPercent = 0.95
mytrain(energyPercent, total_images, modelfile)

# 绘制前10个特征脸，与前10个特征脸的合成，绘制平均脸
plotmodel(modelfile, 'total')

# 输入自己的脸做识别，并保存最相似训练图
mytest(selftest_image, total_images, modelfile, savesimilar=True)

# # 输入自己的脸做重构
myrestruct(selftest_image, modelfile)

# 对训练集中的脸重构
myrestruct(total_images[405], modelfile, 'Re2')

# 将一半数据集用作训练集训练
modelfile_half = 'D:/VS/vscode-py310/ComputerVision/HW3_Eigenface/half_model'
energyPercent_half = 1  # 设置太小会导致特征脸数量不足
mytrain(energyPercent_half, train_images, modelfile_half)

# 输入测试集图片进行识别
PCNs_0, rates_0 = PCN_test(test_images, train_images, modelfile_half)
xshow_save(PCNs_0,
           rates_0,
           title='PC-Rank1 rate curve',
           xlabel='number of PCs',
           ylabel='rate',
           annotationEnable=True)

PCNs_1, rates_1 = PCN_test(nochange_images, train_images, modelfile_half)
xshow_save(PCNs_1,
           rates_1,
           title='PC-Rank1 rate curve Ⅱ',
           xlabel='number of PCs',
           ylabel='rate',
           annotationEnable=True)
