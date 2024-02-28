#!/usr/bin/env python
# coding: utf-8
# writer: 许展风 email: zhanfeng_xu@outlook.com

# In[16]:

import cv2
import numpy as np
import sys

sys.setrecursionlimit(10000)  # 将默认的递归深度修改为3000

# In[17]:


# 计算梯度幅值和方向
def Gradient_1(image):
    '''用大小(2,2)微分算子求图像梯度

    :image: 输入图像
    :return M: 输出梯度幅值,整型
    :return theta: 输出梯度方向
    '''
    Size = np.shape(image)
    P = np.zeros(Size)
    Q = np.zeros(Size)
    M = np.zeros(Size)
    theta = np.zeros(Size)
    image = np.float16(image)
    for i in range(0, Size[0] - 1):
        for j in range(0, Size[1] - 1):
            P[i,
              j] = image[i + 1, j] - image[i, j] + image[i + 1, j +
                                                         1] - image[i, j + 1]
            Q[i,
              j] = image[i, j + 1] - image[i, j] + image[i + 1, j +
                                                         1] - image[i + 1, j]

            M[i, j] = np.sqrt(P[i, j]**2 + Q[i, j]**2)
            if (P[i, j] != 0):
                theta[i, j] = np.arctan(Q[i, j] / P[i, j])
            else:
                theta[i, j] = np.pi / 2
    return np.uint8(M), theta


def Gradient_Sobel(image):
    '''用Sobel微分算子求图像梯度

    :image: 输入图像
    :return M: 输出梯度幅值,整型
    :return theta: 输出梯度方向
    '''
    Size = np.shape(image)
    P = np.zeros(Size)
    Q = np.zeros(Size)
    M = np.zeros(Size)
    theta = np.zeros(Size)
    image = np.float16(image)
    for i in range(1, Size[0] - 1):
        for j in range(1, Size[1] - 1):
            P[i, j] = 2 * (image[i + 1, j] - image[i - 1, j]) + image[
                i + 1, j + 1] - image[i - 1, j + 1] + image[i + 1, j -
                                                            1] - image[i - 1,
                                                                       j - 1]
            Q[i, j] = 2 * (image[i, j + 1] - image[i, j - 1]) + image[
                i + 1, j + 1] - image[i + 1, j - 1] + image[i - 1, j +
                                                            1] - image[i - 1,
                                                                       j - 1]

            M[i, j] = np.sqrt(P[i, j]**2 + Q[i, j]**2)
            if (P[i, j] != 0):
                theta[i, j] = np.arctan(Q[i, j] / P[i, j])
            else:
                theta[i, j] = np.pi / 2
    return np.uint8(M), theta


def gradient_cal(img):
    h, w = img.shape
    gradient = np.zeros([h - 1, w - 1])
    direction = np.zeros([h - 1, w - 1])
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            Gx = np.sum(img[i - 1:i + 2, j - 1:j + 2] * gx)
            Gy = np.sum(img[i - 1:i + 2, j - 1:j + 2] * gy)

            gradient[i, j] = np.sqrt(Gx**2 + Gy**2)
            if Gx == 0:
                direction[i, j] = np.pi / 2
            else:
                direction[i, j] = np.arctan(Gy / Gx)
    gradient = np.uint8(gradient)
    return gradient, direction


# In[18]:


# 方向角离散化
def AngleDiscretization(theta):
    '''对梯度方向进行离散,分为四个方向,东、南北、东南、东北

    :image: 输入梯度方向图
    :return theta: 输出离散梯度方向
    '''
    Size = np.shape(theta)
    theta2 = np.zeros(Size)
    for i in range(1, Size[0] - 1):
        for j in range(1, Size[1] - 1):
            if (theta[i, j] >= -np.pi / 8 and theta[i, j] < np.pi / 8):  # x方向
                theta2[i, j] = 0
            elif (theta[i, j] >= np.pi / 8
                  and theta[i, j] < 3 * np.pi / 8):  # 主对角线
                theta2[i, j] = 1
            elif (theta[i, j] >= 3 * np.pi / 8  # y方向
                  or theta[i, j] < -3 * np.pi / 8):
                theta2[i, j] = 2
            else:
                theta2[i, j] = 3  # 次对角线
    return theta2


# 非极大值抑制
def NonMaxSuppression(M, theta2):
    '''通过离散梯度方向图对梯度图像进行非极大值抑制,抑制为0

    :M: 输入图像
    :theta2: 输入离散梯度方向图
    :return M: 输出处理后的图像
    '''
    Size = np.shape(M)
    temp = np.zeros(Size) + 1
    for i in range(1, Size[0] - 1):
        for j in range(1, Size[1] - 1):
            # if (theta2[i, j] == 0):
            #     if (M[i, j] < M[i + 1, j] or M[i, j] < M[i - 1, j]):
            #         temp[i, j] = 0
            # elif (theta2[i, j] == 1):
            #     if (M[i, j] < M[i + 1, j + 1] or M[i, j] < M[i - 1, j - 1]):
            #         temp[i, j] = 0
            # elif (theta2[i, j] == 2):
            #     if (M[i, j] < M[i, j + 1] or M[i, j] < M[i, j - 1]):
            #         temp[i, j] = 0
            # else:
            #     if (M[i, j] < M[i - 1, j + 1] or M[i, j] < M[i + 1, j - 1]):
            #         temp[i, j] = 0
            if (theta2[i, j] == 0):
                if (M[i, j] < M[i, j + 1] or M[i, j] < M[i, j - 1]):
                    temp[i, j] = 0
            elif (theta2[i, j] == 1):
                if (M[i, j] < M[i - 1, j + 1] or M[i, j] < M[i + 1, j - 1]):
                    temp[i, j] = 0
            elif (theta2[i, j] == 2):
                if (M[i, j] < M[i + 1, j] or M[i, j] < M[i - 1, j]):
                    temp[i, j] = 0
            else:
                if (M[i, j] < M[i + 1, j + 1] or M[i, j] < M[i - 1, j - 1]):
                    temp[i, j] = 0

    return np.uint8(temp * M)


# In[19]:


# 双阈值连接边缘
def EdgeJoint(M, highThreshold, lowThreshold):
    '''双阈值判断后连接边缘

    :M: 输入图像
    :highThreshold: 高阈值
    :lowThreshold: 低阈值
    :return M: 输出处理后图像
    '''
    Size = np.shape(M)
    high = np.zeros(np.shape(M))
    low = np.zeros(np.shape(M))
    for i in range(1, Size[0] - 1):  # 判定得到分布图
        for j in range(1, Size[1] - 1):
            if (M[i, j] > highThreshold):
                high[i, j] = 1
            elif (M[i, j] > lowThreshold):
                low[i, j] = 1

    for i in range(1, Size[0] - 1):
        for j in range(1, Size[0] - 1):
            if (high[i, j] == 1 and low[i, j] != 1):  # 遍历所有强边缘，以其为起点进行边缘连接
                [high, low] = Joint(high, low, i, j)
            # if(low[i,j]==1):
            # for k in range(0,3):
            #     for l in range(0,3):
            #         if(high[i-1+k,j-1+l]==1):
            #             high[i,j] = 1
            #             i = 0
            #             j = 0
    return np.uint8(high * M)


def Joint(high, low, i, j):
    '''连接指定点的边缘

    :high: 高阈值分布图
    :low: 低阈值分布图
    :i: 目标点横坐标
    :j: 目标点纵坐标
    :return [high, low]: 高、低阈值分布图
    '''
    Size = np.shape(high)
    if (i != 0 and j != 0 and i != Size[0] - 1 and j != Size[0] - 1):
        for k in range(0, 3):
            for m in range(0, 3):
                if (high[i - 1 + k, j - 1 + m] != 1
                        and low[i - 1 + k, j - 1 + m]
                        == 1):  # 将弱梯度点标记为强梯度，同时对该弱梯度作为新的连接起点
                    high[i - 1 + k, j - 1 + m] = 1
                    [high, low] = Joint(high, low, i - 1 + k, j - 1 + m)
    return [high, low]


# In[20]:


# 边缘检测函数
def EdgeDect(image, highThreshold, lowThreshold):
    '''对输入图像进行canny边缘检测,输出结果图与中间结果

    :image: 输入彩色图像
    :highThreshold: 高阈值
    :lowThreshold: 低阈值
    :return [imageGray, magGradient, magSuppression, magEdgejoint]: 原灰度图像、梯度图像、非极大值抑制后图像、双阈值连接后最终结果图
    '''

    # 图片转灰度
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行高斯模糊处理
    imageGauss = cv2.GaussianBlur(imageGray, (3, 3), 1)

    # 得到梯度幅值与梯度方向
    magGradient, angleGradient = Gradient_Sobel(imageGauss)

    # 梯度方向离散化
    angleDiscretion = AngleDiscretization(angleGradient)

    # 非极大抑制
    magSuppression = NonMaxSuppression(magGradient, angleDiscretion)

    # 双阈值检测连接边缘
    magEdgejoint = EdgeJoint(magSuppression, highThreshold, lowThreshold)

    return [imageGray, magGradient, magSuppression, magEdgejoint]


def mask(maskImage, image):
    '''将边缘检测结果图作为mask覆盖原彩色图像,得到彩色边缘图

    :maskImage: 边缘检测结果图
    :image: 原彩色图像
    '''
    try:
        if (np.shape(maskImage) != np.shape(image)[0:2]):
            print(np.shape(maskImage))
            print(np.shape(image))
            raise ValueError('!图片大小不一致!')
    except Exception as e:
        print(e)
        return

    Size = np.shape(image)
    for i in range(0, Size[0]):
        for j in range(0, Size[1]):
            if (maskImage[i, j] == 0):  # 覆盖即：让右边缘的地方保留，其他地方置0
                for k in range(0, 3):
                    image[i, j, k] = 0
    return np.uint8(image)


# In[24]:

# 导入图片
imagetext = input('the name of picture fire: ')
image0 = cv2.imread(imagetext)

# 获取阈值设定
highThreshold = int(input('input the highThreshold: '))
lowThreshold = int(input('input the lowThreshold: '))

# canny边缘检测
[imageGray, magGradient, magSuppression,
 magEdgejoint] = EdgeDect(image0, highThreshold, lowThreshold)

# 覆盖得到彩色边缘
colorImage = mask(magEdgejoint, image0)

# In[25]:

print(np.shape(magGradient), np.shape(magSuppression), np.shape(magEdgejoint))

images = np.hstack([magGradient, magSuppression, magEdgejoint])
# 显示最终结果
cv2.imshow(imagetext, images)
k = cv2.waitKey(0)
# 按Esc关闭图像
if k == 27:
    cv2.destroyAllWindows()

cv2.imshow(imagetext, colorImage)
k = cv2.waitKey(0)
# 按Esc关闭图像
if k == 27:
    cv2.destroyAllWindows()

# 保存中间结果图
cv2.imwrite('magGradient.png', magGradient, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite('magsuppression.png', magSuppression,
            [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite('magEdgejoint.png', magEdgejoint, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite('colorImage.png', colorImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
