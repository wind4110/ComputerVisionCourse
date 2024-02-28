#!/usr/bin/env python
# coding: utf-8
# writer: 许展风 email: zhanfeng_xu@outlook.com

# In[44]:

import cv2
import numpy as np

# In[45]:


def removeBlack(blackim):
    '''去除图像黑边

    :blackim: 输入图像
    :res_image: 输出图像
    '''
    blackimgGray = cv2.cvtColor(blackim, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    edges_y, edges_x = np.where(blackimgGray != 0)  # 求非黑的有效区域
    bottom = min(edges_y)
    top = max(edges_y)
    height = top - bottom  # 求有效区域的最大高度

    left = min(edges_x)
    right = max(edges_x)
    width = right - left  # 求有效区域的最小宽度

    res_image = blackim[bottom:bottom + height, left:left + width]  # 裁剪出有效区域
    return res_image


# In[47]:


def imgStitching(imga, imgb, outputName):
    '''拼接两张图片,imgb不变,透视变换imga.输出最终结果与中间过程

    :imga: 拼接图片1
    :imgb: 拼接图片2
    :outputName: 输出的拼接图像名
    :Res: 输出图片
    '''

    # 由于透视变换将imga图片进行平移、旋转等等操作，所以需要扩大图像画布，避免信息丢失
    tw = np.int16(np.max([imga.shape[1], imgb.shape[1]]))  # 确定宽度平移量
    th = np.int16(np.max([imga.shape[0], imgb.shape[0]]))  # 确定高度平移量

    M = np.float32([[1, 0, tw], [0, 1, th]])  # 构造平移变换矩阵
    img1 = cv2.warpAffine(imga, M,
                          (imga.shape[1] + 2 * tw,
                           imga.shape[0] + 2 * th))  # 变换后，保证了图像的最大尺度变换下信息不丢失
    img2 = cv2.warpAffine(imgb, M,
                          (imga.shape[1] + 2 * tw,
                           imga.shape[0] + 2 * th))  # 对imgb作同样处理，便于后续图片直接原位置拼接

    imgGray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 求灰度图像
    imgGray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # sift特征点计算
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgGray1, None)
    kp2, des2 = sift.detectAndCompute(imgGray2, None)

    # 绘制特征点计算图片
    imgkp1 = img1.copy()
    imgkp2 = img2.copy()
    imgkp1 = cv2.drawKeypoints(imgGray1, kp1, imgkp1)
    imgkp2 = cv2.drawKeypoints(imgGray2, kp2, imgkp2)
    imgkpRemove1 = removeBlack(imgkp1)
    imgkpRemove2 = removeBlack(imgkp2)
    cv2.imwrite('Sift' + outputName + 'a.jpg', imgkpRemove1)
    cv2.imwrite('Sift' + outputName + 'b.jpg', imgkpRemove2)

    # 对应特征点配对
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    goodMatch = []  # 配对点集合，用于画图
    good = []  # 配对点坐标序号集合， 用于后续求变换矩阵
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatch.append(m)
            good.append((m.trainIdx, m.queryIdx))

    goodMatch = np.expand_dims(goodMatch, 1)  # 扩展维度，用于画图

    imMatch = cv2.drawMatchesKnn(imgkp1,
                                 kp1,
                                 imgkp2,
                                 kp2,
                                 goodMatch[:100],
                                 None,
                                 flags=2)  # 绘制前100个匹配点
    imMatch = removeBlack(imMatch)
    cv2.imwrite('Match' + outputName + '.jpg', imMatch)

    # 求变换矩阵
    pts1 = np.float32([kp1[i].pt for (_, i) in good])
    pts2 = np.float32([kp2[i].pt for (i, _) in good])
    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    # 用变换矩阵对imga作透视变换
    tranRes = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    # 透视变换后的图片拼接上imgb
    tranRes[th:imgb.shape[0] + th, tw:imgb.shape[1] + tw] = imgb
    # 去除图像多余黑边
    Res = removeBlack(tranRes)
    cv2.imwrite('Res' + outputName + '.jpg', Res)

    return Res


# In[48]:

# 输入接口
N = int(input("输入需要拼接的图片总数：\n"))
image = [[], []] * N
print("依次输入图片路径:")
for i in range(0, N):
    imageName = input()
    image[i] = cv2.imread(imageName)
name = input("输入输出图片名:\n")


# 为了使得中间图片能拥有不变的视角，即以N//2+1这一张为主视角进行变换拼接
# 调整两两拼接的顺序
temp1 = image[0]
for i in range(1, N // 2 + 1):
    temp1 = imgStitching(temp1, image[i], name)

temp2 = image[N - 1]
for i in range(N - 2, N // 2, -1):
    temp2 = imgStitching(temp2, image[i], name)

# 最后一次拼接直接保存结果图片
imgStitching(temp2, temp1, name)
