import numpy as np
import cv2 as cv
import sys  # 导入sys模块

sys.setrecursionlimit(10000)  # 将默认的递归深度修改为3000


# 计算梯度
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


# 非极大值抑制
def NMS(gradient, direction):
    h, w = gradient.shape
    nms = np.copy(gradient)
    for i in range(h):
        for j in range(w):
            if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                nms[i, j] = 0
                continue
            theta = direction[i, j]
            if theta > np.pi * 3 / 8 or theta < -np.pi * 3 / 8:
                if gradient[i, j] <= gradient[i - 1, j] or gradient[
                        i, j] <= gradient[i + 1, j]:
                    nms[i, j] = 0
            elif np.pi / 8 < theta < np.pi * 3 / 8:
                if gradient[i, j] <= gradient[i - 1, j + 1] or gradient[
                        i, j] <= gradient[i + 1, j - 1]:
                    nms[i, j] = 0
            elif -np.pi / 8 < theta < np.pi / 8:
                if gradient[i, j] <= gradient[i, j + 1] or gradient[
                        i, j] <= gradient[i, j - 1]:
                    nms[i, j] = 0
            elif -np.pi * 3 / 8 < theta < -np.pi / 8:
                if gradient[i, j] <= gradient[i + 1, j + 1] or gradient[
                        i, j] <= gradient[i - 1, j - 1]:
                    nms[i, j] = 0
            else:
                nms[i, j] = 0
    nms = np.uint8(nms)
    return nms


# 双阈值
def double_threshold(nms, low_threshold, high_threshold):
    output = np.copy(nms)
    visit = np.zeros_like(nms)

    # 边缘连接
    def edge_link(i, j):
        if i >= h or i < 0 or j >= w or j < 0 or visit[i, j] == 1:
            return
        visit[i, j] = 1
        if output[i, j] > low_threshold:
            output[i, j] = 255
            for x in range(i - 1, i + 2):
                for y in range(j - 1, j + 2):
                    edge_link(x, y)

    h, w = output.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if visit[i, j] == 1:
                continue
            if output[i, j] > high_threshold:
                edge_link(i, j)
            elif output[i, j] <= low_threshold:
                output[i, j] = 0
                visit[i, j] = 1
    for i in range(h):
        for j in range(w):
            if visit[i, j] == 0:
                output[i, j] = 0
    return output


image = cv.imread("lena.jpg", 0)
image_filter = cv.GaussianBlur(image, (5, 5), 0)
gradient, direction = gradient_cal(image_filter)
# nms = NMS(gradient, direction)
# out = double_threshold(nms, 50, 90)

# 显示结果
# cv.imshow('Original Image', image)
# cv.imshow('Gaussian Filter', image_filter)
# cv.imshow('Gradient', gradient)
cv.imshow('NMS', gradient)
# cv.imshow('out', out)
cv.waitKey(0)
# cv.destroyAllWindows()
