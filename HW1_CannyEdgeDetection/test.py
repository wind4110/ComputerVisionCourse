import cv2

image0 = cv2.imread('lena.jpg')
cv2.imshow('NMS', image0)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
