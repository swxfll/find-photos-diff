import cv2
import numpy as np

# 读取两张图片
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# 将两张图片调整为相同的大小
resized_img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

# 计算两张图片的差异
diff = cv2.absdiff(resized_img1, img2)

# 将差异图像转换为灰度图像
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# 对灰度图像进行高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 对灰度图像进行二值化处理
thresh = cv2.threshold(blurred, 49, 255, cv2.THRESH_BINARY)[1]

# 对二值图像进行形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(thresh, kernel, iterations=3)

# 找到二值图像中不同的像素位置
coords = np.column_stack(np.where(dilated > 0))

# 在原始图像中标记不同的像素位置
for coord in coords:
    cv2.circle(img2, (coord[1], coord[0]), 5, (0, 0, 255), -1)

# 显示标记不同像素后的图像
cv2.imshow('Result', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()