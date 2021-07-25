import math
import numpy as np
from PIL import Image


class MyGaussianBlur():
    # 初始化
    def __init__(self, radius=1, sigema=1.5):
        self.radius = radius
        self.sigema = sigema
        # 高斯的计算公式

    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
        return res1 * res2

    # 得到滤波模版
    def template(self):
        sideLength = self.radius * 2 + 1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all = result.sum()
        return result / all
        # 滤波函数

    def filter(self, image, template):
        arr = np.array(image)
        height = arr.shape[0]
        width = arr.shape[1]
        newData = np.zeros((height, width, 3))
        for i in range(self.radius, height - self.radius):
            for j in range(self.radius, width - self.radius):
                t = arr[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]
                r = np.multiply(t[..., 0], template)        # 获得三维矩阵中R通道
                g = np.multiply(t[..., 1], template)      # 获得三维矩阵中G通道
                b = np.multiply(t[..., 2], template)        # 获得三维矩阵中B通道
                newData[i, j] = (r.sum(), g.sum(), b.sum())
        newImage = Image.fromarray(newData.astype('uint8')).convert('RGB')  # 将灰度图像转为彩色图像
        return newImage


r = 3                                      # 模版半径，自己自由调整
s = 2                                       # sigema数值，自己自由调整
GBlur = MyGaussianBlur(radius=r, sigema=s)  # 声明高斯模糊类
temp = GBlur.template()                     # 得到滤波模版
im = Image.open('./pics/1.jpg')             # 打开图片
im.show()
image = GBlur.filter(im, temp)              # 高斯模糊滤波，得到新的图片
image.show()


