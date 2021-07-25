# 导入要用的库
import matplotlib.image as img
import numpy as np
from PIL import Image


class ImageDataProcessing:
    """
        图像数据处理
        传入地址list，从前往后一次对图片进行不同处理
    """
    def __init__(self, path):
        self.path = path
    # T1
    def rgb2hsl(self):
        # 实现rgb到hsl的转换功能
        # 待填, 对self.path[0]操作
        # 请分别打印self.path[0]的rgb和hsl像素值
        # 辅助函数,将一个RGB格式的像素点转化为一个HSL格式的像素点
        def rgbToHsl(R, G, B):
            h, s, l = 0.0, 0.0, 0.0
            r, g, b = float(R) / 255.0, float(G) / 255.0, float(B) / 255.0
            maxz = max([r, g, b])
            minz = min([r, g, b])
            # 求h
            if maxz == minz:
                h = 0.0
            elif maxz == r and g >= b:
                h = 60 * (g - b) / (maxz - minz)
            elif maxz == r and g < b:
                h = 60 * (g - b) / (maxz - minz) + 360.0
            elif maxz == g:
                h = 60 * (b - r) / (maxz - minz) + 120.0
            elif maxz == b:
                h = 60 * (r - g) / (maxz - minz) + 240.0
            # 求l
            l = (maxz + minz) / 2.0
            # 求s
            if l == 0.0 or maxz == minz:
                s = 0.0
            elif l > 0.0 and l <= 0.5:
                s = (maxz - minz) / (2 * l)
            elif l > 0.5:
                s = (maxz - minz) / (2 - 2 * l)
            return h, s, l
        imgArray = img.imread(self.path[0])
        HSL = []
        for i in range(imgArray.shape[0]):
            HSL.append([])
            for j in range(imgArray.shape[1]):
                h, s, l = rgbToHsl(imgArray[i][j][0], imgArray[i][j][1], imgArray[i][j][2])
                HSL[i].append([h, s, l])
        HSL = np.array(HSL)
        print("对第一个图像(path[0])的操作")
        print("RGB:")
        print(imgArray.shape)
        print(imgArray)
        print("HSL:")
        print(HSL.shape)
        print(HSL)
        return HSL
    # T2
    def vague(self):
        # 实现将图片变模糊功能
        # 待填，对self.path[1]进行操作
        # 请分别打印self.path[1]的模糊前和模糊后的图像
        pass
    # T3
    def noise_reduction(self):
        # 实现将图片降噪功能
        # 待填，对self.path[2]进行操作
        # 请分别打印self.path[2]的降噪前和降噪后的图像
        pass
    # T4
    def edge_extraction(self):
        # 实现边缘提取功能
        # 待填，对self.path[3]进行操作
        # 请分别打印self.path[3]的边缘提取前后的图像
        pass
    # T5
    def brightness_adjustment(self):
        # 实现亮度调整功能
        # 待填，对self.path[4]进行操作
        # 请分别打印self.path[4]的原图，变亮后图像，变暗后图像
        pass
    # T6
    def rotate(self):
        # 实现旋转功能
        # 待填，对self.path[5]进行操作
        # 请分别打印self.path[5]的原图，旋转任意角度后图像
        pass
    # T7
    def flip_horizontally(self):
        # 实现水平翻转功能
        # 待填，对self.path[6]进行操作
        # 请分别打印self.path[6]的原图，水平翻转后图像
        pass
    # T8
    def cutting(self):
        # 实现裁切功能
        # 待填，对self.path[7]进行操作
        # 请分别打印self.path[7]的原图，裁切后图像
        pass
    # T9
    def resize(self):
        # 实现调整大小功能
        # 待填，对self.path[8]进行操作
        # 请分别打印self.path[8]的原图，调整任意大小后图像
        pass
    # T10
    def normalization(self):
        # 实现归一化功能
        # 待填，对self.path[9]进行操作
        # 请分别打印self.path[9]的原图，归一化后图像
        pass

    def fit(self):
        self.rgb2hsl()
        self.vague()
        self.noise_reduction()
        self.edge_extraction()
        self.brightness_adjustment()
        self.rotate()
        self.flip_horizontally()
        self.cutting()
        self.resize()
        self.normalization()


if __name__ == '__main__':
    ImageDataProcessing(["pics/0.jpg", "pics/3.jpg", "pics/2.jpg", "pics/8.jpg", "pics/9.jpg", "pics/1.jpg",
                         "pics/7.jpg", "pics/6.jpg", "pics/5.jpg", "pics/4.jpg"]).fit()
