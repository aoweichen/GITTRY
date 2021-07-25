import matplotlib.image as img
from PIL import Image
filename = './pics/0.jpg'
# 将图像转化为np矩阵
imgArray = img.imread(filename)
print(imgArray)
# 将np矩阵转换为图像的格式
imge = Image.fromarray(imgArray)
imge.show()