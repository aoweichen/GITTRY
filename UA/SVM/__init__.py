import random

import numpy as np
import pandas as pd
# 辅助函数，用于将label中0变为-1
def changeY(y):
    y=np.array(y)
    for i in range(y.shape[0]):
        if y[i] == 0:
            y[i] =-1
    return y
# 辅助函数,在某一区间范围内随机选择一个整数
def selectJrand(i,m):
    j = i
    while j==i:
        j = int(random.uniform(0,m))
    return j
# 辅助函数,在数值太大时进行调整
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj
"""
简化版SMO
dataMatIn:输入的数据集矩阵
toler:容错率
maxIter:退出当前循环的最大循环次数
"""

def smoSample(dataMatIn,classLabels,C,toler,maxIter):
    dataMattrrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = dataMattrrix.shape
    alphas = np.mat(np.zeros(m))
    iter = 0
    for iter in range(maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas,labelMat).T*\
                        (dataMattrrix*dataMattrrix[i,:].T)) + b
            Ei = fxi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i]*Ei > toler) and (alphas[i] >0 )):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*\
                                       (dataMattrrix*dataMattrrix[j,:].T))+b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j]+alphas[i])
                    H = min(C,C+alphas[j]+alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0*dataMattrrix[i,:]*dataMattrrix[j,:].T-\
                    dataMattrrix[i,:]*dataMattrrix[i,:].T-\
                    dataMattrrix[j,:]*dataMattrrix[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j]-=labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if abs(alphas[j]-alphaJold)<0.00001:
                    print("j not moving enough.")
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1 = b-Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMattrrix[i,:]*dataMattrrix[i,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*dataMattrrix[i,:]*dataMattrrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMattrrix[i,:]*dataMattrrix[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*dataMattrrix[j,:]*dataMattrrix[j,:].T
                if 0<alphas[i] and C>alphas[i]:
                    b = b1
                elif alphas[j]<0 and alphas[j]<C:
                    b=b2
                else:b = (b1+b2)/2.0
                alphaPairsChanged+=1
                print("iter:%d i:%d,pairs changed %d" %(iter,i,alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter+=1
        else:
            iter = 0
            print("iteration number:%d"%iter)
    return b,alphas
data = pd.read_csv('../Bayes/breast_cancer.csv', header=None)
x_train, y_train,x_test,y_test = np.array(data.iloc[:400, :-1]), np.array(data.iloc[:400, -1]),\
                                 np.array(data.iloc[400:, :-1]), np.array(data.iloc[400:, -1])

b,al = smoSample(x_train,changeY(y_train),0.6,0.001,40)
