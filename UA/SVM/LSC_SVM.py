# maxIter:最大迭代次数
# alphas:svm参数
from random import random, uniform

import numpy as np
# 决策函数
import pandas as pd


class SMO:
    def __init__(self,):
        pass

def f(dataSetX,dataSetY,alphas,b,i):
    w = np.multiply(alphas,dataSetY).T
    x =(dataSetX*(dataSetX[i,:].T)).T
    f = float(x.dot(w).sum())+b
    return f
#
def ff(x_test,dataSetX,dataSetY,alphas,b,i):
    w = np.multiply(alphas,dataSetY).T
    x =(dataSetX*(x_test[i,:].T)).T
    print(x)
    f = float(x.dot(w).sum())+b
    return f
# 随机选择第二个alpha
def selectJrand(i,m):
    j = i
    while j == i:
        j = int(uniform(0,m))
    return j
# 更新alpha中的eta
def ETA(dataSetX,i,j):
    Kii= (dataSetX[i, :] * (dataSetX[i, :].T)).sum()
    Kjj= (dataSetX[j, :] * (dataSetX[j, :].T)).sum()
    Kij =(dataSetX[i, :] *(dataSetX[j, :].T)).sum()
    return (Kii+Kjj-2.0*Kij)
# 得到L,H
def LH(dataSetY,alphas,C,i,j):
    if (dataSetY[i] != dataSetY[j]):
        L = max(0, alphas[j] + alphas[i])
        H = min(C, C + alphas[j] + alphas[i])
    else:
        L = max(0, alphas[j] + alphas[i] - C)
        H = min(C, alphas[j] + alphas[i])
    return L,H
# 得到剪辑后的alpha[j]
def clipAlpha(alp,H,L):
    if alp > H:
        alp = H
    if L > alp:
        alp = L
    return alp
# 核函数
def K(dataSetX,i,j):
    return (dataSetX[i,:]*dataSetX[j,:].T).sum()

def WSMOS(alpha1,alpha2):
    pass

def SVMSMOS(dataSetX,dataSetY,C,toler,maxIter):# toler:容错率
    # 将训练集和标签转化为矩阵形式
    X, Y = np.array(dataSetX),np.array(dataSetY).transpose()
    m, n = X.shape # m表示数据量
    b = 0
    alphas =np.zeros(m)
    iter = 0
    t=0
    while iter < maxIter and t<maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            t+=1
            fxi = f(X,Y,alphas,b,i)
            Ei = fxi - float(Y[i])
            if ((Y[i]*Ei <-toler) and (alphas[i]<C)) or ((int(Y[i])*Ei >toler) and (alphas[i]>C)):
                j = selectJrand(i,m)
                fxj = f(X,Y,alphas,b,j)
                Ej = fxj - float(Y[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                L,H=LH(Y,alphas,C,i,j)
                if L == H :
                    print("L==H")
                #     t+=1
                #     continue
                # if t >= 20:
                #     break
                eta = ETA(X,i,j)
                if eta < 0:
                    print("eta<0!")
                    continue
                alphas[j]+=Y[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if abs(alphas[j]-alphaJold)<=0:
                    print("move not enough!")
                    continue
                alphas[i] = alphaIold+Y[i]*Y[j]*(alphaJold-alphas[j])
                b1=b-Ei-Y[i]*(alphas[i]-alphaIold)*K(X,i,i)-Y[j]*(alphas[j]-alphaJold)*K(X,j,i)
                b2=b-Ei-Y[i]*(alphas[i]-alphaIold)*K(X,i,i)-Y[j]*(alphas[j]-alphaJold)*K(X,j,i)
                if alphas[i]>0 and alphas[i]<C:
                    b=b1
                elif alphas[j]>0 and alphas[j]<C:
                    b = b2
                else:
                    b = (b1+b2)/2
                alphaPairsChanged+=1
                if alphas.all()>0:
                    print("True")
        if alphaPairsChanged==0:
            iter+=1
        else:iter=0
    return b,alphas
# 求系数w
def getW(alphas,dataSetY,dataSetX):
    w=np.zeros(dataSetX[0].shape[0])
    for i in range(dataSetY.shape[0]):
        w+=np.multiply(alphas[i]*dataSetY[0],dataSetX[i])
    return w
#
def pre(w,b,x_test):
    y_pre = w.dot(x_test.T)+b
    for i in range(y_pre.shape[0]):
        if y_pre[i]>=0:
            y_pre[i]=1
        else:
            y_pre[i]=0
    return y_pre


def prefect(X_test,dataSetX,dataSetY,b,alphas):
    w=[]
    y_pre = []
    for i in range(dataSetY.shape[0]):
        w.append(ff(X_test,dataSetX,dataSetY,alphas,b,i))
    for i in range(X_test.shape[0]):
        F = w[i]
        if F>0:
            y_pre.append(1)
        else:
            y_pre.append(-1)
    return y_pre


data = pd.read_csv('../Bayes/breast_cancer.csv', header=None)
x_train, y_train,x_test,y_test = np.array(data.iloc[:400, :-1]), np.array(data.iloc[:400, -1]),\
                                 np.array(data.iloc[400:, :-1]), np.array(data.iloc[400:, -1])

for i in range(x_train.shape[0]):
    if y_train[i] ==0:
        y_train[i] =-1
for i in range(x_test.shape[0]):
    if y_test[i] ==0:
        y_test[i] =-1

b,alp=SVMSMOS(x_train,y_train,1.0,0.1,3)
w=getW(alp,y_train,x_train)
y_pre=pre(w,b,x_test)
print(w.dot(x_test[0]))
print(x_test[0])
print(w)


# num2 =0.0
# num1 =0.0
# # 求正确率，二分类
# for i in range(y_test.shape[0]):
#     num1+=1
#     if aa[i]==y_test[i]:
#         num2+=1
# print(num2/num1)






