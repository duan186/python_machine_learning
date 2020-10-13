import numpy as np 
import matplotlib.pyplot as plt
from tkinter import _flatten
 
x_arange = 0.041 * np.arange(0, 25, 1) #每组数据的25个点
y_True = np.sin(2 * np.pi * x_arange)  #每个数据点对应的值（没有添加噪声）
y_Noise = np.zeros(y_True.shape)       #添加噪声的值
x_Prec = np.linspace(0, 24*0.041, 100) #画图范围
 
mu = 0  #噪声的mu值
sigma = 0.3  #噪声的sigma值
Num = 100  #100组数据集
n = 8  #7阶多项式
lamda = [np.exp(1), np.exp(0), np.exp(-5), np.exp(-10)]  #不同的lambda值
phi = np.mat(np.zeros((x_arange.size, n)))  #phi矩阵
x = np.mat(x_arange).T  #输入数据矩阵
 
#phi矩阵运算
for i_n in range(n):
    for y_n in range(x_arange.size):
        phi[y_n, i_n] = x[y_n, 0] ** i_n
 
plt.figure(figsize=(15, 10))
index = 221
for i_lamda in lamda:
    plt.subplot(index)
    index += 1
    plt.title("lambda = %f" % i_lamda)
    plt.plot(x_Prec, np.sin(2 * np.pi * x_Prec), color='g')
    for k in range(Num):
        for i in range(x_arange.size):
            y_Noise[i] = y_True[i] + np.random.normal(mu, sigma)
        y = np.mat(y_Noise).T
        #求解w参数
        W = (phi.T * phi + i_lamda*np.eye(n)).I * phi.T * y
    
        ploy = list(_flatten(W.T.tolist()))
        ploy.reverse()
        p = np.poly1d(ploy)
        if k%5==0:  #只画20条曲线
            plt.plot(x_Prec, p(x_Prec), color='r')
plt.show()
