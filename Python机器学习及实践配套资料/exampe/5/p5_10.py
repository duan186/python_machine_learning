"""
基于交叉验证的岭回归alpha选择
可以直接获得一个相对不错的alpha
""" 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
 
# 这里设计矩阵X是一个希尔伯特矩阵（Hilbert matrix）
# 其元素A（i,j）=1(i + j -1),i和j分别为其行标和列标
# 希尔伯特矩阵是一种数学变换矩阵，正定，且高度病态
# 任何一个元素发生一点变动，整个矩阵的行列式的值和逆矩阵都会发生巨大变化
# 这里设计矩阵是一个10x5的矩阵，即有10个样本，5个变量
X = 1. / (np.arange(1, 6) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10) 
print('设计矩阵为：')
print(X)
 
# 初始化一个Ridge Cross-Validation Regression
clf = linear_model.RidgeCV(fit_intercept=False) 
# 训练模型
clf.fit(X, y) 
print
print('alpha的数值 : ', clf.alpha_)
print('参数的数值：', clf.coef_)
