"""
岭回归测试代码
这里需要先生成一个线性相关的设计矩阵X，再使用岭回归对其进行建模
岭回归中最重要的就是参数alpha的选择，本例显示了不同的alpha下
模型参数omega不同的结果
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
 
# alpha 取值为10^（-10）到10^（-2）之间的连续的200个值
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
print('\n alpha的值为：')
print(alphas)
 
# 初始化一个Ridge Regression
clf = linear_model.Ridge(fit_intercept=False)
 
# 参数矩阵，即每一个alpha对于的参数所组成的矩阵
coefs = []
# 根据不同的alpha训练出不同的模型参数
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_) 
# 获得绘图句柄
ax = plt.gca()
# 参数中每一个维度使用一个颜色表示
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k'])
 
# 绘制alpha和对应的参数之间的关系图
ax.plot(alphas, coefs)
ax.set_xscale('log')    #x轴使用对数表示
ax.set_xlim(ax.get_xlim()[::-1])  # 将x轴反转，便于显示
plt.grid()
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
