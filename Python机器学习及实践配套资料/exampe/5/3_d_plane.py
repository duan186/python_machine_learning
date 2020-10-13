import numpy as np
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.linspace(0,10,10), np.linspace(0,100,10))
zz = 1.0 * xx + 3.5 * yy + np.random.randint(0,100,(10,10))
# 构建成特征、值的形式
X, Z = np.column_stack((xx.flatten(),yy.flatten())), zz.flatten()
# 建立线性回归模型
regr = linear_model.LinearRegression()
# 拟合
regr.fit(X, Z)
# 不难得到平面的系数、截距
a, b = regr.coef_, regr.intercept_
# 给出待预测的一个特征
x = np.array([[5.8, 78.3]])
# 方式1：根据线性方程计算待预测的特征x对应的值z（注意：np.sum）
print(np.sum(a * x) + b)
# 方式2：根据predict方法预测的值z
print(regr.predict(x))
# 画图
fig = plt.figure()
ax = fig.gca(projection='3d')
# 1.画出真实的点
ax.scatter(xx, yy, zz)

# 2.画出拟合的平面
ax.plot_wireframe(xx, yy, regr.predict(X).reshape(10,10))
ax.plot_surface(xx, yy, regr.predict(X).reshape(10,10), alpha=0.3)


plt.show()
