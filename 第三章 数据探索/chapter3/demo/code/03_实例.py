# -*- coding: utf-8 -*-

# 代码3-10 计算两个列向量的相关系数
import pandas as pd
D = pd.DataFrame([range(1, 8), range(2, 9)])  # 生成样本D，一行为1~7，一行为2~8
print(D.corr(method='spearman'))  # 计算相关系数矩阵
S1 = D.loc[0]  # 提取第一行
S2 = D.loc[1]  # 提取第二行
print(S1.corr(S2, method='pearson'))  # 计算S1、S2的相关系数


# 代码3-11 计算6×5随机矩阵的协方差矩阵

import numpy as np
D = pd.DataFrame(np.random.randn(6, 5))  # 产生6×5随机矩阵
print(D.cov())  # 计算协方差矩阵
print(D[0].cov(D[1]))  # 计算第一列和第二列的协方差


# 代码3-12 计算6×5随机矩阵的偏度（三阶矩）∕峰度（四阶矩）
import numpy as np
D = pd.DataFrame(np.random.randn(6, 5))  # 产生6×5随机矩阵
print(D.skew())  # 计算偏度
print(D.kurt())  # 计算峰度


# 代码3-13 6×5随机矩阵的describe

import numpy as np
D = pd.DataFrame(np.random.randn(6, 5))  # 产生6×5随机矩阵
print(D.describe())


# 代码3-14 pandas累积统计特征函数、移动窗口统计函数示例


D=pd.Series(range(0, 20))  # 构造Series，内容为0~19共20个整数
print(D.cumsum())  # 给出前n项和
print(D.rolling(2).sum())  # 依次对相邻两项求和


# 代码3-15 绘图之前需要加载的代码
import matplotlib.pyplot as plt  # 导入绘图库
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize = (7, 5))  # 创建图像区域，指定比例


# 代码3-16 绘制一条蓝色的正弦虚线

import numpy as np
x = np.linspace(0,2*np.pi,50)  # x坐标输入
y = np.sin(x)  # 计算对应x的正弦值
plt.plot(x, y, 'bp--')  # 控制图形格式为蓝色带星虚线，显示正弦曲线
plt.show()


# 代码3-17 绘制饼图


import matplotlib.pyplot as plt

# The slices will be ordered and plotted counter-clockwise.
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'  # 定义标签
sizes = [15, 30, 45, 10]  # 每一块的比例
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']  # 每一块的颜色
explode = (0, 0.1, 0, 0)  # 突出显示，这里仅仅突出显示第二块（即'Hogs'）

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # 显示为圆（避免比例压缩为椭圆）
plt.show()


# 代码3-18 绘制二维条形直方图

import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(1000)  # 1000个服从正态分布的随机数
plt.hist(x, 10)  # 分成10组进行绘制直方图
plt.show()


# 代码3-19 绘制箱型图


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x = np.random.randn(1000)  # 1000个服从正态分布的随机数
D = pd.DataFrame([x, x+1]).T  # 构造两列的DataFrame
D.plot(kind = 'box')  # 调用Series内置的绘图方法画图，用kind参数指定箱型图box
plt.show()


# 代码3-20 使用plot(logy = True)函数进行绘图


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import numpy as np
import pandas as pd

x = pd.Series(np.exp(np.arange(20)))  # 原始数据
plt.figure(figsize = (8, 9))  # 设置画布大小 
ax1 = plt.subplot(2, 1, 1)
x.plot(label = u'原始数据图', legend = True)

ax1 = plt.subplot(2, 1, 2)
x.plot(logy = True, label = u'对数数据图', legend = True)
plt.show()


# 代码3-21 绘制误差棒图


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import numpy as np
import pandas as pd

error = np.random.randn(10)  # 定义误差列
y = pd.Series(np.sin(np.arange(10)))  # 均值数据列
y.plot(yerr = error)  # 绘制误差图
plt.show()




