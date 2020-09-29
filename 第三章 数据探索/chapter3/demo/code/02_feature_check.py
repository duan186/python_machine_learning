# -*- coding: utf-8 -*-

# 代码3-3 捞起生鱼片的季度销售情况
import pandas as pd
import numpy as np
catering_sale = '../data/catering_fish_congee.xls'  # 餐饮数据
data = pd.read_excel(catering_sale,names=['date','sale'])  # 读取数据，指定“日期”列为索引

bins = [0,500,1000,1500,2000,2500,3000,3500,4000]
labels = ['[0,500)','[500,1000)','[1000,1500)','[1500,2000)',
       '[2000,2500)','[2500,3000)','[3000,3500)','[3500,4000)'] 

data['sale分层'] = pd.cut(data.sale, bins, labels=labels)
aggResult = data.groupby(by=['sale分层'])['sale'].agg({'sale': np.size})

pAggResult = round(aggResult/aggResult.sum(), 2, ) * 100

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))  # 设置图框大小尺寸
pAggResult['sale'].plot(kind='bar',width=0.8,fontsize=10)  # 绘制频率直方图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.title('季度销售额频率分布直方图',fontsize=20)
plt.show()



# 代码3-4 不同菜品在某段时间的销售量的分布情况
import pandas as pd
import matplotlib.pyplot as plt
catering_dish_profit = '../data/catering_dish_profit.xls'  # 餐饮数据
data = pd.read_excel(catering_dish_profit)  # 读取数据，指定“日期”列为索引

# 绘制饼图
x = data['盈利']
labels = data['菜品名']
plt.figure(figsize = (8, 6))  # 设置画布大小
plt.pie(x,labels=labels)  # 绘制饼图
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title('菜品销售量分布（饼图）')  # 设置标题
plt.axis('equal')
plt.show()

# 绘制条形图
x = data['菜品名']
y = data['盈利']
plt.figure(figsize = (8, 4))  # 设置画布大小
plt.bar(x,y)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.xlabel('菜品')  # 设置x轴标题
plt.ylabel('销量')  # 设置y轴标题
plt.title('菜品销售量分布（条形图）')  # 设置标题
plt.show()  # 展示图片



# 代码3-5 不同部门在各月份的销售对比情况
# 部门之间销售金额比较
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_excel("../data/dish_sale.xls")
plt.figure(figsize=(8, 4))
plt.plot(data['月份'], data['A部门'], color='green', label='A部门',marker='o')
plt.plot(data['月份'], data['B部门'], color='red', label='B部门',marker='s')
plt.plot(data['月份'], data['C部门'],  color='skyblue', label='C部门',marker='x')
plt.legend() # 显示图例
plt.ylabel('销售额（万元）')
plt.show()


#  B部门各年份之间销售金额的比较
data=pd.read_excel("../data/dish_sale_b.xls")
plt.figure(figsize=(8, 4))
plt.plot(data['月份'], data['2012年'], color='green', label='2012年',marker='o')
plt.plot(data['月份'], data['2013年'], color='red', label='2013年',marker='s')
plt.plot(data['月份'], data['2014年'],  color='skyblue', label='2014年',marker='x')
plt.legend() # 显示图例
plt.ylabel('销售额（万元）')
plt.show()


# 代码3-6 餐饮销量数据统计量分析

# 餐饮销量数据统计量分析
import pandas as pd

catering_sale = '../data/catering_sale.xls'  # 餐饮数据
data = pd.read_excel(catering_sale, index_col = u'日期')  # 读取数据，指定“日期”列为索引列
data = data[(data[u'销量'] > 400)&(data[u'销量'] < 5000)]  # 过滤异常数据
statistics = data.describe()  # 保存基本统计量

statistics.loc['range'] = statistics.loc['max']-statistics.loc['min']  # 极差
statistics.loc['var'] = statistics.loc['std']/statistics.loc['mean']  # 变异系数
statistics.loc['dis'] = statistics.loc['75%']-statistics.loc['25%']  # 四分位数间距

print(statistics)


# 代码3-7 某单位日用电量预测分析

import pandas as pd
import matplotlib.pyplot as plt

df_normal = pd.read_csv("../data/user.csv")
plt.figure(figsize=(8,4))
plt.plot(df_normal["Date"],df_normal["Eletricity"])
plt.xlabel("日期")
plt.ylabel("每日电量")
# 设置x轴刻度间隔
x_major_locator = plt.MultipleLocator(7)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.title("正常用户电量趋势")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.show()  # 展示图片

# 窃电用户用电趋势分析
df_steal = pd.read_csv("../data/Steal user.csv")
plt.figure(figsize=(10, 9))
plt.plot(df_steal["Date"],df_steal["Eletricity"])
plt.xlabel("日期")
plt.ylabel("日期")
# 设置x轴刻度间隔
x_major_locator = plt.MultipleLocator(7)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.title("窃电用户电量趋势")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.show()  # 展示图片


# 代码3-8 菜品盈利帕累托图

# 菜品盈利数据 帕累托图
import pandas as pd

# 初始化参数
dish_profit = '../data/catering_dish_profit.xls'  # 餐饮菜品盈利数据
data = pd.read_excel(dish_profit, index_col = u'菜品名')
data = data[u'盈利'].copy()
data.sort_values(ascending = False)

import matplotlib.pyplot as plt  # 导入图像库
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.figure()
data.plot(kind='bar')
plt.ylabel(u'盈利（元）')
p = 1.0*data.cumsum()/data.sum()
p.plot(color = 'r', secondary_y = True, style = '-o',linewidth = 2)
plt.annotate(format(p[6], '.4%'), xy = (6, p[6]), xytext=(6*0.9, p[6]*0.9), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))  # 添加注释，即85%处的标记。这里包括了指定箭头样式。
plt.ylabel(u'盈利（比例）')
plt.show()


# 代码3-9 餐饮销量数据相关性分析

# 餐饮销量数据相关性分析
from __future__ import print_function
import pandas as pd

catering_sale = '../data/catering_sale_all.xls'  # 餐饮数据，含有其他属性
data = pd.read_excel(catering_sale, index_col = u'日期')  # 读取数据，指定“日期”列为索引列

print(data.corr())  # 相关系数矩阵，即给出了任意两款菜式之间的相关系数
print(data.corr()[u'百合酱蒸凤爪'])  # 只显示“百合酱蒸凤爪”与其他菜式的相关系数
# 计算“百合酱蒸凤爪”与“翡翠蒸香茜饺”的相关系数
print(data[u'百合酱蒸凤爪'].corr(data[u'翡翠蒸香茜饺']))







