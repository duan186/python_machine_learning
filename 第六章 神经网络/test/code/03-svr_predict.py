#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR

inputfile = '../tmp/new_reg_data_GM11.xls'  # 灰色预测后保存的路径
data = pd.read_excel(inputfile)  # 读取数据
feature = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']  # 属性所在列
data_train = data.loc[range(1994,2014)].copy()  # 取2014年前的数据建模
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std  # 数据标准化
x_train = data_train[feature].as_matrix()  # 属性数据
y_train = data_train['y'].as_matrix()  # 标签数据

linearsvr = LinearSVR()  # 调用LinearSVR()函数
linearsvr.fit(x_train,y_train)
x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()  # 预测，并还原结果。
data[u'y_pred'] = linearsvr.predict(x) * data_std['y'] + data_mean['y']
outputfile = '../tmp/new_reg_data_GM11_revenue.xls'  # SVR预测后保存的结果
data.to_excel(outputfile)

print('真实值与预测值分别为：\n',data[['y','y_pred']])

fig = data[['y','y_pred']].plot(subplots = True, style=['b-o','r-*'])  # 画出预测结果图
plt.show()