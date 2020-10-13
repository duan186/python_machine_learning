# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.manifold import TSNE
# 参数初始化
inputfile = '../data/consumption_data.xls'  # 销量及其他属性数据
inputfile1 = '../tmp/data_type.xls'  
data = pd.read_excel(inputfile, index_col = 'Id')  # 读取数据
data_zs = 1.0*(data - data.mean())/data.std() 
r = pd.read_excel(inputfile1,index_col='Id')

tsne = TSNE(random_state=105)
tsne.fit_transform(data_zs)  # 进行数据降维
tsne = pd.DataFrame(tsne.embedding_, index = data_zs.index)  # 转换数据格式

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 不同类别用不同颜色和样式绘图
d = tsne[r['聚类类别'] == 0]
plt.plot(d[0], d[1], 'r.')
d = tsne[r['聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')
d = tsne[r['聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
plt.show()


