# -*- coding: utf-8 -*-

# 代码4-6 主成分分析降维
import pandas as pd

# 参数初始化
inputfile = '../data/principal_component.xls'
outputfile = '../tmp/dimention_reducted.xls'  # 降维后的数据

data = pd.read_excel(inputfile, header = None)  # 读入数据

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(data)

pca.components_  # 返回模型的各个特征向量


pca.explained_variance_ratio_  # 返回各个成分各自的方差百分比




# 代码4-7 计算成分结果
pca = PCA(3)
pca.fit(data)
low_d = pca.transform(data)  # 用它来降低维度
pd.DataFrame(low_d).to_excel(outputfile)  # 保存结果
pca.inverse_transform(low_d)  # 必要时可以用inverse_transform()函数来复原数据

low_d







