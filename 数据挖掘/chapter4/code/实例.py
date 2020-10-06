# -*- coding: utf-8 -*-

# 代码4-8 求向量D中的单值元素，并返回相关索引
import pandas as pd
import numpy as np
D = pd.Series([1, 1, 2, 3, 5])
D.unique()
np.unique(D)




# 代码 4-9 对一个10×4维的随机矩阵进行主成分分析
from sklearn.decomposition import PCA
D = np.random.rand(10,4)
pca = PCA()
pca.fit(D)
pca.components_  # 返回模型的各个特征向量
pca.explained_variance_ratio_  # 返回各个成分各自的方差百分比






