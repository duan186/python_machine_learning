import pandas as pd
import numpy as np
datafile ='/Users/wangduan/python_machine_learning/数据挖掘/data/normalization_data.xls' # 参数初始化
data = pd.read_excel(datafile, header = None)  # 读取数据
print(data)
 # 最小-最大规范化
print((data - data.min()) / (data.max() - data.min()))
# 零-均值规范化
print((data - data.mean()) / data.std())
# 小数定标规范化
print(data / 10 ** np.ceil(np.log10(data.abs().max())))

