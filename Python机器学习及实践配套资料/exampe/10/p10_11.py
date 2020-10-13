#-*- coding: utf-8 -*-
#数据规范化
import pandas as pd
import numpy as np

datafile ='data.xls' #参数初始化
data = pd.read_excel(datafile, header = None) #读取数据

print((data - data.min())/(data.max() - data.min())) #最小-最大规范化
print((data - data.mean())/data.std()) #零-均值规范化
