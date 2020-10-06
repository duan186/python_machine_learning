import pandas as pd
from scipy.interpolate import lagrange
inputfile = '/Users/wangduan/python_machine_learning/数据挖掘/data/catering_sale.xls'
outputfile = '/Users/wangduan/python_machine_learning/数据挖掘/catering_sale.xls'
data = pd.read_excel(inputfile, header=None)


def ployinterp_column(s, n, k=5):
    y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] # 取数
    y = y[y.notnull()]
    return lagrange(y.index,list(y))(n)  # 差值并返回差值结果
# 逐个元素判断是否需要插值


for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:
            data[i][j] = ployinterp_column(data[i], j)
data.to_excel(outputfile)

