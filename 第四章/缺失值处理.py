import pandas as pd
import numpy as np
inputfile = '/Users/wangduan/python_machine_learning/数据挖掘/data/catering_sale.xls'
outputfile = '/Users/wangduan/python_machine_learning/数据挖掘/catering_sale.xls'
data = pd.read_excel(inputfile, header=None)


def newton_interpolation(X, Y, x):
    """
    计算x点的插值
    """
    sum = Y[0]
    temp=np.zeros((len(X),len(X)))
    #将第一行赋值
    for i in range(0,len(X)):
        temp[i,0]=Y[i]
    temp_sum=1.0
    for i in range(1,len(X)):
        #x的多项式
        temp_sum=temp_sum*(x-X[i-1])
        #计算均差
        for j in range(i,len(X)):
            temp[j,i]=(temp[j,i-1]-temp[j-1,i-1])/(X[j]-X[j-i])
        sum+=temp_sum*temp[i,i]
    return sum



def ployinterp_column(s, n, k=5):
    y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))]  # 取数
    y = y[y.notnull()]
    return newton_interpolation(y.index, list(y), n)  # 差值并返回差值结果


# 逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:
            data[i][j] = ployinterp_column(data[i], j)
data.to_excel(outputfile)


