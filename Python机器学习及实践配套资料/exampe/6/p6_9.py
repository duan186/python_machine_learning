import numpy as np
x=np. mat([1,2,3])
y=np.mat([4,5,7])
#合并
"""
stack()函数 
函数原型：stack(arrays, axis=0)
axis=0时"增加一维，新维度的下标为0 即按行合并；
axis=0时"增加一维，新维度的下标为1" 即按行合并；
等同于hstack()函数 
函数原型：hstack(tup) ，参数tup可以是元组，列表，
或者numpy数组，返回结果为numpy的数组 
作用：在水平方向把元素堆起来
vstack()函数 
函数原型：vstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组 
作用：在垂直方向把元素堆叠起来
"""
X=np.vstack([x,y])
print(X)
#方法一：根据公式求解
sk=np.var(X,axis=0,ddof=0)#方差
#numpy.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<class numpy._globals._NoValue>)
print(sk) 
d1=np.sqrt(((np.power((x-y),2)/sk).sum()))
print(d1)
#方法二：根据scipy库求解
"""
pdist
"""
from scipy.spatial.distance import pdist
d2=pdist(X,'seuclidean',V=None)
print(d2)
from scipy.spatial.distance import seuclidean
d3= seuclidean(x, y, sk)
print(d3)
