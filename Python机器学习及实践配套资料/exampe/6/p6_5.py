from math import*
import numpy as np
def manhattann(a, b):
    """
    n维空间曼哈顿距离
    """
    distance = 0 
    for i in range(len(a)):
        distance += np.abs(a[i]-b[i])
    return distance
print ('n维空间a, b两点之间的曼哈顿距离为： ', manhattann((1,1,2,2),(2,2,4,4))) 
def manhattann2(a, b):
    """
    n维空间曼哈顿距离, 不使用循环
    """
    A = np.array(a)
    B = np.array(b)
    distance = sum(np.abs(A-B))
    return distance
print ('n维空间a, b两点之间的曼哈顿距离为： ', manhattann2((1,1,2,2),(2,2,4,4)))
