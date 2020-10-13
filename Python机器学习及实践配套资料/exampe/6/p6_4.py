from math import*
import numpy as np
def manhattan2(a, b):
    """
    二维空间曼哈顿距离
    """
    distance = np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])
    return distance
print ('二维空间a, b两点之间的曼哈顿距离为： ', manhattan2((1,1),(2,2)))
