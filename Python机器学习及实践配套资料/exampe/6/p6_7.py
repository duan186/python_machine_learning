from math import*
import numpy as np
def chebyshevn(a, b):
    """
    n维空间切比雪夫距离
    """
    distance = 0
    for i in range(len(a)):
        if (abs(a[i]-b[i]) > distance):
            distance = abs(a[i]-b[i])
    return distance
print ('n维空间a, b两点之间的切比雪夫距离为：' , chebyshevn((1,1,1,1),(3,4,3,4))) 
def chebyshevn2(a, b):
    """
    n维空间切比雪夫距离, 不使用循环
    """
    distance = 0
    A = np.array(a)
    B = np.array(b)
    distance = max(abs(A-B))
    return distance 
print ('n维空间a, b两点之间的切比雪夫距离为：' , chebyshevn2((1,1,1,1),(3,4,3,4)))
