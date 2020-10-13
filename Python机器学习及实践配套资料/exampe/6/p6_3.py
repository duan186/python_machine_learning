from math import*
import numpy as np
def euclideann(a, b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])** 2
    distance = np.sqrt(sum)
    return distance
print ('n 维空间a, b两点之间的欧式距离为： ', euclideann((1,1,2,2),(2,2,4,4))) 
def euclideann2(a, b):
    """
    不使用循环
    """
    A = np.array(a)
    B = np.array(b)
    c = (A - B) ** 2
    distance = np.sqrt(sum(c))
    return distance
print ('n 维空间a, b两点之间的欧式距离为： ', euclideann2((1,1,2,2),(2,2,4,4)))
