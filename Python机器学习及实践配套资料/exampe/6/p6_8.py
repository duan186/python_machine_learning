from math import*
import numpy as np
def minkowski(a, b):
    """
    闵可夫斯基距离
    """
    A = np.array(a)
    B = np.array(b)
    #方法一：根据公式求解
    distance1 = np.sqrt(np.sum(np.square(A-B)))
 
    #方法二：根据scipy库求解
    from scipy.spatial.distance import pdist
    X = np.vstack([A,B])
    distance2 = pdist(X)[0]
    return distance1, distance2
print ('二维空间a, b两点之间的闵可夫斯基距离为：' , minkowski((1,1),(2,2))[0])
