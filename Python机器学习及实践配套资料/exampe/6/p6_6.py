from math import*
def chebyshev2(a, b):
    """
    二维空间切比雪夫距离
    """
    distance = max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    return distance
print ('二维空间a, b两点之间的欧式距离为： ', chebyshev2((1,2),(3,4)))
