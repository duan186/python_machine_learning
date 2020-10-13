from math import*
def euclidean2(a, b):
    distance = sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )
    return distance
print ('a, b两点之间的欧式距离为： ', euclidean2((1,1),(2,2)))
