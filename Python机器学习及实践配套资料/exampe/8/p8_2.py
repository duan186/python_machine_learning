A (1,1,1,1,1,1,1,1,0,0)
B (1,0,1,1,1,1,1,1,1,1)
实现的Python代码为：
import math
# 余弦定理
def cos(v1, v2):
    l = 0.0
    for i in range(0, len(v1)):
        l += v1[i] * v2[i]
    v = 0.0
    w = 0.0
    for i in range(0, len(v1)):
        v += math.pow(v1[i], 2)
        w += math.pow(v2[i], 2)
    cos = l / (math.sqrt(v) * math.sqrt(w))
    return cos

if __name__ == '__main__':
    v1 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    v2 = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    l = cos(v1, v2)
    print(l)
