# coding = utf8
import matplotlib.pyplot as plt
import numpy as np
#        1    2     3   4   5
shop1 = [3.3, 5.8, 3.6, 3.4, 5.2]
shop2 = [6.5, 2.6, 6.3, 5.8, 3.1]
for i in range(0, len(shop1) - 1):
    for j in range(i + 1, len(shop2)):
        distance = (shop1[i] - shop1[j]) ** 2 + (shop2[i] - shop2[j]) ** 2
        correlation = 1 / (1 + distance) # 将范围缩小至0 ~ 1 之间
        if correlation > 0.60:  # 根据需求更换相关系数
            print("第 {} and {}  distance is {}".format(i + 1, j + 1, correlation))
