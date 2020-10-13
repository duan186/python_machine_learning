# coding = utf8
import matplotlib.pyplot as plt
import numpy as np
# 皮尔孙相关系数计算
def PearsonCorrelationSimilarity(vec1, vec2):
    value = range(len(vec1))
    sum_vec1 = sum([vec1[i] for i in value])
    sum_vec2 = sum([vec2[i] for i in value])
    square_sum_vec1 = sum([pow(vec1[i], 2) for i in value])
    square_sum_vec2 = sum([pow(vec2[i], 2) for i in value])
    product = sum([vec1[i] * vec2[i] for i in value])
    numerator = product - (sum_vec1 * sum_vec2 / len(vec1))
    dominator = ((square_sum_vec1 - pow(sum_vec1, 2) / len(vec1))*(
        square_sum_vec2 - pow(sum_vec2, 2) / len(vec2))) ** 0.5
    if dominator == 0:
        return 0
    result = numerator / (dominator * 1.0)
    return result

if __name__ == '__main__':
    # 五个用户对五个商品的评价
    user1 = [3.3, 5.8, 3.6, 3.4, 5.2]
    user2 = [6.5, 2.6, 6.3, 5.8, 3.1]
    user3 = [5.5, 3.2, 6.5, 4.7, 4.4]
    user4 = [4.4, 6.2, 2.3, 5.1, 3.3]
    user5 = [2.1, 5.2, 4.2, 2.2, 4.1]
    userlist = [user1, user2, user3, user4, user5]
    for i in range(0, len(userlist) - 1):
        for j in range(i + 1, len(userlist)):
            result = PearsonCorrelationSimilarity(userlist[i], userlist[j])
            print("user%d 和 user%d 的相关系数是%f" % (i + 1, j + 1, result))
