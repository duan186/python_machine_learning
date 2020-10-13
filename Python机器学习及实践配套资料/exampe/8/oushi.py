# coding = utf8
import matplotlib.pyplot as plt
import numpy as np

# 散点图分析五个用户之间的相关程度
# x轴为对于商品 1 的喜欢程度
# y轴是对于商品 2 的喜欢程度
# 数据如下
# 商品1 [3.3, 5.8, 3.6, 3.4, 5.2]
# 商品2 [6.5, 2.6, 6.3, 5.8, 3.1]

shop1 = [3.3, 5.8, 3.6, 3.4, 5.2]
shop2 = [6.5, 2.6, 6.3, 5.8, 3.1]
color = ['r', 'b', 'c', 'g', 'y', 'k', 'm', '0xff0012']
for i in range(0, len(shop1)):
    plt.scatter(shop1[i], shop2[i], c=color[i])
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.ylabel("shopping 2")
plt.xlabel("shopping 1")
plt.legend('12345')
plt.show()
