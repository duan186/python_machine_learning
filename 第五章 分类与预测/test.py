import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.001)
y = (((x - 1) * (x - 1) + (x * 2 - 2) * (x * 2 - 2) + (x * 3 - 3) * (x * 3 - 3)) * 1 / 6.0)
plt.plot(x, y)


# plt.show()  #显示图形

def sum(x):
    return (x * 1 - 1) * 1 + (x * 2 - 2) * 2 + (x * 3 - 3) * 3


def fun(x):
    return (1 / 3.0) * sum(x)


old = 0
new = 5
step = 0.01
pre = 0.00000001


def src_fun(x):
    print(((x - 1) * (x - 1) + (x * 2 - 2) * (x * 2 - 2) + (x * 3 - 3) * (x * 3 - 3)) * 1 / 6.0)


while abs(new - old) > pre:
    old = new
    # src_fun(old)   #输出每次迭代的损失值
    new = new - step * fun(old)
print(new)
print(src_fun(new))
