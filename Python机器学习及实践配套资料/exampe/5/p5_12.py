# 基于DFP的拟牛顿法
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt 
 
def compute_original_fun(x):
    """ 1. 计算原函数的值 
    input:  x, 一个向量
    output: value, 一个值
    """
    value = x[0]**2 + 2*x[1]**2
    return value 
 
def compute_gradient(x):
    """ 2. 计算梯度 
    input:  x, 一个向量
    output: value, 一个向量
    """
    value = np.mat([[0],[0]], np.double)
    value[0] = 2*x[0]
    value[1] = 4*x[1]
    return value 
 
def draw_result(result):
    """ 3. 将收敛过程(即最小值的变化情况)画图 """
    plt.figure("min value")
    plt.plot(range(len(result)), result, "y", label="min value")
    plt.title("min value's change")
    plt.legend()
    return plt 
 
def main(x0, H, epsilon = 1e-6, max_iter = 1000):   
    """
    x0: 初始迭代点
    H: 校正的对角正定矩阵
    eplison: 最小值上限
    max_iter: 最大迭代次数
    result: 最小值
    alpha**m: 步长
    d: 方向
    """
    result = [compute_original_fun(x0)[0,0]]
    for k in range(max_iter):
        # 计算梯度
        g = compute_gradient(x0)        
        # 终止条件
        if linalg.norm(g) < epsilon:
            break            
        # 计算搜索方向
        d = -H*g        
        # 简单线搜索求步长
        alpha = 1/2
        for m in range(max_iter):
            if compute_original_fun(x0 + alpha**m*d) <= (compute_original_fun(x0) + (1/2)*alpha**m*g.T*d):
                break
        x = x0 + alpha**m*d        
        # DFP校正迭代矩阵
        s = x - x0
        y = compute_gradient(x) - g
        if s.T * y > 0:
            H = H - (H*y*y.T*H)/(y.T*H*y) + (s*s.T)/(s.T*y)        
        x0 = x
        result.append(compute_original_fun(x0)[0,0])
    return result   
 
if __name__ == "__main__":
    x0 = np.asmatrix(np.ones((2,1)))
    H = np.asmatrix(np.eye(x0.size))
    result = main(x0, H)
    draw_result(result).show()
