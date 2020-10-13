import numpy as np
#导入Python的矩阵计算模块
import matplotlib.pyplot as plt

def fun2ploy(x,n):
    '''
    数据转化为[x^0,x^1,x^2,...x^n]
    首列变1
    '''
    lens = len(x)
    X = np.ones([1,lens])
    for i in range(1,n):
        X = np.vstack((X,np.power(x,i)))#按行堆叠
    return X  

def leastseq_byploy(x,y,ploy_dim):
    '''
    最小二乘求解
    '''
    #散点图
    plt.scatter(x,y,color="r",marker='o',s = 50)
    X = fun2ploy(x,ploy_dim);
    #直接求解
    Xt = X.transpose();#转置变成列向量
    XXt=X.dot(Xt);#矩阵乘
    XXtInv = np.linalg.inv(XXt)#求逆
    XXtInvX = XXtInv.dot(X)
    coef = XXtInvX.dot(y.T)    
    y_est = Xt.dot(coef)    
    return y_est,coef

def fit_fun(x):
    '''
    要拟合的函数
    '''
    #return np.power(x,5)
    return np.sin(x) 

if __name__ == '__main__':
    data_num = 100;
    ploy_dim =10;#拟合参数个数，即权重数量
    noise_scale = 0.2;
    ## 数据准备
    x = np.array(np.linspace(-2*np.pi,2*np.pi,data_num))   #数据 
    y = fit_fun(x)+noise_scale*np.random.rand(1,data_num)  #添加噪声
    # 最小二乘拟合
    [y_est,coef] = leastseq_byploy(x,y,ploy_dim)    
    #显示拟合结果
    org_data = plt.scatter(x,y,color="r",marker='o',s = 50)
    est_data = plt.plot(x,y_est,color="g",linewidth= 3)    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Fit funtion with leastseq method")
    plt.legend(["Noise data","Fited function"]);
    plt.show()
