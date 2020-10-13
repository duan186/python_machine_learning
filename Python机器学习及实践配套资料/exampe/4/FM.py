def stocGradAscent(dataMatrix, classLabels, k, max_iter, alpha):
    '''利用随机梯度下降法训练FM模型
    input:  dataMatrix(mat)特征
            classLabels(mat)标签
            k(int)v的维数
            max_iter(int)最大迭代次数
            alpha(float)为随机梯度下降法的学习率
    output: w0(float),w(mat),v(mat):权重
    '''
    m, n = np.shape(dataMatrix)
    # 1、初始化参数
    w = np.zeros((n, 1))  # 其中n是特征的个数
    w0 = 0  # 偏置项
    v = initialize_v(n, k)  # 初始化V
    
    # 2、训练
    for it in range(max_iter):
        for x in range(m):  # 随机优化，对每一个样本而言的
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
             np.multiply(v, v)  # multiply对应元素相乘
            # 完成交叉项
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            p = w0 + dataMatrix[x] * w + interaction  # 计算预测的输出
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
        
            w0 = w0 - alpha * loss * classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    
                    for j in range(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * \
                        (dataMatrix[x, i] * inter_1[0, j] -\
                          v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
        
        # 计算损失函数的值
        if it % 1000 == 0:
            print ("\t------- iter: ", it, " , cost: ",             getCost(getPrediction(np.mat(dataMatrix), w0, w, v), classLabels)))
    
    # 3、返回最终的FM模型的参数
    return w0, w, v

def initialize_v(n, k):
    '''初始化交叉项
    input:  n(int)特征的个数
            k(int)FM模型的超参数
    output: v(mat):交叉项的系数权重
    '''
    v = np.mat(np.zeros((n, k)))    
    for i in range(n):
        for j in range(k):
            # 利用正态分布生成每一个权重
            v[i, j] = normalvariate(0, 0.2)
    return v

from random import normalvariate

def sigmoid(inx):
	return 1.0/(1+np.exp(-inx))

def getCost(predict, classLabels):
    '''计算预测准确性
    input:  predict(list)预测值
            classLabels(list)标签
    output: error(float)计算损失函数的值
    '''
    m = len(predict)
    error = 0.0
    for i in range(m):
        error -=  np.log(sigmoid(predict[i] * classLabels[i] ))  
    return error
