import numpy as np 
class CSVD(object):
    '''
    实现svd分解降维应用示例的Python代码
    '''    
    def __init__(self, data):
        self.data = data       #用户数据
        self.S = []  #用户数据矩阵的奇异值序列 singular values
        self.U = []  #svd后的单位正交向量
        self.VT = []  #svd后的单位正交向量
        self.k = 0   #满足self.p的最小k值(k表示奇异值的个数)
        self.SD = [] #对角矩阵，对角线上元素是奇异值 singular values diagonal matrix        
    def _svd(self):
        '''
        用户数据矩阵的svd奇异值分解
        '''
        self.U, self.S, self.VT = np.linalg.svd(self.data)
        return self.U, self.S, self.VT        
    def _calc_k(self, percentge):
    '''确定k值：前k个奇异值的平方和占比 >=percentage, 求满足此条件的最小k值
        :param percentage, 奇异值平方和的占比的阈值
        :return 满足阈值percentage的最小k值
        '''
        self.k = 0
        #用户数据矩阵的奇异值序列的平方和
        total = sum(np.square(self.S))
        svss = 0 #奇异值平方和 singular values square sum
        for i in range(np.shape(self.S)[0]):
            svss += np.square(self.S[i])
            if (svss/total) >= percentge:
                self.k = i+1
                break
        return self.k
 
    def _buildSD(self, k):
        '''构建由奇异值组成的对角矩阵
        :param k,根据奇异值开放和的占比阈值计算出来的k值
        :return 由k个前奇异值组成的对角矩阵
        '''
        #方法1：用数组乘方法
        self.SD = np.eye(self.k) * self.S[:self.k] 
        #方法2：用自定义方法
        e = np.eye(self.k)
        for i in range(self.k):
            e[i,i] = self.S[i] 
        return self.SD
    
    def DimReduce(self, percentage):
        '''
        SVD降维
        :param percentage, 奇异值开方和的占比阈值
        :return 降维后的用户数据矩阵
        '''
        #Step1:svd奇异值分解
        self._svd()
        #Step2:计算k值
        self._calc_k(percentage)
        print('\n按照奇异值开方和占比阈值percentage=%d, 求得降维的k=%d'%(percentage, self.k))
        #Step3:构建由奇异值组成的对角矩阵singular values diagonal
        self._buildSD(self.k)
        k,U,SD,VT = self.k,self.U, self.SD, self.VT
        #Step4:按照svd分解公式对用户数据矩阵进行降维，得到降维压缩后的数据矩阵        
        a = U[:len(U), :k]
        b = np.dot(SD, VT[:k, :len(VT)])
        newData = np.dot(a,b)
        return newData
        
def CSVD_manual():
    ##训练数据集，用户对商品的评分矩阵，行为多个用户对单个商品的评分，列为用户对每个商品的评分
    data = np.array([[5, 5, 0, 5],
                     [5, 0, 3, 4],
                     [3, 4, 0, 3],
                     [0, 0, 5, 3],
                     [5, 4, 4, 5],
                     [5, 4, 5, 5]])
    percentage = 0.9
    svdor = CSVD(data)
    ret = svdor.DimReduce(percentage)
    print('====================================================')
    print('原始用户数据矩阵:\n', data)
    print('降维后的数据矩阵:\n', ret)
    print('====================================================')    
if __name__=='__main__':
    CSVD_manual()
