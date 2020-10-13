from scipy import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt

class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)

    def _calcAct(self, X):
        # 计算RBFs的激活函数值
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: n x indim维的矩阵
            y: n x 1维的列向量"""

        # 从训练集随机选择中心向量
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]

        print("center", self.centers)
        # 计算RBFs的激活函数值
        G = self._calcAct(X)
        print(G)

        # 计算输出层的权值
        self.W = dot(pinv(G), Y)

    def test(self, X):
        """ X: n x indim维的矩阵 """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

if __name__ == '__main__':
    n = 100
    x = mgrid[-1:1:complex(0,n)].reshape(n, 1) #设置x的值
    y = sin(3*(x+0.5)**3 - 1) # 设置y的值
    # rbf回归
    rbf = RBF(1, 10, 1)
    rbf.train(x, y)
    z = rbf.test(x)
    # 画原始图像
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')
    # 画学习后的图像
    plt.plot(x, z, 'r-', linewidth=2)
    # 画RBF
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')
    for c in rbf.centers:
        # RF的预测线条
        cx = arange(c-0.7, c+0.7, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
    plt.xlim(-1.2, 1.2)
    plt.show()
