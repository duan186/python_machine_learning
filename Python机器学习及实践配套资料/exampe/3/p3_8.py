#导入必要的编程库
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
#载入数据
X, _ = load_iris(True)
# 最重要的参数是n_components、alpha、l1_ratio、solver
nmf = NMF(n_components=2,  # k value,默认会保留全部特征
          init=None,  # W H 的初始化方法，包括'random' | 'nndsvd'(默认) |  'nndsvda' | 'nndsvdar' | 'custom'.
          solver='cd',  # 'cd' | 'mu'
          #{'frobenius', 'kullback-leibler', 'itakura-saito'}，一般默认就好
          beta_loss='frobenius', 
          tol=1e-4,  # 停止迭代的极限条件
          max_iter=200,  # 最大迭代次数
          random_state=None,
          alpha=0.,  # 正则化参数
          l1_ratio=0.,  # 正则化参数
          verbose=0,  # 冗长模式
          shuffle=False  # 针对"cd solver"
          )
# -----------------函数------------------------
print('params:', nmf.get_params())  # 获取构造函数参数的值，也可以nmf.attr得到，所以下面我会省略这些属性
# 下面四个函数很简单，也最核心，例子中见
nmf.fit(X)
W = nmf.fit_transform(X)
W = nmf.transform(X)
nmf.inverse_transform(W)
# -----------------属性------------------------
H = nmf.components_  # H矩阵
print('reconstruction_err_', nmf.reconstruction_err_)  # 损失函数值
print('n_iter_', nmf.n_iter_)  # 实际迭代次数
