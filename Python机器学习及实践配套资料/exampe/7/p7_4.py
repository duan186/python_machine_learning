from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
import numpy as np
#利用算法进行创建数据集
def creatdata(): 
 x,y = make_classification(n_samples=1000, n_features=2,n_redundant=0,n_informative=1,n_clusters_per_class=1)
 '''
 #n_samples:生成样本的数量 
 #n_features=2:生成样本的特征数，特征数=n_informative（） + n_redundant + n_repeated 
 #n_informative：多信息特征的个数 
 #n_redundant：冗余信息，informative特征的随机线性组合 
 #n_clusters_per_class ：某一个类别是由几个cluster构成的 
 make_calssification默认生成二分类的样本，上面的代码中，x代表生成的样本空间（特征空间）
 y代表了生成的样本类别，使用1和0分别表示正例和反例 
 y=[0 0 0 1 0 1 1 1... 1 0 0 1 1 0]
 '''
 return x,y
 
if __name__ == '__main__':
 x,y=creatdata()
 
 #将生成的样本分为训练数据和测试数据，并将其中的正例和反例分开
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 
 #正例和反例
 positive_x1=[x[i,0]for i in range(len(y)) if y[i]==1]
 positive_x2=[x[i,1]for i in range(len(y)) if y[i]==1]
 negetive_x1=[x[i,0]for i in range(len(y)) if y[i]==0]
 negetive_x2=[x[i,1]for i in range(len(y)) if y[i]==0]
 
 #定义感知机
 clf=Perceptron(fit_intercept=True,n_iter=50,shuffle=False)
 # 使用训练数据进行训练
 clf.fit(x_train,y_train)
 #得到训练结果，权重矩阵
 weights=clf.coef_
 #得到截距
 bias=clf.intercept_
 
 #到此时，我们已经得到了训练出的感知机模型参数，下面用测试数据对其进行验证
 acc=clf.score(x_test,y_test)#Returns the mean accuracy on the given test data and labels.
 print('平均精确度为：%.2f'%(acc*100.0))
 
 #最后，我们将结果用图像显示出来，直观的看一下感知机的结果
 #画出正例和反例的散点图
 plt.scatter(positive_x1,positive_x2,c='red')
 plt.scatter(negetive_x1,negetive_x2,c='blue')
 
 #画出超平面（在本例中即是一条直线）
 line_x=np.arange(-4,4)
 line_y=line_x*(-weights[0][0]/weights[0][1])-bias
 plt.plot(line_x,line_y)
 plt.show()
