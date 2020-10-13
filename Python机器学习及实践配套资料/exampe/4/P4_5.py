#载入模型
from sklearn import datasets,cross_validation,naive_bayes
import numpy as np
import matplotlib.pyplot as plt

#显示Digit Dataset数据
def show_digits():
    digits=datasets.load_digits()
    fig=plt.figure()
    print("vector from images 0:",digits.data[0])
    for i in range(25):
        ax=fig.add_subplot(5,5,i+1)
        ax.imshow(digits.images[i],cmap=plt.cm.gray_r,interpolation='nearest')
    plt.show()
show_digits()

#加载数据
def load_data():
    digits=datasets.load_digits()
    return cross_validation.train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)

#测试高斯贝叶斯分类器
def test_GaussianNB(*data):
    X_train,X_test,y_train,y_test=data
    cls=naive_bayes.GaussianNB()
    cls.fit(X_train,y_train)
    print("Training Score:%.2f"%cls.score(X_train,y_train))
    print("Testing Score:%.2f"%cls.score(X_test,y_test))

#调用test_GaussianNB函数
X_train,X_test,y_train,y_test=load_data()
test_GaussianNB(X_train,X_test,y_train,y_test)
