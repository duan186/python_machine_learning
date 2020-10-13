from sklearn import datasets,cross_validation,naive_bayes
import numpy as np
import matplotlib.pyplot as plt

#显示Digit Dataset数据集
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

#测试多项式贝叶斯分类器
def test_MultinomialNB(*data):
    X_train,X_test,y_train,y_test=data
    cls=naive_bayes.MultinomialNB()
    cls.fit(X_train,y_train)
    print('Training Score:%.2f'%cls.score(X_train,y_train))
    print('Testing Score:%.2f'%cls.score(X_test,y_test))

X_train,X_test,y_train,y_test=load_data()
test_MultinomialNB(X_train,X_test,y_train,y_test)

#检验不同的a对多项式贝叶斯分类器的预测性能的影响
def test_MultinomialNB_alpha(*data):
    X_train,X_test,y_train,y_test=data
    alphas=np.logspace(-2,5,num=200)
    train_scores=[]
    test_scores=[]
    for alpha in alphas:
        cls=naive_bayes.MultinomialNB(alpha=alpha)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test,y_test))
    #绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,train_scores,label="Training Score")
    ax.plot(alphas,test_scores,label="Testing Score")
    ax.set_xlabel(r"$alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.0)
    ax.set_title("MultinomialNB")
    ax.set_xscale("log")
    plt.show()

X_train,X_test,y_train,y_test=load_data()
test_MultinomialNB_alpha(X_train,X_test,y_train,y_test)
