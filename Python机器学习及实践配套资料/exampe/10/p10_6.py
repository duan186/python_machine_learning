"""3.嵌入型 ，老的版本没有SelectFromModel"""
from sklearn.svm import  LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

iris=load_iris()
X,y=iris.data,iris.target
print('嵌入式选择：')
print(X.shape)
lsvc=LinearSVC(C=0.01,penalty='l1',dual=False).fit(X,y)
model=SelectFromModel(lsvc,prefit=True)
X_new=model.transform(X)
print (X_new.shape)
