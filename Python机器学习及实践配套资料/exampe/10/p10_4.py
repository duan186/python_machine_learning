"""1.过滤型"""
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

iris=load_iris()
X,y=iris.data,iris.target
print('过滤式选择：')
print(X.shape)
X_new=SelectKBest(chi2,k=2).fit_transform(X,y)
print(X_new.shape)
