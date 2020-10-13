"""2.包裹型"""
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston=load_boston()
X=boston["data"]
Y=boston["target"]
names=boston["feature_names"]
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=1)#选择剔除1个
rfe.fit(X,Y)
print('包裹式选择：')
print ("features sorted by their rank:")
print (sorted(zip(map(lambda x:round(x,4), rfe.ranking_),names)))
