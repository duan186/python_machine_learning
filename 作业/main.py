#先导入python统计分析包“Statsmodels”和数组Numpy模块
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
#模拟一组数据，假设模型符合以下条件: Y=1+0.6X
nsample = 200
x = np.linspace(0.1, 5.0, 200)
beta = np.array([1, 0.6])
e = np.random.normal(size=nsample)
#根据上述我们设定的回归直线模型形式，我们需要在模拟数据中添加一个截距项
X = sm.add_constant(x)
y = np.dot(X, beta) + e
#模型参数估计和拟合
model = sm.OLS(y, X)

results = model.fit()
print(results.summary())