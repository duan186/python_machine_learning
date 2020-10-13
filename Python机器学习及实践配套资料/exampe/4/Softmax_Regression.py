import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

def load_data():
     digits = load_digits()
     data = digits.data
     label = digits.target
     return np.mat(data), label

def gradient_descent(train_x, train_y, k, maxCycle, alpha):
# k 为类别数
     numSamples, numFeatures = np.shape(train_x)
     weights = np.mat(np.ones((numFeatures, k)))
     
     for i in range(maxCycle):
          value = np.exp(train_x * weights)  
          rowsum = value.sum(axis = 1)   # 横向求和
          rowsum = rowsum.repeat(k, axis = 1)  # 横向复制扩展
          err = - value / rowsum  #计算出每个样本属于每个类别的概率
          for j in range(numSamples):     
               err[j, train_y[j]] += 1
          weights = weights + (alpha / numSamples) * (train_x.T * err)
     return weights  

def test_model(test_x, test_y, weights):
     results = test_x * weights
     predict_y = results.argmax(axis = 1)
     count = 0
     for i in range(np.shape(test_y)[0]):
          if predict_y[i,] == test_y[i,]:
               count += 1   
     return count / len(test_y), predict_y 

if __name__ == "__main__":
     data, label = load_data()
     #data = preprocessing.minmax_scale(data, axis = 0)
     #数据处理之后识别率降低了
     train_x, test_x, train_y, test_y = train_test_split(data, label, test_size = 0.25, random_state=33)
     k = len(np.unique(label))     
     weights = gradient_descent(train_x, train_y, k, 800, 0.01)
     accuracy, predict_y = test_model(test_x, test_y, weights)
     print("Accuracy:", accuracy)
