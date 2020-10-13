import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
 
#创建数据,直接定义数据列表
def creatdata1():
 samples=np.array([[3,-3],[4,-3],[1,1],[1,2]])
 labels=np.array([-1,-1,1,1])
 return samples,labels
 
def MyPerceptron(samples,labels):
 #定义感知机
 clf=Perceptron(fit_intercept=True,n_iter=30,shuffle=False)
 #训练感知机
 clf.fit(samples,labels)
 #得到权重矩阵
 weigths=clf.coef_ 
 #得到截距bisa
 bias=clf.intercept_ 
 return weigths,bias
 
#画图描绘
class Picture:
 def __init__(self,data,w,b):
  self.b=b
  self.w=w
  plt.figure(1)
  plt.title('Perceptron Learning Algorithm',size=14)
  plt.xlabel('x0-axis',size=14)
  plt.ylabel('x1-axis',size=14)
 
  xData=np.linspace(0,5,100)
  yData=self.expression(xData)
  plt.plot(xData,yData,color='r',label='sample data')
 
  plt.scatter(data[0][0],data[0][1],s=50)
  plt.scatter(data[1][0],data[1][1],s=50)
  plt.scatter(data[2][0],data[2][1],s=50,marker='x')
  plt.scatter(data[3][0],data[3][1],s=50,marker='x')
  plt.savefig('3d.png',dpi=75)
 
 def expression(self,x):
  y=(-self.b-self.w[:,0]*x)/self.w[:,1]
  return y
 
 def Show(self):
  plt.show()
  
if __name__ == '__main__':
 samples,labels=creatdata1()
 weights,bias=MyPerceptron(samples,labels)
 print('最终训练得到的w和b为：',weights,',',bias)
 Picture=Picture(samples,weights,bias)
 Picture.Show()
