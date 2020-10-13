import math
import random
random.seed(0)
def rand(a,b): #随机函数
    return (b-a)*random.random()+a
 
def make_matrix(m,n,fill=0.0):#创建一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill]*n)
    return mat
 
#定义sigmoid函数和它的导数
def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))
def sigmoid_derivate(x):
    return x*(1-x) #sigmoid函数的导数
 
class BPNN:
    def __init__(self):#初始化变量
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []
    #三个列表维护：输入层，隐含层，输出层神经元
    def setup(self,ni,nh,no):
        self.input_n = ni+1 #输入层+偏置项
        self.hidden_n = nh  #隐含层
        self.output_n = no  #输出层 
        #初始化神经元
        self.input_cells = [1.0]*self.input_n
        self.hidden_cells= [1.0]*self.hidden_n
        self.output_cells= [1.0]*self.output_n
 
        #初始化连接边的边权
        self.input_weights = make_matrix(self.input_n,self.hidden_n) #邻接矩阵存边权：输入层->隐藏层
        self.output_weights = make_matrix(self.hidden_n,self.output_n) #邻接矩阵存边权：隐藏层->输出层 
        #随机初始化边权：为了反向传导做准备--->随机初始化的目的是使对称失效
        for i in range(self.input_n):
            for h in range(self.hidden_n):
#由输入层第i个元素到隐藏层第j个元素的边权为随机值 
                self.input_weights[i][h] = rand(-0.2 , 0.2) 
        for h in range(self.hidden_n):
            for o in range(self.output_n):
#由隐藏层第i个元素到输出层第j个元素的边权为随机值
                self.output_weights[h][o] = rand(-2.0, 2.0) 
        #保存校正矩阵，为了以后误差做调整
        self.input_correction = make_matrix(self.input_n , self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n,self.output_n)
 
    #输出预测值
    def predict(self,inputs):
        #对输入层进行操作转化样本
        for i in range(self.input_n-1):
            self.input_cells[i] = inputs[i] #n个样本从0~n-1
        #计算隐藏层的输出，每个节点最终的输出值就是权值*节点值的加权和
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total+=self.input_cells[i]*self.input_weights[i][j]
            # 此处为何是先i再j，以隐含层节点做大循环，输入样本为小循环，是为了每一个隐藏节点计算一个输出值，传输到下一层
            self.hidden_cells[j] = sigmoid(total) #此节点的输出是前一层所有输入点和到该点之间的权值加权和
 
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total+=self.hidden_cells[j]*self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total) #获取输出层每个元素的值
        return self.output_cells[:]  #最后输出层的结果返回
 
    #反向传播算法：调用预测函数，根据反向传播获取权重后前向预测，将结果与实际结果返回比较误差
    def back_propagate(self,case,label,learn,correct):
        #对输入样本做预测
        self.predict(case) #对实例进行预测
        output_deltas = [0.0]*self.output_n #初始化矩阵
        for o in range(self.output_n):
           error = label[o] - self.output_cells[o] #正确结果和预测结果的误差：0,1，-1
       output_deltas[o]= sigmoid_derivate(self.output_cells[o])*error#误差稳定在0~1内
 
        #隐含层误差
        hidden_deltas = [0.0]*self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error+=output_deltas[o]*self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivate(self.hidden_cells[h])*error
        #反向传播算法求W
        #更新隐藏层->输出权重
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o]*self.hidden_cells[h]
                #调整权重：上一层每个节点的权重学习*变化+矫正率
                self.output_weights[h][o] += learn*change + 
correct*self.output_correction[h][o]
        #更新输入->隐藏层的权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h]*self.input_cells[i]
                self.input_weights[i][h] += learn*change + 
correct*self.input_correction[i][h]
                self.input_correction[i][h] =  change
        #获取全局误差
        error = 0.0
        for o in range(len(label)):
            error = 0.5*(label[o]-self.output_cells[o])**2 #平方误差函数
        return error
 
    def train(self,cases,labels,limit=10000,learn=0.05,correct=0.1):
        for i in range(limit): #设置迭代次数
            error = 0.0
            for j in range(len(cases)):#对输入层进行访问
                label = labels[j]
                case = cases[j]
#样例，标签，学习率，正确阈值
                error+=self.back_propagate(case,label,learn,correct)
        print(error) #误差
 
def test(cases,labels): #学习异或       
        B=BPNN()
        B.setup(2,5,1) #初始化神经网络：输入层，隐藏层，输出层元素个数
        B.train(cases,labels,10000,0.05,0.1) #可以更改
        for case in  cases:
            print(B.predict(case))
 
if __name__ == '__main__':
    cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ] #测试样例
    labels = [[0], [1], [1], [0]] #标签
    test(cases,labels)
