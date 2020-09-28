from numpy import *
import operator
def createDataSet():
    group = array([[1.0, 1.1], [2.0, 2.0], [0, 0], [4.1, 5.1]])
    labels = ['A', 'B', 'C', 'D']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    :param inX: 用于分类的输出向量
    :param dataSet:输入的样本集
    :param labels:标签向量
    :param k:用于选择最近邻居的树目
    :return:
    """
    dataSetsize = dataSet.shape[0]  # 得到数据集的行数
    diffMat = tile(inX, (dataSetsize, 1)) - dataSet  # tile生成和训练样本对应的矩阵，并与训练样本求差
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 将矩阵的每一行相加
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # 从小到大排序 返回对应的索引位置
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 找到该样本的类型
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 在字典中将该类型加一
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # reverse = True代表降序
    return sortedClassCount[0][0]  # 排序并返回出现最多的那个类型
#测试
group,labels = createDataSet()
print(classify0([0,0],group,labels,3))
print(classify0([1,2],group,labels,3))
print(classify0([3,3],group,labels,3))
print(classify0([5,5],group,labels,3))
