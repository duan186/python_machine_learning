import numpy as np 
import math
# 使用词集法进行贝叶斯分类
# 构造数据集,分类是侮辱性or非侮辱性
def loadDataset () :
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList, classVec

# 创建一个包涵所有词汇的列表 , 为后面建立词条向量使用
def createlist (dataset) :
    vovabset = set ([])
    for vec in dataset :
        vovabset = vovabset | set (vec)
    return list (vovabset)

# 将词条转化为向量的形式
def changeword2vec (inputdata, wordlist) :
    returnVec = [0] * len (wordlist)
    for word in inputdata :
        if word in wordlist :
            returnVec[wordlist.index(word)] = 1
    return returnVec

# 创建贝叶斯分类器 
def trainNBO (dataset, classlebels) :
    num_of_sample = len (dataset)
    num_of_feature = len (dataset[0])
    pAusuive = sum (classlebels) / num_of_sample # 侮辱性语言的概率
    p0Num = np.ones (num_of_feature)
    p1Num = np.ones (num_of_feature)
    p0tot = num_of_feature
    p1tot = num_of_feature
    for i in range (num_of_sample) :
        if classlebels[i] == 1 :
            p1Num += dataset[i]
            p1tot += sum (dataset[i])
        else :
            p0Num += dataset[i]
            p0tot += sum (dataset[i])   
    p0Vec = p0Num / p0tot
    p1Vec = p1Num / p1tot
    for i in range (num_of_feature) :
        p0Vec[i] = math.log (p0Vec[i])
        p1Vec[i] = math.log (p1Vec[i])
    return p0Vec, p1Vec, pAusuive

#  定义分类器 
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

# 测试代码 
dataset,classlebels = loadDataset ()
wordlist = createlist (dataset)
print (wordlist)
print (changeword2vec (dataset[0], wordlist))
trainmat = []
for temp in dataset :
    trainmat.append (changeword2vec (temp,wordlist))
p0V, p1V, pAb = trainNBO (trainmat, classlebels)
print (p0V)
print (p1V)
print (pAb)
