#encoding:utf-8
import numpy as py
from numpy import *
from audioop import reverse
#fp树的定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue #树节点的名字
        self.count = numOccur #树节点的计数
        self.nodeLink = None #链接下一棵树的线索，好像指针
        self.parent = parentNode    #该树节点的的父亲
        self.children = {}  #树节点的孩子
    
    def inc(self, numOccur): #计算该树节点出现的支持度数，就是出现了几次
        self.count += numOccur
        
    def disp(self, ind=1): #遍历这颗树
        print ('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)
#构建fp树
def createTree(dataSet, minSup=1): 
    headerTable = {} #头结点的字典
    ##1.统计元素出现的次数
    for trans in dataSet:#字典中的每条记录    网上俗称的事务
        for item in trans: #遍历的是事务中的每个元素
			#dict.get(key, default = None ) 有过有该键值就返回对应的价值,如果不存在键值就返回用户指定的值
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    #2 删除小于该支持度的元素项
    for k in headerTable.keys():  
        if headerTable[k] < minSup: 			
            del(headerTable[k])
     #3 对元素项去重得到频繁集       
    freqItemSet = set(headerTable.keys())
    ##如果为空 则返回无需进行下一项
    if len(freqItemSet) == 0: return None, None  
    #在头指针中  保留计数的数值以及指向每种类型第一个指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] 
    #根节点为空  并且出现的次数为一
    retTree = treeNode('Null Set', 1, None) 
    '''第二次遍历数据集 建立fp树'''
    for tranSet, count in dataSet.items():#transSet代表事务（一条物品的组合）  也就是一个集合啦 dataSet的键值，count代表出现的次数
        localD = {} #key每项物品，count 商品出现的次数
        for item in tranSet:  
            if item in freqItemSet:
                localD[item] = headerTable[item][0] #记录商品出现的次数
        if len(localD) > 0:
			#对这个事务（一条物品的组合）按照出现的度数  从大到小进行排序  为插入进行准备
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            #构建树
            updateTree(orderedItems, retTree, headerTable, count)
            #返回FP树结构，头指针
    return retTree, headerTable 
'''已经排好序的物品items，构建fp树'''
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#若该项已经在树中出现则计数加一，
        inTree.children[items[0]].inc(count)#如果没有这个元素项，那么创建一个子节点
    else:   ##如果没有这个元素项，那么创建一个子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #如果头指针没有指向任何元素，那么指向该节点
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:#如果已经指向，那么就继续加入这个链表  updateHeader这个函数的作用就是让已经加入的该链表的最后一项，指向这个新的节点 
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    #对剩下的元素项迭代调用，updateTree
    #不断调用自身，每次调用就会去掉列表的第一个元素
    #通过items[1::]实现 
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
'''更新相似元素的链表，相当于延长这个链表啦'''          
def updateHeader(nodeToTest, targetNode):   #this version does not use recursion
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
'''挖掘频繁集'''
#向上搜索，寻找到leafNode当前节点的路径        
def ascendTree(leafNode, prefixPath): #如果父节点不为空就继续向上寻找并记录
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath) #递归向上查找
'''为给定的元素项找到前缀路径（条件模式基）'''    
def findPrefixPath(basePat, treeNode): #basePat 要发掘的元素  treeNode发掘的节点
    condPats = {} #存放条件模式基，即含元素项basePat的前缀路径以及计数
    #<key,value>  key:前缀路径 value：路径计数
    while treeNode != None:
        prefixPath = [] #存放不同路线的  前缀路径    包含basePat自身   在下面会去掉自身
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats
#inTree为生成的Fp树，头指针表headerTable， preFix空集合Set（）
#freItemList保存生成的频繁集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
# 对头指针出现的元素按照出现的频率从小到大进行排序
#遍历头指针表，挖掘频繁集
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]#(sort header table)
    for basePat in bigL:  ##保存当前前缀路径basePat
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #将每个频繁项添加到频繁项集列表freqItemList
        freqItemList.append(newFreqSet)
        #递归调用findPrefixPath函数找到到元素项basePat的前缀路径
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #根据当前元素项生成的前缀路径和最小支持度生成条件树
        myCondTree, myHead = createTree(condPattBases, minSup)
        #若条件fp树有元素项，可以再次递归生成条件树
        if myHead != None: #3. mine cond. FP-tree
            #递归挖掘该条件树             
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat
#对数据进行格式化处理转化成字典类型，<交易记录，count = 1>
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)    
    resultsPages = []
    for i in range(1,15):
        print ("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList
    print (mat(freItemList))

