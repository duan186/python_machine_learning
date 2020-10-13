import math  
uid_score_bid = ['A,1,a','A,1,b','A,1,d','B,1,b','B,1,c','B,1,e','C,1,c','C,1,d','D,1,b','D,1,c','D,1,d','E,1,a','E,1,d']
class ItemBasedCF:  
    def __init__(self,train_file):  
        self.train_file = train_file  
        self.readData()  
    def readData(self):  
        #读取文件，并生成数据集（用户，兴趣程度，物品）  
        self.train = dict()       
        for line in self.train_file:  
            user,score,item = line.strip().split(",")  
            self.train.setdefault(user,{})  
            self.train[user][item] = int(float(score))
        print(self.train) #输出数据集

    def ItemSimilarity(self): 
        C = dict()  #物品-物品的共现矩阵
        N = dict()  #物品被多少个不同用户购买  
        for user,items in self.train.items():  
            for i in items.keys():  
                N.setdefault(i,0)  
                N[i] += 1  #物品i出现一次就计数加一
                C.setdefault(i,{})  
                for j in items.keys():  
                    if i == j : continue  
                    C[i].setdefault(j,0)  
                    C[i][j] += 1  #物品i和j共现一次就计数加一

        print ('N:',N) 
        print ('C:',C)

        #计算相似度矩阵  
        self.W = dict()  
        for i,related_items in C.items():  
            self.W.setdefault(i,{})  
            for j,cij in related_items.items():  
#按上述物品相似度公式计算相似度
                self.W[i][j] = cij / (math.sqrt(N[i] * N[j]))  
        for k,v in self.W.items():
            print (k+':'+str(v))
        return self.W  

    #给用户user推荐前N个最感兴趣的物品  
    def Recommend(self,user,K=3,N=10):  
        rank = dict() #记录user的推荐物品（没有历史行为的物品）和兴趣程度
        action_item = self.train[user]   #用户user购买的物品和兴趣评分r_ui  
        for item,score in action_item.items():  
#使用与物品item最相似的K个物品进行计算
            for j,wj in sorted(self.W[item].items(),key=lambda x:x[1],reverse=True)[0:K]:  
                if j in action_item.keys():  #如果物品j已经购买过，则不进行推荐
                    continue  
                rank.setdefault(j,0)  
 #如果物品j没有购买过，则累计物品j与item的相似度*兴趣评分，作为user对物品j的兴趣程度
                rank[j] += score * wj 
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:N]) 

#声明一个ItemBased推荐的对象      
Item = ItemBasedCF(uid_score_bid) 
Item.ItemSimilarity()  
recommedDic = Item.Recommend("A")  #计算给用户A的推荐列表
for k,v in recommedDic.items():  
    print (k,"\t",v  )
