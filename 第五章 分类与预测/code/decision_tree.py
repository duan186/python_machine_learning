# -*- coding: utf-8 -*-

# 代码5-2

import pandas as pd
# 参数初始化
filename = '../data/sales_data.xls'
data = pd.read_excel(filename, index_col = u'序号')  # 导入数据

# 数据是类别标签，要将它转换为数据
# 用1来表示“好”“是”“高”这三个属性，用-1来表示“坏”“否”“低”
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = -1
x = data.iloc[:,:3].astype(int)
y = data.iloc[:,3].astype(int)


from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy')  # 建立决策树模型，基于信息熵
dtc.fit(x, y)  # 训练模型

# 导入相关函数，可视化决策树。
# 导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。
from sklearn.tree import export_graphviz
x = pd.DataFrame(x)

"""
string1 = '''
edge [fontname="NSimSun"];
node [ fontname="NSimSun" size="15,15"];
{
''' 
string2 = '}'
"""
 
with open("../tmp/tree.dot", 'w') as f:
    export_graphviz(dtc, feature_names = x.columns, out_file = f)
    f.close()



'''
from IPython.display import Image  
from sklearn import tree
import pydotplus 

dot_data = tree.export_graphviz(dtc, out_file=None,  #regr_1 是对应分类器
                         feature_names=data.columns[:3],   #对应特征的名字
                         class_names=data.columns[3],    #对应类别的名字
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png('../tmp/example.png')    #保存图像
Image(graph.create_png()) 
'''