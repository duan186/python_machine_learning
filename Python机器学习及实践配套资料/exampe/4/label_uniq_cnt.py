from math import pow
def label_uniq_cnt(data):
	'''统计数据集中不同的类标签label的个数
	input:data(list)：原始数据集
	output:label_uniq_cnt(int)：样本中的标签的个数
	'''
	label_uniq_cnt={}
	for x in data:
		label=x[len(x)-1]  #取得每一个样本的类标签label
		if label not in label_uniq_cnt:
			label_uniq_cnt[label]=0
			label_uniq_cnt[label]=label_uniq_cnt[label]+1
		return label_uniq_cnt
