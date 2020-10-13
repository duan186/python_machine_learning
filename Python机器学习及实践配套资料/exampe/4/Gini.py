def cal_gini_index(data):
	''' 计算给定数据集的Gini指数
	input:data(list)数据集
	outout:gini(float):Gini指数
	'''
	total_sample=len(data)  #样本的总个数
	if len(data)==0:
		return 0
		label_counts=label_uniq_cnt(data) #统计数据集中不同标签的个数
		
		#计算数据集的Gini指数
		gini=0
		for label in label_counts:
			gini=gini+pow(label_counts[label],2)
		gini=1-float(gini)/pow(total_sample,2)
		return gini
