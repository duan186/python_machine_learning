# 需要初始化一个Node类，表示kd树中的一个节点，主要包括节点本身的data值，以及其左右子节点
class KDNode(object):
    """kd树节点"""
    def __init__(self,data = None,label = None,left = None,right = None,axis = None,parant = None):
        """
        构造函数
        :param data:数据
        :param label: 数据标签
        :param left: 左孩子节点
        :param right: 右孩子节点
        :param axis: 分割轴
        :param parant: 父节点
        """
        self.data = data
        self.label = label
        self.left = left
        self.right = right
        self.axis = axis
        self.paraent = parant