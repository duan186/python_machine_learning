import random
from copy import deepcopy
from time import time
import numpy as np
from numpy.linalg import norm
from collections import Counter

Counter([0, 1, 1, 2, 2, 3, 3, 4, 3, 3]).most_common(1)


def partition_sort(arr, k, key=lambda x: x):
    """
    以枢纽（位置k）为中心把数组划分为两部分，枢纽左侧的元素不大于枢纽右侧的元素
    :param arr: 待划分数组
    :param k: 枢纽前部元素个数
    :param key: 比较方式
    :return: 无
    """
    start, end = 0, len(arr) - 1
    assert 0 <= k <= end
    while True:
        i, j, pivot = start, end, deepcopy(arr[start])
        while i < j:
            # 从右向左查找最小元素
            while i < j and key(pivot) <= key(arr[j]):
                j -= 1
            if i == j:
                break
            arr[i] = arr[j]
            i += 1
            # 从左向右查找较大元素
            while i < j and key(pivot) >= key(arr[i]):
                i += 1
            if i == j:
                break
            arr[j] = arr[i]
            j -= 1
        arr[i] = pivot
        if i == k:
            return
        elif i < k:
            start = i + 1
        else:
            end = i - 1


def max_heapreplace(heap, new_code, key=lambda x: x[1]):
    """
    大根堆替换堆顶元素
    :param heap:大根堆/列表
    :param new_code: 新节点
    :param key: 无
    :return: 无
    """
    heap[0] = new_code
    root, child = 0, 1
    end = len(heap) - 1
    while child <= end:
        if child < end and key(heap[child]) < key(heap[child + 1]):
            child += 1
        if key(heap[child]) <= key(new_code):
            break
        heap[root] = heap[child]
        root, child = child, 2 * child + 1
    heap[root] = new_code


def max_heappush(heap, new_code, key=lambda x: x[1]):
    """
    大根堆插入元素
    :param heap:大根堆/列表
    :param new_code: 新节点
    :param key:
    :return: 无
    """
    heap.append(new_code)
    pos = len(heap) - 1
    while 0 < pos:
        parent_pos = pos - 1 >> 1
        if key(new_code) <= key(heap[parent_pos]):
            break
        heap[pos] = heap[parent_pos]
        pos = parent_pos
    heap[pos] = new_code


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


class KDTree(object):
    """kd树"""

    def __init__(self, X, y=None):
        """
        构造函数
        :param X:输入特征集，n_samples * n_features
        :param y:l * n_samples
        """
        self.root = None
        self.y_vaild = False if y is None else True
        self.create(X, y)

    def create(self, X, y=None):
        """
        构建KD树
        :param X:输入特征集，n_samples * n_features
        :param y:l * n_samples
        :return: KDNode
        """

        def create_(X, axis, parent=None):
            """
            递归生成kd树
            :param X: 合并标签后输入集
            :param anxi: 切分轴
            :param parent: 父节点
            :return: KDNode
            """
            n_samples = np.shape(X)[0]
            if n_samples == 0:
                return None
            mid = n_samples >> 1
            partition_sort(X, mid, key=lambda x: x[axis])
            if self.y_vaild:
                kd_node = KDNode(X[mid][:-1], X[mid][-1], axis=axis)
            else:
                kd_node = KDNode(X[mid], axis=axis)
            next_axis = (axis + 1) % k_dimensions
            kd_node.left = create_(X[:mid], next_axis, kd_node)
            kd_node.right = create_(X[mid + 1:], next_axis, kd_node)
            return kd_node

        print('building kd-tree...')
        k_dimensions = np.shape(X)[1]
        if y is not None:
            X = np.hstack((np.array(X), np.array([y]).T)).tolist()
        self.root = create_(X, 0)

    def search_knn(self, point, k, dist=None):
        """
        kd树中搜索k个最近邻样本
        :param point: 样本点
        :param dist: 度量方式
        :return:
        """

        def search_knn_(kd_node):
            """
            搜索k近邻节点
            :param kd_node:KDNode
            :return:
            """
            if kd_node is None:
                return
            data = kd_node.data
            distance = p_dist(data)
            if len(heap) < k:
                # 向大根堆插入新元素
                max_heappush(heap, (kd_node, distance))
            elif distance < heap[0][1]:
                # 替换大根堆堆顶元素
                max_heapreplace(heap, (kd_node, distance))
            axis = kd_node.axis
            if abs(point[axis] - data[axis]) < heap[0][1] or len(heap) < k:
                # 当分割最小球体与分割超平面相交或堆中元素小于K个
                search_knn_(kd_node.left)
                search_knn_(kd_node.right)
            elif point[axis] < data[axis]:
                search_knn_(kd_node.left)
            else:
                search_knn_(kd_node.right)

        if self.root is None:
            raise Exception("kd_tree must be null.")
        if k < 1:
            raise ValueError("k must be greater than 0.")
        # 默认使用2个范式度量距离
        if dist is None:
            p_dist = lambda x: norm(np.array(x) - np.array(point))
        else:
            p_dist = lambda x: dist(x, point)
        heap = []
        search_knn_(self.root)
        return sorted(heap, key=lambda x: x[1])

    def search__nn(self, point, dist=None):
        """
        搜索point在样本中的最近邻
        :param self:
        :param point:
        :param dist:
        :return:
        """
        return self.search_knn(point, 1, dist)[0]

    def pre_order(self, root=KDNode()):
        """先序遍历"""
        if root is None:
            return
        elif root.data is None:
            root = self.root
        yield root
        for x in self.pre_order(root.left):
            yield x
        for x in self.pre_order(root.right):
            yield x

    def lev_order(self, root=KDNode(), queue=None):
        """层次遍历"""
        if root is None:
            return
        elif root.data is None:
            root = self.root
        if queue is None:
            queue = []
        yield root
        if root.left:
            queue.append(root.left)
        if root.right:
            queue.append(root.right)
        if queue:
            for x in self.lev_order(queue.pop(0), queue):
                yield x

    @classmethod
    def height(cls, root):
        """kd树深度"""
        if root is None:
            return 0
        else:
            return max(cls.height(root.left), cls.height(root.right)) + 1


class KNeighborsClassifier(object):
    """K近邻分类器"""

    def __init__(self, k, dist=None):
        """K近邻分类器"""
        self.k = k
        self.dist = dist
        self.kd_tree = None

    def fit(self, X, y):
        """构建kd树"""
        print("fitting....")
        X = self._data_processing(X)
        self.kd_tree = KDTree(X, y)

    def predict(self, X):
        """预测分类"""
        if self.kd_tree is None:
            raise TypeError("Classifier must be fitted before predict !")
        search_knn = lambda x: self.kd_tree.search_knn(point=x, k=self.k, dist=self.dist)
        y_ptd = []
        X = (X - self.x_min) / (self.X_max - self.x_min)
        for x in X:
            y = Counter(r[0].label for r in search_knn(x)).most_common(1)[0][0]
            y_ptd.append(y)
        return y_ptd

    def score(self, X, y):
        """预测正确率"""
        y_psd = self.predict(X)
        correct_nums = len(np.where(np.array(y_psd) == np.array(y))[0])
        return correct_nums / len(y)

    def _data_processing(self, X):
        X = np.array(X)
        self.x_min = np.min(X, axis=0)
        self.x_max = np.max(X, axis=0)
        X = (X - self.x_min) / (self.x_max - self.x_min)
        return X


# 代码测试
if __name__ == '__main__':
    """测试程序正确
    使用 kd-tree和计算距离，比较两种结果是否全部一致
    """
    N = 100000
    X = [[np.random.random() * 100 for _ in range(3)] for _ in range(N)]
    kd_tree = KDTree(X)

    for x in X[:10]:
        resl = ([list(node[0].data) for node in kd_tree.search_knn(x, 20)])
        distances = norm(np.array(X) - np.array(x), axis=1)
        res2 = ([list(X[i]) for _, i in sorted(zip(distances, range(N)))[:20]])
        if all(x in res2 for x in resl):
            print('correct ^_^^_^')
        else:
            print('error >-< >_<')
    print('\n')

    """10万个样本查找10个实例的最近邻"""
    n = 10
    indices = random.sample(range(N), n)
    # 1.kd-tree搜索，
    tm = time()
    for i, index in enumerate(indices):
        kd_tree.search__nn(X[index])
    print('kd - tree search:{}s'.format(time() - tm))

    # 2.numpy 计算全部样本与新实例的距离
    tm = time()
    for i, index in enumerate(indices):
        min(norm(X - np.array(X[index]), axis=0))
    print('numpy search:{}s'.format(time() - tm))

    # 3.python循环计算距离
    tm = time()
    for i, index in enumerate(indices):
        min([norm(np.array(X[index]) - np.array(x)) for x in X])
        print('python search : {}s'.format(time() - tm))
        print()

if __name__ == '__main__':
    """模型测算"""
    X, y = [], []
    with open(r'knn_dataset.txt') as f:
        for line in f:
            tmp = line.strip().split('\t')
            X.append(tmp[:-1])
            y.append(tmp[-1])
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    """训练误差"""
    X_train, X_test = X[:980], X[-20:]
    y_train, y_test = y[:980], y[-20:]
    knc = KNeighborsClassifier(10)
    knc.fit(X_train,y_train)
    print(knc.score(X_test,y_test)) # 1.0
