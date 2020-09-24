import random
from copy import deepcopy
from time import time
import numpy as np
from numpy.linalg import norm
from collections import Counter
from .KDNode import KDNode

Counter([0, 1, 1, 2, 2, 3, 3, 4, 3, 3]).most_common(1)

def partition_sort(arr,k,key = lambda x : x):
    """
    以枢纽（位置k）为中心把数组划分为两部分，枢纽左侧的元素不大于枢纽右侧的元素
    :param arr: 待划分数组
    :param k: 枢纽前部元素个数
    :param key: 比较方式
    :return: 无
    """
    start ,end = 0 ,len(arr)-1
    assert 0<= k <=end
    while True:
        i, j, pivot = start, end ,deepcopy(arr(start))
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


def max_heapreplace(heap,new_code,key = lambda x : x[1]):
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
    while child <= end :
        if child < end and key(heap[child]<key(heap[child+1])):
            child += 1
        if key(heap[child]) <= key(new_code):
            break
        heap[root] = heap[child]
        root, child = child, 2 * child + 1
    heap[root] = new_code



def max_heappush(heap,new_code,key = lambda x: x[1]):
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
        parent_pos = pos - 1 >>1
        if key(new_code) <= key(heap[parent_pos]):
            break
        heap[pos] = heap[parent_pos]
        pos = parent_pos
    heap[pos] = new_code


