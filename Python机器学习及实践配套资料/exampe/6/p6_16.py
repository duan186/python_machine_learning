import numpy as np
import matplotlib.pyplot as plt
cs = ['black', 'blue', 'brown', 'red', 'yellow', 'green']
class NpCluster(object):
    def __init__(self):
        self.key = []
        self.value = []

    def append(self, data):
        if str(data) in self.key:
            return
        self.key.append(str(data))
        self.value.append(data)

    def exist(self, data):
        if str(data) in self.key:
            return True
        return False

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        self.times = 0
        return self

    def __next__(self):
        try:
            ret = self.value[self.times]
            self.times += 1
            return ret
        except IndexError:
            raise StopIteration()
def create_sample():
    np.random.seed(10)  # 随机数种子，保证随机数生成的顺序一样
    n_dim = 2
    num = 100
    a = 3 + 5 * np.random.randn(num, n_dim)
    b = 30 + 5 * np.random.randn(num, n_dim)
    c = 60 + 10 * np.random.randn(1, n_dim)
    data_mat = np.concatenate((np.concatenate((a, b)), c))
    ay = np.zeros(num)
    by = np.ones(num)
    label = np.concatenate((ay, by))
    return {'data_mat': list(data_mat), 'label': label}
    
def region_query(dataset, center_point, eps):
    result = NpCluster()
    for point in dataset:
        if np.sqrt(sum(np.power(point - center_point, 2))) <= eps:
            result.append(point)
    return result

def dbscan(dataset, eps, min_pts):
    noise = NpCluster()
    visited = NpCluster()
    clusters = []
    for point in dataset:
        cluster = NpCluster()
        if not visited.exist(point):
            visited.append(point)
            neighbors = region_query(dataset, point, eps)
            if len(neighbors) < min_pts:
                noise.append(point)
            else:
                cluster.append(point)
                expand_cluster(visited, dataset, neighbors, cluster, eps, min_pts)
                clusters.append(cluster)
    for data in clusters:
        print(data.value)
        plot_data(np.mat(data.value), cs[clusters.index(data)])
    if noise.value:
        plot_data(np.mat(noise.value), 'green')
    plt.show()

def plot_data(samples, color, plot_type='o'):
    plt.plot(samples[:, 0], samples[:, 1], plot_type, markerfacecolor=color, markersize=14)

def expand_cluster(visited, dataset, neighbors, cluster, eps, min_pts):
    for point in neighbors:
        if not visited.exist(point):
            visited.append(point)
            point_neighbors = region_query(dataset, point, eps)
            if len(point_neighbors) >= min_pts:
                for expand_point in point_neighbors:
                    if not neighbors.exist(expand_point):
                        neighbors.append(expand_point)
                if not cluster.exist(point):
                    cluster.append(point)
                    
init_data = create_sample()
dbscan(init_data['data_mat'], 10, 3)
