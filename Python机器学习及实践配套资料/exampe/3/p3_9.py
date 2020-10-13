#导入必要的编程库
from time import time
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition
#设置参数
n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)
# 载入face数据
dataset = fetch_olivetti_faces('./', True, random_state=rng)
faces = dataset.data
n_samples, n_features = faces.shape
print("Dataset consists of %d faces, features is %s" % (n_samples, n_features))
#显示原始图像
def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
#显示压缩后的图像
        plt.imshow(comp.reshape(image_shape), cmap=cmap,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

estimators = [
    ('Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3))
]
# 绘制输入数据的示例
plot_gallery("First centered Olivetti faces", faces[:n_components])

# 估算并绘制它
for name, estimator in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()
    data = faces
    estimator.fit(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    components_ = estimator.components_
    print('components_:', components_.shape, '\n**\n', components_)
    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                 components_)
plt.show()
