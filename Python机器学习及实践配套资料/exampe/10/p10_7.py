#导入time模块，用于测算一些步骤的时间消耗。
from time import time
#导入Python科学计算的基本需求模块，主要包括NumPy（矩阵计算模块）、SciPy（科学计算模块）和matplotlib.pyplot模块（画图）。有了这三个模块，Python俨然已是基础版的Matlab
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp 
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
#导入稀疏字典学习所需要的函数
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
#检测SciPy版本，如果版本太低就抛出一个异常
if sp_version < (0, 12):
    raise SkipTest("Skipping because SciPy version earlier than 0.12.0 and "
                   "thus does not include the scipy.misc.face() image.")
#尝试打开样本测试用例，如果打不开就抛出一个异常
try:
    from scipy import misc
    face = misc.face(gray=True)
except AttributeError:
    # 旧版本的scipy在顶层包中有face
    face = sp.face(gray=True)
#读入的face大小在0~255之间，所以通过除以255将face的大小映射到0~1上去
face = face / 255.0 
#对图形进行采样，把图片的长和宽各缩小一般。记住array矩阵的访问方式      array[起始点：终结点（不包括）：步长]
face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
#图片的长宽大小
face = face / 4.0
height, width = face.shape
print('Distorting image...')
#将face的内容复制给distorted，这里不用等号因为等号在python中其实是地址的引用
distorted = face.copy()
distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2) 
print('Extracting reference patches...')
#开始计时，并保存在t0中
t0 = time()
#tuple格式的pitch大小
patch_size = (7, 7)
#对图片的左半部分（未加噪声的部分）提取pitch
data = extract_patches_2d(distorted[:, :width // 2], patch_size)
#用reshape函数对data(94500,7,7)进行整形，reshape中如果某一位是-1，则这一维会根据（元素个数/已指明的维度）来计算这里经过整形后data变成（94500，49）
data = data.reshape(data.shape[0], -1)
#每一行的data减去均值除以方差，这是zscore标准化的方法
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print('done in %.2fs.' % (time() - t0))
print('Learning the dictionary...')
t0 = time()
#初始化MiniBatchDictionaryLearning类，并按照初始参数初始化类的属性
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
#调用fit方法对传入的样本集data进行字典提取，components_返回该类fit方法的运算结果，也就是我们想要的字典V
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)
#画出V中的字典
plt.figure(figsize=(4.2, 4)) #figsize方法指明图片的大小，4.2英寸宽，4英寸高。其中一英寸的定义是80个像素点
for i, comp in enumerate(V[:100]):  #循环画出100个字典V中的字
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from face patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)#6个参数与注释后的6个属性对应
def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 2, 2)
    difference = image - reference
 
    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2) 
show_with_diff(distorted, face, 'Distorted image')
print('Extracting noisy patches... ')
t0 = time()
#提取照片中被污染过的右半部进行字典学习
data = extract_patches_2d(distorted[:, width // 2:], patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print('done in %.2fs.' % (time() - t0))
#四中不同的字典表示策略
transform_algorithms = [
    ('Orthogonal Matching Pursuit\n1 atom', 'omp',
     {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms' , 'omp',
     {'transform_n_nonzero_coefs': 2}),
    ('Least-angle regression\n5 atoms', 'lars',
     {'transform_n_nonzero_coefs': 5}),
    ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha‘': .1})]
 
reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + '...')
    reconstructions[title] = face.copy()
    t0 = time()
    '''transform根据set_params对设完参数的模型进行字典表示，表示结果放在code
中。code总共有100列，每一列对应着V中的一个字典元素，所谓稀疏性就是
code中每一行的大部分元素都是0，这样就可以用尽可能少的字典元素表示回去'''
    code = dico.transform(data)
    #code矩阵乘V得到复原后的矩阵patches
    patches = np.dot(code, V)
 
    patches += intercept
    #将patches从（94500，49）变回（94500，7，7）
    patches = patches.reshape(len(data), *patch_size)
    if transform_algorithm == 'threshold':
        patches -= patches.min()
        patches /= patches.max()
   #通过reconstruct_from_patches_2d函数将patches重新拼接回图片
    reconstructions[title][:, width // 2:] = reconstruct_from_patches_2d(
        patches, (height, width // 2))
    dt = time() - t0
    print('done in %.2fs.' % dt)
    show_with_diff(reconstructions[title], face,
                   title + ' (time: %.1fs)' % dt) 
plt.show()
