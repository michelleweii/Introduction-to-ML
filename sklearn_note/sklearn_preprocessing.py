# -*- coding: UTF-8 -*-  
# from:http://blog.csdn.net/zhangyang10d/article/details/53418227
# sklearn----数据预处理 sklearn.preprocessing   
#1.标准化Standardization（这里指移除均值和方差标准化）
# 1.1 z-score标准化
# z-score标准化指的是将数据转化成均值为0方差为1的高斯分布，也就是通常说的z-score标准化，但是对于不服从标准正态分布的特征，这样做效果会很差。
# 如果目标函数中的一个特征的方差的阶数的量级高于其他特征的方差，那么这一特征就会在目标函数中占主导地位，从而“淹没”其他特征的作用。
#
# # Z-score标准化后的数据的均值为0，方差为1。
# from sklearn import preprocessing
# x = [[1., -1., 2],
#      [2., 0., 0.],
#      [ 0., 1., -1.]]  # 每一行为[feature1, feature2, feature3]
# print(x)
# x_scaled = preprocessing.scale(x)
# print(x_scaled)
# print(x_scaled.mean(axis=0))  #均值为0
# print(x_scaled.std(axis=0))   #方差为1
# # [[1.0, -1.0, 2], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
# # [[ 0.         -1.22474487  1.33630621]
# #  [ 1.22474487  0.         -0.26726124]
# #  [-1.22474487  1.22474487 -1.06904497]]
# # [ 0.  0.  0.]
# # [ 1.  1.  1.]

# # preprocessing模块还提供了一个实用类StandardScaler，这个类实现了一个叫做Transformer的应用程序接口，
# # 能够计算训练数据的均值和标准差，从而在训练数据集上再次使用。
# print('*********************************************')
# scaler = preprocessing.StandardScaler().fit(x)
# print('scaler：');print(scaler)
# print('scaler.mean_:');print(scaler.mean_)
# print('scaler.scale_:');print(scaler.scale_)
# print('scaler.transform(x):');print(scaler.transform(x))
# scaler = preprocessing.StandardScaler().fit(x)
# print(scaler)
# print(scaler.transform([[-1.,1.,0.]]))  # 在其他数据集上使用
# # scaler：
# # StandardScaler(copy=True, with_mean=True, with_std=True)
# # scaler.mean_:
# # [ 1.          0.          0.33333333]
# # scaler.scale_:
# # [ 0.81649658  0.81649658  1.24721913]
# # scaler.transform(x):
# # [[ 0.         -1.22474487  1.33630621]
# #  [ 1.22474487  0.         -0.26726124]
# #  [-1.22474487  1.22474487 -1.06904497]]
# # StandardScaler(copy=True, with_mean=True, with_std=True)
# # [[-2.44948974  1.22474487 -0.26726124]]

#1.2 将特征数据缩放到一个范围 scale to a range
# 利用最大值和最小值进行缩放，通常是将数据缩放到0-1这个范围，或者是将每个特征的绝对值最大值缩放到单位尺度，分别利用MinMaxScaler和MaxAbsScaler实现。
# 使用这一方法的情况一般有两种：
# (1) 特征的标准差较小
# (2) 可以使稀疏数据集中的0值继续为0
import numpy as np
from sklearn import preprocessing
# x = [[1.0, -1.0, 2], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled_minmax = min_max_scaler.fit_transform(x)
# print(x_scaled_minmax)
# # 这个transformer的实例还能够应用于新的数据集，此时的缩放比例与之前训练集上的缩放比例是相同的。
# x_test = np.array([[3.,1.,4.]])
# print('x_test:');print(x_test)
# print(min_max_scaler.transform(x_test))

#
#
#************  缩放比例应用于新的数据集，这个不太明白      ***************
#
#

#MaxAbsScaler与上述用法相似，但是标准化后的数据的取值范围为[-1, 1]。这对于稀疏数据或者是数据中心已经为0的数据很有意义。
# x = [[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]
# max_abs_scaler = preprocessing.MaxAbsScaler()
# print(max_abs_scaler.fit_transform(x))
# x_test = [[-2., 4., 2.]]
# print('max_abs_scaler.transform(x_test):');print(max_abs_scaler.transform(x_test))
# print('max_abs_scaler.scale_:');print(max_abs_scaler.scale_)
# [[ 0.5 -1.   1. ]
#  [ 1.   0.   0. ]
#  [ 0.   1.  -0.5]]
# max_abs_scaler.transform(x_test):
# [[-1.  4.  1.]]
# max_abs_scaler.scale_:
# [ 2.  1.  2.]

# # 2. 规范化（Normalization）
# # 规范化是指将样本缩放成单位向量。如果需要使用二次方程，比如点积或者其他核方法计算样本对之间的相似性，那么这一过程非常有用。
# # 这一假设是常用于文本分类和内容聚类的向量空间模型的基础。
# # normalize函数提供了一个处理单个结构类似数组的数据集的快速简单的方法，可以使用1范数l1或者2范数l2。
#
# from sklearn import preprocessing
# x = [[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]
# x_normalized = preprocessing.normalize(x,norm='l2')
# print(x_normalized)
#
# # 类似的，preprocessing模块也提供了一个实体类Normalizer，
# # 能够利用Transformer API执行相同的操作（虽然fit方法这里是没有意义的，因为规范化是对于每个样本独立进行的）。
# x1 = [[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]
# normalizer = preprocessing.Normalizer().fit(x1)
# print(normalizer)
# preprocessing.Normalizer(copy=True,norm='l2')
# print(normalizer.transform(x1))
# print(normalizer.transform([[1., -1., 0]]))
# # 对于稀疏的输入数据，normalize和Normalizer可以接受非稀疏数组类型和稀疏矩阵类型
# # 左右的输入。稀疏数据将被转化为压缩的稀疏行表示法。
# [[ 0.40824829 -0.40824829  0.81649658]
#  [ 1.          0.          0.        ]
#  [ 0.          0.70710678 -0.70710678]]
# Normalizer(copy=True, norm='l2')
# [[ 0.40824829 -0.40824829  0.81649658]
#  [ 1.          0.          0.        ]
#  [ 0.          0.70710678 -0.70710678]]
# [[ 0.70710678 -0.70710678  0.        ]]

# # 3. 二值化
# # 3.1 特征二值化
# # 这一过程就是定义一个阈值，然后得到数值特征的布尔值。这对于假设输入数据服从多元伯努利分布的概率估计量非常有用。这在文本处理过程中也非常常见。
# # 实例类Binarizer可以实现这一过程。同样的，fit函数没有意义。
# from sklearn import preprocessing
# x = [[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]
# binarizer = preprocessing.Binarizer().fit(x)
# print(binarizer)
# print(binarizer.transform(x))
# binarizer = preprocessing.Binarizer(threshold=1.1)
# print(binarizer.transform(x))
#

#
# 4. 分类特征编码
#
# 没看懂
#
#
# 5. 推定缺失数据
# Imputer类能够提供一些处理缺失值的基本策略，例如使用缺失值所处的一行或者一列的均值、
# 中位数或者出现频率最高的值作为缺失数据的取值。下边举一个使用缺失值所处行的均值作为缺失值的例子：
# import numpy as np
# from sklearn.preprocessing import Imputer
# # 很多情况下，真实的数据集中会存在缺失值，此时数据集中会采用空格、NaNs或者其他占位符进行记录。
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imp.fit([[1,2],[np.nan,3],[7,6]])
# Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
# X = [[np.nan, 2], [6, np.nan], [7, 6]]
# print(imp.transform(X))
# # [[ 4.          2.        ]
# #  [ 6.          3.66666667]
# #  [ 7.          6.        ]]

# # Imputer也支持稀疏矩阵作为输入：
# import scipy.sparse as sp
# X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])
# imp = Imputer(missing_values=0, strategy='mean', axis=0)
# imp.fit(X)
# X_test = sp.csc_matrix([[0, 2], [6, 0], [7, 6]])
# print(imp.transform(X_test))
# # [[ 4.          2.        ]
# #  [ 6.          3.66666667]
# #  [ 7.          6.        ]]

# 6. 产生多项式特征
# 在输入数据存在非线性特征时，这一操作对增加模型的复杂度十分有用。一种常见的用法是生成多项式特征，能够得到特征的高阶项和相互作用项。利用PolynomialFeatures实现：
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# X = np.arange(6).reshape(3,2)
# print(X)
# poly = PolynomialFeatures(2)
# print(poly.fit_transform(X))
# [[0 1]
#  [2 3]
#  [4 5]]
# [[  1.   0.   1.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.]
#  [  1.   4.   5.  16.  20.  25.]]
# 此时，特征向量X=(X1, X2)被转化为(1, X1, X2, X1^2, X1X2, X2^2)。
#
#
# 在有些情况下，我们只需要相互作用项，此时可以通过设定interaction_only=True实现：
# X = np.arange(9).reshape(3,3)
# print(X)
# poly = PolynomialFeatures(degree=3,interaction_only=True)
# print(poly.fit_transform(X))
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
# [[   1.    0.    1.    2.    0.    0.    2.    0.]
#  [   1.    3.    4.    5.   12.   15.   20.   60.]
#  [   1.    6.    7.    8.   42.   48.   56.  336.]]

# 这里，X=(X1, X2, X3)被转化为to (1, X1, X2, X3, X1X2, X1X3, X2X3, X1X2X3)。
# 多项式特征经常用于使用多项式核函数的核方法（比如SVC和KernelPCA）


# 7. 定制转换器
# 我们经常希望将一个Python的函数转变为transformer，用于数据清洗和预处理。
# 可以使用FunctionTransformer方法将任意函数转化为一个Transformer。
# 比如，构建一个对数log的Transformer：
import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
X = np.array([[0, 1], [2, 3]])
print(X)
print(transformer.transform(X))

# [[0 1]
#  [2 3]]
# [[ 0.          0.69314718]
#  [ 1.09861229  1.38629436]]