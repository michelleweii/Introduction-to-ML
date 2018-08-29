# -*- coding: UTF-8 -*-    
# from：http://blog.csdn.net/lilianforever/article/details/53780613 
#sklearn学习笔记
import numpy as np
import scipy
import matplotlib
from sklearn.cluster import KMeans
# data = np.random.rand(100,3) #生成一个随机数据，样本大小为100, 特征数为3
# ##假如我要构造一个聚类数为3的聚类器
#
# estimator = KMeans(n_clusters=3) #构造聚类器
# estimator.fit(data) #聚类
# label_pred = estimator.labels_  #获取聚类标签
# centroids = estimator.cluster_centers_  #获取聚类中心
# inertia = estimator.inertia_  # 获取聚类准则的最后值
#
# print(label_pred)
# print(centroids)
# print(inertia)
# #直接采用kmeans函数：
# from sklearn import cluster
# data = np.random.rand(100,3)
# k=3 # 假如我要聚类为3个clusters
# [centroids,label,inertia] = cluster.KMeans(data,k)

# #Classification
# from sklearn import neighbors,datasets
# iris = datasets.load_iris()
# n_neighbors = 15
# X = iris.data[:,:2]  # we only take the first two features.
# y = iris.target
#
# weights = 'distance' # also set as 'uniform'
# clf = neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
# clf.fit(X,y)
#
# # if you have test data, just predict with the following functions
# # for example, xx, yy is constructed test data
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# #h应该是多少？？
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
# #np.arange(1,10,2)
# #第三个参数是步长，arange()返回的是array([1,3,5,7,9]),是一个list
#
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Z is the label_pred

# #SVM
# from sklearn import svm
# X = [[0,0],[1,1]]
# y = [0,1]
# #建立支持向量分类模型
# clf = svm.SVC()
# #拟合训练数据，得到训练模型参数
# clf.fit(X,y)
# #对测试点[2., 2.], [3., 3.]预测
# res = clf.predict([[2.,2.],[3.,3.]])
# #输出预测结果值
# print(res)
# #get support vectors
# print('support vectors:',clf.support_vectors_)
# #get indices of support vectors
# print('indices of support vectors:',clf.support_)
# #get number of support vectors for each class
# print('number of support vectors for each class:',clf.n_support_)
#
# # [1 1]
# # support vectors: [[ 0.  0.]
# #  [ 1.  1.]]
# # indices of support vectors: [0 1]
# # number of support vectors for each class: [1 1]

# #SVR
# from sklearn import svm
# X = [[0,0],[2,2]]
# y = [0.5,2.5]
# clf = svm.SVR()
# clf.fit(X,y)
# res = clf.predict([[1,1]])
# print(res)
# #[ 1.5]

# #逻辑回归
# from sklearn import linear_model
# X = [[0,0],[1,1]]
# y = [0,1]
# logreg = linear_model.LogisticRegression(C=1e5)
# #we create an instance of Neighbours Classifier and fit the data.
# logreg.fit(X,y)
# res = logreg.predict([[2,2]])
# print(res)
# #[1]

# #preprocessing
# import numpy as np
# from sklearn import preprocessing
# X = np.random.rand(3,4)
# print(X)
# #用scaler的方法
# scaler = preprocessing.MinMaxScaler()
# X_scaled = scaler.fit_transform(X) #fit()与fit_transform()的区别，前者仅训练一个模型，没有返回nmf后的分支，而后者除了训练数据，并返回nmf后的分支
# #用scale函数的方法
# X_scaled_convinent = preprocessing.minmax_scale(X)
# print(X_scaled_convinent)
# # [[ 0.84654775  0.47749349  0.65988479  0.2546447 ]
# #  [ 0.4637643   0.8935383   0.29814965  0.67649929]
# #  [ 0.01350813  0.85737521  0.57029029  0.81949678]]
# # [[ 1.          0.          1.          0.        ]
# #  [ 0.5404979   1.          0.          0.74684082]
# #  [ 0.          0.91307886  0.75232017  1.        ]]

# # Decomposition
# # NMF 非负矩阵分解
# import numpy as np
# X = np.array([[1,1],[2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
# from sklearn.decomposition import NMF
# model = NMF(n_components=2,init='random',random_state=0)
# model.fit(X)
#
# print(model.components_)
# print(model.reconstruction_err_)
# print(model.n_iter_)
# # [[ 2.09783018  0.30560234]
# #  [ 2.13443044  2.13171694]]
# # 0.0011599349216
# # 29

# # Decomposition
# # PCA
# import numpy as np
# X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
# print(X)
# from sklearn.decomposition import PCA
# model = PCA(n_components=2)
# model.fit(X)
#
# print(model.components_)
# print(model.n_components_)
# print(model.explained_variance_)
# print(model.explained_variance_ratio_)
# print(model.mean_)
# print(model.noise_variance_)
# # [[ 1.   1. ]
# #  [ 2.   1. ]
# #  [ 3.   1.2]
# #  [ 4.   1. ]
# #  [ 5.   0.8]
# #  [ 6.   1. ]]
# # [[-0.99973675  0.02294398]
# #  [ 0.02294398  0.99973675]]
# # 2
# # [ 3.501836  0.014164]
# # [ 0.99597156  0.00402844]
# # [ 3.5  1. ]
# # 0.0

# # Metrics (聚类分类任务，都需要最后的评估)
# from sklearn.metrics import accuracy_score
# y_pred = [0, 2, 1, 3]
# y_ture = [0, 1, 2, 3]
# ac = accuracy_score(y_ture,y_pred)
# print(ac)
# ac2 = accuracy_score(y_ture,y_pred,normalize=False)
# print(ac2)
# # 0.5
# # 2

# #求的是聚类结果的NMI（标准互信息），其他指标也类似
# from sklearn.metrics import normalized_mutual_info_score
# y_pred = [0,0,1,1,2,2]
# y_true = [1,1,2,2,3,3]
# nmi = normalized_mutual_info_score(y_true,y_pred)
# print(nmi)
# #output:1.0


