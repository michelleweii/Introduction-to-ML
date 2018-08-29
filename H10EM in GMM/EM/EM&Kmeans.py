# -*- coding: UTF-8 -*-     

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt
#pairwise_distances_argmin: Compute minimum distances between one point and a set of points.

#sklearn.metrics.pairwise_distances_argmin：
 # 对两个二维数组进行距离计算，按顺序（按行）在计算第二个数组中与第一个
 # 数组距离队最近的相应行的下标。




#labels assignment is also called the E-step of EM
#computation of the means is also called the M-step of EM
colors = ['navy', 'turquoise', 'darkorange']
#GaussianMixture
iris = load_iris()
#iris数据集是四维的,图中显示前两个维度
#X = iris.data
#y = iris.target
#print('X:',X)
#print('y：',y)
#用K-means和EM算法对样本进行聚类，并比较结果labels assignment is also called the E-step of EM
estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(iris.data)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
#inertia = estimator.inertia_ # 获取聚类准则的最后值


fig = plt.figure(figsize = (8, 3))
fig.subplots_adjust(left = 0.02, right = 0.98, bottom = 0.05, top = 0.9)
ax1 = fig.add_subplot(1, 2, 1)
'''
for i in range(len(centroids)):
    ax1.plot(centroids[i][0],centroids[i][1],'o',color = colors[i],markersize = 6)
    ax1.scatter(X[:, 0], X[:, 1], color = colors[i]) #scatter绘制散点
'''
for n, color in enumerate(colors):
    data = iris.data[iris.target == n]
    plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
                label=label_pred[n])
#scatter绘制质心
#plt.scatter(centroids[])
#for i in range(len(centroids))


#ax.title("Iris data in Kmeans")   #加标题
plt.show()
#防止聚类卷标不一致,导致统计正确率失真
#order1 = pairwise_distances_argmin(mu,mu1,axis=1,metric='euclidean')
#order2 = pairwise_distances_argmin(mu,gmm.means_,axis=1,metric='euclidean')
