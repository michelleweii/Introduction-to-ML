# -*- coding: UTF-8 -*-     

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt
#pairwise_distances_argmin: Compute minimum distances between one point and a set of points.

iris = load_iris()
data = iris.data
y = iris.target
origin_centers = np.array([np.mean(data[y == i], axis=0) for i in range(3)])
print('origin_centers',origin_centers)
print('origin_target :',y)

#    Kmeans
estimator = KMeans(n_clusters=3)
estimator.fit(iris.data)    #聚类
kmeans_pred = estimator.labels_     #获取聚类标签
kmeans_centers = estimator.cluster_centers_    #获取聚类中心
print('kmeans_pred :',kmeans_pred)
print('kmeans_centers',kmeans_centers)

#---------------------------------------------------------------------------------------------
#    EM
# Try GMMs using full covariances.
estimator2 = GaussianMixture(n_components=3,covariance_type='full', max_iter=20, random_state=0)
estimator2.fit(data)
gmm_pred = estimator2.predict(data)
gmm_centers = estimator2.means_
print('gmm_centers:',gmm_centers)
print('gmm_pred :',gmm_pred)
# after_gmm_pred = pairwise_distances_argmin(data, gmm_centers)

#-----------------------------------------------------------------------------------------------

#防止聚类卷标不一致
order1 = pairwise_distances_argmin(origin_centers,kmeans_centers)
print('order1',order1)
order2 = pairwise_distances_argmin(origin_centers,gmm_centers)
print('order2',order2)
n_sample = y.size  #150
n_types = 3
change = np.empty((n_types, n_sample), dtype=np.bool)
# print(change)
for i in range(n_types):
    change[i] = kmeans_pred == order1[i]
for i in range(n_types):
    kmeans_pred[change[i]] = i
for i in range(n_types):
    change[i] = gmm_pred == order2[i]
for i in range(n_types):
    gmm_pred[change[i]] = i

print('after_kmeans_pred',kmeans_pred)
print('after_gmm_pred',gmm_pred)

#-----------------------------------------------------------------------------------------------


##   iris数据集是四维的,图中显示前两个维度
fig = plt.figure(figsize = (8,3))

ax1 = fig.add_subplot(131)
ax1.set_title('kmeans:')
ax1.scatter(data[:, 0], data[:, 1], s=5, c=kmeans_pred)
ax1.scatter(kmeans_centers[:,0], kmeans_centers[:,1], marker='x', color='r',s=12)

ax2 = fig.add_subplot(132)
ax2.scatter(data[:,0],data[:,1], s=5, c=iris.target)
ax2.set_title('origin:')


ax3 = fig.add_subplot(133)
ax3.scatter(data[:,0],data[:,1], s=5, c=gmm_pred)
ax3.set_title('EM:')
ax3.scatter(gmm_centers[:,0], gmm_centers[:,1], marker='x', color='r',s=12)

plt.show()

