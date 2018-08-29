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
avgx = np.mean(data)
print('avgs:',avgx)


estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(iris.data)#聚类
label_pred = estimator.labels_ #获取聚类标签
kmeans_centroids = estimator.cluster_centers_ #获取聚类中心

print('target :',iris.target)
print('pred_label in Kmeans:',label_pred)
print('centroids',kmeans_centroids)
'''
count = 0
for n in range(150):
    if (iris.target[n] != label_pred[n]):
        count = count + 1
'''

#EM
# Try GMMs using full covariances.
estimator2 = GaussianMixture(n_components=3,covariance_type='full', max_iter=20, random_state=0)
estimator2.fit(iris.data)
y_pred = estimator2.predict(iris.data)
gmm_center = estimator2.means_
print('gmm:',gmm_center)
print('y_pred in EM:',y_pred)
'''
for n in range(150):
    if (iris.target[n] != y_pred[n]):
        count = count + 1
print('count:',count)
'''
order = pairwise_distances_argmin(kmeans_centroids,gmm_center,axis=1,metric='euclidean')
print('order:',order)
#order1 = pairwise_distances_argmin(iris.target,label_pred,axis=1,metric='euclidean')
#order2 = pairwise_distances_argmin(iris.target,y_pred,axis=1,metric='euclidean')

#print('lenofprelabel:',len(label_pred))
##   iris数据集是四维的,图中显示前两个维度
fig = plt.figure(figsize = (8,3))
fig.subplots_adjust(left = 0.02, right = 0.98, bottom = 0.05, top = 0.9)

ax1 = fig.add_subplot(131)
ax1.set_title('kmeans:')
ax1.scatter(data[:, 0], data[:, 1], s=5, c=label_pred)

ax2 = fig.add_subplot(132)
ax2.scatter(data[:,0],data[:,1], s=5, c=iris.target)
ax2.set_title('origin:')
for k in range(3):
    ax3 = fig.add_subplot(133)
    gmm_members = y_pred == order[k]
    cluster_center = gmm_center[order[k]]
    ax3.scatter(data[:,0],data[:,1], s=5, c=y_pred)
print('gmm_pred:',gmm_members)
ax3.set_title('EM:')

plt.show()
#print(count)