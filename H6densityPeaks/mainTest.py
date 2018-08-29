# coding:utf-8
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, \
    AffinityPropagation, spectral_clustering
from sklearn.metrics import euclidean_distances
import myDensityPeaks


n_samples = 900
random_state = 5
centers = [[-0.6, -0.4], [-0.3, 0.5], [0.5, -0.4]]
center_box = [-1, 1]
X, y = make_blobs(n_features=2, n_samples=n_samples,
                  cluster_std=0.15, center_box=center_box,
                  centers=centers, random_state=random_state)

m = np.array(((1, 1), (1, 3)))  #?????
X = X.dot(m)
print('X:',X)
# tmp = y.tolist
print('yOriginal:',y)


# K-means
y_km = KMeans(n_clusters=3, init='k-means++', random_state=3).fit_predict(X)

# DBSCAN
eps = 0.075
min_samples = 5
y_db = DBSCAN(eps=eps, min_samples=5).fit_predict(X)  ##实行聚类并返回标签(n_samples, n_features)

dis = euclidean_distances(X, squared=True)
# MeanShift
bw = np.median(dis)
band_width = 0.4 * bw
y_ms = MeanShift(bin_seeding=True, bandwidth=band_width).fit_predict(X)

# AP
preference = -np.median(dis)
p = 2 * preference
y_ap = AffinityPropagation(affinity='euclidean', preference=p).fit_predict(X)

# Density Peaks
#
y_dp = myDensityPeaks.getIndex()


# 绘图
cm = matplotlib.colors.ListedColormap(list('rgbm'))

plt.figure(figsize=(12, 8), facecolor='w')
plt.subplot(231)
plt.title(u'Original Data')
plt.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))

plt.subplot(232)
plt.title(u'K-Means')
plt.scatter(X[:, 0], X[:, 1], c=y_km, s=5, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))

plt.subplot(233)
plt.title(u'DBSCAN')
plt.scatter(X[:, 0], X[:, 1], c=y_db, s=5, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))

plt.subplot(234)
plt.title(u'MeanShift')
plt.scatter(X[:, 0], X[:, 1], c=y_ms, s=5, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))

plt.subplot(235)
plt.title(u'AP')
plt.scatter(X[:, 0], X[:, 1], c=y_ap, s=5, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))

plt.subplot(236)
plt.title(u'Density peaks')
plt.scatter(X[:, 0], X[:, 1], c=y_dp, s=5, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.xlabel('where rate1 = 0.6,rate2 = 0.2!!! by Michelle :)')

plt.show()
