# -*- coding: UTF-8 -*-  
#选择最常用的高斯核来建立相似矩阵，用K-Means来做最后的聚类。

import numpy as np
from sklearn import datasets
# 500个6维的数据集，分为5个簇
X,y = datasets.make_blobs(n_samples=500,n_features=6,centers=5,cluster_std=[0.4,0.3,0.4,0.3,0.4],random_state=11)
#查看默认的谱聚类的效果
from sklearn.cluster import SpectralClustering
# y_pred = SpectralClustering().fit_predict(X)

from sklearn import metrics
#Calinski-Harabasz分数越大越好
# print('Calinski-Harabasz Score',metrics.calinski_harabaz_score(X,y_pred))


# #用的是高斯核，一般需要对n_clusters和gamma进行调参。选择合适的参数值
# for index,gamma in enumerate((0.01,0.1,1,10)):
#     for index,k in enumerate((3,4,5,6)):
#         y_pred = SpectralClustering(n_clusters=k,gamma=gamma).fit_predict(X)
#         print('y_pred:',y_pred)
        # print('Calinski-Harabasz Score with gamma=',gamma,'n_clusters=',k,'scores:',metrics.calinski_harabaz_score(X,y_pred))


y_pred = SpectralClustering(gamma=0.1).fit_predict(X)
print('y_pred:',y_pred)
print('y:',y)