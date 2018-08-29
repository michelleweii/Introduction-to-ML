# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn import metrics
from sklearn.metrics import euclidean_distances

if __name__ == "__main__":
    fileName = 'Spiral.txt'
    X = []
    y = []
    for line in open(fileName, "r"):
        items = line.strip("\n").split(",")
        y.append(int(items.pop()))
        tmp = []
        for item in items:
            tmp.append(float(item))
        X.append(tmp)
    X = np.array(X)
    y = np.array(y)
    # print('X:',X)
    # print('y:',y)

    from sklearn.cluster import spectral_clustering
    # y_pred = SpectralClustering().fit_predict(X)
    # print('Calinski-Harabasz Score：',metrics.calinski_harabaz_score(X,y_pred))
    #
    # for index,gamma in enumerate((0.01,0.1,1,10)):
    #     for index,k in enumerate((3,4,5,6)):
    #         y_pred = SpectralClustering(n_clusters=k,gamma=gamma).fit_predict(X)
    #         print('Calinski-Harabasz Score with gamma=',gamma,'n_clusters=',k,'scores:',metrics.calinski_harabaz_score(X,y_pred))

    n_clusters = 3
    # y_hat = spectral_clustering(X, n_clusters=n_clusters, assign_labels='kmeans', random_state=1)


    cm = matplotlib.colors.ListedColormap(list('rgbm'))
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.suptitle(u'The spectral clustering results in different gamma.', fontsize=20)


    for index,gamma in enumerate((0.01,0.1,0.2,0.5,0.6,1)):
        y_pred = SpectralClustering(gamma=gamma,n_clusters=n_clusters).fit_predict(X)
        print('Calinski-Harabasz Score with gamma=',gamma,',Calinski-Harabasz Score：',metrics.calinski_harabaz_score(X,y_pred))
        #randindex越大越好,原始数据标签已知
        # print('Rand index with gamma=',gamma,',Rand index Score：',metrics.adjusted_rand_score(y, y_pred))
        # print('silhouette_score with gamma=', gamma, ',silhouette_score Score：', metrics.silhouette_score(X,y_pred,metric='euclidean'))
        # print('y_pred:',y_pred)
        plt.subplot(2, 3,index+1)
        plt.scatter(X[:, 0], X[:, 1], s=40, c=y_pred, edgecolors='k')
        x1_min, x2_min = np.min(X, axis=0)
        x1_max, x2_max = np.max(X, axis=0)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.grid(True)
        plt.title(gamma, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

#Calinski-Harabasz Score的评判标准和这个数据样本好像不符合？？？ Calinski-Harabasz分数值越大则聚类效果越好