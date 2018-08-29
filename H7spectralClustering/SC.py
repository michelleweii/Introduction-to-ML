# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import spectral_clustering
from sklearn.metrics import euclidean_distances
from sklearn import metrics


def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = [u'Verdana']  #用来正常显示中文标签
    matplotlib.rcParams['axes.unicode_minus'] = False   #用来正常显示负号

    t = np.arange(0, 2*np.pi, 0.1)
    data1 = np.vstack((np.cos(t), np.sin(t))).T
    # print('data1:',data1)
    data2 = np.vstack((2*np.cos(t), 2*np.sin(t))).T
    data3 = np.vstack((3*np.cos(t), 3*np.sin(t))).T
    data = np.vstack((data1, data2, data3))  #以上合并
    # print('data:',data)

    n_clusters = 3
    m = euclidean_distances(data, squared=True)
    print('m:',m)
    sigma = np.median(m)
    print('sigma:',sigma)   #sigma: 7.27842327145

    plt.figure(figsize=(12, 8), facecolor='w') #通过figsize参数可以指定绘图对象的宽度和高度
    plt.suptitle(u'The spectral clustering results in different sigma.', fontsize=20)
    clrs = plt.cm.Spectral(np.linspace(0, 0.8, n_clusters))   #等差数列
    # print('clrs：',clrs)
    # clrs： [[0.61960784  0.00392157  0.25882353  1.]
    #        [0.99607843  0.87843137  0.54509804  1.]
    #         [0.4   0.76078431   0.64705882   1.]]

    for i, s in enumerate(np.logspace(-2, 0, 6)):  #等比数列
        # print('np.logspace(-2, 0, 6):',np.logspace(-2, 0, 6))  #这里控制sigma的改变
        #[ 0.01        0.02511886  0.06309573  0.15848932  0.39810717  1.        ]
        # print('i:',i)
        # print('s:',s)  #一个一个取np.logspace(-2, 0, 6)里面的值
        af = np.exp(-m ** 2 / (s ** 2)) + 1e-6
        y_hat = spectral_clustering(af, n_clusters=n_clusters, assign_labels='kmeans', random_state=1)
        # print('Calinski-Harabasz Score with s=', s, ',Calinski-Harabasz Score：',metrics.calinski_harabaz_score(data, y_hat))
        plt.subplot(2, 3, i+1)
        for k, clr in enumerate(clrs):
            cur = (y_hat == k)
            plt.scatter(data[cur, 0], data[cur, 1], s=40, c=clr, edgecolors='k')
        x1_min, x2_min = np.min(data, axis=0)
        x1_max, x2_max = np.max(data, axis=0)
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.grid(True) #显示网格
        plt.title(s, fontsize=16)
    plt.tight_layout()   # 紧凑显示图片，居中显示
    plt.subplots_adjust(top=0.9)
    plt.show()
