# coding=utf-8
import numpy as np
from numpy import *
import random
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X = np.random.rand(1000, 2)

class Canopy:
    def __init__(self, X):
        self.X = X
        self.t1 = 0
        self.t2 = 0
 # 设置初始阈值
    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print('t1 needs to be larger than t2!')
# 使用欧式距离进行距离的计算
    def euclideanDistance(self, vec1, vec2):
        return math.sqrt(((vec1 - vec2)**2).sum())

# 根据当前dataset的长度随机选择一个下标
    def getRandIndex(self):
        return random.randint(0, len(self.X) - 1)

    def clustering(self):
        if self.t1 == 0:
            print('Please set the threshold.')
        else:
            canopies = []  # 用于存放最终归类结果
            while len(self.X) != 0:
                rand_index = self.getRandIndex()
                current_center = self.X[rand_index]  # 随机获取一个中心点，定为P点
                current_center_list = []  # 初始化P点的canopy类容器
                delete_list = []  # 初始化P点的删除容器
                self.X = np.delete(
                    self.X, rand_index, 0)  # 删除随机选择的中心点P
                for datum_j in range(len(self.X)):
                    datum = self.X[datum_j]
                    distance = self.euclideanDistance(
                        current_center, datum)  # 计算选取的中心点P到每个点之间的距离
                    if distance < self.t1:
                        # 若距离小于t1，则将点归入P点的canopy类
                        current_center_list.append(datum)
                    if distance < self.t2:
                        delete_list.append(datum_j)  # 若小于t2则归入删除容器
                # 根据删除容器的下标，将元素从数据集中删除
                self.X = np.delete(self.X, delete_list, 0)
                canopies.append((current_center, current_center_list))
        return canopies



def main():
    t1 = 0.6
    t2 = 0.3
    # X, y = createCluster()
    gc = Canopy(X)
    gc.setThreshold(t1, t2)
    canopies = gc.clustering()
    print('Get %s initial centers.' % len(canopies))
    n_clusters = len(canopies)
    estimator = KMeans(n_clusters=n_clusters) #构造聚类器
    estimator.fit(X) #聚类
    label_pred = estimator.labels_  #获取聚类标签
    centroids = estimator.cluster_centers_  #获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的最后值，值越小聚类效果越好
    # print(label_pred)
    print('centroids:');print(centroids)
    print('inertia:');print(inertia)
    mark = ['or', 'ob','Dr','og', 'ok', '^y', '+r', 'sb', 'dr', '<r', 'pr']
    # 这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
    color = 0
    j = 0

    for i in label_pred:
        plt.plot([X[j:j + 1, 0]], [X[j:j + 1, 1]], mark[i], markersize=1.5)
        plt.plot(centroids[i][0],centroids[i][1],mark[i], markersize=8)
        j += 1
    plt.show()



if __name__ == '__main__':
    main()

# output:
# Get 11 initial centers.
# centroids:
# [[ 0.51076904  0.35781027]
#  [ 0.87683538  0.84981096]
#  [ 0.56583216  0.87187067]
#  [ 0.89119835  0.34848059]
#  [ 0.17623325  0.8465058 ]
#  [ 0.76637142  0.57265549]
#  [ 0.78806928  0.10789874]
#  [ 0.41505514  0.12332844]
#  [ 0.3873335   0.64509885]
#  [ 0.14601429  0.18853685]
#  [ 0.14145831  0.49114034]]
# inertia:
# 14.8591352455
# 结果每次都不一样，有时候还会越界ToT