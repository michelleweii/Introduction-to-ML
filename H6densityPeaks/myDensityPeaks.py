# -*- coding: utf-8 -*-
"""
Created on 2017-11-07

@author: Michelle
"""
from sklearn.datasets import make_blobs
import numpy as np
import math
import operator


MAX = 1000000
n_samples = 900
random_state = 5
centers = [[-0.6, -0.4], [-0.3, 0.5], [0.5, -0.4]]
center_box = [-1, 1]
X, y = make_blobs(n_features=2, n_samples=n_samples,
                  cluster_std=0.15, center_box=center_box,
                  centers=centers, random_state=random_state)
m = np.array(((1, 1), (1, 3)))
X = X.dot(m)
# print('X:',X)
# print('y:',y)

#Read data
location = X
label = y
length = len(location)


def nearestNeighbor(index):
    dd = MAX
    neighbor = -1
    for i in range(length):
        if dist[index, i] < dd and rho[index] < rho[i]:
            dd = dist[index, i]
            neighbor = i
    if result[neighbor] == -1:
        result[neighbor] = nearestNeighbor(neighbor)
    return result[neighbor]


#Caculate distance
dist = np.zeros((length, length))
ll = []
begin = 0
while begin < length-1:
    end = begin + 1
    while end < length:
        dd = np.linalg.norm(location[begin]-location[end])
        dist[begin][end] = dd
        dist[end][begin] = dd
        ll.append(dd)
        end = end + 1
    begin = begin + 1
ll = np.array(ll)

# Algorithm

percent = 2.0
position = int(len(ll) * percent / 100)
sortedll = np.sort(ll)
dc = sortedll[position] #阈值

#求点的局部密度(local density)
rho = np.zeros((length, 1))
begin = 0
while begin < length-1:
    end = begin + 1
    while end < length:
        rho[begin] = rho[begin] + math.exp(-(dist[begin][end]/dc) ** 2)
        rho[end] = rho[end] + math.exp(-(dist[begin][end]/dc) ** 2)
        #if dist[begin][end] < dc:
        #    rho[begin] = rho[begin] + 1
        #    rho[end] = rho[end] + 1
        end = end + 1
    begin = begin + 1

#求比点的局部密度大的点到该点的最小距离
delta = np.ones((length, 1)) * MAX
maxDensity = np.max(rho)
begin = 0
while begin < length:
    if rho[begin] < maxDensity:
        end = 0
        while end < length:
            if rho[end] > rho[begin] and dist[begin][end] < delta[begin]:
                delta[begin] = dist[begin][end]
            end = end + 1
    else:
        delta[begin] = 0.0
        end = 0
        while end < length:
            if dist[begin][end] > delta[begin]:
                delta[begin] = dist[begin][end]
            end = end + 1
    begin = begin + 1

rate1 = 0.6
# rate1 = 0.75
thRho = rate1 * (np.max(rho) - np.min(rho)) + np.min(rho)

rate2 = 0.2
# rate2 = 0.05
thDel = rate2 * (np.max(delta) - np.min(delta)) + np.min(delta)

#确定聚类中心
result = np.ones(length, dtype=np.int) * (-1)
center = 0
for i in range(length): #items:
    if rho[i] > thRho and delta[i] > thDel:
        result[i] = center
        center = center + 1

#赋予每个点聚类类标
for i in range(length):
    dist[i][i] = MAX

for i in range(length):
    if result[i] == -1:
        result[i] = nearestNeighbor(i)
    else:
        continue

#求聚类结果的下标
y_index = []
for i in range(length):
    index = result[i]
    y_index.append(index)
    # plt.plot(location[i][0], location[i][1], color = colors[index], marker = '.')

def getIndex():
    # print('1111')
    y_dp = y_index
    return y_dp

# getIndex()
# print(y_index)

# if __name__ == '__main__':
#     main(y_index)



'''
Compare:
'''
# print('lenOfOriginal:',len(y))
# print('lenOfDensity:',len(y_index))
# print(operator.lt(y,y_index))