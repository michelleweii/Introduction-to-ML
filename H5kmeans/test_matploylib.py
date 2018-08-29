# coding=utf-8
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from matplotlib import pyplot as plt

# def createCluster():
#     X,y = make_blobs(n_samples=500,random_state=170)
#     return X.tolist(),y.tolist()
# data,target = make_blobs(n_samples=100,n_features=2,centers=4)
# #n_samples是待生成的样本的总数。
# # n_features是每个样本的特征数。
# # centers表示类别数。
#
# # X, y = createCluster()
# print('X:');print(data)
# print('y:');print(target)
# # print('data[:,0]:');print(data[:,0]) #第一列数据
# plt.scatter(data[:,0],data[:,1],c=target)
# # plt.plot(data)
# # plt.plot(target)
# plt.show()

#
#
# scatter散布图
#
#
x = np.arange(1,10)
y = x
fig = plt.figure()
ax1 = fig.add_subplot(111)
#设置标题
ax1.set_title('Scatter Plot')
#设置X轴标签
plt.xlabel('X')
#设置Y轴标签
plt.ylabel('Y')
#画散点图
cValue = ['r','y','g','b','r','y','g','b','r']
ax1.scatter(x,y,c=cValue,marker='s')
#设置图标
# plt.legend('x1')
#显示所画的图
plt.show()

