from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
x = random.rand(50,30)      #创建一个50行30列的多维数组(ndarray)
#basic
f1 = plt.figure(1)          #创建显示图形输出的窗口对象
plt.subplot(211)            #创建子坐标系
#x[:,1]获取第二列作为一维数组，x[:,0]获取第一列作为一维数组
plt.scatter(x[:,1],x[:,0])  #画散点图


# with label
plt.subplot(212)            #c重新创建坐标系

#list(ones(20))     创建1行20列的全为1列表
#list(2*ones(15))   创建1行15列全为2的列表
#list(3*ones(15)    创建1行15列全为3的列表

label = list(ones(20))+list(2*ones(15))+list(3*ones(15))    #将列表合并到一起，共50列
label = array(label)                                        #将列表转为数组

#15.0*label         将数组的每个值都乘以15.0
#x[:,1]             将x的第2列50行转为1行50列
#x[:,0]             将x的第1列50行转为1行50列

#x轴和y轴均50个点，两个Label都是1行50列的数组
#从第一个点到第20个点的样式相同，从第21到第35个点相同，从第36到第50个点相同
plt.scatter(x[:,1],x[:,0],15.0*label,15.0*label)

# with legend
f2 = plt.figure(2)              #创建显示图形输出的窗口对象
idx_1 = np.where(label==1)      #找label中为1的位置

#画图  marker标识散点图样式 color标识颜色  label表示图例的解释   s表示散点的大小
p1 = plt.scatter(x[idx_1,1], x[idx_1,0], marker = 'x', color = 'm', label='1', s = 30)
idx_2 = np.where(label==2)      #找label中为2的位置

p2 = plt.scatter(x[idx_2,1], x[idx_2,0], marker = '+', color = 'c', label='2', s = 50)
idx_3 = np.where(label==3)      #找label中为3的位置

p3 = plt.scatter(x[idx_3,1], x[idx_3,0], marker = 'o', color = 'r', label='3', s = 15)
plt.legend(loc = 'upper right')  #图例的位置

plt.show()