import numpy as np
#numpy中meshgrid是使用
xnums = np.arange(4)
ynums = np.arange(5,10)
print('xnums：')
print(xnums)
print('ynums：')
print(ynums)
print('*************************************')
data_list = np.meshgrid(xnums,ynums)
print('after meshgrid:')
print(data_list)
x,y = data_list
print('meshgrid生成的两个数组，第一个数组的维数：')
print(x.shape)
print('meshgrid生成的两个数组，第二个数组的维数：')
print(y.shape)
print('meshgrid生成的两个数组，第一个数组：')
print(x)
print('meshgrid生成的两个数组，第二个数组：')
print(y)

# meshgrid的作用是根据传入的两个一维数组参数生成两个数组元素的列表。
# 如果第一个参数是xarray，维度是xdimesion，第二个参数是yarray，维度是ydimesion。
# 那么生成的第一个二维数组是以xarray为行，ydimesion行的向量；
# 而第二个二维数组是以yarray的转置为列，xdimesion列的向量。