# 《机器学习导论》课后作业 Introduction to ML

另外：
[《PRML》模式识别期末总结](https://blog.csdn.net/weixin_31866177/article/details/83060061)

## episode2 用python+sklearn实现随机森林
## episode3 在KNN中使用托梅克连接算法 P21

使用托梅克连接去除干扰样例（噪音）之前

<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h3-tomek/before.png" width="450" alt="knn1">

使用托梅克连接去除干扰样例（噪音）之后
                                                                                                                      
<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h3-tomek/after.png" width="450" alt="knn2">

噪音去除了~😊                                                                                                                       
## episode4 朴素贝叶斯代码实现分类任务 P66
## episode5 Canopy+KMeans混合聚类算法 P46

该数据集应该取k=5进行聚类

<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h5-canopy/%E8%BF%90%E8%A1%8C%E7%BB%93%E6%9E%9Ccanopy.png" width="550" alt="Canopy">

## episode6 实现密度最小值聚类算法 P61

rate1=0.75 rate2=0.05

<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h6-densityPeaks/%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C%E5%9B%BE1.png" width="650" alt="密度聚类1">

rate1=0.8 rate2=0.4

<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h6-densityPeaks/%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C%E5%9B%BE2.png" width="650" alt="密度聚类2">

最佳参数：rate1=0.6 rate2=0.2

<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h6-densityPeaks/%E6%9C%80%E4%BD%B3%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C.png" width="650" alt="密度聚类best case">


## episode7 用谱聚类算法进行聚类 P41

可以解决kmeans在非球状数据上聚类效果不好的问题

The spectral clustering results in different sigma, Case 1

<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h7-spectralclustering/from.png" width="650" alt="谱聚类1">

The spectral clustering results in different sigma, Case 2

<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h7-spectralclustering/%E8%B0%B1%E8%81%9A%E7%B1%BB%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C.png" width="650" alt="谱聚类2">

## episode8 推导SVM对偶公式
## episode9 用Libsvm训练出model以及找到数据集的最佳参数 P86

采用grid search最佳参数

<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h9-libsvm/libsvm-pic.png" width="550" alt="EM Kmeans对比">

## episode10 分别用KMeans算法和EM算法对样本进行聚类 P39

Kmeans和GMM的对比

<img src="https://github.com/michelleweii/Introduction-to-ML/blob/master/pic/h10-em/%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C%E5%9B%BE.png" width="800" alt="EM Kmeans对比">

#### else sklearn库的练习以及机器学习导论上课课件
