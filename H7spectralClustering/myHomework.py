import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import spectral_clustering
from sklearn.metrics import euclidean_distances
a = np.array([[10, 7, 4], [3, 2, 1]])
m = np.median(a, axis=0)
print('m:',m) #m: [ 6.5  4.5  2.5]
