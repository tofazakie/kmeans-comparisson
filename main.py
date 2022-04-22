from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from scipy.spatial import distance
from kmeans import *

import numpy as np
import statistics
import pandas as pd
import gc

gc.collect()

# load datasets
ds_iris = datasets.load_iris()
# ds_wine = datasets.load_wine()

# k-means
kmeans_ri, kmeans_er = kmeans(3, ds_iris.data, ds_iris.target)
# print 'Rand Index: ', kmeans_ri
# print 'Error Rate: ', kmeans_er, '%'

# k-means maximin
kmeans_maximin_ri, kmeans_maximin_er = kmeans_maximin(3, ds_iris.data, ds_iris.target)
# print 'Rand Index: ', kmeans_maximin_ri
# print 'Error Rate: ', kmeans_maximin_er, '%'

kmeans_aldaoud_ri, kmeans_aldaoud_er = al_daoud(3, ds_iris.data, ds_iris.target)
# print 'Rand Index: ', kmeans_aldaoud_ri
# print 'Error Rate: ', kmeans_aldaoud_er, '%'

kmeans_goyal_ri, kmeans_goyal_er = goyal(3, ds_iris.data, ds_iris.target)
# print 'Rand Index: ', kmeans_goyal_ri
# print 'Error Rate: ', kmeans_goyal_er, '%'

print ''
print 'Summary:'
print 'K-Means'
print 'Rand Index: ', kmeans_ri
print 'Error Rate: ', kmeans_er, '%'

print ''
print 'K-Means Maximin'
print 'Rand Index: ', kmeans_maximin_ri
print 'Error Rate: ', kmeans_maximin_er, '%'

print ''
print 'Al-Daoud'
print 'Rand Index: ', kmeans_aldaoud_ri
print 'Error Rate: ', kmeans_aldaoud_er, '%'

print ''
print 'Goyal'
print 'Rand Index: ', kmeans_goyal_ri
print 'Error Rate: ', kmeans_goyal_er, '%'