from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from scipy.spatial import distance
from kmeans import *

import numpy as np
import statistics
import pandas as pd


# load datasets
ds_iris = datasets.load_iris()

kmeans_ri, kmeans_er, kmeans_dbi = kmeans(3, ds_iris.data, ds_iris.target, True)
kmeans_maximin_ri, kmeans_maximin_er, kmeans_maximin_dbi = kmeans_maximin(3, ds_iris.data, ds_iris.target, True)
kmeans_aldaoud_ri, kmeans_aldaoud_er, kmeans_aldaoud_dbi = al_daoud(3, ds_iris.data, ds_iris.target, True)
kmeans_goyal_ri, kmeans_goyal_er, kmeans_goyal_dbi = goyal(3, ds_iris.data, ds_iris.target, True)
prop_ri, prop_er, prop_dbi = proposed(3, ds_iris.data, ds_iris.target, True)

print 'Summary:'

print 'K-Means'
print 'Rand Index: ', kmeans_ri
print 'Error Rate: ', kmeans_er, '%'
print 'DBI: ', kmeans_dbi

print ''
print 'K-Means Maximin'
print 'Rand Index: ', kmeans_maximin_ri
print 'Error Rate: ', kmeans_maximin_er, '%'
print 'DBI: ', kmeans_maximin_dbi

print ''
print 'Al-Daoud'
print 'Rand Index: ', kmeans_aldaoud_ri
print 'Error Rate: ', kmeans_aldaoud_er, '%'
print 'DBI: ', kmeans_aldaoud_dbi

print ''
print 'Goyal'
print 'Rand Index: ', kmeans_goyal_ri
print 'Error Rate: ', kmeans_goyal_er, '%'
print 'DBI: ', kmeans_goyal_dbi

print ''
print 'Proposed'
print 'Rand Index: ', prop_ri
print 'Error Rate: ', prop_er, '%'
print 'DBI: ', prop_dbi