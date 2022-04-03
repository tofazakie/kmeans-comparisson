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
ds = datasets.load_iris()

# k-means
kmeans_ri, kmeans_er = kmeans(3, ds.data, ds.target)
print 'Rand Index: ', kmeans_ri
print 'Error Rate: ', kmeans_er, '%'