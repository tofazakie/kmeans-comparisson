import csv
import random
import math
import operator
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import f1_score
import statistics
import pandas as pd
from collections import OrderedDict


##############################  GLOBAL FUNCTION  #################################

def euclideanDistance(instance1, instance2):
    distance = 0
    length = len(instance1)
    for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
        
    return math.sqrt(distance)


def removearray(List_arr, arr):
    print List_arr
    print arr
    ind = 0
    size = len(List_arr)
    while ind != size and not np.array_equal(List_arr[ind],arr):
        ind += 1
    if ind != size:
        List_arr.pop(ind)
    else:
		# print L
		print arr
		raise ValueError('array not found in list.')


def getRandIndex(y, predictions):
    correct = 0
    for i in range(len(y)):
        if y[i] == predictions[i]:
			correct += 1

	ri = (correct/float(len(y))) 
    return ri


def getErrorRate(y, predictions):
    wrong = 0
    for j in range(len(y)):
        if y[j] != predictions[j]:
			wrong += 1

	er = (wrong/float(len(y))) * 100.0
    return er


def isEqual(old_centroid, new_centroid):
    equal = True
    # print old_centroid
    # print new_centroid
    for c in range(len(old_centroid)):
        if ((old_centroid[c] != new_centroid[c]).all()):
            equal = False
            break

    return equal

############################################################################



############################################################################
##############################  MAIN PROGRAM  ##############################
############################################################################


# ****************************** K-Means 1967 ******************************
# MacQueen
def kmeans(k, data, y):
    n_data       = len(data)-1
    centroid_idx = []
    centroid     = []
    last_centroid= []
    new_centroid = []
    cluster      = []
    cluster_dt   = []
    
    predictions  = []

    # initialize random clusters
    for c in range(k): 
        r = random.randint(0, n_data)
        while r in centroid_idx:
            r = random.randint(0, n_data)

        centroid_idx.append(r)
        centroid.append(data[r])
        new_centroid.append([])
        cluster.append(y[r])
        cluster_dt.append([])
    
    last_centroid = centroid

    # determine cluster for each data
    for x in data:
        dist = []
        for i in range(k):
            d = euclideanDistance(x, centroid[i])
            dist.append(d)

        min_dist = min(dist)
        cluster_idx = dist.index(min_dist)
        cluster_dt[cluster_idx].append(x)

        pred_cluster = cluster[cluster_idx]
        predictions.append(pred_cluster)

    # measure new centroid means
    for c in range(k):
        new_centroid[c] = np.array(cluster_dt[c]).mean(axis=0)

    print '==== LOOP 1 ===='
    print 'Last centroid:'
    print last_centroid
    print 'New centroid:'
    print new_centroid

    # centroid movements
    itr = 2
    while (not isEqual(last_centroid, new_centroid)):
        last_centroid = new_centroid

        centroid_idx = []
        centroid     = []
        cluster_dt   = []
        for c in range(k):
            cluster_dt.append([])
        predictions  = []

        # determine cluster for each data accroding new cluster
        for x in data:
            dist = []
            for i in range(k):
                d = euclideanDistance(x, new_centroid[i])
                dist.append(d)

            min_dist = min(dist)
            cluster_idx = dist.index(min_dist)
            cluster_dt[cluster_idx].append(x)

            pred_cluster = cluster[cluster_idx]
            predictions.append(pred_cluster)

        # measure new centroid means
        new_centroid = []
        for c in range(k):
            new_centroid.append([])
        for c in range(k):
            new_centroid[c] = np.array(cluster_dt[c]).mean(axis=0)

        print 'Result: Not Equal'
        print
        print '==== LOOP ', itr,' ===='
        print 'Last Centroid: ', last_centroid
        print 'New Centroid: ', new_centroid

        itr += 1
        
    print 'Result: Equal'
    print

    ri = getRandIndex(y, predictions)
    er = getErrorRate(y, predictions)
    return ri, er



# ****************************** Maximin 1985 ******************************
# Gonzalez

def kmeans_maximin(k, data, y):
    n_data       = len(data)-1
    centroid_idx = []
    centroid     = []
    last_centroid= []
    new_centroid = []
    cluster      = []
    cluster_dt   = []
    
    predictions  = []

    no_centroid = data
    no_centroid_y = y

    for i in range(k):
        new_centroid.append([])
        cluster_dt.append([])

    # initialize first centroid
    r = random.randint(0, n_data)
    centroid_idx.append(r)
    centroid.append(data[r])
    cluster.append(y[r])
    centroid_point = data[r]
    np.delete(no_centroid, r)
    np.delete(no_centroid_y, r)
    
    # determine next centroid
    for i in range(k-1):
        # get farthest instance from last centroid_point
        dist = []
        for dt in no_centroid:
            d = euclideanDistance(dt, centroid_point)
            dist.append(d)

        max_dist = max(dist)
        new_centroid_idx = dist.index(max_dist)

        centroid_idx.append(new_centroid_idx)
        centroid.append(no_centroid[new_centroid_idx])
        cluster.append(no_centroid_y[new_centroid_idx])
        
        np.delete(no_centroid, centroid_point)
        np.delete(no_centroid_y, centroid_point)

        centroid_point = no_centroid[new_centroid_idx]

    last_centroid = centroid

    # determine cluster for each data
    for x in data:
        dist = []
        for i in range(k):
            d = euclideanDistance(x, centroid[i])
            dist.append(d)

        min_dist = min(dist)
        cluster_idx = dist.index(min_dist)
        cluster_dt[cluster_idx].append(x)

        pred_cluster = cluster[cluster_idx]
        predictions.append(pred_cluster)

    # measure new centroid means
    for c in range(k):
        new_centroid[c] = np.array(cluster_dt[c]).mean(axis=0)

    print '==== LOOP 1 ===='
    print 'Last centroid:'
    print last_centroid
    print 'New centroid:'
    print new_centroid

    # centroid movements
    itr = 2
    while (not isEqual(last_centroid, new_centroid)):
        last_centroid = new_centroid

        centroid_idx = []
        centroid     = []
        cluster_dt   = []
        for c in range(k):
            cluster_dt.append([])
        predictions  = []

        # determine cluster for each data accroding new cluster
        for x in data:
            dist = []
            for i in range(k):
                d = euclideanDistance(x, new_centroid[i])
                dist.append(d)

            min_dist = min(dist)
            cluster_idx = dist.index(min_dist)
            cluster_dt[cluster_idx].append(x)

            pred_cluster = cluster[cluster_idx]
            predictions.append(pred_cluster)

        # measure new centroid means
        new_centroid = []
        for c in range(k):
            new_centroid.append([])
        for c in range(k):
            new_centroid[c] = np.array(cluster_dt[c]).mean(axis=0)

        print 'Result: Not Equal'
        print
        print '==== LOOP ', itr,' ===='
        print 'Last Centroid: ', last_centroid
        print 'New Centroid: ', new_centroid

        itr += 1
        
    print 'Result: Equal'
    print

    ri = getRandIndex(y, predictions)
    er = getErrorRate(y, predictions)
    return ri, er



# ****************************** Al-Daoud 1985 ******************************
# Al-Daoud
def al_daoud(k, data, y):
    n_data       = len(data)-1
    centroid_idx = []
    centroid     = []
    last_centroid= []
    new_centroid = []
    cluster      = []
    cluster_dt   = []
    subset_data  = []
    
    predictions  = []

    for i in range(k):
        new_centroid.append([])
        cluster_dt.append([])
        subset_data.append([])

    # measure cvmax
    cv = []
    for c in range(len(data[0])):
        data_column = [item for sublist in data[:, c:c+1] for item in sublist]
        cv.append(statistics.stdev(data_column) / statistics.mean(data_column))

    max_cv = max(cv)
    max_cv_idx = cv.index(max_cv)
    print 'CV:', cv
    print 'Max CV Col:', max_cv_idx

    # sort data ordered by max cv column
    sorted_data = sorted(data, key=lambda x: x[max_cv_idx])
    # print sorted_data

    # split to subsets
    subset_n = int(np.floor(len(data) / k))
        
    for i in range(k):
        subset_data[i] = (sorted_data[i*subset_n:subset_n*(i+1)])

    # determine clusters from subset
    for c in range(k):
        centro = []
        for f in range(len(data[0])):
            # print subset_data[c]
            subset_data = np.asarray(subset_data, dtype=np.float32)
            centro.append(np.median(subset_data[c][:, f:f+1]))
            
        centroid.append(centro)
        cluster.append(c)

    print centroid

    last_centroid = centroid

    # determine cluster for each data
    for x in data:
        dist = []
        for i in range(k):
            d = euclideanDistance(x, centroid[i])
            dist.append(d)

        min_dist = min(dist)
        cluster_idx = dist.index(min_dist)
        cluster_dt[cluster_idx].append(x)

        pred_cluster = cluster[cluster_idx]
        predictions.append(pred_cluster)

    # measure new centroid means
    for c in range(k):
        new_centroid[c] = np.array(cluster_dt[c]).mean(axis=0)

    print '==== LOOP 1 ===='
    print 'Last centroid:'
    print last_centroid
    print 'New centroid:'
    print new_centroid

    # centroid movements
    itr = 2
    while (not isEqual(last_centroid, new_centroid)):
        last_centroid = new_centroid

        centroid_idx = []
        centroid     = []
        cluster_dt   = []
        for c in range(k):
            cluster_dt.append([])
        predictions  = []

        # determine cluster for each data according new cluster
        for x in data:
            dist = []
            for i in range(k):
                d = euclideanDistance(x, new_centroid[i])
                dist.append(d)

            min_dist = min(dist)
            cluster_idx = dist.index(min_dist)
            cluster_dt[cluster_idx].append(x)

            pred_cluster = cluster[cluster_idx]
            predictions.append(pred_cluster)

        # measure new centroid means
        new_centroid = []
        for c in range(k):
            new_centroid.append([])
        for c in range(k):
            new_centroid[c] = np.array(cluster_dt[c]).mean(axis=0)

        print 'Result: Not Equal'
        print
        print '==== LOOP ', itr,' ===='
        print 'Last Centroid: ', last_centroid
        print 'New Centroid: ', new_centroid

        itr += 1
        
    print 'Result: Equal'
    print

    ri = getRandIndex(y, predictions)
    er = getErrorRate(y, predictions)
    return ri, er



# ****************************** Goyal 2004 ******************************
# Goyal
def goyal(k, data, y):
    n_data       = len(data)-1
    n_col        = len(data[0])
    centroid_idx = []
    centroid     = []
    last_centroid= []
    new_centroid = []
    cluster      = []
    cluster_dt   = []
    subset_data  = []
    origin       = []
    
    predictions  = []

    for i in range(k):
        new_centroid.append([])
        cluster_dt.append([])
        subset_data.append([])

    # set origin
    for j in range(n_col):
        origin.append(0)

    data_completed = []
    # determine cluster for each data
    t = 0
    for x in data:
        d = euclideanDistance(x, origin)
        full = x.tolist()
        full.append(y[t])
        full.append(d)
        data_completed.append(full)

    # sort data ordered by max cv column
    sorted_data = sorted(data_completed, key=lambda x: x[-1])

    # split to subsets
    subset_n = int(np.floor(n_data / k))
        
    for i in range(k):
        subset_data[i] = (sorted_data[i*subset_n:subset_n*(i+1)])

    # determine clusters from subset
    for c in range(k):
        centro = []
        for f in range(n_col):
            # print subset_data[c]
            subset_data = np.asarray(subset_data, dtype=np.float32)
            centro.append(np.mean(subset_data[c][:, f:f+1]))
            
        centroid.append(centro)
        cluster.append(c)

    print centroid

    last_centroid = centroid

    # determine cluster for each data
    for x in data:
        dist = []
        for i in range(k):
            d = euclideanDistance(x, centroid[i])
            dist.append(d)

        min_dist = min(dist)
        cluster_idx = dist.index(min_dist)
        cluster_dt[cluster_idx].append(x)

        pred_cluster = cluster[cluster_idx]
        predictions.append(pred_cluster)

    # measure new centroid means
    for c in range(k):
        new_centroid[c] = np.array(cluster_dt[c]).mean(axis=0)

    print '==== LOOP 1 ===='
    print 'Last centroid:'
    print last_centroid
    print 'New centroid:'
    print new_centroid

    # centroid movements
    itr = 2
    while (not isEqual(last_centroid, new_centroid)):
        last_centroid = new_centroid

        centroid_idx = []
        centroid     = []
        cluster_dt   = []
        for c in range(k):
            cluster_dt.append([])
        predictions  = []

        # determine cluster for each data accroding new cluster
        for x in data:
            dist = []
            for i in range(k):
                d = euclideanDistance(x, new_centroid[i])
                dist.append(d)

            min_dist = min(dist)
            cluster_idx = dist.index(min_dist)
            cluster_dt[cluster_idx].append(x)

            pred_cluster = cluster[cluster_idx]
            predictions.append(pred_cluster)

        # measure new centroid means
        new_centroid = []
        for c in range(k):
            new_centroid.append([])
        for c in range(k):
            new_centroid[c] = np.array(cluster_dt[c]).mean(axis=0)

        print 'Result: Not Equal'
        print
        print '==== LOOP ', itr,' ===='
        print 'Last Centroid: ', last_centroid
        print 'New Centroid: ', new_centroid

        itr += 1
        
    print 'Result: Equal'
    print

    ri = getRandIndex(y, predictions)
    er = getErrorRate(y, predictions)
    return ri, er