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


def getAccuracy(y, predictions):
    correct = 0
    for i in range(len(y)):
        if y[i] == predictions[i]:
			correct += 1

	ri = (correct/float(len(y))) 
    return ri

def getRandIndex(actuals, predictions):
    set_act, count_act  = np.unique(actuals, return_counts=True)
    set_pred, count_pred = np.unique(predictions, return_counts=True)

    count_act.sort()
    count_pred.sort()

    corrects = 0
    for i in range(len(set_act)):
        if(count_pred[i] > count_act[i]):
            corrects += count_act[i]
        else:
            corrects += count_pred[i]
    
    ri = (corrects/float(len(actuals))) 
    return ri


def getErrorRate(actuals, predictions):
    set_act, count_act  = np.unique(actuals, return_counts=True)
    set_pred, count_pred = np.unique(predictions, return_counts=True)

    count_act.sort()
    count_pred.sort()

    wrongs = 0
    for i in range(len(count_act)):
        wrongs += abs(count_pred[i] - count_act[i])
    
    ri = (wrongs/2/float(len(actuals)))  * 100.0
    return ri
    


def isEqual(old_centroid, new_centroid):
    equal = True
    # print old_centroid
    # print new_centroid
    for c in range(len(old_centroid)):
        if ((old_centroid[c] != new_centroid[c]).all()):
            equal = False
            break

    return equal



def getDBI(all_datas, centroids):
    n_cluster = len(centroids)

    # determine SSW
    ssw = []
    for i in range(n_cluster):
        ssw.append( np.average([t[-2] for t in all_datas if t[-1] == i]) )

    print 'SSW: ', ssw

    # determine SSB
    ssb = [[0 for x in range(n_cluster)] for x in range(n_cluster)]
    for i in range(n_cluster):
        for j in range(n_cluster):
            if(i != j):
                ssb[i][j] = euclideanDistance(centroids[i], centroids[j])

    print 'SSB: ', ssb

    # determine ratio
    ratio = [[0 for x in range(n_cluster)] for x in range(n_cluster)]
    for i in range(n_cluster):
        for j in range(n_cluster):
            if(i != j):
                ratio[i][j] = (ssw[i] + ssw[j]) / ssb[i][j]

    print 'Ratio: ', ratio

    # determine DBI
    dbi = 0
    for i in range(n_cluster):
        dbi += max(ratio[i])

    dbi = dbi / n_cluster

    return dbi



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
        cluster.append(c)
        cluster_dt.append([])
    
    last_centroid = centroid

    all_datas = []
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

        dt = x.tolist()
        dt.append(min_dist)
        dt.append(pred_cluster)
        all_datas.append(dt)

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
        all_datas    = []

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

            dt = x.tolist()
            dt.append(min_dist)
            dt.append(pred_cluster)
            all_datas.append(dt)

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
    dbi = getDBI(all_datas, new_centroid)
    return ri, er, dbi



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
    cluster.append(0)
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
        cluster.append(i+1)
        
        np.delete(no_centroid, centroid_point)
        np.delete(no_centroid_y, centroid_point)

        centroid_point = no_centroid[new_centroid_idx]

    last_centroid = centroid

    # determine cluster for each data
    all_datas = []
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

        dt = x.tolist()
        dt.append(min_dist)
        dt.append(pred_cluster)
        all_datas.append(dt)

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
        all_datas = []
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

            dt = x.tolist()
            dt.append(min_dist)
            dt.append(pred_cluster)
            all_datas.append(dt)

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
    dbi = getDBI(all_datas, new_centroid)
    return ri, er, dbi



# ****************************** Al-Daoud 1985 ******************************
# Al-Daoud
def al_daoud(k, data, y):
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
    all_datas = []
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

        dt = x.tolist()
        dt.append(min_dist)
        dt.append(pred_cluster)
        all_datas.append(dt)

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
        all_datas = []
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

            dt = x.tolist()
            dt.append(min_dist)
            dt.append(pred_cluster)
            all_datas.append(dt)

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
    dbi = getDBI(all_datas, new_centroid)
    return ri, er, dbi



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

    # sort data ordered by distance to origin
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
    all_datas = []
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

        dt = x.tolist()
        dt.append(min_dist)
        dt.append(pred_cluster)
        all_datas.append(dt)

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
        all_datas = []
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

            dt = x.tolist()
            dt.append(min_dist)
            dt.append(pred_cluster)
            all_datas.append(dt)

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
    dbi = getDBI(all_datas, new_centroid)
    return ri, er, dbi