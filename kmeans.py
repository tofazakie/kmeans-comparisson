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