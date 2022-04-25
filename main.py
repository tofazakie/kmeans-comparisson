from sklearn import cluster, datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from scipy.spatial import distance
from kmeans import *

import numpy as np
import statistics
import pandas as pd


# load datasets
all_datasets = [None] * 10
all_datasets[0] = datasets.load_iris() # 150 - 4
all_datasets[1] = datasets.load_wine() # 178 - 13
all_datasets[2] = datasets.load_sonar() # 208 - 60
all_datasets[3] = datasets.load_seeds() # 210 - 7
all_datasets[4] = datasets.load_vehicle() # 846 - 18
all_datasets[5] = datasets.load_thyroid() # 215 - 5
all_datasets[6] = datasets.load_haberman() # 306 - 2
all_datasets[7] = datasets.load_ecoli() # 336 - 7
all_datasets[8] = datasets.load_ionosphere() # 351-34
all_datasets[9] = datasets.load_vowel() # 990 - 11


datasets_name = ['IRIS', 'WINE', 'SONAR', 'SEEDS', 'VEHICLE']  
datasets_name += ['HABERMAN', 'IONOSPHERE', 'BREASTCANCER', 'BLOOD', 'VOWEL']

selected_datasets = [1, 1, 1, 1, 1,    1, 1, 1, 1, 1]

cluster_methods_name = [ 'kmeans', 'maximin', 'al-daoud', 'goyal', 'proposed']
cluster_methods      = [        1,         1,          1,       1,          1]

cols = []
for i in range(len(cluster_methods)):
    if(cluster_methods[i] == 1):
        cols = cols + [cluster_methods_name[i]]

# set file log
log_file = 'log.txt'

# loop all datasets
with open(log_file, "a") as text_file:
    text_file.write("\n")
    text_file.write("DATASETS \n")
    text_file.write("************************************")
    text_file.write("\n")

# Write datasets name
id = 0
for i in range(len(all_datasets)):
    if(selected_datasets[i] == 0):
        continue

    with open(log_file, "a") as text_file:
        text_file.write("%d %s" % (id, datasets_name[i]))
        text_file.write("\n")

    id += 1


# Davies-Bouldin Index
dbi_kmeans      = []
dbi_maximin     = []
dbi_aldaoud     = []
dbi_goyal       = []
dbi_proposed    = []

# Rand Index
ri_kmeans       = []
ri_maximin      = []
ri_aldaoud      = []
ri_goyal        = []
ri_proposed     = []

# Error Rate
er_kmeans       = []
er_maximin      = []
er_aldaoud      = []
er_goyal        = []
er_proposed     = []

data_row_d 	    = []
data_row_r 	    = []
data_row_e 	    = []
means_d		    = []
means_r	        = []
means_e	        = []



# test for all datasets
for i in range(len(all_datasets)):
    if(selected_datasets[i] == 0):
        continue

    dataset = all_datasets[i]
    print ''
    print datasets_name[i]

    # number of clusters
    k = len(list(dict.fromkeys(dataset.target)))

    row_d = []
    row_r = []
    row_e = []

    # cluster methods
    if(cluster_methods[0]):
        d_kmeans, r_kmeans, e_kmeans = kmeans(k, dataset.data, dataset.target)
        dbi_kmeans.append(d_kmeans)
        ri_kmeans.append(r_kmeans)
        er_kmeans.append(e_kmeans)

        row_d.append(d_kmeans)
        row_r.append(r_kmeans)
        row_e.append(e_kmeans)

        print '> K-Means finished'

    if(cluster_methods[1]):
        d_maximin, r_maximin, e_maximin = kmeans_maximin(k, dataset.data, dataset.target)
        dbi_maximin.append(d_maximin)
        ri_maximin.append(r_maximin)
        er_maximin.append(e_maximin)

        row_d.append(d_maximin)
        row_r.append(r_maximin)
        row_e.append(e_maximin)

        print '> Maximin finished'

    if(cluster_methods[2]):
        d_aldaoud, r_aldaoud, e_aldaoud = al_daoud(k, dataset.data, dataset.target)
        dbi_aldaoud.append(d_aldaoud)
        ri_aldaoud.append(r_aldaoud)
        er_aldaoud.append(e_aldaoud)

        row_d.append(d_aldaoud)
        row_r.append(r_aldaoud)
        row_e.append(e_aldaoud)

        print '> Al-Daoud finished'

    if(cluster_methods[3]):
        d_goyal, r_goyal, e_goyal = goyal(k, dataset.data, dataset.target)
        dbi_goyal.append(d_goyal)
        ri_goyal.append(r_goyal)
        er_goyal.append(e_goyal)

        row_d.append(d_goyal)
        row_r.append(r_goyal)
        row_e.append(e_goyal)

        print '> Goyal finished'

    if(cluster_methods[4]):
        d_prop, r_prop, e_prop = proposed(k, dataset.data, dataset.target)
        dbi_proposed.append(d_prop)
        ri_proposed.append(r_prop)
        er_proposed.append(e_prop)

        row_d.append(d_prop)
        row_r.append(r_prop)
        row_e.append(e_prop)

        print '> Proposed finished'
        
    print ''
    data_row_d = data_row_d + [row_d]
    data_row_r = data_row_r + [row_r]
    data_row_e = data_row_e + [row_e]


# MEANS
if(cluster_methods[0]):
    means_d.append(statistics.mean(dbi_kmeans))
    means_r.append(statistics.mean(ri_kmeans))
    means_e.append(statistics.mean(er_kmeans))

if(cluster_methods[1]):
    means_d.append(statistics.mean(dbi_maximin))
    means_r.append(statistics.mean(ri_maximin))
    means_e.append(statistics.mean(er_maximin))

if(cluster_methods[2]):
    means_d.append(statistics.mean(dbi_aldaoud))
    means_r.append(statistics.mean(ri_aldaoud))
    means_e.append(statistics.mean(er_aldaoud))

if(cluster_methods[3]):
    means_d.append(statistics.mean(dbi_goyal))
    means_r.append(statistics.mean(ri_goyal))
    means_e.append(statistics.mean(er_goyal))

if(cluster_methods[4]):
    means_d.append(statistics.mean(dbi_proposed))
    means_r.append(statistics.mean(ri_proposed))
    means_e.append(statistics.mean(er_proposed))


data_row_d = data_row_d + [means_d]
data_row_pd_d = pd.DataFrame(data = data_row_d, columns=cols)

data_row_r = data_row_r + [means_r]
data_row_pd_r = pd.DataFrame(data = data_row_r, columns=cols)

data_row_e = data_row_e + [means_e]
data_row_pd_e = pd.DataFrame(data = data_row_e, columns=cols)

with open(log_file, "a") as text_file:
    text_file.write("\n")
    text_file.write("\n")
    text_file.write("DBI \n")
    text_file.write("%s" % data_row_pd_d)
    text_file.write("\n")
    text_file.write("\n")
    text_file.write("Rand Index \n")
    text_file.write("%s" % data_row_pd_r)
    text_file.write("\n")
    text_file.write("\n")
    text_file.write("Error Rate \n")
    text_file.write("%s" % data_row_pd_e)
    text_file.write("\n")
    text_file.write("\n")

