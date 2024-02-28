import warnings
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf

from scipy import spatial

import os
import pandas as pd

from sklearn import linear_model, cluster

def get_img_metadata(meta, idx, nsamples=5, buffx=0, buffX=1475, buffy=50, buffz=50):
    tapes = meta.loc[idx, ['tapes_{}'.format(i) for i in range(nsamples)]].values.astype(int)
    peaks = meta.loc[idx, ['peaks_{}'.format(i) for i in range(nsamples)]].values.astype(int)
    peaks = np.hstack((peaks, peaks[-1] + np.max(np.ediff1d(peaks)) ))

    coefs = meta.loc[idx, ['coef0_{}'.format(i) for i in range(nsamples)] + ['coef1_{}'.format(i) for i in range(nsamples)] ].values
    coefs = np.reshape(coefs, (nsamples, 2), order='F')

    xvals = np.arange(buffX)
    lines = np.zeros((nsamples, buffX))

    v = np.column_stack([coefs[:,1], np.ones(len(coefs))])
    v /= np.linalg.norm(v, axis=1).reshape(-1,1)
    n = np.column_stack((v[:,1], -v[:,0]))

    for i in range(len(lines)):
        b0 = peaks[i] + coefs[i,0] - buffy
        lines[i] = b0 + coefs[i,1]*(xvals - buffx)

    plants = [ np.s_[buffx:tapes[0]+buffx, 0:peaks[1]-buffz] ]
    for i in range(1,len(tapes)):
        plants.append(np.s_[buffx:tapes[i]+buffx, peaks[i-1]+buffz:peaks[i+1]-buffz])
    
    return n, lines, xvals, plants
    
def get_plant_metadata(pidx, plants, xvals, lines, n):
    xlen = xvals[ plants[pidx][0] ] - plants[pidx][0].start
    line = lines[ pidx, plants[pidx][0] ] - plants[pidx][1].start
    b = n[pidx,0] * line[0]
    return xlen, line, b

def get_cluster_seeds(skeleton, dist, mask):
    clustering = cluster.DBSCAN(eps=10, min_samples=5)
    clusts = clustering.fit_predict(skeleton[:, mask].T, sample_weight=100-dist[mask])
    uq = np.unique(clusts[clusts >= 0])
    seeds = np.zeros((len(uq), len(skeleton)), dtype=int)
    for i in range(len(uq)):
        foo = np.argmin(dist[mask][clusts==uq[i]])
        seeds[i] = skeleton[:,mask][:, clusts==uq[i] ][:, foo]

    dm = np.triu(spatial.distance.squareform(spatial.distance.pdist(seeds)))
    dm[dm == 0] = 100
    merge = np.asarray(np.nonzero(dm < 35))
    if merge.shape[1] > 0:
        for i in range(len(merge[0])):
            clusts[clusts == merge[1,i]] = merge[0,i]
    uq = np.unique(clusts[clusts >= 0])
    
    return clusts, uq, seeds
    
def expand_cluster_labels(skeleton, clusts, mask_original, mask_expanded):
    dm = spatial.distance.cdist(skeleton[:, mask_original].T, skeleton[:, mask_expanded].T)
    nclusts = clusts[np.argmin(dm, axis=0)]
    
    return nclusts

def get_coil_metadata(skeleton, mask, nclusts, uq, dist, nvec):
    coefs = np.zeros((len(uq), 2))
    angle = np.zeros(len(uq))

    for i in range(len(uq)):
        y,X = skeleton[:, mask][:, nclusts == uq[i]]
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                ransac = linear_model.RANSACRegressor(random_state=42, min_samples=4)
                reg = ransac.fit(X.reshape(-1,1), y, (np.max(dist) - dist[mask][nclusts==uq[i]]))
                
                coef1 = np.diff(reg.predict([[0],[1]]))[0]
                coef0 = reg.predict([[0]])[0]
                
                w = np.array([1, coef1])
                w /= np.linalg.norm(w)
                
                coefs[i] = [coef0, coef1]
                angle[i] = -np.sign(coef1)*np.arccos(np.abs(np.sum(w*nvec)))
            except Warning:
                print('warning')
        
    return coefs, angle
