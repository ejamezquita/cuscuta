from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf

from scipy import spatial

import os
import pandas as pd

from sklearn import linear_model, cluster

import utils

def main():
    
    colors = [None, '#117733', '#aa4499', '#332288', '#ddcc77', '#88ccee', '#44aa99']
    buffx = 0
    buffX = 1475
    buffy = 200
    buffz = 50
    nsamples = 5


    isrc = '../raw/'
    rsrc = '../reference/'
    ssrc = '../proc/skel/'
    time = '4pm'
    rep = 7

    meta = pd.read_csv(rsrc + 'reference_positions_{}_rep{}.csv'.format(time, rep))
    meta = meta.set_index('numidx')
    meta.head()

    idx = 400
    n, lines, xvals, plants = utils.get_img_metadata(meta, idx, nsamples, buffx, buffX, buffy, buffz)
    pidx = 0
    xlen, line, b = utils.get_plant_metadata(pidx, plants, xvals, lines, n)
    
    filename = ssrc + '{}_rep{}/plant_{:02d}/{}_rep{}_p{:02d}_{:04d}.csv'.format(time, rep, pidx, time, rep, pidx, idx)
    skeleton = np.loadtxt(filename, dtype=int, delimiter=',')

    dist = np.abs(skeleton[1]*n[pidx, 0] + skeleton[0]*n[pidx, 1] - b)
    print('Min dist: {:.2f}\nMax dist: {:.2f}'.format(np.min(dist),np.max(dist)))
    dmask = dist < 15
    mmask = dist < 25

    print('Number of pixels close to center:\t', np.sum(dmask))
    if np.sum(dmask) > 5:
        clusts, uq, seeds = utils.get_cluster_seeds(skeleton, dist, dmask)
        nclusts = utils.expand_cluster_labels(skeleton, clusts, dmask, mmask)
        lcoefs, angle = utils.get_coil_metadata(skeleton, mmask, nclusts, uq, dist, n[pidx])

        fig, ax = plt.subplots(1,4,figsize=(6,5), sharex=True, sharey=True)
        ax = np.atleast_1d(ax).ravel()

        ax[0].scatter(skeleton[1], skeleton[0], c=dist, cmap='inferno_r', vmin=0, zorder=3)

        ax[1].scatter(skeleton[1, dmask], skeleton[0, dmask], c=clusts, cmap='inferno_r', zorder=3)
        ax[1].scatter(seeds[:,1], seeds[:,0], marker='D', c='white', edgecolor='k', alpha=0.5, zorder=4)

        ax[2].scatter(skeleton[1, mmask], skeleton[0, mmask], c=nclusts, cmap='inferno_r', zorder=3)

        ax[3].scatter(skeleton[1,~mmask], skeleton[0,~mmask], c='gainsboro',zorder=1)
        ax[3].scatter(skeleton[1,mmask], skeleton[0,mmask], c=(np.max(dist) - dist[mmask])**4, cmap='inferno', vmin=0, zorder=3)
        for i in range(len(uq)):
            xy1 = (seeds[i,1], seeds[i,0])
            ax[3].axline(xy1, slope=lcoefs[i,1], c=colors[i+4], lw = 3, zorder=2)
            ax[3].scatter(*xy1, c='w', s=150, marker='*', zorder=4, edgecolor='k', linewidths=1)

        for i in range(len(ax)):
            ax[i].plot(line, xlen, c='r', zorder=1)
            ax[i].set_aspect('equal')
            ax[i].margins(x=0.5, y=0)
            ax[i].set_facecolor('gray')
        ax[0].invert_yaxis()

        ax[0].set_title('Skeleton')
        ax[1].set_title('Find crossings')
        ax[2].set_title('Merge and Expand')
        ax[3].set_title('RANSAC')

        fig.tight_layout();
        plt.show()

if __name__ == '__main__':
    main()
