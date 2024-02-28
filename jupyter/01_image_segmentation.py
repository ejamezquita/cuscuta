from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology

import os
import pandas as pd
import argparse

buffx = 0
buffX = 1475
buffy = 200
nsamples = 5
minsize = 100
buffz = 50
mind = 50

src = '../raw/'

def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('time', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('rep', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('init', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('endn', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()
    
    time = args.time
    rep = args.rep
    init = args.init
    endn = args.endn

    dst = '../proc/skel/'
    
    foldername = time + ' Inc Rep ' + str(rep)
    dst = dst + '{}_rep{}/'.format(time,rep)
    if not os.path.isdir(dst):
        os.mkdir(dst)
        for i in range(nsamples):
            os.mkdir(dst + 'plant_{:02d}'.format(i))

    meta = pd.read_csv('../reference/reference_positions_{}_rep{}.csv'.format(time, rep))
    meta = meta.set_index('numidx')

    for idx in range(init,endn):
        pidx = 0
        filename = dst + 'plant_{:02d}/{}_rep{}_p{:02d}_{:04d}.csv'.format(pidx, time, rep, pidx, idx)
        
        if not os.path.isfile(filename):
            filename = glob(src + foldername + '/*({})*.JPG'.format(idx))[0]
            raw = cv2.imread(filename)
            rawstd = np.var(raw, axis=2, ddof=1)
            rawmean = np.mean(raw, axis = 2)

            graw = raw[:,:,1]
            hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            # # Remove most of the background and foreground

            stdmask = (rawstd > 60) | (rawmean > 210)
            hmask = ((h < 40) & (h > 0)) & (v > 60) & stdmask

            img = raw.copy()
            for i in range(3):
                img[:,:,i] *= hmask

            # # Find the y-coord for the sticks

            tapes = meta.loc[idx, ['tapes_{}'.format(i) for i in range(nsamples)]].values.astype(int)
            peaks = meta.loc[idx, ['peaks_{}'.format(i) for i in range(nsamples)]].values.astype(int)
            peaks = np.hstack((peaks, min([ raw.shape[1], peaks[-1] + np.max(np.ediff1d(peaks)) ])))

            coefs = meta.loc[idx, ['coef0_{}'.format(i) for i in range(nsamples)] + ['coef1_{}'.format(i) for i in range(nsamples)] ].values
            coefs = np.reshape(coefs, (nsamples, 2), order='F')

            xvals = np.arange(len(raw))
            lines = np.zeros((nsamples, len(raw)))
            for i in range(len(lines)):
                b0 = peaks[i] + coefs[i,0] - buffy
                lines[i] = b0 + coefs[i,1]*(xvals - buffx)

            # # Separate the sticks and cuscuta

            plants = [ np.s_[buffx:tapes[0]+buffx, 0:peaks[1]-buffz] ]

            for i in range(1,len(tapes)):
                plants.append(np.s_[buffx:tapes[i]+buffx, peaks[i-1]+buffz:peaks[i+1]-buffz])

            for pidx in range(nsamples):
                line = lines[ pidx, plants[pidx][0] ] - plants[pidx][1].start
                patch = img[plants[pidx]][:,:,::-1]
                skewers = np.zeros((len(line), 2*buffz, 3), dtype=np.uint8)

                for i in range(len(skewers)):
                    foo = int(line[i])
                    skewers[i] = patch[i, foo-buffz:foo+buffz, :]

                # # Only keep the large chunks that are close to the central axis
                # 
                # - We are going to work only with the green channel
                # - There might be still bits of skewer left
                # - We are going to get a sense of what's the 80th quantile value of all the remaining non-zero skewer pixels in the red channel
                # - Then threshold the hold patch

                median = ndimage.median_filter(patch[:,:,1], size=7)
                skewer = skewers[:,:,1].copy()
                    
                nonzeroratio = np.sum(skewer > 0)/skewer.size
                
                if len(skewer[skewer > 0]) < 10:
                    threshold = mind
                else:
                    q = np.max([0.25, (1 - nonzeroratio)*0.5])
                    threshold = np.quantile(skewer[skewer > mind], q)
                    threshold = np.min([115, threshold])
                    threshold = np.max([mind, threshold])
                
                median[median < threshold] = 0
                median[median > 0] = 1
                
                labels,num = ndimage.label(median, structure=ndimage.generate_binary_structure(2,1))

                comp_size = np.zeros(num, dtype=int)
                feret = np.zeros((num,2))
                touch = np.zeros((num,4))
                dtouch = np.zeros(num, dtype=int)

                # Compute geometrical shape descriptors for each connected component
                # We will later drop those that are either:
                # - Too oblong
                # - Far away from the central axis
                # - Too small
                # - Too narrow

                for i in range(num):
                    box = median.copy()
                    box[labels != i+1] = 0

                    coords = np.asarray(np.nonzero(box))
                    feret[i] = np.max(coords, axis=1) - np.min(coords, axis=1) + np.array([1,1])
                    comp_size[i] = len(coords[1])
                    
                    foo = np.abs(line[coords[0]] - coords[1])
                    
                    bar = np.argmin(foo)
                    dtouch[i] = foo[bar]
                    touch[i,:2] = coords[:,bar]
                    
                f_ratio = np.divide(*np.sort(feret, axis=1).T) 
                comp_mask = ( (feret[:,1] > 30) | (f_ratio > 0.35) | (dtouch < 5) ) & (comp_size > 100) & (dtouch < 75) & (f_ratio > 0.075)
                size_mask = comp_size/np.sum(comp_size[comp_mask]) > 0.04
                mask = comp_mask & size_mask
                box = np.zeros_like(labels).astype(bool)
                comp_labels = np.nonzero(mask)[0]

                for i in comp_labels:
                    box[labels == i+1] = True

                # # Skeletonize and reduce box size

                skel = morphology.thin(box)
                coords = np.asarray(np.nonzero(skel))
                filename = dst + 'plant_{:02d}/{}_rep{}_p{:02d}_{:04d}.csv'.format(pidx, time, rep, pidx, idx)
                np.savetxt(filename, coords, fmt='%d', delimiter=',')


if __name__ == '__main__':
    main()

