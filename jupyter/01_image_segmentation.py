from glob import glob
import cv2
import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib.pyplot as plt
import tifffile as tf

from scipy import ndimage, signal
from skimage import morphology

import os
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Normalize density values of walnuts')
parser.add_argument('src', metavar='raw_walnut_src', type=str, help='path to raw walnut images')
parser.add_argument('dst', metavar='clean_img_dst', type=str, help='path to store clean images')
parser.add_argument('time', metavar='scan_id', type=str, help='walnut batch scan id')
parser.add_argument('rep', metavar='repetition', type=int, help='walnut batch scan id')

args = parser.parse_args()

boundary = ndimage.generate_binary_structure(2,2).astype(int) - 2
boundary[1,1] = -(np.sum(boundary) + 1)
struc1 = ndimage.generate_binary_structure(2,1)
struc2 = ndimage.generate_binary_structure(2,2)

its = 15
pad = its+10
buffx = 200
buffy = 200
buffz = 50
minsize = 100
w = 5

#src = '../raw/'
#dst = '../proc/'
#time = '4pm'
#rep = 7

src = args.src
dst = args.dst
time = args.time
rep = args.rep

foldername = time + ' Inc Rep ' + str(rep)

adst = dst + 'anchory/' + time + '_rep{}/'.format(rep)
pdst = dst + 'prelim/' + time + '_rep{}/'.format(rep)

if not os.path.isdir(adst):
    os.mkdir(adst)
if not os.path.isdir(pdst):
    os.mkdir(pdst)

for i in range(5):
    for bar in [adst, pdst]:
        foo = bar + '/plant_{:02d}/'.format(i)
        if not os.path.isdir(foo):
            os.mkdir(foo)

numfiles = len(glob(src + foldername + '/*.JPG'))

for idx in range(681, 800):
#for idx in [58,122,179]:
    filename = src + foldername + '/' + foldername + ' ({}).JPG'.format(idx+1)
    print(filename)

    raw = cv2.imread(filename)
    rawstd = np.var(raw, axis=2, ddof=1)
    rawmean = np.mean(raw, axis = 2)

    graw = raw[:,:,1]
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hmask = ((h < 40) & (h > 0)) & (v > 60) & ((rawstd > 60) | (rawmean > 210))

    img = raw.copy()
    for i in range(3):
        img[:,:,i] *= hmask

    # Find the y-coord for the sticks

    medians = np.median(raw[:,:,0], axis=0)
    peaks, _ = signal.find_peaks(255 - medians, distance=300, height=150, prominence=50)
    peaks = peaks[:6]

    # Find the tape on each stick

    stick = [np.s_[buffx:, peaks[i]-buffy:peaks[i]+buffy] for i in range(len(peaks)-1)]
    erode = [None for i in range(len(stick))]
    tapes = np.zeros(len(stick), dtype=int)

    for i in range(len(stick)):
        foo = np.max([50, np.quantile(graw[buffx:, peaks[i]-w:peaks[i]+w], 0.3)])
        rmask = graw[stick[i]] < foo
        rmask = ndimage.median_filter(rmask, 11)
        
        dilation = ndimage.binary_dilation(rmask, structure=struc2, iterations=10, border_value=0)
        dilation = ndimage.binary_erosion(dilation, structure=struc2, iterations=5, border_value=1)
        tapes[i] = np.argmin(np.sum(dilation, axis=1)[450:1450]) + 450

    # Find the sticks above the tape and find the best-fit line

    stick = [np.s_[buffx:tapes[i]+buffx, peaks[i]-buffy:peaks[i]+buffy] for i in range(len(peaks)-1)]

    for i in range(len(stick)):
        rmask = graw[stick[i]] < np.quantile(graw[buffx:, peaks[i]-w:peaks[i]+w], 0.75)
        rmask = ndimage.median_filter(rmask, 11)
        rmask = np.pad(rmask, pad)

        bound = ndimage.convolve(rmask, boundary, mode='constant', cval=0)
        bound = ndimage.binary_dilation(bound, structure = struc2, iterations=its)

        fill = ndimage.binary_fill_holes(bound)
        erod = ndimage.binary_erosion(fill, structure = struc2, iterations=its, border_value=1)
        erode[i] = erod[pad:-pad, pad:-pad]

    coef = np.zeros((len(erode), 3))

    for i in range(len(coef)):
        medial = np.zeros(len(erode[i]) - 25)
        
        for j in range(len(medial)):
            foo = np.nonzero(erode[i][j])[0]

            if len(foo) > 10:
                medial[j] = np.median(foo)

        medial[medial == 0] = np.mean(medial)
        
        coef[i,0] = np.mean(medial)
        coef[i, 1:] = P.polyfit(range(len(medial)), medial,1,full=False)


    # Separate the sticks and cuscuta

    plants = [ np.s_[buffx:tapes[0]+buffx, 0:peaks[1]-buffz] ]

    for i in range(1,len(tapes)):
        plants.append(np.s_[buffx:tapes[i]+buffx, peaks[i-1]+buffz:peaks[i+1]-buffz])


    # Only keep the large chunks that are close to the central axis
    for pidx in range(len(plants)):

        anchory = peaks[pidx]-plants[pidx][1].start

        patch = img[plants[pidx]]

        median = ndimage.median_filter(patch[:,:,1], size=11)
        foo = median[:, anchory-10:anchory+10]

        if np.sum(foo > 0)/foo.size > 1e-2:
            foo = foo[foo > 0]
            foo = np.quantile(foo, 0.25)
            print(foo)
        else:
            foo = 100
        bar = np.min([foo, 55])
        median[median < bar] = 0
        median[median > 0] = 1

        labels,num = ndimage.label(median, structure=ndimage.generate_binary_structure(median.ndim, 1))


        # Compute geometrical shape descriptors for each connected component
        # We will later drop those that are either:
        # - Too oblong
        # - Far away from the central axis
        # - Too small
        # - Too narrow

        comp_size = np.zeros(num, dtype=int)
        feret = np.zeros((num,2))
        dtouch = np.zeros(num, dtype=int)

        for i in range(num):
            box = median.copy()
            box[labels != i+1] = 0

            coords = np.asarray(np.nonzero(box))
            feret[i] = np.max(coords, axis=1) - np.min(coords, axis=1) + np.array([1,1])
            comp_size[i] = len(coords[1])
            
            foo = np.abs(anchory - coords[1])
            
            bar = np.argmin(foo)
            dtouch[i] = foo[bar]
                
        f_ratio = np.divide(*np.sort(feret, axis=1).T) 
        
        touch = 50; i = 0; itermax = 10
        comp_mask = ((feret[:,1] > 50) | (f_ratio > 0.35)) & (comp_size > 100) & (dtouch < touch)
        
        while (np.sum(comp_mask) == 0) & (i < itermax):
            i += 1
            touch += 5
            comp_mask = ((feret[:,1] > 50) | (f_ratio > 0.35)) & (comp_size > 100) & (dtouch < touch)
        
        if np.sum(comp_mask) > 0:
            size_mask = comp_size/np.sum(comp_size[comp_mask]) > 0.05

            mask = comp_mask & size_mask

            box = np.zeros_like(labels).astype(bool)
            comp_labels = np.nonzero(mask)[0]

            for i in comp_labels:
                box[labels == i+1] = True


            # Skeletonize and combine stick with plant

            skel = morphology.thin(box)

            ceros = np.zeros(4, dtype=int)
            cero = np.nonzero(np.any(box, axis = 1))[0][np.asarray([0,-1])]
            ceros[:2] = cero

            comb = box[ceros[0]:ceros[1], : ].copy().astype(np.uint8)
            comb[:, anchory-buffy:anchory+buffy] += 2*(erode[pidx][ceros[0]:ceros[1], :]).astype(np.uint8)

            cero = np.nonzero(np.any(comb != 0, axis = 0))[0][np.asarray([0,-1])]
            ceros[2:] = cero
            ss = np.s_[ceros[0]:ceros[1], ceros[2]:ceros[3]]

            comb = comb[:, ceros[2]:ceros[3]]*2
            comb[comb == 4] = 1
            comb[comb == 2] = 3
            comb[comb == 6] = 4

            comb[ skel[ss] ] += 2

            filename = pdst + 'plant_{:02d}/{}_rep{:02d}_{:04d}'.format(pidx, time, rep, idx+1)
            tf.imwrite(filename + '.tif', comb*40, photometric='minisblack')


            meta = [
                peaks[pidx],
                buffy,
                plants[pidx][0].start, plants[pidx][0].stop, plants[pidx][1].start, plants[pidx][1].stop,
                *ceros,
                anchory
            ]

            df = pd.DataFrame(meta, dtype=int).T
            df[11] = coef[pidx, 0]
            df[12] = coef[pidx, 1]
            df[13] = coef[pidx, 2]

            filename = adst + 'plant_{:02d}/{}_rep{:02d}_{:04d}'.format(pidx, time, rep, idx+1)
            print(filename)
            df.to_csv(filename + '.csv', index=False, header=False)
        else:
            filename = adst + 'plant_{:02d}/{}_rep{:02d}_{:04d}'.format(pidx, time, rep, idx+1)
            print('Failed to produce ' + filename)

