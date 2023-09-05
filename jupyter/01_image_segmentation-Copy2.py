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
parser.add_argument('repetition', metavar='repetition', type=int, help='walnut batch scan id')

args = parser.parse_args()

boundary = ndimage.generate_binary_structure(2,2).astype(int) - 2
boundary[1,1] = -(np.sum(boundary) + 1)
struc1 = ndimage.generate_binary_structure(2,1)
struc2 = ndimage.generate_binary_structure(2,2)

its = 15
pad = its+10
buffx = 0
buffX = 1475
buffy = 200
buffz = 50
minsize = 100
w = 5
mind = 50
tolpx = 75

#src = '../raw/'
#dst = '../proc/'
#time = '4pm'
#rep = 7

src = args.src
dst = args.dst
time = args.time
rep = args.repetition

reference = pd.read_csv(src + 'reference_positions.csv')
ref = reference[reference.id == '{}_{}'.format(time,rep)].iloc[0]

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

filenames = glob(src + foldername + '/*.JPG')

nums = np.zeros(len(filenames), dtype=int) - 1
for i in range(len(nums)):
    num = os.path.splitext(os.path.split(filenames[i])[1])[0].split('(')[1][:-1]
    if num.isdigit():
        num = int(num)
        nums[i] = num

nums = np.argsort(nums)
#rng = np.random.default_rng(42)

for idx in range(len(nums)):
    
    if nums[idx] != -1:
        filename = filenames[nums[idx]]
        numidx = int(os.path.splitext(os.path.split(filename)[1])[0].split('(')[1][:-1])
        print(filename, numidx, sep='\t')
        if len(glob(adst + 'plant_*/{}_rep{:02d}_{:04d}.csv'.format(time, rep, numidx))) == 5:
            print('already computed. skip')
        else:
            
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

            medians = np.median(raw[:,:,0], axis=0)[300: ]
            peaks, foo = signal.find_peaks(255 - medians, distance=300, height=150, prominence=50)
            peaks += 300
            while len(peaks) < 6:
                peaks = np.hstack((peaks, [raw.shape[1] - 1]))
            
            peaks = peaks[:6]
            for i in range(len(peaks)):
                foo = 'peaks_{}'.format(i)
                if np.abs(ref[foo] - peaks[i]) > tolpx:
                    peaks[i] = ref[foo]

            erode = [None for i in range(5)]
            tapes = np.zeros(len(erode), dtype=int)
            stick = [np.s_[buffx:buffX+buffx, peaks[i]-buffy:peaks[i]+buffy] for i in range(len(erode))]
            coef = np.zeros((len(erode), 2))
            xvals = np.arange(len(raw))
            lines = np.zeros((len(coef), len(xvals)))
            medial = np.zeros(buffX)    
            xmvals = np.arange(len(medial))
            lightissue = False

            for i in range(len(stick)):
                foo = np.quantile(graw[buffx:, peaks[i]-w:peaks[i]+w], 0.75)
                foo = np.min([90, foo])
                
                rmask = graw[stick[i]] < foo
                rmask = ndimage.median_filter(rmask, 11)
                rmask = np.pad(rmask, pad)

                bound = ndimage.convolve(rmask, boundary, mode='constant', cval=0)
                bound = ndimage.binary_dilation(bound, structure = struc2, iterations=its)

                fill = ndimage.binary_fill_holes(bound)
                erod = ndimage.binary_erosion(fill, structure = struc2, iterations=its//2, border_value=1)
                erode[i] = erod[pad:-pad, pad:-pad]
                
                tapes[i] = np.argmin(np.sum(erode[i][400:buffX], axis = 1)) + 400 - 50

                for j in range(200, len(medial)):
                    foo = np.nonzero(erode[i][j])[0]
                    if len(foo) > 10:
                        medial[j] = np.median(foo)
                    else:
                        medial[j] = 0
                        
                mask = medial > 0
                if np.sum(mask) < 300:
                    lightissue = True
                    break
                
                else:
                    coef[i] = P.polyfit(xmvals[mask], medial[mask], 1,full=False)
                    b0 = peaks[i] + coef[i,0] - buffy
                    lines[i] = b0 + coef[i,1]*(xvals - buffx)
            
            if not lightissue:
                
                for i in range(len(tapes)):
                    foo = 'tapes_{}'.format(i)
                    if np.abs(ref[foo] - tapes[i]) > tolpx:
                        tapes[i] = ref[foo]

        # # Separate the sticks and cuscuta

                plants = [ np.s_[buffx:tapes[0]+buffx, 0:peaks[1]-buffz] ]
                for i in range(1,len(tapes)):
                    plants.append(np.s_[buffx:tapes[i]+buffx, peaks[i-1]+buffz:peaks[i+1]-buffz])

                for pidx in range(len(plants)):
                    
                    anchory = peaks[pidx]-plants[pidx][1].start
                    xlen = xvals[ plants[pidx][0] ] - plants[pidx][0].start
                    line = lines[ pidx, plants[pidx][0] ] - plants[pidx][1].start
                    patch = img[plants[pidx]][:,:,::-1]

        # # Only keep the large chunks that are close to the central axis

                    median = ndimage.median_filter(patch[:,:,1], size=11)

                    skewer = np.zeros((len(line), 2*buffz))
                    for i in range(len(skewer)):
                        foo = int(line[i])
                        skewer[i] = patch[i, foo-buffz:foo+buffz, 1]
            
                    bar = np.sum(skewer > 0)/skewer.size
                    if len(skewer[skewer > 0]) < 10:
                        bar = mind
                    else:
                        foo = np.quantile(skewer[skewer > mind], 0.8)
                        foo = np.min([115, foo])
                        bar = np.floor(mind + bar*(foo - mind))
                    median[median < bar] = 0
                    median[median > 0] = 1

                    labels,num = ndimage.label(median, structure=struc1)

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
                
                    touch = 55; i = 0; itermax = 10
                    comp_mask = ( (feret[:,1] > 30) | (f_ratio > 0.35) | (dtouch < 5) ) & (comp_size > minsize) & (dtouch < touch) & (f_ratio > 0.075)

                    while (np.sum(comp_mask) == 0) & (i < itermax):
                        i += 1
                        touch += 5
                        comp_mask = ( (feret[:,1] > 30) | (f_ratio > 0.35) | (dtouch < 5) ) & (comp_size > minsize) & (dtouch < touch) & (f_ratio > 0.075)

                    if np.sum(comp_mask) > 0:
                        size_mask = comp_size/np.sum(comp_size[comp_mask]) > 0.05

                        mask = comp_mask & size_mask

                        box = np.zeros_like(labels).astype(bool)
                        comp_labels = np.nonzero(mask)[0]

                        for i in comp_labels:
                            box[labels == i+1] = True

                        # # Skeletonize and reduce box size

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

                        filename = pdst + 'plant_{:02d}/{}_rep{:02d}_{:04d}'.format(pidx, time, rep, numidx)
                        tf.imwrite(filename + '.tif', comb*40, photometric='minisblack')


                        meta = [
                            raw.shape[0], raw.shape[1],
                            tapes[pidx],
                            peaks[pidx],
                            buffx, buffy,
                            plants[pidx][0].start, plants[pidx][0].stop, plants[pidx][1].start, plants[pidx][1].stop,
                            *ceros,
                            anchory
                        ]

                        df = pd.DataFrame(meta, dtype=int).T
                        df[15] = coef[pidx, 0]
                        df[16] = coef[pidx, 1]

                        filename = adst + 'plant_{:02d}/{}_rep{:02d}_{:04d}'.format(pidx, time, rep, numidx)
                        print(filename)
                        df.to_csv(filename + '.csv', index=False, header=False)
                    
                    else:
                        
                        filename = adst + 'plant_{:02d}/{}_rep{:02d}_{:04d}'.format(pidx, time, rep, numidx)
                        print('Failed to produce ' + filename)

