from glob import glob
import cv2
import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib.pyplot as plt
from scipy import ndimage, signal

import os
import pandas as pd
import argparse

boundary = ndimage.generate_binary_structure(2,2).astype(int) - 2
boundary[1,1] = -(np.sum(boundary) + 1)

struc1 = ndimage.generate_binary_structure(2,1)
struc2 = ndimage.generate_binary_structure(2,2)


w = 5
its = 15
pad = its+10
fs = 20

buffx = 0
buffX = 1475
buffy = 200

src = '../raw/'
imgsets = sorted(glob(src + '*/'))


def get_repetition_metadata(imgsetname):
    filenames = glob(imgsetname + '*')
    reptime,_,_,repnum = imgsetname.split('/')[-2].split(' ')
    
    nums = np.zeros(len(filenames), dtype=int) - 1
    for i in range(len(nums)):
        num = os.path.splitext(os.path.split(filenames[i])[1])[0].split('(')[1][:-1]
        num = int(num)
        nums[i] = num
    nums = np.argsort(nums)

    return filenames, reptime, repnum, nums

def get_peaks_slices(raw, buffx=0, buffX=1450, buffy=50, nsamples=5):
    
    medians = np.median(raw[:,:,0], axis=0)[300: ]
    peaks, foo = signal.find_peaks(255 - medians, distance=300, height=140, prominence=50)
    peaks += 300
    if len(peaks) < 6:
        peaks = np.hstack((peaks, [raw.shape[1]]))
    stickss = [np.s_[buffx:buffX+buffx, peaks[i]-buffy:peaks[i]+buffy] for i in range(nsamples)]

    return peaks, stickss

horz = np.array([[0,0,0],[1,1,1],[0,0,0]], dtype=bool)
def approx_skewer(graw, slic, peak, buffx=0, buffX=1450, buffy=50, w=5, struc2=ndimage.generate_binary_structure(2,2), its=15, pad=25):
    
    q75 = np.quantile(graw[ buffx:, peak-w:peak+w], 0.75)
    q75 = np.min([90, q75])
    
    rmask = graw[slic] < q75
    rmask = ndimage.median_filter(rmask, 11)
    rmask = np.pad(rmask, pad)
    
    bound = ndimage.convolve(rmask, boundary, mode='constant', cval=0)
    bound = ndimage.binary_dilation(bound, structure = struc2, iterations=its)
    
    fill = ndimage.binary_fill_holes(bound)
    erod = ndimage.binary_erosion(fill, structure = struc2, iterations=its//2, border_value=1)
    erode = erod[pad:-pad, pad:-pad]
    
    sumvals = np.sum(erode[400:buffX], axis = 1)
    tape = np.argmin(sumvals) + 400

    cdt = ndimage.distance_transform_cdt(erod, metric=horz)
    cdt = cdt[pad:-pad, pad:-pad]
    medial = np.argmax(cdt, axis=1)
    
    mask = medial > 0
    coef = P.polyfit(np.arange(len(medial))[mask], medial[mask], 1,full=False)
    
    return coef, tape, erode

nsamples = 5

def main():
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('jdx', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('init', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('endn', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()
    
    jdx = args.jdx
    init = args.init
    endn = args.endn
    
    filenames, reptime, repnum, nums = get_repetition_metadata(imgsets[jdx])
    
    dst = '../reference/'
    dst = dst + '{}_rep{}/'.format(reptime, repnum)
    if not os.path.isdir(dst):
        os.mkdir(dst)
        
    for idx in range(init, endn):
        filename = filenames[nums[idx]]
        numidx = int(os.path.splitext(os.path.split(filename)[1])[0].split('(')[1][:-1])
        fname = dst + 'metapx_{}_rep{}_{:04d}.csv'.format(reptime, repnum, numidx)

        if not os.path.isfile(fname):
        
            raw = cv2.imread(filename)
            
            peaks, stick = get_peaks_slices(raw, buffx, buffX, buffy)
            
            coefs = np.zeros((nsamples,2))
            tapes = np.zeros(len(coefs), dtype=int)
            for i in range(len(coefs)):
                coefs[i], tapes[i], _ = approx_skewer(raw[:,:,1], stick[i], peaks[i])
                
            np.savetxt(fname, np.hstack((peaks[:nsamples], tapes)), fmt='%d', delimiter=',', newline=',')
            fname = dst + 'metacf_{}_rep{}_{:04d}.csv'.format(reptime, repnum, numidx)
            np.savetxt(fname, np.ravel(coefs, order='F'), fmt='%.5e', delimiter=',', newline=',')

if __name__ == '__main__':
    main()

