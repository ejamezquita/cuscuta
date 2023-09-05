from scipy import ndimage
import numpy as np

def get_largest_element(comp, thr=0.1, minsize=None, outlabels=False):
    tot = np.sum(comp > 0)
    labels,num = ndimage.label(comp, structure=ndimage.generate_binary_structure(comp.ndim, 1))
    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    argsort_hist = np.argsort(hist)[::-1]

    if minsize is None:
        minsize = np.max(hist) + 1

    where = np.where((hist/tot > thr) | (hist > minsize))[0] + 1
    print(num,'components\t',len(where),'preserved')
    print(np.sort(hist)[::-1][:20])

    mask = labels == where[0]
    for w in where[1:]:
        mask = mask | (labels == w)
    box0 = comp.copy()
    box0[~mask] = 0

    if outlabels:
        return box0, labels, where

    return box0

def clean_zeroes(img, pad=2):
    dim = img.ndim
    orig_size = img.size

    cero = np.arange(2*dim)

    for k in range(dim):
        ceros = np.all(img == 0, axis = (k, (k+1)%dim))

        for i in range(len(ceros)):
            if(~ceros[i]):
                break
        for j in range(len(ceros)-1, 0, -1):
            if(~ceros[j]):
                break
        cero[k] = i
        cero[k+dim] = j+1
    for i in range(dim):
        cero[i] -= 2
    for i in range(dim, len(cero)):
        cero[i] += 2
    cero[cero < 0] = 0
    img = img[cero[1]:cero[4], cero[2]:cero[5], cero[0]:cero[3]]

    print(round(100-100*img.size/orig_size),'% reduction from input')

    return img, cero
