import itertools as it
from glob import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf

from scipy import ndimage, signal, spatial, interpolate

import os
import pandas as pd

from sklearn import linear_model as linear

import argparse

parser = argparse.ArgumentParser(description='Normalize density values of walnuts')
parser.add_argument('src', metavar='clean_img_dst', type=str, help='path to store clean images')
parser.add_argument('time', metavar='scan_id', type=str, help='walnut batch scan id')
parser.add_argument('rep', metavar='repetition', type=int, help='walnut batch scan id')

args = parser.parse_args()

#src = '../proc/'
#time = '9am'
#rep = 7
#pidx = 1

src = args.src
time = args.time
rep = args.rep

dst = src + 'results/'

colors = [None, '#117733', '#aa4499', '#332288', '#44aa99', '#88ccee', '#ddcc77']
markers = ['s','D','o','^','v','<','>','*']

perms = [None, np.asarray([[0]]) ]
for i in range(len(perms), 7):
    perms.append(np.asarray(list(it.permutations(range(i)))))

boundary = ndimage.generate_binary_structure(2,2).astype(int) - 2
boundary[1,1] = -(np.sum(boundary) + 1)
unique = np.array([40, 120, 160, 200, 240], dtype=np.uint8)

struc1 = ndimage.generate_binary_structure(2,1)
struc2 = ndimage.generate_binary_structure(2,2)
wlen = 15
fs = 18
qvals = [0.2, .5, 0.8]

asrc = src + 'anchory/' + time + '_rep{}/'.format(rep)

for pidx in range(5):
	psrc = src + 'prelim/' + time + '_rep{}/plant_{:02d}/'.format(rep,pidx)
	print(psrc)
	filenames = glob(psrc + '*.tif')
	nums = np.zeros(len(filenames), dtype=int) - 1
	for i in range(len(nums)):
		num = os.path.splitext(os.path.split(filenames[i])[1])[0].split('_')[-1]
		num = int(num)
		nums[i] = num
	anums = np.argsort(nums)

	metafile = '{}{}_rep{}_plant_{:02d}.csv'.format(asrc, time, rep, pidx)
	meta = pd.read_csv(metafile, header=None)
	meta.index = nums[anums]

	lenraw = meta.iloc[0,0]
	tape = meta.iloc[:,2].values
	peak = meta.iloc[:,3].values
	buffx = meta.iloc[:,4].values
	buffy = meta.iloc[:,5].values

	plant = meta.iloc[:, 6:10].values.astype(int)
	pss = [ np.s_[plant[i,0]:plant[i,1], plant[i,2]:plant[i,3]] for i in range(len(plant)) ]

	ceros = meta.iloc[:, 10:14].values.astype(int)
	css = [ np.s_[ceros[i, 0]:ceros[i,1], ceros[i,2]:ceros[i,3]] for i in range(len(ceros)) ]
	coef = meta.iloc[:,15:].values

	rawx = ceros[:, 2] + plant[:, 2] - peak + buffy
	rawy = ceros[:, 0] + plant[:, 0] + buffx

	xvals = np.arange(lenraw)
	b0 = peak + coef[:, 0] - buffy
	lines = b0.reshape(-1,1) + np.outer(coef[:,1], xvals)

	line = [ lines[i][pss[i][0]] - plant[i,2] for i in range(len(pss)) ]
	ll = [ line[i][np.arange(ceros[i,0], ceros[i,1])] - ceros[i,2] for i in range(len(ceros))]
	xx = [ np.arange(ceros[i,1] - ceros[i,0]) for i in range(len(ceros)) ]

	v0 = np.column_stack((coef[:,1], np.ones(len(coef))))
	v = v0/(np.linalg.norm(v0, axis=1)).reshape(-1,1)
	n = np.column_stack((-v[:,1], v[:,0]))
	p = np.column_stack(([ll[i][0] for i in range(len(ll))], np.zeros(len(ll))))
	b = np.sum(p*n, axis=1)

	mint = np.min(tape)
	
	deets = []
	maxL = 0

	for idx in range(len(anums)):
		
		img = tf.imread(filenames[anums[idx]])
		skewer = (img == 40) | (img == 160) | (img == 240)
		contour = ndimage.grey_erosion(skewer, size=(1,21), mode='constant', cval=255).astype(bool)
		img = img*contour
		cross = np.asarray(np.nonzero(img > 180))
		dist = np.abs(cross[1]*n[idx,0] + cross[0]*n[idx,1] - b[idx])
		mask = dist < 10
		
		if np.sum(mask) > 5:
			skel = np.zeros(img.shape, dtype=bool)
			skel[cross[0, mask], cross[1, mask]] = True
			labels,num = ndimage.label(skel, structure=struc2)
			hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
			where = np.nonzero((hist/np.sum(hist) > 0.1))[0]
			
			if len(where) > 1:
				merged = dict()
				coords = dict()
				for i in range(len(where)):
					coords[where[i]] = np.asarray(np.nonzero(labels == where[i] + 1))
					merged[where[i]] = []

				combs = list(it.combinations(where, 2))
				for i in range(len(combs)):
					dxy = np.min(spatial.distance.cdist(coords[combs[i][0]].T, coords[combs[i][1]].T, metric='euclidean'))

					if dxy < 20:
						merged[combs[i][1]].append(combs[i][0])

				for j,i in enumerate(where):
					k = i  
					while len(merged[k]) > 0:
						k = min(merged[k])
					
					labels[labels == i+1] = k+1
					where[j] = k

				where = np.unique(where)
			
			coms = np.asarray(ndimage.center_of_mass(skel, labels, where+1))
			coms = coms[(tape[idx] - (coms[:,0] + rawy[idx])) > 100]

			if len(coms) > 0:
				angle = np.zeros(len(coms))
				coef = np.zeros(len(coms))
		
				for i in range(len(coms)):
					com = coms[i]
					dd = np.sqrt(np.sum((cross - com.reshape(-1,1))**2, axis=0))
					dmask = dd < 40
					foo = dd[dmask]
					dcom = np.max(foo) + 1 - foo
		
					X = cross[1, dmask].reshape(-1,1)
					with warnings.catch_warnings():
						warnings.filterwarnings('error')
						try:
							reg = linear.RANSACRegressor().fit(X, cross[0,dmask], sample_weight = dcom)
							coef1 = np.diff(reg.predict([[0],[1]]))[0]
							w0 = np.array([1, coef1])
							w = w0/np.linalg.norm(w0)
						except Warning:
							print(idx, filenames[anums[idx]])
			
					coef[i] = coef1
					angle[i] = -np.sign(coef1)*np.arccos(np.abs(np.sum(w*n[idx])))
			
				if len(coms) > maxL:
					maxL = len(coms)
				deet = dict()
				deet['idx'] = idx
				deet['num'] = nums[anums[idx]]
				deet['comsx'] = coms[:,1]+rawx[idx]
				deet['comsy'] = coms[:,0]+rawy[idx]
				deet['raw'] = np.array([rawx[idx],rawy[idx]])
				deet['coef1'] = coef
				deet['angles'] = np.rad2deg(angle)
				
				deets.append(deet)
	print(len(deets), maxL, sep='\t')
	fidx = deets[0]['idx']
	print('First coil at timestamp:\t', fidx)

	pos = np.zeros((len(meta) - fidx, 1+maxL))
	pos[:,0] = np.arange(deets[0]['num'], deets[0]['num'] + pos.shape[0])
	ang = np.copy(pos)

	for i in range(len(deets)):
		j = deets[i]['idx']-fidx
		pad = maxL - len(deets[i]['comsy'])
		pos[j, 1:] = np.pad(mint - deets[i]['comsy'], (pad,0), constant_values = -100.)
		ang[j, 1:] = np.pad(deets[i]['angles'], (pad,0), constant_values = -100.)

	pos = pos.T
	ang = ang.T

	mask = np.hstack(([True], np.sum(pos[1:] > 0, axis=1) > 50))
	pos = pos[mask]
	ang = ang[mask]

	for j in range(1, len(pos)):
		mask = pos[j] > 0
		labels,num = ndimage.label(mask)
		hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
		where = np.nonzero((hist/np.sum(hist) > 0.10))[0]
		
		mask = np.zeros_like(mask)
		for k in range(len(where)):
			mask[labels == where[k] + 1] = True
		pos[j, ~mask] = -100

	for j in range(1, len(pos)):
		mask = pos[j] > 0
		qq = np.quantile(ang[j, mask], qvals)
		iqr = qq[2] - qq[0]
		qq[0] - 1.5*iqr
		pos[j, ang[j] < qq[0] - 1.5*iqr] = -100
		pos[j, ang[j] > qq[2] + 1.5*iqr] = -100

	for j in range(1, len(pos)):
		mask = pos[j] > 0
		qq = np.quantile(pos[j, mask], qvals)
		iqr = qq[2] - qq[0]
		qq[0] - 1.5*iqr
		pos[j, pos[j] < qq[0] - 1.5*iqr] = -100
		pos[j, pos[j] < qq[2] + 1.5*iqr] = -100

	ipos = pos.copy()
	iang = ang.copy()
	mss = [ None ]
	for j in range(1, len(pos)):
		mss.append( np.s_[np.argmax(pos[j] > 0) : len(pos[j]) - np.argmax(np.flip(pos[j] > 0))] )
	
	for j in range(1, len(pos)):
		mask = pos[j] > 0
		interpolated = interpolate.interp1d(ang[0,mask], ang[j, mask], bounds_error=True, kind='linear', assume_sorted=True)
		iang[j, mss[j]] = interpolated(pos[0, mss[j]])

		interpolated = interpolate.interp1d(pos[0,mask], pos[j, mask], bounds_error=True, kind='linear', assume_sorted=True)
		ipos[j, mss[j]] = interpolated(pos[0, mss[j]])

	mpos = pos.copy()
	mang = ang.copy()

	weights = np.ones(wlen, dtype=bool)
	for j in range(1, len(pos)):
		mang[j,mss[j]] = signal.savgol_filter(iang[j,mss[j]], window_length=wlen, polyorder=2, deriv=0, delta=1, mode='nearest')
		mpos[j,mss[j]] = signal.savgol_filter(ipos[j,mss[j]], window_length=wlen, polyorder=2, deriv=0, delta=1, mode='nearest')


	fig, ax = plt.subplots(2,1,figsize=(13,8), sharex=True, sharey=False)
	ax = np.atleast_1d(ax).ravel()

	for j in range(1, len(pos)):
		mask = pos[j] > 0
		label = 'coil {}'.format(maxL + 1 - j)
		ax[0].scatter(pos[0, mask], ang[j, mask], c=colors[-j], marker=markers[j],  zorder=j+1, alpha=0.35, s=50)
		ax[1].scatter(pos[0, mask], pos[j, mask], c=colors[-j], marker=markers[j],  zorder=j+1, alpha=0.35, s=50)
		
		ax[0].plot(pos[0, mss[j]], iang[j, mss[j]], c=colors[-j], zorder=20)
		ax[0].plot(pos[0, mss[j]], mang[j, mss[j]], c=colors[j], zorder=23, lw=3, label=label)
		ax[1].plot(pos[0, mss[j]], mpos[j, mss[j]], c=colors[j], zorder=23, lw=3, label=label)

	ax[0].legend(fontsize=fs, markerscale=2);
	ax[0].axhline(0, c='gray')

	for i in range(len(ax)):
		ax[i].tick_params(labelsize=fs-3)

	ax[0].set_ylabel('Angle wrt base [$\circ$]', fontsize=fs)
	ax[1].set_ylabel('Height from tape [px]', fontsize=fs)

	fig.supxlabel('Timestamp', fontsize=fs)
	fig.suptitle('Inc @ {}, Repetition {}, Skewer {}'.format(time, rep, pidx), fontsize=fs+5)
	fig.tight_layout()

	filename = dst + time + '_rep{}_plant_{:02d}'.format(rep,pidx)
	plt.savefig(filename + '.png', dpi=100, format='png', pil_kwargs={'optimize':True}, bbox_inches='tight')
	plt.close()

	filename = src + 'prelim/' + time + '_rep{}/{}_rep{}_plant_{:02d}.csv'.format(rep,time,rep,pidx)
	print(filename)

	df = pd.DataFrame()
	df['timestamp'] = pos[0].astype(int)
	for i in range(1, len(pos)):
		j = maxL + 1 - i
		foo = 'angle_raw_{}'.format(j)
		df[foo] = ang[i]
		foo = 'angle_interp_{}'.format(j)
		df[foo] = iang[i]
		foo = 'angle_sg_{}'.format(j)
		df[foo] = mang[i]

	for i in range(1, len(pos)):
		j = maxL + 1 - i
		foo = 'pos_raw_{}'.format(j)
		df[foo] = pos[i]
		foo = 'pos_interp_{}'.format(j)
		df[foo] = ipos[i]
		foo = 'pos_sg_{}'.format(j)
		df[foo] = mpos[i]

	print(df.shape)
	df.to_csv(filename, index=False)
	df.head()
