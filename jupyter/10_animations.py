from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import json
import pandas as pd
import utils
import argparse

#### DEFINE VARIABLES #####

colors = ['#ddcc77', '#117733','#aa4499', '#88ccee', '#CC6677', '#44aa99']
markers = ['s','D','o','^','v','<','>','*']
buffx = 0
buffX = 1475
buffy = 200
buffz = 50
nsamples = 5
pxmm = 28
FS,fs = 20,16
llen = 175

isrc = '../raw/'
rsrc = '../reference/'
ssrc = '../proc/skel/'
dst = '../proc/anim/'
time = '4pm'
rep = 7

for i in range(nsamples):
    ddst = dst + '{}_rep{}_plant_{:02d}/'.format(time, rep, i)
    if not os.path.isdir(ddst):
        os.mkdir(ddst)

#### GRAB AND COMPUTE AS MUCH DATA UPFRONT ####

infodicts = [dict() for i in range(nsamples)]
for pidx in range(nsamples):
    filename = ssrc + '{}_rep{}/{}_rep{}_plant_{:02d}_posang.json'.format(time, rep, time, rep, pidx)
    with open(filename) as f:
        d = json.load(f)
    for key in d:
        d[key] = np.array(d[key])
    infodicts[pidx] = d

maxdict = np.zeros((nsamples, 4))
for pidx in range(nsamples):
    info = infodicts[pidx]
    j = 0
    for col in ['mang','mpos']:
        vals = np.hstack([info['{}_{}'.format(col, i+1)] for i in range(len(info['bookends']))])
        maxdict[pidx, j:j+2] = [np.min(vals), np.max(vals)]
        j += 2

maxdict[:, :2] = np.rad2deg(maxdict[:,:2])
maxdict[:, 2:] = maxdict[:, 2:]/28

meta = pd.read_csv(rsrc + 'reference_positions_{}_rep{}.csv'.format(time, rep))
meta = meta.set_index('numidx')

skeletons = [ dict() for j in range(nsamples) ]
for pidx in range(nsamples):
    for idx in meta.index:
        filename = ssrc + '{}_rep{}/plant_{:02d}/{}_rep{}_p{:02d}_{:04d}.csv'.format(time, rep, pidx, time, rep, pidx, idx)
        skeletons[pidx][idx] = np.loadtxt(filename, dtype=int, delimiter=',')

maxvals = np.zeros((nsamples, len(meta), 4), dtype=int)
for pidx in range(1, nsamples):
    for i,idx in enumerate(meta.index):
        foo = skeletons[pidx][idx][1] - ( meta.loc[idx]['peaks_{}'.format(pidx)] - meta.loc[idx]['peaks_{}'.format(pidx-1)] - buffz)
        maxvals[pidx, i, 0] = np.min(skeletons[pidx][idx][0])
        maxvals[pidx, i, 1] = np.max(skeletons[pidx][idx][0])
        maxvals[pidx, i, 2] = np.abs(np.min([np.min(foo), 0]))
        maxvals[pidx, i, 3] = np.max([np.max(foo),0])
pidx = 0
for i,idx in enumerate(meta.index):
    foo = skeletons[pidx][idx][1] - meta.loc[idx]['peaks_{}'.format(pidx)]
    maxvals[pidx, i, 0] = np.min(skeletons[pidx][idx][0])
    maxvals[pidx, i, 1] = np.max(skeletons[pidx][idx][0])
    maxvals[pidx, i, 2] = np.abs(np.min([np.min(foo), 0]))
    maxvals[pidx, i, 3] = np.max([np.max(foo),0])

alpha = 0.025
pad = 25
topbot = np.zeros((nsamples, 2), dtype = int)
for pidx in range(nsamples):
    topbot[pidx] = [max([np.quantile(maxvals[pidx, :,0], alpha) - pad, 0]) , 
                    min([np.quantile(maxvals[pidx, :,1], 1-alpha) + pad, min(meta['tapes_{}'.format(pidx)])])]

adjustby = np.diff(topbot, axis=1).squeeze() - np.mean(np.diff(topbot, axis=1)).astype(int) - pad
adjustby[adjustby < 0 ] = 0
topbot[:,0] += adjustby

yticks = [None for pidx in range(nsamples)]
ylims = np.zeros((nsamples, 2))
for pidx in range(nsamples):
    tape = np.min(meta['tapes_{}'.format(pidx)])
    minit = (tape - topbot[pidx,1] + pad)//pxmm+2
    mfin = (tape - topbot[pidx,0] - pad)//pxmm
    ylabs = np.arange(minit, mfin+1, 4)
    yvals = tape - ylabs*pxmm
    yticks[pidx] = np.vstack((yvals,ylabs))
    ylims[pidx] = [(tape - topbot[pidx,1])/pxmm, (tape - topbot[pidx,0])/pxmm]

def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('init', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('endn', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()
    
    init = args.init
    endn = args.endn

    ### FIX A SNAPSHOT ###

    for idx in range(init,endn):
        filename = glob(isrc + time + '*{}/*{}*({})*'.format(rep,rep,idx))[0]
        raw = cv2.imread(filename)[:,:,::-1]
        n, lines, xvals, plants = utils.get_img_metadata(meta, idx, nsamples, buffx, buffX, buffy, buffz)

        ### FIX A SKEWER ###

        for pidx in range(nsamples):
            filename = dst + '{}_rep{}_plant_{:02}/{}_rep{}_plant_{:02d}_{:04d}.png'.format(time, rep, pidx, time, rep, pidx, idx)
            
            if not os.path.isfile(filename):
                xlen, line, b = utils.get_plant_metadata(pidx, plants, xvals, lines, n)
                peak = int(np.mean(line))
                tape = np.min(meta['tapes_{}'.format(pidx)])

                upperleft=[topbot[pidx,0], peak - np.max(maxvals[pidx, :, 2]) - pad]
                lowerright=[topbot[pidx,1], peak + np.max(maxvals[pidx, :, 3]) + pad]
                skeleton = skeletons[pidx][idx]
                info = infodicts[pidx]
                rows = idx - info['bookends'][:, 0]

                posang = np.zeros((np.sum(rows >= 0), 2))
                for i in range(len(posang)):
                    posang[i] = [ xlen[-1] - info['mpos_{}'.format(i+1)][rows[i]] , info['mang_{}'.format(i+1)][rows[i]] ]

                data = [None for i in range(len(posang))]
                for i in range(len(data)):
                    data[i] = [ info['{}_{}'.format(col, i+1)][:rows[i] + 1] for col in ['mang','mpos'] ]
                    data[i][0] = np.rad2deg(data[i][0])
                    data[i][1] = data[i][1].astype(float)/28

                fig = plt.figure(layout='tight', figsize=(11.25,6.3))
                gs = GridSpec(2,3, figure=fig)
                ax0 = fig.add_subplot(gs[0,0])
                ax1 = fig.add_subplot(gs[1,0])
                ax2 = fig.add_subplot(gs[0,1:])
                ax3 = fig.add_subplot(gs[1,1:])
                ax = [ax0, ax1, ax2, ax3]

                ax[1].scatter(skeleton[1], skeleton[0], c='yellow', marker='.', zorder=3)
                ax[1].plot(line, xlen, c='r', lw=2, zorder=2)
                    
                for i in range(len(posang)):
                    xy1 = (line[int(posang[i,0])], posang[i,0])
                    m = -np.tan(posang[i,1])
                    b = xy1[1] - m*xy1[0]
                    x = np.array([xy1[0]-llen, xy1[0]+llen])
                    ax[1].plot(x, m*x+b, c=colors[i+3], lw = 3, zorder=4)
                    ax[1].scatter(*xy1, c='w', s=150, marker='*', zorder=5, edgecolor='k', linewidths=1)

                for i in [0,1]:
                    ax[i].imshow(raw[plants[pidx]], origin='upper', zorder=1)
                    ax[i].set_aspect('equal')
                    ax[i].margins(x=0.5, y=0)
                    ax[i].set_facecolor('gray')
                    ax[i].set_xlim(upperleft[1], lowerright[1])
                    ax[i].set_ylim(lowerright[0], upperleft[0])
                    ax[i].set_yticks(*yticks[pidx])
                    ax[i].tick_params(left=True, labelleft=True, bottom=False, labelbottom=False, width=2, labelsize=fs)

                ax[2].tick_params(bottom=True, labelbottom=False, width=2, labelsize=fs)
                ax[3].tick_params(bottom=True, labelbottom=True, width=2, labelsize=fs)

                ax[2].set_ylim(min([-5, maxdict[pidx,0]-5]), maxdict[pidx,1]+5)
                ax[3].set_ylim(*ylims[pidx])
                ax[3].plot([0, idx*2/75], [ylims[pidx,0], ylims[pidx,0]], lw=10, c='k')
                ax[3].set_yticks(yticks[pidx][1], yticks[pidx][1]);

                ax[2].axhline(0, c='gray', zorder=1)

                bar = []
                for i in [2,3]:
                    ax[i].set_facecolor('gainsboro')
                    ax[i].set_xlim(0,24)
                    ax[i].set_xticks(range(3,25,3),range(3,25,3))
                    for j in range(len(posang)):
                        foo = ax[i].plot(np.arange(info['bookends'][j,0],idx+1)*2/75, data[j][i-2], color=colors[j+3], lw=3, zorder=2)
                        bar.append(foo)

                ax[0].set_title('Raw', fontsize=FS)
                ax[1].set_xlabel('Processed', fontsize=FS)
                ax[3].set_xlabel('HAI', fontsize=FS)

                ax[2].set_title('Inoculation at {}, Repetition {}, Skewer {}'.format(time, rep, pidx+1), fontsize=FS)
                ax[2].set_ylabel('Angle wrt. base [$\circ$]', fontsize=fs)
                ax[3].set_ylabel('Dist. to point of inoc. [mm]', fontsize=fs);

                fig.align_xlabels(ax)
                fig.align_ylabels(ax)

                label = ['Coil {}'.format(j+1) for j in range(2)]
                foo = [ bar[i][0] for i in range(len(posang)) ]
                if len(foo) > 0:
                    leg = fig.legend(foo, label[:len(foo)], fontsize=fs, ncols=2, bbox_to_anchor=(0.5, 0., 0.5, 0.09));

                plt.savefig(filename, format='png', dpi=96, bbox_inches='tight')
                plt.close()
                
if __name__ == '__main__':
    main()


