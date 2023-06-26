''' contains various useful program snippets for neuron analysis:

  get_nFolder   get number of folders in path
  pathcat       attach strings to create proper paths
  _hsm          half sampling mode to obtain baseline


'''

import os, pickle, cmath, time, cv2, h5py
import scipy as sp
import scipy.stats as sstats
from scipy import signal, cluster
import numpy as np
import matplotlib.pyplot as plt
from fastcluster import linkage
from scipy.spatial.distance import squareform

from .utils import gauss_smooth


def get_performance(pathMouse,s_arr,rw_pos=[50,70],rw_delay=0,f=15,plot_bool=False):
    
    nSes = len(s_arr)
    L = 120         ## length of track in cm
    nbin = 100
    if len(rw_pos) <= 2:
        rw_pos = np.ones((len(s_arr),1))*rw_pos
    if np.isscalar(rw_delay):
        rw_delay = np.ones(len(s_arr))*rw_delay
    range_approach = [-2,4]      ## in secs
    ra = range_approach[1]-range_approach[0]
    vel_arr = np.linspace(0.5,30,51)

    dataStore = {}
    ### can only get performance from mouse behavior: stopping / velocity
    for i,s in enumerate(s_arr):

        pathSession = os.path.join(pathMouse,'Session%02d'%s)

        dataBH = define_active(pathSession,plot_bool=plot_bool)
        if dataBH is None:
            continue

        dataBH['velocity'] *= L/nbin
        try:
            hist = gauss_smooth(np.histogram(dataBH['velocity'],vel_arr)[0],2,mode='nearest')

            vel_run_idx = signal.find_peaks(hist,distance=10,prominence=10)[0][-1]
            vel_run = vel_arr[vel_run_idx]
            vel_min_idx = signal.find_peaks(-hist,distance=5)[0]
            vel_min_idx = vel_min_idx[vel_min_idx<vel_run_idx][-1]
            vel_thr = vel_arr[vel_min_idx]
        except:
            vel_thr = dataBH['velocity'].mean()

        dataStore[i] = {}
        dataStore[i]['trials'] = {}
        dataStore[i]['trials']['RW_reception'] = np.zeros(dataBH['trials']['ct'],'bool')
        dataStore[i]['trials']['RW_frame'] = np.zeros(dataBH['trials']['ct'],'int')
        dataStore[i]['trials']['slowDown'] = np.zeros(dataBH['trials']['ct'],'bool')
        dataStore[i]['trials']['frame_slowDown'] = np.zeros(dataBH['trials']['ct'],'int')
        dataStore[i]['trials']['pos_slowDown'] = np.zeros(dataBH['trials']['ct'])*np.NaN
        dataStore[i]['trials']['t_slowDown_beforeRW'] = np.zeros(dataBH['trials']['ct'])*np.NaN

        dataBH['RW_approach_time'] = np.zeros((dataBH['trials']['ct'],ra*f))
        dataBH['RW_approach_space'] = np.zeros((dataBH['trials']['ct'],nbin))*np.NaN
        for t in range(dataBH['trials']['ct']):

            pos_trial = dataBH['position'][dataBH['trials']['frame'][t]:dataBH['trials']['frame'][t+1]].astype('int')
            vel_trial = dataBH['velocity'][dataBH['trials']['frame'][t]:dataBH['trials']['frame'][t+1]]
            time_trial = dataBH['time'][dataBH['trials']['frame'][t]:dataBH['trials']['frame'][t+1]]
            for j,p in enumerate(range(nbin)):
                dataBH['RW_approach_space'][t,j] = vel_trial[pos_trial==p].mean()

            try:        ## fails, when last trial is cut off, due to measure end
                idx_enterRW = np.where(pos_trial>rw_pos[i,0])[0][0]         ## find, where first frame within rw position is
                idx_RW_reception = int(idx_enterRW + rw_delay[i]*f)

                if pos_trial[idx_RW_reception]<rw_pos[i,1]:
                    dataStore[i]['trials']['RW_frame'][t] = dataBH['trials']['frame'][t] + idx_RW_reception
                    dataStore[i]['trials']['RW_reception'][t] = True
                    idx_trough_tmp = signal.find_peaks(-vel_trial,prominence=2,height=-vel_thr,distance=f)[0]
                    # print(idx_enterRW)
                    idx_trough_tmp = idx_trough_tmp[idx_trough_tmp>idx_enterRW]
                    if len(idx_trough_tmp)>0:
                        idx_trough = idx_enterRW + idx_trough_tmp[0]
                        ### slowing down should occur before this - defined by drop below threshold velocity
                        slow_down = np.where((vel_trial[:idx_trough]>vel_thr) & (pos_trial[:idx_trough]<=rw_pos[i,1]) & (pos_trial[:idx_trough]>5))[0]# & (pos_trial[:idx_trough]<rw_pos[i,1]))[0]
                        if len(slow_down) > 0:
                            slow_down = slow_down[-1]
                            # print(pos_trial[slow_down])
                            if vel_trial[slow_down+1] < vel_thr :#vel_trial[slow_down+1]<vel_thr:
                                dataStore[i]['trials']['slowDown'][t] = True
                                dataStore[i]['trials']['frame_slowDown'][t] = dataBH['trials']['frame'][t] + slow_down
                                dataStore[i]['trials']['pos_slowDown'][t] = pos_trial[slow_down]
                                dataStore[i]['trials']['t_slowDown_beforeRW'][t] = time_trial[idx_RW_reception] - time_trial[slow_down]
            except:
                continue

            idx_enterRW = int(dataBH['trials']['frame'][t]+np.where(pos_trial>rw_pos[i,0])[0][0] + rw_delay[i]*f)     ## find, where first frame within rw position is

            dataBH['RW_approach_time'][t,:ra*f+np.minimum(0,len(dataBH['velocity'])-(idx_enterRW+f*range_approach[1]))] = dataBH['velocity'][idx_enterRW+f*range_approach[0]:idx_enterRW+f*range_approach[1]]

        # plot_fig = False
        if plot_bool:
            plt.figure()
            plt.subplot(221)
            plt.plot(np.linspace(range_approach[0],range_approach[1],f*ra),dataBH['RW_approach_time'].T,color=[0.5,0.5,0.5],alpha=0.5)
            plt.plot(np.linspace(range_approach[0],range_approach[1],f*ra),dataBH['RW_approach_time'].mean(0),color='k')
            plt.plot(-dataStore[i]['trials']['t_slowDown_beforeRW'][dataStore[i]['trials']['slowDown'][:]],dataBH['velocity'][dataStore[i]['trials']['frame_slowDown'][dataStore[i]['trials']['slowDown'][:]]],'rx')
            plt.plot(range_approach,[vel_thr,vel_thr],'k--',linewidth=0.5)
            plt.xlim(range_approach)
            plt.subplot(222)
            plt.plot(np.linspace(0,nbin-1,nbin),dataBH['RW_approach_space'].T,color=[0.5,0.5,0.5],alpha=0.5)
            plt.plot(np.linspace(0,nbin-1,nbin),np.nanmean(dataBH['RW_approach_space'],0),color='k')
            plt.plot(dataStore[i]['trials']['pos_slowDown'][dataStore[i]['trials']['slowDown']],dataBH['velocity'][dataStore[i]['trials']['frame_slowDown'][dataStore[i]['trials']['slowDown'][:]]],'rx')
            plt.plot([0,nbin],[vel_thr,vel_thr],'k--',linewidth=0.5)
            ax = plt.subplot(223)
            ax.hist(dataBH['velocity'],np.linspace(0.5,30,51))
            ax.plot(np.linspace(0.5,30,50),hist)
            ax.plot([vel_thr,vel_thr],[0,ax.get_ylim()[-1]],'k--')
            plt.show(block=False)

    return dataStore

def define_active(pathSession,f=15,plot_bool=False):

    data = {}
    pathBH = None
    for file in os.listdir(pathSession):
    #   if file.endswith("aligned.mat"):
      if file.endswith("aligned_behavior.pkl"):
          pathBH = os.path.join(pathSession, file)
    if pathBH is None:
        return
    
    ### load data
    with open(pathBH,'rb') as f:
        data = pickle.load(f)

    # min_val,max_val = np.nanpercentile(data['bin_position'],(0.1,99.9))
    # loc_dist = max_val - min_val

    # ## define trials    
    # data['trials'] = {}
    # data['trials']['start'] = np.append(0,np.where(np.diff(data['bin_position'])<(-loc_dist/2))[0] + 1)

    # ## remove half trials from data
    # if not (data['bin_position'][0] < 10):
    #     data['active'][:max(0,data['trials']['start'][0])] = False

    # if not (data['bin_position'][-1] >= 90):
    #     data['active'][data['trials']['start'][-1]:] = False
    
    # pos_active = data['bin_position'][data['active']]
    # data['trials']['start_active'] = np.append(0,np.where(np.diff(pos_active)<(-loc_dist/2))[0] + 1)
    # data['trials']['start_active_t'] = data['time'][data['active']][data['trials']['start_active']]
    # data['trials']['ct'] = len(data['trials']['start_active'])

    if plot_bool:
        plt.figure(dpi=300)
        plt.plot(data['time'],data['position'],'r.',markersize=1,markeredgecolor='none')
        plt.plot(data['time'][data['active']],data['position'][data['active']],'k.',markersize=2,markeredgecolor='none')
        plt.show(block=False)
    return data
