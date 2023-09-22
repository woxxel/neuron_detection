''' contains several functions, defining data and parameters used for analysis

  set_para

'''
import numpy as np
from .utils import pathcat
from .get_t_measures import *

def set_para(basePath,mouse,s,nP=0,nbin=100,plt_bool=False,sv_bool=False,suffix='2'):

  ## set paths:
  pathMouse = pathcat([basePath,mouse])
  pathSession = pathcat([pathMouse,'Session%02d'%s])

  coarse_factor = int(nbin/20)
  #nbin = 100
  #coarse_factor = 5
  qtl_steps = 4


  fact = 1 ## factor from path length to bin number


  # gate_mice = ["34","35","65","66","72","839","840","841","842","879","882","884","886","67","68","91","549","551","756","757","758","918shKO","931wt","943shKO"]
  # nogate_mice = ["231","232","236","243","245","246","762","",""]

  # zone_idx = {}
  # if any(mouse==m for m in gate_mice):        ## gate
  #   zone_idx['gate'] = [18,33]
  #   zone_idx['reward'] = [75,95]
  #   have_gt = True;
  # elif any(mouse==m for m in nogate_mice):    ## no gate
  #   zone_idx['reward'] = [50,70]#[50,66]#
  #   zone_idx['gate'] = [np.NaN,np.NaN]
  #   have_gt = False;

  # zone_mask = {}
  # zone_mask['reward'] = np.zeros(nbin).astype('bool')#range(zone_idx['reward'][0],zone_idx['reward'][-1])
  # zone_mask['gate'] = np.zeros(nbin).astype('bool')
  # zone_mask['others'] = np.ones(nbin).astype('bool')

  # zone_mask['reward'][zone_idx['reward'][0]:zone_idx['reward'][-1]] = True
  # zone_mask['others'][zone_mask['reward']] = False
  # if have_gt:
  #   zone_mask['gate'][zone_idx['gate'][0]:zone_idx['gate'][-1]] = True
  #   zone_mask['others'][zone_mask['gate']] = False

  # # zone_mask['others'][40:50] = False  ## remove central wall pattern change?!
  # zone_mask['active'] = nbin+1
  # zone_mask['silent'] = nbin+2

  print('now')

  para = {'nbin':nbin,'f':15,
          'bin_array':np.linspace(0,nbin-1,nbin),
          'bin_array_centers':np.linspace(0,nbin,nbin+1)-0.5,
          'coarse_factor':coarse_factor,
          'nbin_coarse':int(nbin/coarse_factor),
          'pxtomu':536/512,
          'L_track':120,

          'rate_thr':4,
          'width_thr':5,

          'trials_min_count':3,
          'trials_min_fraction':0.2,

          'Ca_thr':0,

          # 't_measures': get_t_measures(mouse),

          'nP':nP,
          'N_bs':10000,
          'repnum':1000,
          'qtl_steps':qtl_steps,'sigma':5,
          'qtl_weight':np.ones(qtl_steps)/qtl_steps,
          'names':['A_0','A','SD','theta'],
          #'CI_arr':[0.001,0.025,0.05,0.159,0.5,0.841,0.95,0.975,0.999],
          'CI_arr':[0.025,0.05,0.95,0.975],

          'plt_bool':plt_bool&(nP==0),
          # 'plt_theory_bool':True&(nP==0),
          'plt_theory_bool':False,
          'plt_sv':sv_bool&(nP==0),

          'mouse':mouse,
          'session':s,
          'pathSession':pathSession,
          'pathMouse':pathMouse,
          'pathFigs':'/home/wollex/Data/Science/PhD/Thesis/pics/Methods',#'/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Results/pics/Methods',

          ### provide names for figures
          'svname_status':          'PC_fields%s_status.mat'%suffix,
          'svname_fields':          'PC_fields%s_para.mat'%suffix,
          'svname_firingstats':     'PC_fields%s_firingstats.mat'%suffix,

          ### modes, how to perform PC detection
          'modes':{'activity':'calcium',#'spikes',#          ## data provided: 'calcium' or 'spikes'
                    'info':'MI',                   ## information calculated: 'MI', 'Isec' (/second), 'Ispike' (/spike)
                    'shuffle':'shuffle_trials'     ## how to shuffle: 'shuffle_trials', 'shuffle_global', 'randomize'
                  },

          # 'zone_idx':zone_idx,
          # 'zone_mask':zone_mask
          }


## -----------------------------------------------------------------------------------------------------------------------

  #if nargin == 3:

    #para.t_s = get_t_measures(mouse);
    #para.nSes = length(para.t_s);

    #time_real = false;
  #if time_real
    #t_measures = get_t_measures(mouse);
    #t_mask_m = false(1,t_measures(nSes));
    #for s = 1:nSes-1
      #for sm = s+1:nSes
        #dt = t_measures(sm)-t_measures(s);
        #t_mask_m(dt) = true;
      #end
    #end
    #t_data_m = find(t_mask_m);
    #t_ses = t_measures;
    #t_mask = t_mask_m;
    #t_data = t_data_m;
    #nT = length(t_data);
  #else
    #t_measures = get_t_measures(mouse);
    #t_measures = t_measures(s_offset:s_offset+nSes-1);
##      t_measures = 1:nSes;    ## remove!
##      t_measures
    #t_ses = linspace(1,nSes,nSes);
    #t_data = linspace(1,nSes,nSes);
    #t_mask = true(nSes,1);
    #nT = nSes;
  #end

  return para
