import os, random, h5py, time, math, cmath, copy, importlib, warnings, pickle
import multiprocessing as mp
from multiprocessing import get_context

from skimage import measure
from collections import Counter
import scipy as sp
import scipy.stats as sstats
import scipy.io as sio
from scipy.io import savemat, loadmat

import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#from scipy.signal import savgol_filter

### UltraNest toolbox ###
### from https://github.com/JohannesBuchner/UltraNest
### Documentation on https://johannesbuchner.github.io/UltraNest/performance.html
import ultranest
from ultranest.plot import cornerplot
import ultranest.stepsampler

from .spike_shuffling import shuffling

from .utils import pathcat, _hsm, get_nPaths, extend_dict, compute_serial_matrix, corr0, gauss_smooth, get_reliability, get_firingrate, add_number, pickleData
from .utils_data import set_para
from .utils_analysis import define_active

warnings.filterwarnings("ignore")



class detect_PC:

  def __init__(self,basePath,mouse,s,nP,nbin=100,plt_bool=False,sv_bool=False,suffix=''):

    print('----------- mouse %s --- session %d -------------'%(mouse,s))

    ### set global parameters and data for all processes to access
    self.para = set_para(basePath,mouse,s,nP=nP,nbin=nbin,plt_bool=plt_bool,sv_bool=sv_bool,suffix=suffix)
    self.get_behavior()       ## load and process behavior


  def run_detection(self,S=None,rerun=False,f_max=1,return_results=False,specific_n=None,artificial=False,dataSet='redetect',mode_info='MI',mode_activity='spikes',assignment=None):

    global t_start
    t_start = time.time()
    self.dataSet = dataSet
    self.f_max = f_max
    self.para['modes']['info'] = mode_info
    self.para['modes']['activity'] = mode_activity
    self.tmp = {}   ## dict to store some temporary variables in

    if S is None:
        S, other = load_activity(self.para['pathSession'],dataSet=dataSet)

        if dataSet == 'redetect':
            idx_evaluate = other[0]
            idx_previous = other[1]
            SNR = other[2]
            r_values = other[3]
    else:
        nCells = S.shape[0]
        SNR = np.zeros(nCells)*np.NaN
        r_values = np.zeros(nCells)*np.NaN

    nCells = S.shape[0]

    if not (specific_n is None):
      self.para['n'] = specific_n
      #self.S = S[specific_n,:]
      result = self.PC_detect(S[specific_n,:])
      return result
    if rerun:
      if artificial:
        #nDat,pathData = get_nPaths(self.para['pathSession'],'artificialData_analyzed_n')
        #for i in range(nDat):

        f = open(self.para['svname_art'],'rb')
        PC_processed = pickle.load(f)
        f.close()

        #PC_processed = extend_dict(PC_processed,ld_tmp['fields']['parameter'].shape[0],ld_tmp)
      else:
        PC_processed = {}
        PC_processed['status'] = loadmat(os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_status'])[0]+'.pkl'),squeeze_me=True)
        PC_processed['fields'] = loadmat(os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_fields'])[0]+'.pkl'),squeeze_me=True)
        PC_processed['firingstats'] = loadmat(os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_firingstats'])[0]+'.pkl'),squeeze_me=True)

      #self.para['modes']['info'] = False

      #idx_process = np.where(np.isnan(PC_processed['status']['Bayes_factor'][:,0]))[0]
      idx_process = np.where(PC_processed['status']['MI_value']<=0.1)[0]
      print(idx_process)
    elif not (assignment is None):
        idx_process = assignment
        print(idx_process)
    else:
      idx_process = np.arange(nCells)
    nCells_process = len(idx_process)
    if nCells_process:
      #print(idx_process)
      print('run detection on %d neurons'%nCells_process)

      ### run multiprocessing from here, obtain list of result-dictionaries
      result_tmp = []
      if self.para['nP'] > 0:

        pool = get_context("spawn").Pool(self.para['nP'])
        batchSz = 500
        nBatch = nCells_process//batchSz

        for i in range(nBatch+1):
          idx_batch = idx_process[i*batchSz:min(nCells_process,(i+1)*batchSz)]
          #result_tmp.extend(pool.map(self.PC_detect,S[idx_batch,:]))
          #res = pool.starmap(self.PC_detect,zip(S[idx_batch,:],SNR[idx_batch],r_values[idx_batch]))
          #result_tmp.extend(res)
          result_tmp.extend(pool.starmap(self.PC_detect,zip(S[idx_batch,:],SNR[idx_batch],r_values[idx_batch])))
          print('\n\t\t\t ------ mouse %s --- session %d ------ %d / %d neurons processed\t ------ \t time passed: %7.2fs\n'%(self.para['mouse'],self.para['session'],min(nCells_process,(i+1)*batchSz),nCells_process,time.time()-t_start))
      else:
        for n0 in range(nCells_process):
          n = idx_process[n0]
          result_tmp.append(self.PC_detect(S[n,:],SNR[n],r_values[n]))
          print('\t\t\t ------ mouse %s --- session %d ------ %d / %d neurons processed\t -----\t time passed: %7.2fs'%(self.para['mouse'],self.para['session'],n0+1,nCells_process,time.time()-t_start))

      results = self.build_PC_results(nCells)   ## pre-allocate array
      #nCells = batchSz*(nBatch+1)
      for n in range(nCells):
        for key_type in result_tmp[0].keys():
          for key in result_tmp[0][key_type].keys():
            if key[0] == '_':
              continue
            if rerun:
                if n in idx_process:
                    n0 = np.where(idx_process==n)[0][0]
                    results[key_type][key][n,...] = result_tmp[n0][key_type][key]
                else:
                    #((~np.isnan(PC_processed['status']['Bayes_factor'][n,0])) | (key in ['MI_value','MI_p_value','MI_z_score','Isec_value','Isec_p_value','Isec_z_score'])):# | (n>=idx_process[10])):
                    results[key_type][key][n,...] = PC_processed[key_type][key][n,...]
            elif not (assignment is None):
                if n<len(idx_process):
                    n0 = idx_process[n]
                    results[key_type][key][n0,...] = result_tmp[n][key_type][key]
            else:
                n0 = np.where(idx_process==n)[0][0]
                results[key_type][key][n,...] = result_tmp[n0][key_type][key]

      #for (r,n) in zip(result_tmp,range(nCells)):
        #for key_type in r.keys():
          #for key in r[key_type].keys():
            #results[key_type][key][n,...] = r[key_type][key]

      print('time passed (overall): %7.2f'%(time.time()-t_start))

      if return_results:
        return results
      else:
        print('saving results...')
        # savemat(self.para['svname_status'],results['status'])
        # savemat(self.para['svname_fields'],results['fields'])
        # savemat(self.para['svname_firingstats'],results['firingstats'])
        status_path = os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_status'])[0] + '.pkl')
        print(status_path)
        field_path = os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_fields'])[0] + '.pkl')
        print(field_path)
        firingstats_path = os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_firingstats'])[0] + '.pkl')
        print(firingstats_path)
        pickleData(results['status'],status_path,mode='save')
        pickleData(results['fields'],field_path,mode='save')
        pickleData(results['firingstats'],firingstats_path,mode='save')
        return
    else:
      print('nothing here to process')


  def get_behavior(self,T=None):

    print(self.para['pathSession'])
    data = define_active(self.para['pathSession'])
    
    if T is None:
      T = data['time'].shape[0]
    self.dataBH = {}
    self.dataBH['active'] = data['active']
    self.dataBH['time'] = data['time']
    self.dataBH['velocity'] = data['velocity']/self.para['nbin']*self.para['L_track']

    self.dataBH['position'] = data['position']/data['position'].max()*self.para['L_track']
    self.dataBH['binpos'] = data['position'].astype('int')

    nbin_coarse = (self.para['nbin']/self.para['coarse_factor'])
    self.dataBH['binpos_coarse'] = (data['position']*nbin_coarse/(data['position'].max()*1.001)).astype('int')

    self.dataBH['binpos_active'] = self.dataBH['binpos'][data['active']]
    self.dataBH['binpos_coarse_active'] = self.dataBH['binpos_coarse'][data['active']]

    self.dataBH['time_active'] = self.dataBH['time'][data['active']]
    self.dataBH['T'] = np.count_nonzero(data['active'])


    ###### define trials
    self.dataBH['trials'] = {}
    #self.dataBH['trials']['frame_raw'] = np.where(np.diff(self.dataBH['binpos'])<-10)[0]+1
    self.dataBH['trials']['frame'] = np.hstack([0, np.where(np.diff(self.dataBH['binpos_active'])<-10)[0]+1,len(self.dataBH['time_active'])])

    self.dataBH['trials']['t'] = np.hstack([self.dataBH['time_active'][self.dataBH['trials']['frame'][:-1]],self.dataBH['time_active'][self.dataBH['trials']['frame'][-1]-1]])
    #dt = np.diff(self.dataBH['trials']['t'])

    self.dataBH['trials']['ct'] = len(self.dataBH['trials']['frame'])-1
    self.dataBH['trials']['dwelltime'] = np.zeros((self.dataBH['trials']['ct'],self.para['nbin']))
    self.dataBH['trials']['T'] = np.zeros(self.dataBH['trials']['ct']).astype('int')

    t_offset = 0
    self.dataBH['trials']['trial'] = {}
    for t in range(self.dataBH['trials']['ct']):
      self.dataBH['trials']['trial'][t] = {}
      self.dataBH['trials']['trial'][t]['binpos_active'] = self.dataBH['binpos_active'][self.dataBH['trials']['frame'][t]:self.dataBH['trials']['frame'][t+1]]
      self.dataBH['trials']['dwelltime'][t,:] = np.histogram(self.dataBH['trials']['trial'][t]['binpos_active'],self.para['bin_array_centers'])[0]/self.para['f']
      self.dataBH['trials']['T'][t] = len(self.dataBH['trials']['trial'][t]['binpos_active'])
    return self.dataBH


  def PC_detect(self,S,SNR=None,r_value=None):
  #def PC_detect(varin):
    t_start = time.time()
    result = self.build_PC_fields()
    S[S<0] = 0

    if not (SNR is None):
      result['status']['SNR'] = SNR
      result['status']['r_value'] = r_value

    T = S.shape[0]
    try:
      active, result['firingstats']['rate'] = self.get_active_Ca(S)

      # if (result['status']['SNR'] < 2) | (results['r_value'] < 0):
      #     print('component not considered to be proper neuron')
      #     return

      if result['firingstats']['rate']==0:
        print('no activity for this neuron')
        return result

      ### get trial-specific activity and overall firingmap stats
      trials_S, result['firingstats']['trial_map'] = self.get_trials_activity(active)

      ## obtain mutual information first - check if (computational cost of) finding fields is worth it at all
      t_start = time.time()
      if self.para['modes']['info']:
        MI_tmp = self.test_MI(active,trials_S)
        for key in MI_tmp.keys():
          result['status'][key] = MI_tmp[key]
      #print('time taken (information): %.4f'%(time.time()-t_start))

      result = self.get_correlated_trials(result,smooth=2)
      # print(result['firingstats']['trial_field'])
      # return

      firingstats_tmp = self.get_firingstats_from_trials(result['firingstats']['trial_map'])
      for key in firingstats_tmp.keys():
        result['firingstats'][key] = firingstats_tmp[key]
      # return
      if np.any(result['firingstats']['trial_field']) and ((result['status']['SNR']>2) or np.isnan(result['status']['SNR'])):  # and (result['status']['MI_value']>0.1)     ## only do further processing, if enough trials are significantly correlated
        for t in range(5):
            trials = np.where(result['firingstats']['trial_field'][t,:])[0]
            if len(trials)<1:
                continue

            firingstats_tmp = self.get_firingstats_from_trials(result['firingstats']['trial_map'],trials,complete=False)

          #print(gauss_smooth(firingstats_tmp['map'],2))

          # if (gauss_smooth(firingstats_tmp['map'],4)>(self.para['rate_thr']/2)).sum()>self.para['width_thr']:

            ### do further tests only if there is "significant" mutual information

            for f in range(self.f_max+1):
                field = self.run_nestedSampling(result['firingstats'],firingstats_tmp['map'],f)

            ## pick most prominent peak and store into result, if bayes factor > 1/2
            if field['Bayes_factor'][0] > 0:

                dTheta = np.abs(np.mod(field['parameter'][3,0]-result['fields']['parameter'][:t,3,0]+self.para['L_track']/2,self.para['L_track'])-self.para['L_track']/2)
                if not np.any(dTheta < 10):   ## should be in cm
                    ## store results into array index "t"
                    for key in field.keys():#['parameter','p_x','posterior_mass']:
                        result['fields'][key][t,...] = field[key]
                    result['fields']['nModes'] += 1

                    ## reliability is calculated later
                    result['fields']['reliability'][t], _, _ = get_reliability(result['firingstats']['trial_map'],result['firingstats']['map'],result['fields']['parameter'],t)

                    if self.para['plt_bool']:
                        self.plt_results(result,t)

      t_process = time.time()-t_start

      #print('get spikeNr - time taken: %5.3g'%(t_end-t_start))
      print_msg = 'p-value: %.2f, value (MI/Isec): %.2f / %.2f, '%(result['status']['MI_p_value'],result['status']['MI_value'],result['status']['Isec_value'])

      if result['fields']['nModes']>0:
        print_msg += ' \t Bayes factor (reliability) :'
        for f in np.where(result['fields']['Bayes_factor']>1/2)[0]:#range(result['fields']['nModes']):
          print_msg += '\t (%d): %.2f+/-%.2f (%.2f), '%(f+1,result['fields']['Bayes_factor'][f,0],result['fields']['Bayes_factor'][f,1],result['fields']['reliability'][f])
      if not(SNR is None):
        print_msg += '\t SNR: %.2f, \t r_value: %.2f'%(SNR,r_value)
      print_msg += ' \t time passed: %.2fs'%t_process
      print(print_msg)

      #except (KeyboardInterrupt, SystemExit):
        #raise
    except:# KeyboardInterrupt: #:# TypeError:#
      print('analysis failed: (-)')# p-value (MI): %.2f, \t bayes factor: %.2fg+/-%.2fg'%(result['status']['MI_p_value'],result['status']['Bayes_factor'][0,0],result['status']['Bayes_factor'][0,1]))
      #result['fields']['nModes'] = -1

    return result#,sampler


  def get_active_Ca(self,S):

    active = {}
    active['S'] = S[self.dataBH['active']]    ### only consider activity during continuous runs

    ### calculate firing rate
    frate, S_thr, _ = get_firingrate(active['S'],f=self.para['f'],sd_r=self.para['Ca_thr'])
    if self.para['modes']['activity']=='spikes':
      active['S'] = np.floor(S / S_thr)[self.dataBH['active']]#(S>S_thr).astype('float')[self.dataBH['active']]#

    if frate>0:
      if self.para['modes']['info']:
        if self.para['modes']['info'] == 'MI':
          ## obtain quantized firing rate for MI calculation
          active['qtl'] = sp.ndimage.gaussian_filter(np.floor(S / S_thr).astype('float')*self.para['f'],self.para['sigma'])#(S>S_thr).astype('float')#
          active['qtl'] = active['qtl'][self.dataBH['active']]
          qtls = np.quantile(active['qtl'][active['qtl']>0],np.linspace(0,1,self.para['qtl_steps']+1))
          active['qtl'] = np.count_nonzero(active['qtl'][:,np.newaxis]>=qtls[np.newaxis,1:-1],1)
    return active, frate


  def get_correlated_trials(self,result,smooth=None):

    ## check reliability
    corr = corr0(gauss_smooth(result['firingstats']['trial_map'],smooth=(0,smooth*self.para['nbin']/self.para['L_track'])))

    # corr = np.corrcoef(gauss_smooth(result['firingstats']['trial_map'],smooth=(0,smooth*self.para['nbin']/self.para['L_track'])))
    # corr = sstats.spearmanr(gauss_smooth(result['firingstats']['trial_map'],smooth=(0,smooth*self.para['nbin']/self.para['L_track'])),axis=1)[0]

    #result['firingstats']['trial_map'] = gauss_smooth(result['firingstats']['trial_map'],(0,2))
    corr[np.isnan(corr)] = 0
    ordered_corr,res_order,res_linkage = compute_serial_matrix(-(corr-1),'average')
    cluster_idx = sp.cluster.hierarchy.cut_tree(res_linkage,height=0.5)
    _, c_counts = np.unique(cluster_idx,return_counts=True)
    c_trial = np.where((c_counts>self.para['trials_min_count']) & (c_counts>(self.para['trials_min_fraction']*self.dataBH['trials']['ct'])))[0]

    for (i,t) in enumerate(c_trial):
        fmap = gauss_smooth(np.nanmean(result['firingstats']['trial_map'][cluster_idx.T[0]==t,:],0),2)
        # baseline = np.percentile(fmap[fmap>0],20)
        baseline = np.nanpercentile(fmap[fmap>0],30)
        fmap2 = np.copy(fmap)
        fmap2 -= baseline
        fmap2 *= -1*(fmap2 <= 0)

        Ns_baseline = (fmap2>0).sum()
        noise = np.sqrt((fmap2**2).sum()/(Ns_baseline*(1-2/np.pi)))
        if (fmap>(baseline+4*noise)).sum()>5:
            result['firingstats']['trial_field'][i,:] = (cluster_idx.T==t)

    testing = False
    if testing and self.para['plt_bool']:
      plt.figure()
      plt.subplot(121)
      plt.pcolormesh(corr[res_order,:][:,res_order],cmap='jet')
      plt.clim([0,1])
      plt.colorbar()
      plt.subplot(122)
      corr = sstats.spearmanr(gauss_smooth(result['firingstats']['trial_map'],smooth=(0,smooth*self.para['nbin']/self.para['L_track'])),axis=1)[0]
      # print(corr)
      ordered_corr,res_order,res_linkage = compute_serial_matrix(-(corr-1),'average')
      # Z = sp.cluster.hierarchy.linkage(-(corr-1),method='average')
      # print(Z)
      plt.pcolormesh(corr[res_order,:][:,res_order],cmap='jet')
      plt.clim([0,1])
      plt.colorbar()
      plt.show(block=False)
      plt.figure()
      color_t = plt.cm.rainbow(np.linspace(0,1,self.dataBH['trials']['ct']))
      for i,r in enumerate(res_order):
        if i<25:
            col = color_t[int(res_linkage[i,3]-2)]
            plt.subplot(5,5,i+1)
            plt.plot(np.linspace(0,self.para['L_track'],self.para['nbin']),gauss_smooth(result['firingstats']['trial_map'][r,:],smooth*self.para['nbin']/self.para['L_track']),color=col)
            plt.ylim([0,20])
            plt.title('trial # %d'%r)
      plt.show(block=False)
    return result


  def run_nestedSampling(self,firingstats,firingmap,f):

    hbm = HierarchicalBayesModel(firingmap,self.para['bin_array'],firingstats['parNoise'],f)

    ### test models with 0 vs 1 fields
    paramnames = [self.para['names'][0]]
    for ff in range(f):
      paramnames.extend(self.para['names'][1:]*f)

    ## hand over functions for sampler
    my_prior_transform = hbm.transform_p
    my_likelihood = hbm.set_logl_func()

    sampler = ultranest.ReactiveNestedSampler(paramnames, my_likelihood, my_prior_transform,wrapped_params=hbm.pTC['wrap'],vectorized=True,num_bootstraps=20)#,log_dir='/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Programs/PC_analysis/test_ultra')   ## set up sampler...
    num_samples = 400
    if f>1:
      sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=3)#, adaptive_nsteps='move-distance')
      num_samples = 200

    sampling_result = sampler.run(min_num_live_points=num_samples,max_iters=10000,cluster_num_live_points=20,max_num_improvement_loops=3,show_status=False,viz_callback=False)  ## ... and run it #max_ncalls=500000,(f+1)*100,
    #t_end = time.time()
    #print('nested sampler done, time: %5.3g'%(t_end-t_start))

    Z = [sampling_result['logz'],sampling_result['logzerr']]    ## store evidences
    field = {'Bayes_factor':np.zeros(2)*np.NaN}
    if f > 0:

      fields_tmp = self.detect_modes_from_posterior(sampler)

      if len(fields_tmp)>0:

        for key in fields_tmp.keys():
          field[key] = fields_tmp[key]
        field['Bayes_factor'][0] = Z[0]-self.tmp['Z'][0]
        field['Bayes_factor'][1] = np.sqrt(Z[1]**2 + self.tmp['Z'][1]**2)
      else:
        field['Bayes_factor'] = np.zeros(2)*np.NaN
    self.tmp['Z'] = Z
      #if f==2:
        ##try:
        #if np.any(~np.isnan(fields_tmp['posterior_mass'])):

          #if np.any(~np.isnan(result['fields']['posterior_mass'])):
            #f_major = np.nanargmax(result['fields']['posterior_mass'])
            #theta_major = result['fields']['parameter'][f_major,3,0]
            #dTheta = np.abs(np.mod(theta_major-fields_tmp['parameter'][:,3,0]+self.para['nbin']/2,self.para['nbin'])-self.para['nbin']/2)
            #if result['fields']['Bayes_factor'][f-1,0] > 0:
              #for key in fields_tmp.keys():
                #result['fields'][key] = fields_tmp[key]
              #result['fields']['major'] = np.nanargmin(dTheta)
          #else:
            #if result['fields']['Bayes_factor'][f-1,0] > 0:
              #for key in fields_tmp.keys():
                #result['fields'][key] = fields_tmp[key]
              #result['fields']['major'] = np.NaN

          #print('peaks to compare:')
          #print(result['fields']['parameter'][:,3,0])
          #print(fields_tmp['parameter'][:,3,0])
          #print(dTheta)
          #if np.any(dTheta<(self.para['nbin']/10)):

          #else:
            #print('peak detection for 2-field model was not in line with 1-field model')
            #print(result['fields']['parameter'][:,3,0])
            #print(fields_tmp['parameter'][:,3,0])
            #result['status']['Bayes_factor'][-1,:] = np.NaN
          #return result, sampler
        #except:
          #pass
        #return result, sampler
      #else:

      #if result['status']['Bayes_factor'][f-1,0]<=0:
        #break_it = True

    return field


  def detect_modes_from_posterior(self,sampler,plt_bool=False):
    ### handover of sampled points
    data_tmp = ultranest.netiter.logz_sequence(sampler.root,sampler.pointpile)[0]
    logp_prior = np.log(-0.5*(np.diff(np.exp(data_tmp['logvol'][1:]))+np.diff(np.exp(data_tmp['logvol'][:-1])))) ## calculate prior probabilities (phasespace-slice volume from change in prior-volume (trapezoidal form)

    data = {}
    data['logX'] = np.array(data_tmp['logvol'][1:-1])
    data['logl'] = np.array(data_tmp['logl'][1:-1])
    data['logz'] = np.array(data_tmp['logz'][1:-1])
    data['logp_posterior'] = logp_prior + data['logl'] - data['logz'][-1]   ## normalized posterior weight
    data['samples'] = data_tmp['samples'][1:-1,:]

    if False:#self.para['plt_bool']:
      plt.figure(figsize=(2.5,1.5),dpi=300)
      ## plot weight
      ax1 = plt.subplot(111)
      dZ = np.diff(np.exp(data['logz']))
      ax1.fill_between(data['logX'][1:],dZ/dZ.max(),color=[0.5,0.5,0.5],zorder=0,label='$\Delta Z$')

      w = np.exp(logp_prior)
      ax1.plot(data['logX'],w/w.max(),'r',zorder=5,label='$w$')

      L = np.exp(data['logl'])
      ax1.plot(data['logX'],L/L.max(),'k',zorder=10,label='$\mathcal{L}$')

      ax1.set_yticks([])
      ax1.set_xlabel('ln X')
      ax1.legend(fontsize=8,loc='lower left')
      plt.tight_layout()
      plt.show(block=False)

      if self.para['plt_sv']:
        pathSv = pathcat([self.para['pathFigs'],'PC_analysis_NS_contributions.png'])
        plt.savefig(pathSv)
        print('Figure saved @ %s'%pathSv)

      print('add colorbar to other plot')

    nPars = data_tmp['samples'].shape[-1]
    nf = int((nPars - 1)/3)

    testing = False
    bins = 2*self.para['nbin']
    offset = self.para['nbin']

    fields = {}
    for f in range(nf):

      #fields[f] = {}
      #fields[f]['nModes'] = 0
      #fields[f]['posterior_mass'] = np.zeros(3)*np.NaN
      #fields[f]['parameter'] = np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN
      #fields[f]['p_x'] = np.zeros((3,self.para['nbin']))*np.NaN

      data['pos_samples'] = np.array(data['samples'][:,3+3*f])
      logp = np.exp(data['logp_posterior'])   ## even though its not logp, but p!!

      ### search for baseline (where whole prior space is sampled)
      x_space = np.linspace(0,self.para['nbin'],11)
      logX_top = -(data['logX'].min())
      logX_bottom = -(data['logX'].max())
      for i in range(10):
        logX_base = (logX_top + logX_bottom)/2
        mask_logX = -data['logX']>logX_base
        cluster_hist = np.histogram(data['pos_samples'][mask_logX],bins=x_space)[0]>0
        if np.mean(cluster_hist) > 0.9:
          logX_bottom = logX_base
        else:
          logX_top = logX_base
        i+=1

      post,post_bin = np.histogram(data['pos_samples'],bins=np.linspace(0,self.para['nbin'],bins+1),weights=logp*(np.random.rand(len(logp))<(logp/logp.max())))
      post /= post.sum()

      # construct wrapped and smoothed histogram
      post_cat = np.concatenate([post[-offset:],post,post[:offset]])
      post_smooth = sp.ndimage.gaussian_filter(post,2,mode='wrap')
      post_smooth = np.concatenate([post_smooth[-offset:],post_smooth,post_smooth[:offset]])

      ## find peaks and troughs
      mode_pos, prop = sp.signal.find_peaks(post_smooth,distance=self.para['nbin']/5,height=post_smooth.max()/3)
      mode_pos = mode_pos[(mode_pos>offset) & (mode_pos<(bins+offset))]
      trough_pos, prop = sp.signal.find_peaks(-post_smooth,distance=self.para['nbin']/5)


      if testing and self.para['plt_bool']:
        plt.figure()
        plt.subplot(211)
        #bin_arr = np.linspace(-25,125,bins+2*offset)
        bin_arr = np.linspace(0,bins+2*offset,bins+2*offset)
        plt.bar(bin_arr,post_smooth)
        plt.plot(bin_arr[mode_pos],post_smooth[mode_pos],'ro')
        plt.subplot(212)
        plt.bar(post_bin[:-1],post,width=0.5,facecolor='b',alpha=0.5)
        plt.plot(post_bin[np.mod(mode_pos-offset,bins)],post[np.mod(mode_pos-offset,bins)],'ro')
        plt.plot(post_bin[np.mod(trough_pos-offset,bins)],post[np.mod(trough_pos-offset,bins)],'bo')
        plt.show(block=False)

      modes = {}
      #c_ct = 0
      p_mass = np.zeros(len(mode_pos))
      for (i,p) in enumerate(mode_pos):
        try:
          ## find neighbouring troughs
          dp = trough_pos-p
          t_right = p+dp[dp>0].min()
          t_left = p+dp[dp<0].max()
        except:
          try:
            t_left = np.where(post_smooth[:p]<(post_smooth[p]*0.01))[0][-1]
          except:
            t_left = 0
          try:
            t_right = p+np.where(post_smooth[p:]<(post_smooth[p]*0.01))[0][0]
          except:
            nbin+2*offset

        p_mass[i] = post_cat[t_left:t_right].sum()    # obtain probability mass between troughs
        if p_mass[i] > 0.05:
          modes[i] = {}
          modes[i]['p_mass'] = p_mass[i]
          modes[i]['peak'] = post_bin[p-offset]
          modes[i]['left'] = post_bin[np.mod(t_left-offset,bins)]
          modes[i]['right'] = post_bin[np.mod(t_right-offset,bins)]
          #c_ct += 1

        if testing and self.para['plt_bool']:
          print('peak @x=%.1f'%post_bin[p-offset])
          print('\ttroughs: [%.1f, %.1f]'%(post_bin[np.mod(t_left-offset,bins)],post_bin[np.mod(t_right-offset,bins)]))
          print('\tposterior mass: %5.3g'%p_mass[i])

      nsamples = len(logp)
      #if testing and self.para['plt_bool']:
        #plt.figure()
        #plt.subplot(311)
        #plt.scatter(data['pos_samples'],-data['logX'],c=np.exp(data['logp_posterior']),marker='.',label='samples')
        #plt.plot([0,self.para['nbin']],[logX_base,logX_base],'k--')
        #plt.xlabel('field position $\\theta$')
        #plt.ylabel('-ln(X)')
        #plt.legend(loc='lower right')
        ###plt.show(block=False)
      if len(p_mass)<1:
        return {}

      if np.max(p_mass)>0.05:
        p = np.argmax(p_mass)
        m = modes[p]
        #for (p,m) in enumerate(modes.values()):
        if m['p_mass'] > 0.3 and ((m['p_mass']<p_mass).sum()<3):

          field = self.define_field(data,logX_base,modes,p,f)
          field['posterior_mass'] = m['p_mass']
        else:
          field = {}
      else:
        field = {}
      #plt.show(block=False)

          #print('val: %5.3g, \t (%5.3g,%5.3g)'%(val[c,i],CI[c,i,0],CI[c,i,1]))
      #print('time took (post-process posterior): %5.3g'%(time.time()-t_start))
      #print(fields[f]['parameter'])
      if False:#self.para['plt_bool'] or plt_bool:
        #plt.figure()
        #### plot nsamples
        #### plot likelihood
        #plt.subplot(313)
        #plt.plot(-data['logX'],np.exp(data['logl']))
        #plt.ylabel('likelihood')
        #### plot importance weight
        #plt.subplot(312)
        #plt.plot(-data['logX'],np.exp(data['logp_posterior']))
        #plt.ylabel('posterior weight')
        #### plot evidence
        #plt.subplot(311)
        #plt.plot(-data['logX'],np.exp(data['logz']))
        #plt.ylabel('evidence')
        #plt.show(block=False)

        col_arr = ['tab:blue','tab:orange','tab:green']

        fig = plt.figure(figsize=(7,4),dpi=300)
        ax_NS = plt.axes([0.1,0.11,0.2,0.85])
        #ax_prob = plt.subplot(position=[0.6,0.675,0.35,0.275])
        #ax_center = plt.subplot(position=[0.6,0.375,0.35,0.275])
        ax_phase_1 = plt.axes([0.4,0.11,0.125,0.2])
        ax_phase_2 = plt.axes([0.55,0.11,0.125,0.2])
        ax_phase_3 = plt.axes([0.4,0.335,0.125,0.2])
        ax_hist_1 = plt.axes([0.7,0.11,0.1,0.2])
        ax_hist_2 = plt.axes([0.55,0.335,0.125,0.15])
        ax_hist_3 = plt.axes([0.4,0.56,0.125,0.15])


        ax_NS.scatter(data['pos_samples'],-data['logX'],c=np.exp(data['logp_posterior']),marker='.',label='samples')
        ax_NS.plot([0,self.para['nbin']],[logX_base,logX_base],'k--')
        ax_NS.set_xlabel('field position $\\theta$')
        ax_NS.set_ylabel('-ln(X)')
        ax_NS.legend(loc='lower right')
        for c in range(fields[f]['nModes']):
          #if fields[f]['posterior_mass'][c] > 0.05:
          #ax_center.plot(logX_arr,blob_center[:,c],color=col_arr[c])
          #ax_center.fill_between(logX_arr,blob_center_CI[:,0,c],blob_center_CI[:,1,c],facecolor=col_arr[c],alpha=0.5)

          ax_phase_1.plot(data['samples'][clusters[c_arr[c]]['mask'],2+3*f],data['samples'][clusters[c_arr[c]]['mask'],3+3*f],'k.',markeredgewidth=0,markersize=1)
          ax_phase_2.plot(data['samples'][clusters[c_arr[c]]['mask'],1+3*f],data['samples'][clusters[c_arr[c]]['mask'],3+3*f],'k.',markeredgewidth=0,markersize=1)
          ax_phase_3.plot(data['samples'][clusters[c_arr[c]]['mask'],2+3*f],data['samples'][clusters[c_arr[c]]['mask'],1+3*f],'k.',markeredgewidth=0,markersize=1)

          ax_hist_1.hist(data['samples'][clusters[c_arr[c]]['mask'],3+3*f],np.linspace(0,self.para['nbin'],50),facecolor='k',orientation='horizontal')
          ax_hist_2.hist(data['samples'][clusters[c_arr[c]]['mask'],1+3*f],np.linspace(0,10,20),facecolor='k')
          ax_hist_3.hist(data['samples'][clusters[c_arr[c]]['mask'],2+3*f],np.linspace(0,5,20),facecolor='k')

          #ax_phase.plot(logX_arr,blob_phase_space[:,c],color=col_arr[c],label='mode #%d'%(c+1))
          #ax_prob.plot(logX_arr,blob_probability_mass[:,c],color=col_arr[c])

          #if c < 3:
            #ax_NS.annotate('',(fields[f]['parameter'][c,3,0],logX_top),xycoords='data',xytext=(fields[f]['parameter'][c,3,0]+5,logX_top+2),arrowprops=dict(facecolor=ax_center.lines[-1].get_color(),shrink=0.05))

        nsteps = 5
        logX_arr = np.linspace(logX_top,logX_base,nsteps)
        for (logX,i) in zip(logX_arr,range(nsteps)):
          ax_NS.plot([0,self.para['nbin']],[logX,logX],'--',color=[1,i/(2*nsteps),i/(2*nsteps)],linewidth=0.5)

        #ax_center.set_xticks([])
        #ax_center.set_xlim([logX_base,logX_top])
        #ax_prob.set_xlim([logX_base,logX_top])
        #ax_center.set_ylim([0,self.para['nbin']])
        #ax_center.set_ylabel('$\\theta$')
        #ax_prob.set_ylim([0,1])
        #ax_prob.set_xlabel('-ln(X)')
        #ax_prob.set_ylabel('posterior')

        ax_phase_1.set_xlabel('$\\sigma$')
        ax_phase_1.set_ylabel('$\\theta$')
        ax_phase_2.set_xlabel('$A$')
        ax_phase_3.set_ylabel('$A$')
        ax_phase_2.set_yticks([])
        ax_phase_3.set_xticks([])

        ax_hist_1.set_xticks([])
        ax_hist_2.set_xticks([])
        ax_hist_3.set_xticks([])
        ax_hist_1.set_yticks([])
        ax_hist_2.set_yticks([])
        ax_hist_3.set_yticks([])

        ax_hist_1.spines['top'].set_visible(False)
        ax_hist_1.spines['right'].set_visible(False)
        ax_hist_1.spines['bottom'].set_visible(False)

        ax_hist_2.spines['top'].set_visible(False)
        ax_hist_2.spines['right'].set_visible(False)
        ax_hist_2.spines['left'].set_visible(False)

        ax_hist_3.spines['top'].set_visible(False)
        ax_hist_3.spines['right'].set_visible(False)
        ax_hist_3.spines['left'].set_visible(False)

        #ax_phase_1.set_xticks([])
        #ax_phase.set_xlim([logX_base,logX_top])
        #ax_phase.set_ylim([0,1])
        #ax_phase.set_ylabel('% phase space')
        #ax_phase.legend(loc='upper right')

        if self.para['plt_sv']:
            pathSv = pathcat([self.para['pathFigs'],'PC_analysis_NS_results.png'])
            plt.savefig(pathSv)
            print('Figure saved @ %s'%pathSv)
        plt.show(block=False)


    #if nf > 1:

      #if testing and self.para['plt_bool']:
        #print('detected from nested sampling:')
        #print(fields[0]['parameter'][:,3,0])
        #print(fields[0]['posterior_mass'])
        #print(fields[1]['parameter'][:,3,0])
        #print(fields[1]['posterior_mass'])

      #fields_return = {}
      #fields_return['nModes'] = 0
      #fields_return['posterior_mass'] = np.zeros(3)*np.NaN
      #fields_return['parameter'] = np.zeros((3,4,1+len(self.para['CI_arr'])))*np.NaN
      #fields_return['p_x'] = np.zeros((3,self.para['nbin']))*np.NaN

      #for f in range(fields[0]['nModes']):
        #p_cluster = fields[0]['posterior_mass'][f]
        #dTheta = np.abs(np.mod(fields[0]['parameter'][f,3,0]-fields[1]['parameter'][:,3,0]+self.para['L_track']/2,self.para['L_track'])-self.para['L_track']/2)
        #if np.any(dTheta<5):  ## take field with larger probability mass to have better sampling
          #f2 = np.nanargmin(dTheta)
          #if fields[0]['posterior_mass'][f] > fields[1]['posterior_mass'][f2]:
            #handover_f = 0
            #f2 = f
          #else:
            #handover_f = 1
            #p_cluster = fields[1]['posterior_mass'][f2]
        #else:
          #handover_f = 0
          #f2 = f

        #if p_cluster>0.3:
          #fields_return['parameter'][fields_return['nModes'],...] = fields[handover_f]['parameter'][f2,...]
          #fields_return['p_x'][fields_return['nModes'],...] = fields[handover_f]['p_x'][f2,...]
          #fields_return['posterior_mass'][fields_return['nModes']] = fields[handover_f]['posterior_mass'][f2]
          #fields_return['nModes'] += 1
          #if fields_return['nModes']>=3:
            #break

      #for f in range(fields[1]['nModes']):
        #if fields_return['nModes']>=3:
          #break

        #dTheta = np.abs(np.mod(fields[1]['parameter'][f,3,0]-fields[0]['parameter'][:,3,0]+self.para['L_track']/2,self.para['L_track'])-self.para['L_track']/2)

        #dTheta2 = np.abs(np.mod(fields[1]['parameter'][f,3,0]-fields_return['parameter'][:,3,0]+self.para['L_track']/2,self.para['L_track'])-self.para['L_track']/2)


        #if (not np.any(dTheta<5)) and (not np.any(dTheta2<5)) and (fields[1]['posterior_mass'][f]>0.3):  ## take field with larger probability mass to have better sampling
          #fields_return['parameter'][fields_return['nModes'],...] = fields[1]['parameter'][f,...]
          #fields_return['p_x'][fields_return['nModes'],...] = fields[1]['p_x'][f,...]
          #fields_return['posterior_mass'][fields_return['nModes']] = fields[1]['posterior_mass'][f]
          #fields_return['nModes'] += 1
      #if testing and self.para['plt_bool']:
        #print(fields_return['parameter'][:,3,0])
        #print(fields_return['posterior_mass'])
    #else:
    return field


  def define_field(self,data,logX_base,modes,p,f):

    field = {}
    field['posterior_mass'] = np.NaN
    field['parameter'] = np.zeros((4,1+len(self.para['CI_arr'])))*np.NaN
    field['p_x'] = np.zeros(self.para['nbin'])*np.NaN
    #fields[f]['posterior_mass'][fields[f]['nModes']]

    logp = np.exp(data['logp_posterior'])
    nsamples = len(logp)
    samples = data['samples']
    mask_mode = np.ones(nsamples,'bool')

    ## calculate further statistics
    for (p2,m2) in enumerate(modes.values()):
      if not (p==p2):
        # obtain samples first
        if m2['left']<m2['right']:
          mask_mode[(samples[:,3]>m2['left']) & (samples[:,3]<m2['right']) & (-data['logX']>logX_base)] = False
        else:
          mask_mode[((samples[:,3]>m2['left']) | (samples[:,3]<m2['right'])) & (-data['logX']>logX_base)] = False

    mode_logp = logp[mask_mode]#/posterior_mass
    mode_logp /= mode_logp.sum()#logp.sum()#

    #if testing and self.para['plt_bool']:
      ##plt.figure()
      #plt.subplot(312)
      #plt.scatter(data['pos_samples'][mask_mode],-data['logX'][mask_mode],c=np.exp(data['logp_posterior'][mask_mode]),marker='.',label='samples')
      #plt.plot([0,self.para['nbin']],[logX_base,logX_base],'k--')
      #plt.xlabel('field position $\\theta$')
      #plt.ylabel('-ln(X)')
      #plt.legend(loc='lower right')

    ## obtain parameters
    field['parameter'][0,0] = get_average(samples[mask_mode,0],mode_logp)
    field['parameter'][1,0] = get_average(samples[mask_mode,1+3*f],mode_logp)
    field['parameter'][2,0] = get_average(samples[mask_mode,2+3*f],mode_logp)
    field['parameter'][3,0] = get_average(samples[mask_mode,3+3*f],mode_logp,True,[0,self.para['nbin']])
    #print(field['parameter'][2,0])
    for i in range(4):
      ### get confidence intervals from cdf
      if i==0:
        samples_tmp = samples[mask_mode,0]
      elif i==3:
        samples_tmp = (samples[mask_mode,3+3*f]+self.para['nbin']/2-field['parameter'][3,0])%self.para['nbin']-self.para['nbin']/2        ## shift whole axis such, that peak is in the center, to get proper errorbars
      else:
        samples_tmp = samples[mask_mode,i+3*f]

      x_cdf_posterior, y_cdf_posterior = ecdf(samples_tmp,mode_logp)
      for j in range(len(self.para['CI_arr'])):
        field['parameter'][i,1+j] = x_cdf_posterior[np.where(y_cdf_posterior>=self.para['CI_arr'][j])[0][0]]

    field['p_x'],_ = np.histogram(samples[mask_mode,3],bins=np.linspace(0,self.para['nbin'],self.para['nbin']+1),weights=mode_logp*(np.random.rand(len(mode_logp))<(mode_logp/mode_logp.max())),density=True)
    field['p_x'][field['p_x']<(0.001*field['p_x'].max())] = 0

    field['parameter'][3,0] = field['parameter'][3,0] % self.para['nbin']
    field['parameter'][3,1:] = (field['parameter'][3,0] + field['parameter'][3,1:]) % self.para['nbin']

    ## rescaling to length 100
    field['parameter'][2,:] *= self.para['L_track']/self.para['nbin']
    field['parameter'][3,:] *= self.para['L_track']/self.para['nbin']

    return field


  def get_trials_activity(self,active):

    ## preallocate
    trials_map = np.zeros((self.dataBH['trials']['ct'],self.para['nbin']))

    #frate = gauss_smooth(active['S']*self.para['f'],self.para['sigma'])

    trials_S = {}
    for t in range(self.dataBH['trials']['ct']):
      trials_S[t] = {}
      trials_S[t]['S'] = active['S'][self.dataBH['trials']['frame'][t]:self.dataBH['trials']['frame'][t+1]]#gauss_smooth(active['S'][self.dataBH['trials']['frame'][t]:self.dataBH['trials']['frame'][t+1]]*self.para['f'],self.para['f']);    ## should be quartiles?!
      if self.para['modes']['info'] == 'MI':
        trials_S[t]['qtl'] = active['qtl'][self.dataBH['trials']['frame'][t]:self.dataBH['trials']['frame'][t+1]];    ## should be quartiles?!

      if self.para['modes']['activity'] == 'spikes':
        trials_S[t]['spike_times'] = np.where(trials_S[t]['S']);
        trials_S[t]['spikes'] = trials_S[t]['S'][trials_S[t]['spike_times']];
        trials_S[t]['ISI'] = np.diff(trials_S[t]['spike_times']);

      trials_S[t]['rate'] = trials_S[t]['S'].sum()/(self.dataBH['trials']['T'][t]/self.para['f']);

      if trials_S[t]['rate'] > 0:
        trials_map[t,:] = self.get_firingmap(trials_S[t]['S'],self.dataBH['trials']['trial'][t]['binpos_active'],self.dataBH['trials']['dwelltime'][t,:])#/trials_S[t]['rate']

      #[spikeNr,md,sd_r] = get_spikeNr(trials_S[t]['S'][trials_S[t]['S']>0]);
      #trials_rate[t] = spikeNr/(self.dataBH['trials']['T'][t]/self.para['f']);
    #trials_map /= np.nansum(trials_map,1)[:,np.newaxis]

    return trials_S, trials_map#, trials_rate


  def get_firingstats_from_trials(self,trials_firingmap,trials=None,complete=True):

    ### construct firing rate map from bootstrapping over (normalized) trial firing maps
    if trials is None:
      trials = np.arange(self.dataBH['trials']['ct'])

    #trials_firingmap = trials_firingmap[trials,:]
    dwelltime = self.dataBH['trials']['dwelltime'][trials,:]
    firingstats = {}
    firingmap_bs = np.zeros((self.para['nbin'],self.para['N_bs']))

    base_sample = np.random.randint(0,len(trials),(self.para['N_bs'],len(trials)))

    for L in range(self.para['N_bs']):
      #dwelltime = self.dataBH['trials']['dwelltime'][base_sample[L,:],:].sum(0)
      firingmap_bs[:,L] = np.nanmean(trials_firingmap[trials[base_sample[L,:]],:],0)#/dwelltime
      #mask = (dwelltime==0)
      #firingmap_bs[mask,L] = 0

      #firingmap_bs[:,L] = np.nanmean(trials_firingmap[base_sample[L,:],:]/ self.dataBH['trials']['dwelltime'][base_sample[L,:],:],0)
    firingstats['map'] = np.nanmean(firingmap_bs,1)

    if complete:
      ## parameters of gamma distribution can be directly inferred from mean and std
      firingstats['std'] = np.nanstd(firingmap_bs,1)
      firingstats['std'][firingstats['std']==0] = np.nanmean(firingstats['std'])
      prc = [2.5,97.5]
      firingstats['CI'] = np.nanpercentile(firingmap_bs,prc,1);   ## width of gaussian - from 1-SD confidence interval

      ### fit linear dependence of noise on amplitude (with 0 noise at fr=0)
      firingstats['parNoise'] = jackknife(firingstats['map'],firingstats['std'])
      print()
      if self.para['plt_theory_bool'] and self.para['plt_bool']:
        self.plt_model_selection(firingmap_bs,firingstats,trials_firingmap)

    firingstats['map'] = np.maximum(firingstats['map'],1/dwelltime.sum(0))#1/(self.para['nbin'])     ## set 0 firing rates to lowest possible (0 leads to problems in model, as 0 noise, thus likelihood = 0)
    firingstats['map'][dwelltime.sum(0)<0.2] = np.NaN#1/(self.para['nbin']*self.dataBH['T'])
    ### estimate noise of model
    return firingstats


  def get_info_value(self,activity,dwelltime,mode='MI'):

    if mode == 'MI':
      p_joint = self.get_p_joint(activity)   ## need activity trace
      return get_MI(p_joint,dwelltime/dwelltime.sum(),self.para['qtl_weight'])

    elif mode == 'Isec':
      fmap = self.get_firingmap(activity,self.dataBH['binpos_coarse_active'],dwelltime,coarse=True)
      Isec_arr = dwelltime/dwelltime.sum()*(fmap/np.nanmean(fmap))*np.log2(fmap/np.nanmean(fmap))

      #return np.nansum(Isec_arr[-self.para['nbin']//2:])
      return np.nansum(Isec_arr)


  def get_p_joint(self,activity):

    ### need as input:
    ### - activity (quantiled or something)
    ### - behavior trace
    p_joint = np.zeros((self.para['nbin_coarse'],self.para['qtl_steps']))

    for q in range(self.para['qtl_steps']):
      for (x,ct) in Counter(self.dataBH['binpos_coarse_active'][activity==q]).items():
        p_joint[x,q] = ct;
    p_joint = p_joint/p_joint.sum();    ## normalize
    return p_joint


  def calc_Icorr(self,S,trials_S):

    S /= S[S>0].mean()
    lag = [0,self.para['f']*2]
    nlag = lag[1]-lag[0]
    T = S.shape[0]

    print('check if range is properly covered in C_cross (and PSTH)')
    print('speed, speed, speed!! - what can be vectorized? how to get rid of loops?')
    print('spike train generation: generate a single, long one at once (or smth)')
    print('how to properly generate surrogate data? single trials? conjoint trials? if i put in a rate only, sums will even out to homogenous process for N large')

    PSTH = np.zeros((self.para['nbin'],nlag))
    C_cross = np.zeros((self.para['nbin'],nlag))
    for x in range(self.para['nbin']):
      for t in range(self.dataBH['trials']['ct']):
        idx_x = np.where(self.dataBH['trials']['trial'][t]['binpos_active']==x)[0]
        #print(idx_x)
        if len(idx_x):
          i = self.dataBH['trials']['frame'][t] + idx_x[0]    ## find entry to position x in trial t
          #print('first occurence of x=%d in trial %d (start: %d) at frame %d'%(x,t,self.dataBH['trials']['frame'][t],i))
          PSTH[x,:min(nlag,T-(i+lag[0]))] += S[i+lag[0]:min(T,i+lag[1])]
      for i in range(1,nlag):
        C_cross[x,i] = np.corrcoef(PSTH[x,:-i],PSTH[x,i:])[0,1]
      C_cross[np.isnan(C_cross)] = 0
      #C_cross[x,:] = np.fft.fft(C_cross[x,:])
    #PSTH /= nlag/self.para['f']*self.dataBH['trials']['ct']
    fC_cross = np.fft.fft(C_cross)

    rate = PSTH.sum(1)/(nlag*self.dataBH['trials']['ct'])
    #print(rate)

    Icorr = np.zeros(self.para['nbin'])
    Icorr_art = np.zeros(self.para['nbin'])
    Icorr_art_std = np.zeros(self.para['nbin'])

    for x in range(self.para['nbin']):
      print(x)
      Icorr[x] = -1/2*rate[x] * np.log2(1 - fC_cross[x,:]/(rate[x]+fC_cross[x,:])).sum()
      Icorr_art[x], Icorr_art_std[x] = self.calc_Icorr_data(rate[x],nlag)

    plt.figure()
    plt.plot(Icorr)
    plt.errorbar(range(100),Icorr_art,yerr=Icorr_art_std)
    #plt.plot(Icorr_art,'r')
    plt.show(block=False)

    return PSTH, C_cross, Icorr
      #self.dataBH['trials']['frame'][t]

  def calc_Icorr_data(self,rate,T,N_bs=10):

    t = np.linspace(0,T-1,T)
    Icorr = np.zeros(N_bs)

    nGen = int(math.ceil(1.1*T*rate))
    u = np.random.rand(N_bs,nGen)   ## generate random variables to cover the whole time
    t_AP = np.cumsum(-(1/rate)*np.log(u),1) ## generate points of homogeneous pp
    #print(t_AP)
    for L in range(N_bs):
      t_AP_now = t_AP[L,t_AP[L,:]<T];
      idx_AP = np.argmin(np.abs(t_AP_now[:,np.newaxis]-t[np.newaxis,:]),1)

      PSTH = np.zeros(T)
      for AP in idx_AP:
        PSTH[AP] += 1

      #C_cross = np.correlate(PSTH,PSTH)
      #print(C_cross)
      C_cross = np.zeros(T)
      for i in range(1,T):
        C_cross[i] = np.corrcoef(PSTH[:-i],PSTH[i:])[0,1]
      C_cross[np.isnan(C_cross)] = 0
      fC_cross = np.fft.fft(C_cross)
      Icorr[L] = -1/2*rate * np.log2(1 - fC_cross/(rate+fC_cross)).sum()

    return Icorr.mean(), Icorr.std()


  def test_MI(self,active,trials_S):

    shuffle_peaks = False;
    if self.para['modes']['info'] == 'MI':
      S_key = 'qtl'
    else:
      S_key = 'S'

    MI = {'MI_p_value':np.NaN,'MI_value':np.NaN,'MI_z_score':np.NaN,
          'Isec_p_value':np.NaN,'Isec_value':np.NaN,'Isec_z_score':np.NaN}
    MI_rand_distr = np.zeros(self.para['repnum'])*np.NaN
    Isec_rand_distr = np.zeros(self.para['repnum'])*np.NaN

    ### first, get actual MI value
    norm_dwelltime = self.dataBH['trials']['dwelltime'].sum(0)#/self.dataBH['trials']['dwelltime'].sum()#*self.para['f']/self.dataBH['T']

    if self.para['nbin_coarse'] == self.para['nbin']:
      norm_dwelltime_coarse = norm_dwelltime
    else:
      norm_dwelltime_coarse = np.zeros(self.para['nbin_coarse'])
      for i in range(self.para['nbin_coarse']):
        norm_dwelltime_coarse[i] = norm_dwelltime[i*self.para['coarse_factor']:(i+1)*self.para['coarse_factor']].sum()

    frate = gauss_smooth(active['S']*self.para['f'],self.para['sigma'])

    pos_act = self.dataBH['position'][self.dataBH['active']]

    MI['MI_value'] = self.get_info_value(active[S_key],norm_dwelltime_coarse,mode='MI')
    MI['Isec_value'] = self.get_info_value(frate,norm_dwelltime_coarse,mode='Isec')

    ### shuffle according to specified mode
    trial_ct = self.dataBH['trials']['ct']
    for L in range(self.para['repnum']):

      ## shift single trials to destroy characteristic timescale
      if self.para['modes']['shuffle'] == 'shuffle_trials':

        ## trial shuffling
        trials = np.random.permutation(trial_ct)

        shuffled_activity_qtl = np.roll(np.hstack([np.roll(trials_S[t][S_key],int(random.random()*self.dataBH['trials']['T'][t])) for t in trials]),int(random.random()*self.dataBH['T']))

        #shuffled_activity_S = np.roll(np.hstack([np.roll(trials_S[t]['S'],int(random.random()*self.dataBH['trials']['T'][t])) for t in trials]),int(random.random()*self.dataBH['T']))

        #shuffled_activity_S = gauss_smooth(shuffled_activity_S*self.para['f'],self.para['sigma'])

      elif self.para['modes']['shuffle'] == 'shuffle_global':
        if self.para['modes']['activity'] == 'spikes':
          shuffled_activity = shuffling('dithershift',shuffle_peaks,spike_times=spike_times,spikes=spikes,T=self.dataBH['T'],ISI=ISI,w=2*self.para['f'])
        else:
          shuffled_activity = shuffling('shift',shuffle_peaks,spike_train=active[S_key])

      elif self.para['modes']['shuffle'] == 'randomize':
        shuffled_activity = active[S_key][np.random.permutation(len(active[S_key]))];

      #t_start_info = time.time()
      MI_rand_distr[L] = self.get_info_value(shuffled_activity_qtl,norm_dwelltime_coarse,mode='MI')
      #Isec_rand_distr[L] = self.get_info_value(shuffled_activity_S,norm_dwelltime_coarse,mode='Isec')

      #print('info calc: time taken: %5.3g'%(time.time()-t_start_info))
    #print('shuffle: time taken: %5.3g'%(time.time()-t_start_shuffle))

    MI_mean = np.nanmean(MI_rand_distr)
    MI_std = np.nanstd(MI_rand_distr)
    MI['MI_z_score'] = (MI['MI_value'] - MI_mean)/MI_std
    if MI['MI_value'] > MI_rand_distr.max():
      MI['MI_p_value'] = 1e-10#1/self.para['repnum']
    else:
      x,y = ecdf(MI_rand_distr)
      min_idx = np.argmin(abs(x-MI['MI_value']))
      MI['MI_p_value'] = 1 - y[min_idx]


    #Isec_mean = np.nanmean(Isec_rand_distr)
    #Isec_std = np.nanstd(Isec_rand_distr)
    #MI['Isec_z_score'] = (MI['Isec_value'] - Isec_mean)/Isec_std
    #if MI['Isec_value'] > Isec_rand_distr.max():
      #MI['Isec_p_value'] = 1e-10#1/self.para['repnum']
    #else:
      #x,y = ecdf(Isec_rand_distr)
      #min_idx = np.argmin(abs(x-MI['Isec_value']))
      #MI['Isec_p_value'] = 1 - y[min_idx]



    #Isec_mean = np.nanmean(Isec_rand_distr,0)
    #Isec_std = np.nanstd(Isec_rand_distr,0)
    #p_val = np.zeros(self.para['nbin_coarse'])
    #if ~np.any(MI['Isec_value'] > (Isec_mean+Isec_std)):
      #MI['Isec_p_value'][:] = 1#1/self.para['repnum']
    #else:
      #for i in range(self.para['nbin_coarse']):
        #if MI['Isec_value'][i] > Isec_rand_distr[:,i].max():
          #p_val[i] = 1/self.para['repnum']
        #else:
          #x,y = ecdf(Isec_rand_distr[:,i])
          #min_idx = np.argmin(abs(x-MI['Isec_value'][i]))
          #p_val[i] = 1 - y[min_idx]
    #p_val.sort()
    ##print(p_val)
    #MI['Isec_p_value'] = np.exp(np.log(p_val[:5]).mean())
    ##MI['Isec_z_score'] = np.max((MI['Isec_value'] - Isec_mean)/Isec_std)


    #plt.figure()
    #plt.subplot(211)
    #plt.plot(MI['Isec_value'])
    #plt.errorbar(np.arange(self.para['nbin_coarse']),Isec_mean,Isec_std)
    #plt.subplot(212)
    #plt.plot(MI['Isec_p_value'])
    #plt.yscale('log')
    #plt.show(block=False)


    #print('p_value: %7.5g'%MI['MI_p_value'])

    #if pl['bool']:
    #plt.figure()
    #plt.hist(rand_distr)
    #plt.plot(MI['MI_value'],0,'kx')
    #plt.show(block=True)

    return MI



  def build_PC_fields(self):

    result = {}
    result['status'] = {'MI_value':np.NaN,
                        'MI_p_value':np.NaN,
                        'MI_z_score':np.NaN,
                        'Isec_value':np.NaN,
                        'Isec_p_value':np.NaN,
                        'Isec_z_score':np.NaN,
                        #'Z':np.zeros((self.f_max+1,2))*np.NaN,
                        #'Bayes_factor':np.zeros((self.f_max,2))*np.NaN,
                        'SNR':np.NaN,
                        'r_value':np.NaN}

    result['fields'] = {'parameter':np.zeros((5,4,1+len(self.para['CI_arr'])))*np.NaN,          ### (field1,field2) x (A0,A,std,theta) x (mean,CI_low,CI_top)
                        'posterior_mass':np.zeros(5)*np.NaN,
                        'p_x':np.zeros((5,self.para['nbin']))*np.NaN,
                        'reliability':np.zeros(5)*np.NaN,
                        'Bayes_factor':np.zeros((5,2))*np.NaN,
                        'nModes':0}
                        #'major':np.NaN}

    result['firingstats'] = {'rate':np.NaN,
                             'map':np.zeros(self.para['nbin'])*np.NaN,
                             'std':np.zeros(self.para['nbin'])*np.NaN,
                             'CI':np.zeros((2,self.para['nbin']))*np.NaN,
                             'trial_map':np.zeros((self.dataBH['trials']['ct'],self.para['nbin']))*np.NaN,
                             'trial_field':np.zeros((5,self.dataBH['trials']['ct']),'bool'),
                             'parNoise':np.zeros(2)*np.NaN}
    return result


  def build_PC_results(self,nCells):
    results = {}
    results['status'] = {'MI_value':np.zeros(nCells)*np.NaN,
                         'MI_p_value':np.zeros(nCells)*np.NaN,
                         'MI_z_score':np.zeros(nCells)*np.NaN,
                         'Isec_value':np.zeros(nCells)*np.NaN,
                         'Isec_p_value':np.zeros(nCells)*np.NaN,
                         'Isec_z_score':np.zeros(nCells)*np.NaN,
                         #'Z':np.zeros((nCells,self.f_max+1,2))*np.NaN,
                         #'Bayes_factor':np.zeros((nCells,self.f_max,2))*np.NaN,
                         'SNR':np.zeros(nCells)*np.NaN,
                         'r_value':np.zeros(nCells)*np.NaN}

    results['fields'] = {'parameter':np.zeros((nCells,5,4,1+len(self.para['CI_arr'])))*np.NaN,          ### (mean,std,CI_low,CI_top)
                         'p_x':np.zeros((nCells,5,self.para['nbin'])),##sp.sparse.COO((nCells,3,self.para['nbin'])),#
                         'posterior_mass':np.zeros((nCells,5))*np.NaN,
                         'reliability':np.zeros((nCells,5))*np.NaN,
                         'Bayes_factor':np.zeros((nCells,5,2))*np.NaN,
                         'nModes':np.zeros(nCells).astype('int')}
                         #'major':np.zeros(nCells)*np.NaN}

    results['firingstats'] = {'rate':np.zeros(nCells)*np.NaN,
                              'map':np.zeros((nCells,self.para['nbin']))*np.NaN,
                              'std':np.zeros((nCells,self.para['nbin']))*np.NaN,
                              'CI':np.zeros((nCells,2,self.para['nbin']))*np.NaN,
                              'trial_map':np.zeros((nCells,self.dataBH['trials']['ct'],self.para['nbin']))*np.NaN,
                              'trial_field':np.zeros((nCells,5,self.dataBH['trials']['ct']),'bool'),
                              'parNoise':np.zeros((nCells,2))*np.NaN}
    return results


  #def set_para(self,basePath,mouse,s,nP,plt_bool,sv_bool):

    ### set paths:
    #pathMouse = pathcat([basePath,mouse])
    #pathSession = pathcat([pathMouse,'Session%02d'%s])

    #nbin = 100
    #qtl_steps = 5
    #coarse_factor = 4
    #self.para = {'nbin':nbin,'f':15,
                #'bin_array':np.linspace(0,nbin-1,nbin),
                #'bin_array_centers':np.linspace(0,nbin,nbin+1)-0.5,
                #'coarse_factor':coarse_factor,
                #'nbin_coarse':int(nbin/coarse_factor),

                #'nP':nP,
                #'N_bs':10000,'repnum':1000,
                #'qtl_steps':qtl_steps,'sigma':1,
                #'qtl_weight':np.ones(qtl_steps)/qtl_steps,
                #'names':['A_0','A','SD','theta'],
                #'CI_arr':[0.001,0.025,0.05,0.159,0.5,0.841,0.95,0.975,0.999],

                #'plt_bool':plt_bool&(nP==0),
                #'plt_theory_bool':False&(nP==0),
                #'plt_sv':sv_bool&(nP==0),

                #'mouse':mouse,
                #'session':s,
                #'pathSession':pathSession,
                #'pathFigs':'/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Results/pics/Methods',

                #### provide names for figures
                #'svname_status':pathcat([pathSession,'PC_fields_status.mat']),
                #'svname_fields':pathcat([pathSession,'PC_fields_para.mat']),
                #'svname_firingstats':pathcat([pathSession,'PC_fields_firingstats.mat']),

                #### modes, how to perform PC detection
                #'modes':{'activity':'calcium',          ## data provided: 'calcium' or 'spikes'
                          #'info':'MI',                   ## information calculated: 'MI', 'Isec' (/second), 'Ispike' (/spike)
                          #'shuffle':'shuffle_trials'     ## how to shuffle: 'shuffle_trials', 'shuffle_global', 'randomize'
                        #}
                #}

  def get_firingmap(self,S,binpos,dwelltime=None,coarse=False):

    ### calculates the firing map
    spike_times = np.where(S)
    spikes = S[spike_times]
    binpos = binpos[spike_times]#.astype('int')

    firingmap = np.zeros(int(self.para['nbin']/self.para['coarse_factor'])) if coarse else np.zeros(self.para['nbin'])
    for (p,s) in zip(binpos,spikes):#range(len(binpos)):
      firingmap[p] = firingmap[p]+s

    if not (dwelltime is None):
      firingmap = firingmap/dwelltime
      firingmap[dwelltime==0] = np.NaN

    return firingmap

  def plt_results(self,result,t):

    #print('for display: draw tuning curves from posterior distribution and evaluate TC-value for each bin. then, each bin has distribution of values and can be plotted! =)')
    style_arr = ['--','-']
    #col_arr = []
    #fig,ax = plt.subplots(figsize=(5,3),dpi=150)

    hbm = HierarchicalBayesModel(result['firingstats']['map'],self.para['bin_array'],result['firingstats']['parNoise'],0)

    plt.figure()
    ax = plt.axes([0.6,0.625,0.35,0.25])
    ax.bar(self.para['bin_array'],result['firingstats']['map'],facecolor='b',width=1,alpha=0.2)
    ax.errorbar(self.para['bin_array'],result['firingstats']['map'],result['firingstats']['CI'],ecolor='r',linestyle='',fmt='',elinewidth=0.3)#,label='$95\\%$ confidence')

    ax.plot(self.para['bin_array'],hbm.TC(np.array([result['fields']['parameter'][t,0,0]])),'k',linestyle='--',linewidth=1)#,label='$log(Z)=%4.1f\\pm%4.1f$ (non-coding)'%(result['status']['Z'][0,0],result['status']['Z'][0,1]))

    #try:
    #print(result['fields']['nModes'])
    #for c in range(min(2,result['fields']['nModes'])):
    #if result['fields']['nModes']>1:
      #if c==0:
      #label_str = '(mode #%d)\t$log(Z)=%4.1f\\pm%4.1f$'%(c+1,result['status']['Z'][1,0],result['status']['Z'][1,1])
      #else:
    label_str = '(mode #%d)'%(t+1)
    #else:
      #label_str = '$log(Z)=%4.1f\\pm%4.1f$ (coding)'%(result['status']['Z'][1,0],result['status']['Z'][1,1])

    para = result['fields']['parameter'][t,...]

    para[2,:] *= self.para['nbin']/self.para['L_track']
    para[3,:] *= self.para['nbin']/self.para['L_track']

    ax.plot(self.para['bin_array'],hbm.TC(para[:,0]),'r',linestyle='-',linewidth=0.5+result['fields']['posterior_mass'][t]*2,label=label_str)
      #except:
        #1
    #ax.plot(self.para['bin_array'],hbm.TC(par_results[1]['mean']),'r',label='$log(Z)=%5.3g\\pm%5.3g$'%(par_results[1]['Z'][0],par_results[1]['Z'][1]))
    ax.legend(title='evidence',fontsize=8,loc='upper left',bbox_to_anchor=[0.05,1.4])
    ax.set_xlabel('Location [bin]')
    ax.set_ylabel('$\\bar{\\nu}$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if self.para['plt_sv']:
      pathSv = pathcat([self.para['pathFigs'],'PC_analysis_fit_results.png'])
      plt.savefig(pathSv)
      print('Figure saved @ %s'%pathSv)
    plt.show(block=False)


  def plt_data(self,n,S=None,results=None,ground_truth=None,activity_mode='calcium',sv=False,suffix=''):

    rc('font',size=10)
    rc('axes',labelsize=12)
    rc('xtick',labelsize=8)
    rc('ytick',labelsize=8)

    highlight_trial = False
    if (S is None) and ~hasattr(self,'S'):
        pathDat = os.path.join(self.para['pathSession'],'results_redetect.mat')
        ld = loadmat(pathDat,variable_names=['S','C'])
        self.S = ld['S']
        self.C = ld['C']
        S_raw = self.S[n,:]
        C = self.C[n,:]
    else:
        S_raw = np.squeeze(S[n,:].toarray())
        C = np.squeeze(S[n,:].toarray())
    # print(S_raw.shape)

    rate,S_thr,_ = get_firingrate(S_raw[self.dataBH['active']],f=self.para['f'],sd_r=self.para['Ca_thr'])
    if activity_mode == 'spikes':
        S = np.floor(S_raw/S_thr).astype('float')
    else:
        S = S_raw

    # if ~hasattr(self,'results'):
    self.results = {}
    if results is None:
        self.results['status'] = loadmat(os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_status'])[0]+'.pkl'))
        self.results['fields'] = loadmat(os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_fields'])[0]+'.pkl'))
        self.results['firingstats'] = loadmat(os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_firingstats'])[0]+'.pkl'))
    else:
        self.results['status'] = results['status']
        self.results['fields'] = results['fields']
        self.results['firingstats'] = results['firingstats']

    t_start = 0#200
    t_end = 600#470
    n_trial = 12

    fig = plt.figure(figsize=(7,3),dpi=150)
    if ground_truth is None:
        ax_Ca = plt.axes([0.1,0.7,0.6,0.25])
        ax_dwell = plt.axes([0.7,0.9,0.2,0.075])
        ax_loc = plt.axes([0.1,0.1,0.6,0.6])
        ax1 = plt.axes([0.7,0.1,0.25,0.5])
    else:
        ax_Ca = plt.axes([0.125,0.75,0.7,0.225])
        ax_trial_act = plt.axes([0.125,0.65,0.7,0.1])
        ax_loc = plt.axes([0.125,0.15,0.7,0.5])
        ax1 = plt.axes([0.825,0.15,0.15,0.5])
    #ax2 = plt.axes([0.6,0.275,0.35,0.175])
    #ax3 = plt.axes([0.1,0.08,0.4,0.25])
    #ax4 = plt.axes([0.6,0.08,0.35,0.175])

    idx_longrun = self.dataBH['active']
    t_longrun = self.dataBH['time'][idx_longrun]
    t_stop = self.dataBH['time'][~idx_longrun]
    ax_Ca.bar(t_stop,np.ones(len(t_stop))*1.2*S.max(),color=[0.9,0.9,0.9],zorder=0)

    ax_Ca.plot(self.dataBH['time'],C,'k',linewidth=0.2)
    ax_Ca.plot(self.dataBH['time'],S_raw,'r',linewidth=0.3)
    ax_Ca.plot([0,self.dataBH['time'][-1]],[S_thr,S_thr])
    ax_Ca.set_ylim([0,1.2*S_raw.max()])
    ax_Ca.set_xlim([t_start,t_end])
    ax_Ca.set_xticks([])
    ax_Ca.set_ylabel('Ca$^{2+}$')
    ax_Ca.set_yticks([])

    if not (ground_truth is None):
        trials_frame = np.where(np.diff(self.dataBH['binpos'])<-10)[0]+1
        trial_act = np.zeros(self.dataBH['binpos'].shape+(2,))
        f = np.where(~np.isnan(self.results['fields']['parameter'][n,:,3,0]))[0][0]
        ff = np.where(np.abs(self.results['fields']['parameter'][n,f,3,0]-ground_truth['theta'][n,...])<5)[0]
        rel,_,trial_field = get_reliability(self.results['firingstats']['trial_map'][n,...],self.results['firingstats']['map'][n,:],self.results['fields']['parameter'][n,...],f)
        for i in range(len(trials_frame)-1):
            trial_act[trials_frame[i]:trials_frame[i+1],0] = ground_truth['activation'][n,ff,i]
            trial_act[trials_frame[i]:trials_frame[i+1],1] = trial_field[i]#self.results['firingstats']['trial_field'][n,f,i]

        ax_trial_act.bar(self.dataBH['time'],trial_act[:,0],facecolor=[0,1,0],bottom=1)
        ax_trial_act.bar(self.dataBH['time'],trial_act[:,1],facecolor=[0.4,1,0.4],bottom=0)
        ax_trial_act.plot([t_start,t_end],[1,1],color=[0.5,0.5,0.5],lw=0.3)
        ax_trial_act.text(t_end-75,0.15,'detected',fontsize=8)
        ax_trial_act.text(t_end-100,1.15,'ground truth',fontsize=8)
        ax_trial_act.set_ylim([0,2])
        ax_trial_act.set_xlim([t_start,t_end])
        ax_trial_act.set_xticks([])
        ax_trial_act.set_yticks([])

    ax_loc.plot(self.dataBH['time'],self.dataBH['position'],'.',color=[0.6,0.6,0.6],zorder=5,markeredgewidth=0,markersize=1)
    idx_active = (S>0) & self.dataBH['active']
    idx_inactive = (S>0) & ~self.dataBH['active']

    t_active = self.dataBH['time'][idx_active]
    pos_active = self.dataBH['position'][idx_active]
    S_active = S[idx_active]

    t_inactive = self.dataBH['time'][idx_inactive]
    pos_inactive = self.dataBH['position'][idx_inactive]
    S_inactive = S[idx_inactive]
    if activity_mode == 'spikes':
        ax_loc.scatter(t_active,pos_active,s=S_active,c='r',zorder=10)
        ax_loc.scatter(t_inactive,pos_inactive,s=S_inactive,c='k',zorder=10)
    else:
        ax_loc.scatter(t_active,pos_active,s=(S_active/S.max())**2*10+0.1,color='r',zorder=10)
        ax_loc.scatter(t_inactive,pos_inactive,s=(S_inactive/S.max())**2*10+0.1,color='k',zorder=10)
    ax_loc.bar(t_stop,np.ones(len(t_stop))*self.para['L_track'],color=[0.9,0.9,0.9],zorder=0)
    if highlight_trial:
        ax_loc.fill_between([self.dataBH['trials']['t'][n_trial],self.dataBH['trials']['t'][n_trial+1]],[0,0],[self.para['L_track'],self.para['L_track']],color=[0,0,1,0.2],zorder=1)
        ax_Ca.fill_between([self.dataBH['trials']['t'][n_trial],self.dataBH['trials']['t'][n_trial+1]],[0,0],[1.2*S_raw.max(),1.2*S_raw.max()],color=[0,0,1,0.2],zorder=1)
    ax_loc.set_ylim([0,self.para['L_track']])
    ax_loc.set_xlim([t_start,t_end])
    ax_loc.set_xlabel('t [s]')
    ax_loc.set_ylabel('Location [bin]')

    if ground_truth is None:
        ax_dwell.bar(self.para['bin_array'],self.dataBH['trials']['dwelltime'].sum(0))

    # fmap = self.get_firingmap(S[self.dataBH['active']],self.dataBH['binpos'][self.dataBH['active']],self.dataBH['trials']['dwelltime'].sum(0))
    bin_array = np.linspace(0,self.para['L_track'],self.para['nbin'])
    # ax1.barh(bin_array,fmap,facecolor='r',alpha=0.5,height=self.para['L_track']/self.para['nbin'],label='$\\bar{\\nu}$')

    # fmap = self.results['firingstats']['map'][n,:]
    # fmap_norm = np.nansum(self.results['firingstats']['map'][n,:])
    ax1.barh(bin_array,self.results['firingstats']['map'][n,:],height=self.para['L_track']/self.para['nbin'],facecolor='r',alpha=0.5,label='(all)')

    trials_frame = np.where(np.diff(self.dataBH['binpos'])<-10)[0]+1
    active = np.copy(self.dataBH['active'])
    rel,_,trial_field = get_reliability(self.results['firingstats']['trial_map'][n,...],self.results['firingstats']['map'][n,:],self.results['fields']['parameter'][n,...],f)
    for i in range(len(trials_frame)-1):
        active[trials_frame[i]:trials_frame[i+1]] &= trial_field[i]#self.results['firingstats']['trial_field'][n,f,i]
    fmap = self.get_firingmap(S[active],self.dataBH['binpos'][active])
    ax1.barh(bin_array,fmap,facecolor='b',alpha=0.5,height=self.para['L_track']/self.para['nbin'],label='(active)')
    # ax1.legend(fontsize=10,bbox_to_anchor=[0.1,1.1],loc='lower left')
    ax1.set_xlabel('$\\bar{\\nu}$')
    #ax1.barh(self.para['bin_array'][i],fr_mu[i],facecolor='b',height=1)
    # ax1.errorbar(self.results['firingstats']['map'][n,:],bin_array,xerr=self.results['firingstats']['CI'][n,...],ecolor='r',linewidth=0.2,linestyle='',fmt='',label='1 SD confidence')

    ##flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5)
    ##h_bp = ax1.boxplot(Y,flierprops=flierprops)#,positions=self.para['bin_array'])
    ax1.set_yticks([])#np.linspace(0,100,6))
    ##ax1.set_yticklabels(np.linspace(0,100,6).astype('int'))
    ax1.set_ylim([0,self.para['L_track']])
    ax1.set_xlim([0,np.nanmax(self.results['firingstats']['map'][n,:])*1.2])
    ax1.set_xticks([])
    ##ax1.set_xlabel('Ca$^{2+}$-event rate $\\nu$')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ##ax1.set_ylabel('Position on track')
    # ax1.legend(title='# trials = %d'%self.dataBH['trials']['ct'],loc='lower left',bbox_to_anchor=[0.55,0.025],fontsize=8)#[h_bp['boxes'][0]],['trial data'],
    plt.tight_layout()
    plt.show(block=False)
    if sv:
      pathSv = pathcat([self.para['pathFigs'],'PC_detection_example_%s.png'%suffix])
      plt.savefig(pathSv)
      print('Figure saved @ %s'%pathSv)


  def plt_model_selection(self,fmap_bs,firingstats,trials_fmap):

    rc('font',size=10)
    rc('axes',labelsize=12)
    rc('xtick',labelsize=8)
    rc('ytick',labelsize=8)

    prc = [15.8,84.2]

    fr_mu = firingstats['map']#gauss_smooth(np.nanmean(fmap_bs,1),2)
    fr_CI = firingstats['CI']
    fr_std = firingstats['std']

    #print(fr_mu)
    #print()

    fig = plt.figure(figsize=(7,5),dpi=150)

    ## get data
    pathDat = os.path.join(self.para['pathSession'],'results_redetect.mat')
    ld = loadmat(pathDat,variable_names=['S','C'])

    C = ld['C'][self.para['n'],:]

    #S_raw = self.S
    S_raw = ld['S'][self.para['n'],:]
    _,S_thr,_ = get_firingrate(S_raw[self.dataBH['active']],f=self.para['f'],sd_r=self.para['Ca_thr'])
    if self.para['modes']['activity'] == 'spikes':
      S = S_raw>S_thr
    else:
      S = S_raw

    t_start = 200#0#
    t_end = 470# 600#
    n_trial = 12

    ax_Ca = plt.axes([0.1,0.75,0.5,0.175])
    add_number(fig,ax_Ca,order=1)
    ax_loc = plt.axes([0.1,0.5,0.5,0.25])
    ax1 = plt.axes([0.6,0.5,0.35,0.25])
    ax2 = plt.axes([0.6,0.26,0.35,0.125])
    add_number(fig,ax2,order=4,offset=[-75,10])
    ax3 = plt.axes([0.1,0.08,0.35,0.225])
    add_number(fig,ax3,order=3)
    ax4 = plt.axes([0.6,0.08,0.35,0.125])
    add_number(fig,ax4,order=5,offset=[-75,10])
    ax_acorr = plt.axes([0.7,0.85,0.2,0.1])
    add_number(fig,ax_acorr,order=2,offset=[-50,10])

    idx_longrun = self.dataBH['active']
    t_longrun = self.dataBH['time'][idx_longrun]
    t_stop = self.dataBH['time'][~idx_longrun]
    ax_Ca.bar(t_stop,np.ones(len(t_stop))*1.2*S_raw.max(),color=[0.9,0.9,0.9],zorder=0)

    ax_Ca.fill_between([self.dataBH['trials']['t'][n_trial],self.dataBH['trials']['t'][n_trial+1]],[0,0],[1.2*S_raw.max(),1.2*S_raw.max()],color=[0,0,1,0.2],zorder=1)

    ax_Ca.plot(self.dataBH['time'],C,'k',linewidth=0.2)
    ax_Ca.plot(self.dataBH['time'],S_raw,'r',linewidth=1)
    ax_Ca.plot([0,self.dataBH['time'][-1]],[S_thr,S_thr])
    ax_Ca.set_ylim([0,1.2*S_raw.max()])
    ax_Ca.set_xlim([t_start,t_end])
    ax_Ca.set_xticks([])
    ax_Ca.set_ylabel('Ca$^{2+}$')
    ax_Ca.set_yticks([])


    ax_loc.plot(self.dataBH['time'],self.dataBH['position'],'.',color='k',zorder=5,markeredgewidth=0,markersize=1.5)
    idx_active = (S>0) & self.dataBH['active']
    idx_inactive = (S>0) & ~self.dataBH['active']

    t_active = self.dataBH['time'][idx_active]
    pos_active = self.dataBH['position'][idx_active]
    S_active = S[idx_active]

    t_inactive = self.dataBH['time'][idx_inactive]
    pos_inactive = self.dataBH['position'][idx_inactive]
    S_inactive = S[idx_inactive]
    if self.para['modes']['activity'] == 'spikes':
      ax_loc.scatter(t_active,pos_active,s=3,color='r',zorder=10)
      ax_loc.scatter(t_inactive,pos_inactive,s=3,color='k',zorder=10)
    else:
      ax_loc.scatter(t_active,pos_active,s=(S_active/S.max())**2*10+0.1,color='r',zorder=10)
      ax_loc.scatter(t_inactive,pos_inactive,s=(S_inactive/S.max())**2*10+0.1,color='k',zorder=10)
    ax_loc.bar(t_stop,np.ones(len(t_stop))*self.para['L_track'],width=1/15,color=[0.9,0.9,0.9],zorder=0)
    ax_loc.fill_between([self.dataBH['trials']['t'][n_trial],self.dataBH['trials']['t'][n_trial+1]],[0,0],[self.para['L_track'],self.para['L_track']],color=[0,0,1,0.2],zorder=1)

    ax_loc.set_ylim([0,self.para['L_track']])
    ax_loc.set_xlim([t_start,t_end])
    ax_loc.set_xlabel('time [s]')
    ax_loc.set_ylabel('position [bins]')

    nC,T = ld['C'].shape
    n_arr = np.random.randint(0,nC,20)
    lags = 300
    t=np.linspace(0,lags/15,lags+1)
    for n in n_arr:
      acorr = np.zeros(lags+1)
      acorr[0] = 1
      for i in range(1,lags+1):
        acorr[i] = np.corrcoef(ld['C'][n,:-i],ld['C'][n,i:])[0,1]
      #acorr = np.correlate(ld['S'][n,:],ld['S'][n,:],mode='full')[T-1:T+lags]
      #ax_acorr.plot(t,acorr/acorr[0])
      ax_acorr.plot(t,acorr,linewidth=0.5)
    for T in self.dataBH['trials']['T']:
        ax_acorr.annotate(xy=(T/self.para['f'],0.5),xytext=(T/self.para['f'],0.9),s='',arrowprops=dict(arrowstyle='->',color='k'))
    ax_acorr.set_xlabel('$\Delta t$ [s]')
    ax_acorr.set_ylabel('corr.')
    ax_acorr.spines['right'].set_visible(False)
    ax_acorr.spines['top'].set_visible(False)

    # i = random.randint(0,self.para['nbin']-1)
    i=72
    #ax1 = plt.subplot(211)
    #for i in range(3):
      #trials = np.where(self.result['firingstats']['trial_field'][i,:])[0]
      #if len(trials)>0:
        #fr_mu_trial = gauss_smooth(np.nanmean(self.result['firingstats']['trial_map'][trials,:],0),2)
        #ax1.barh(self.para['bin_array'],fr_mu_trial,alpha=0.5,height=1,label='$\\bar{\\nu}$')

    ax1.barh(self.para['bin_array'],fr_mu,facecolor='b',alpha=0.2,height=1,label='$\\bar{\\nu}$')
    ax1.barh(self.para['bin_array'][i],fr_mu[i],facecolor='b',height=1)

    # ax1.errorbar(fr_mu,self.para['bin_array'],xerr=fr_CI,ecolor='r',linewidth=0.2,linestyle='',fmt='',label='1 SD confidence')
    Y = trials_fmap/self.dataBH['trials']['dwelltime']
    mask = ~np.isnan(Y)
    Y = [y[m] for y, m in zip(Y.T, mask.T)]

    #flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5)
    #h_bp = ax1.boxplot(Y,flierprops=flierprops)#,positions=self.para['bin_array'])
    ax1.set_yticks([])#np.linspace(0,100,6))
    #ax1.set_yticklabels(np.linspace(0,100,6).astype('int'))
    ax1.set_ylim([0,self.para['nbin']])

    ax1.set_xlim([0,np.nanmax(fr_mu[np.isfinite(fr_mu)])*1.2])
    ax1.set_xticks([])
    #ax1.set_xlabel('Ca$^{2+}$-event rate $\\nu$')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    #ax1.set_ylabel('Position on track')
    ax1.legend(title='# trials = %d'%self.dataBH['trials']['ct'],loc='lower left',bbox_to_anchor=[0.55,0.025],fontsize=8)#[h_bp['boxes'][0]],['trial data'],

    #ax2 = plt.subplot(426)
    ax2.plot(np.linspace(0,5,101),firingstats['parNoise'][1]+firingstats['parNoise'][0]*np.linspace(0,5,101),'--',color=[0.5,0.5,0.5],label='lq-fit')
    ax2.plot(fr_mu,fr_std,'r.',markersize=1)#,label='$\\sigma(\\nu)$')
    ax2.set_xlim([0,np.nanmax(fr_mu[np.isfinite(fr_mu)])*1.2])
    #ax2.set_xlabel('firing rate $\\nu$')
    ax2.set_xticks([])
    ax2.set_ylim([0,np.nanmax(fr_std)*1.2])
    ax2.set_ylabel('$\\sigma_{\\nu}$')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax1.bar(self.para['bin_array'][i],fr_mu[i],color='b')

    x_arr = np.linspace(0,fmap_bs[i,:].max()*1.2,self.para['nbin']+1)
    offset = (x_arr[1]-x_arr[0])/2
    act_hist = np.histogram(fmap_bs[i,:],x_arr,normed=True)
    ax3.bar(act_hist[1][:-1],act_hist[0],width=x_arr[1]-x_arr[0],color='b',alpha=0.2,label='data (bin %d)'%i)

    alpha, beta = gamma_paras(fr_mu[i],fr_std[i])
    mu, shape = lognorm_paras(fr_mu[i],fr_std[i])

    def gamma_fun(x,alpha,beta):
      return beta**alpha * x**(alpha-1) * np.exp(-beta*x) / sp.special.gamma(alpha)

    ax3.plot(x_arr,sstats.gamma.pdf(x_arr,alpha,0,1/beta),'r',label='fit: $\\Gamma(\\alpha,\\beta)$')
    #ax3.plot(x_arr,gamma_fun(x_arr,alpha,beta),'r',label='fit: $\\Gamma(\\alpha,\\beta)$')

    #D,p = sstats.kstest(fmap_bs[i,:1000],'gamma',args=(alpha,0,1/beta))

    #sstats.gamma.rvs()
    #ax3.plot(x_arr,sstats.lognorm.pdf(x_arr,s=shape,loc=0,scale=np.exp(mu)),'b',label='fit: $lognorm(\\alpha,\\beta)$')
    #ax3.plot(x_arr,sstats.truncnorm.pdf(x_arr,(0-fr_mu[i])/fr_std[i],np.inf,loc=fr_mu[i],scale=fr_std[i]),'g',label='fit: $gauss(\\mu,\\sigma)$')

    ax3.set_xlabel('$\\nu$')
    ax3.set_ylabel('$p_{bs}(\\nu)$')
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)


    ax2.legend(fontsize=8)
    #ax2.set_title("estimating noise")

    ax3.legend(fontsize=8)

    D_KL_gamma = np.zeros(self.para['nbin'])
    D_KL_gauss = np.zeros(self.para['nbin'])
    D_KL_lognorm = np.zeros(self.para['nbin'])

    D_KS_stats = np.zeros(self.para['nbin'])
    p_KS_stats = np.zeros(self.para['nbin'])

    for i in range(self.para['nbin']):
      x_arr = np.linspace(0,fmap_bs[i,:].max()*1.2,self.para['nbin']+1)
      offset = (x_arr[1]-x_arr[0])/2
      act_hist = np.histogram(fmap_bs[i,:],x_arr,normed=True)
      alpha, beta = gamma_paras(fr_mu[i],fr_std[i])
      mu, shape = lognorm_paras(fr_mu[i],fr_std[i])

      D_KS_stats[i],p_KS_stats[i] = sstats.kstest(fmap_bs[i,:],'gamma',args=(alpha,offset,1/beta))

    ax4.plot(fr_mu,D_KS_stats,'k.',markersize=1)
    ax4.set_xlim([0,np.nanmax(fr_mu[np.isfinite(fr_mu)])*1.2])
    ax4.set_xlabel('$\\bar{\\nu}$')
    ax4.set_ylabel('$D_{KS}$')
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show(block=False)
    if self.para['plt_sv']:
        pathSv = pathcat([self.para['pathFigs'],'PC_analysis_HBM.png'])
        plt.savefig(pathSv)
        print('Figure saved @ %s'%pathSv)


    if True:
      x_arr = np.linspace(0,self.para['nbin'],1001)
      hbm = HierarchicalBayesModel(np.random.rand(1001),x_arr,firingstats['parNoise'],1)
      A_0 = 0.01
      A = 0.03
      std = 6
      theta = 63
      TC = hbm.TC(np.array([A_0,A,std,theta]))

      plt.figure(figsize=(3,2),dpi=150)
      ax = plt.axes([0.15,0.25,0.8,0.7])
      # plt.bar(np.linspace(0,self.para['nbin'],self.para['nbin']),fr_mu,color='b',alpha=0.2,width=1,label='$\\bar{\\nu}$')
      ax.plot(x_arr,TC,'k',label='tuning curve model TC($x$;$A_0$,A,$\\sigma$,$\\theta$)')

      y_arr = np.linspace(0,0.1,1001)
      x_offset = 10
      alpha, beta = gamma_paras(A_0,A_0/2)
      x1 = sstats.gamma.pdf(y_arr,alpha,0,1/beta)
      x1 = -10*x1/x1.max()+x_offset
      x2 = x_offset*np.ones(1001)
      ax.plot(x_offset,A_0,'ko')
      ax.fill_betweenx(y_arr,x1,x2,facecolor='b',alpha=0.2,edgecolor=None)

      idx = 550
      x_offset = x_arr[idx]
      plt.plot(x_offset,TC[idx],'ko')

      alpha, beta = gamma_paras(TC[idx],TC[idx]/2)
      x1 = sstats.gamma.pdf(y_arr,alpha,0,1/beta)
      x1 = -10*x1/x1.max()+x_offset
      x2 = x_offset*np.ones(1001)
      ax.fill_betweenx(y_arr,x1,x2,facecolor='b',alpha=0.2,edgecolor=None,label='assumed noise')

      ### add text to show parameters
      plt_text = True
      if plt_text:
        ax.annotate("", xy=(theta+3*std, A_0), xytext=(theta+3*std, A_0+A),
              arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
        ax.text(theta+3*std+2,A_0+A/2,'A')

        ax.annotate("", xy=(theta, A_0*0.9), xytext=(theta+2*std, A_0*0.9),
              arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
        ax.text(theta+2,A_0*0.3,'$2\\sigma$')

        ax.annotate("", xy=(90, 0), xytext=(90, A_0),
              arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
        ax.text(92,A_0/3,'$A_0$')

        ax.annotate("", xy=(theta, 0), xytext=(theta,A_0+A),
              arrowprops=dict(arrowstyle="-"))
        ax.text(theta,A_0+A*1.1,'$\\theta$')

      ax.set_xlabel('Position $x$ [bins]')
      ax.set_ylabel('Ca$^{2+}$ event rate')
      ax.set_yticks([])
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      #plt.legend(loc='upper left')
      ax.set_ylim([0,0.055])
      ax.set_yticks([])
      plt.tight_layout()
      plt.show(block=False)

      if self.para['plt_sv']:
          pathSv = pathcat([self.para['pathFigs'],'PC_analysis_HBM_model.png'])
          plt.savefig(pathSv)
          print('Figure saved @ %s'%pathSv)


#### ---------------- end of class definition -----------------


class HierarchicalBayesModel:

  ### possible speedup through...
  ###   parallelization of code

  def __init__(self, data, x_arr, parsNoise, f):

    self.data = data
    self.nbin = data.shape[0]
    self.parsNoise = parsNoise
    self.x_arr = x_arr
    self.x_max = x_arr.max()
    self.Nx = len(self.x_arr)
    self.change_model(f)

    ### steps for lookup-table (if used)
    #self.lookup_steps = 100000
    #self.set_beta_prior(5,4)

  def set_logl_func(self):
    def get_logl(p):
      if len(p.shape)==1:
        p = p[np.newaxis,:]
      p = p[...,np.newaxis]

      mean_model = np.ones((p.shape[0],self.Nx))*p[:,0,:]
      if p.shape[1] > 1:
        for j in [-1,0,1]:   ## loop, to have periodic boundary conditions

          mean_model += (p[:,slice(1,None,3),:]*np.exp(-(self.x_arr[np.newaxis,np.newaxis,:]-p[:,slice(3,None,3),:]+self.x_max*j)**2/(2*p[:,slice(2,None,3),:]**2))).sum(1)

      #plt.figure()
      #for i in range(p.shape[0]):
        #plt.subplot(p.shape[0],1,i+1)
        #plt.plot(self.x_arr,np.squeeze(mean_model[i,:]))
      #plt.show(block=False)

      SD_model = self.parsNoise[1] + self.parsNoise[0]*mean_model

      alpha = (mean_model/SD_model)**2
      beta = mean_model/SD_model**2

      logl = np.nansum(alpha*np.log(beta) - np.log(sp.special.gamma(alpha)) + (alpha-1)*np.log(self.data) - beta*self.data ,1)#.sum(1)
      if self.f>1:
        p_theta = p[:,slice(3,None,3)]
        dTheta = np.squeeze(np.abs(np.mod(p_theta[:,1]-p_theta[:,0]+self.nbin/2,self.nbin)-self.nbin/2))
        logl[dTheta<(self.nbin/10)] = -1e300

      return logl

    return get_logl

  ### want beta-prior for std - costly, though, so define lookup-table
  def set_beta_prior(self,a,b):
    self.lookup_beta = sp.stats.beta.ppf(np.linspace(0,1,self.lookup_steps),a,b)
    #return sp.special.gamma(a+b)/(sp.special.gamma(a)*sp.special.gamma(b))*x**(a-1)*(1-x)**(b-1)

  def transform_p(self,p):


    if p.shape[-1]>1:
      p_out = p*self.prior_stretch + self.prior_offset
      #p_out[...,2] = self.prior_stretch[2]*self.lookup_beta[(p[...,2]*self.lookup_steps).astype('int')]
    else:
      p_out = p*self.prior_stretch[0] + self.prior_offset[0]
    #print(p_out[:,slice(3,None,3)])
    return p_out

  def set_priors(self):
    self.prior_offset = np.array(np.append(0,[2,2,0]*self.f))
    prior_max = np.array(np.append(10,[100,20,self.nbin]*self.f))
    self.prior_stretch = prior_max-self.prior_offset
    #print(self.prior_stretch)
    #self.prior_stretch = np.array(np.append(1,[1,6-1,self.nbin]*self.f))

  def change_model(self,f):
    self.f = f
    self.nPars = 1+3*f
    self.TC = self.build_TC_func()
    self.pTC = {}
    self.set_priors()
    self.pTC['wrap'] = np.zeros(self.nPars).astype('bool')
    self.pTC['wrap'][slice(3,None,3)] = True

    self.transform_ct = 0

  def build_TC_func(self):        ## general function to build tuning curve model
    def TC_func(p):
      if len(p.shape)==1:
        p = p[np.newaxis,:]
      p = p[...,np.newaxis]
      TC = np.ones((p.shape[0],self.Nx))*p[:,0,:]
      if p.shape[1] > 1:
        for j in [-1,0,1]:   ## loop, to have periodic boundary conditions
          #TC += p[:,1]*np.exp(-(self.x_arr[np.newaxis,:]-p[:,3]+self.x_max*j)**2/(2*p[:,2]**2))
          TC += (p[:,slice(1,None,3),:]*np.exp(-(self.x_arr[np.newaxis,np.newaxis,:]-p[:,slice(3,None,3),:]+self.x_max*j)**2/(2*p[:,slice(2,None,3),:]**2))).sum(1)
          #TC += (p[:,slice(1,None,3)]*np.exp(-(self.x_arr[np.newaxis,:]-p[:,slice(3,None,3)]+self.x_max*j)**2/(2*p[:,slice(2,None,3)]**2))).sum(-1)

      return np.squeeze(TC)

    return TC_func

####------------------------ end of HBM definition ----------------------------- ####






def load_activity(pathSession,dataSet='redetect'):
  ## load activity data from CaImAn results

  pathAct = pathcat([pathSession,'results_%s.mat'%dataSet])

  ld = sio.loadmat(pathAct,squeeze_me=True)
  S = ld['S']
  if S.shape[0] > 8000:
    S = S.transpose()

  if dataSet=='redetect':
    idx_evaluate = ld['idx_evaluate'].astype('bool')
    idx_previous = ld['idx_previous'].astype('bool')
    SNR = ld['SNR']
    r_values = ld['r_values']
    other = [idx_evaluate,idx_previous,SNR,r_values]
  else:
    other = None

  return S, other

def get_MI(p_joint,p_x,p_f):

  ### - joint distribution
  ### - behavior distribution
  ### - firing rate distribution
  ### - all normalized, such that sum(p) = 1

  p_tot = p_joint * np.log2(p_joint/(p_x[:,np.newaxis]*p_f[np.newaxis,:]))
  #return p_tot[p_joint>0].sum()
  #plt.figure()
  #plt.plot(p_tot.sum(1))
  #plt.show(block=False)
  return np.nansum(p_tot)

def _hsm(data,sort_it=True):
  ### adapted from caiman
  ### Robust estimator of the mode of a data set using the half-sample mode.
  ### versionadded: 1.0.3

  ### Create the function that we can use for the half-sample mode
  ### sorting done as first step, if not specified else

  data = data[np.isfinite(data)]
  if sort_it:
    data = np.sort(data)

  if data.size == 1:
      return data[0]
  elif data.size == 2:
      return data.mean()
  elif data.size == 3:
      i1 = data[1] - data[0]
      i2 = data[2] - data[1]
      if i1 < i2:
          return data[:2].mean()
      elif i2 > i1:
          return data[1:].mean()
      else:
          return data[1]
  else:

      wMin = np.inf
      N = data.size//2 + data.size % 2
      for i in range(N):
          w = data[i + N - 1] - data[i]
          if w < wMin:
              wMin = w
              j = i

      return _hsm(data[j:j + N])


def get_spikeNr(data):

  if np.count_nonzero(data)==0:
    return 0,np.NaN,np.NaN
  else:
    md = _hsm(data,True);       #  Find the mode

    # only consider values under the mode to determine the noise standard deviation
    ff1 = data - md;
    ff1 = -ff1 * (ff1 < 0);

    # compute 25 percentile
    ff1.sort()
    ff1[ff1==0] = np.NaN
    Ns = round((ff1>0).sum() * .5).astype('int')

    # approximate standard deviation as iqr/1.349
    iqr_h = ff1[-Ns];
    sd_r = 2 * iqr_h / 1.349;
    data_thr = md+2*sd_r;
    spikeNr = np.floor(data/data_thr).sum();
    return spikeNr,md,sd_r



def ecdf(x,p=None):

  if type(p)==np.ndarray:
    #assert abs(1-p.sum()) < 10**(-2), 'probability is not normalized, sum(p) = %5.3g'%p.sum()
    #if abs(1-p.sum()) < 10**(-2):
    p /= p.sum()
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = np.cumsum(p[sort_idx])
  else:
    x = np.sort(x)
    y = np.cumsum(np.ones(x.shape)/x.shape)

  return x,y


def get_average(x,p,periodic=False,bounds=None):

  #assert abs(1-p.sum()) < 10**(-2), 'probability not normalized, sum(p) = %5.3g'%p.sum()
  if abs(1-p.sum()) < 10**(-2):
      p /= p.sum()
  if periodic:
    assert bounds, 'bounds not specified'
    L = bounds[1]-bounds[0]
    scale = L/(2*np.pi)
    avg = (cmath.phase((p*np.exp(+complex(0,1)*(x-bounds[0])/scale)).sum())*scale + bounds[0]) % L
  else:
    avg = (x*p).sum()
  return avg


#def get_average(x,p,periodic=False,bounds=None):

  ##assert abs(1-p.sum()) < 10**(-2), 'probability not normalized, sum(p) = %5.3g'%p.sum()
  #if abs(1-p.sum()) < 10**(-2):
      #p /= p.sum()
  #if periodic:
    #assert bounds, 'bounds not specified'
    #L = bounds[1]-bounds[0]
    #scale = L/(2*np.pi)
    #avg = (cmath.phase((p*periodic_to_complex(x,bounds)).sum())*scale) % L + bounds[0]
  #else:
    #avg = (x*p).sum()
  #return avg


def periodic_difference(x,y,bounds):
  scale = (bounds[1]-bounds[0])/(2*np.pi)
  print(scale)
  print(periodic_to_complex(x,bounds))
  print(periodic_to_complex(y,bounds))
  print(cmath.phase(periodic_to_complex(y,bounds) - periodic_to_complex(x,bounds)))
  diff = cmath.phase(periodic_to_complex(y,bounds) - periodic_to_complex(x,bounds))*scale + bounds[0]
  return diff


def periodic_to_complex(x,bounds):
  scale = (bounds[1]-bounds[0])/(2*np.pi)
  return np.exp(complex(0,1)*(x-bounds[0])/scale)

def complex_to_periodic(phi,bounds):
  L = bounds[1]-bounds[0]
  scale = L/(2*np.pi)

  return (cmath.phase(phi)*scale) % L + bounds[0]



def jackknife(X,Y,W=None,rank=1):

  ## jackknifing a linear fit (with possible weights)
  ## W_i = weights of value-tuples (X_i,Y_i)

  if type(W) == np.ndarray:
    print('weights given (not working)')
    W = np.ones(Y.shape)
    Xw = X * np.sqrt(W[:,np.newaxis])
    Yw = Y * np.sqrt(W)
  else:
    if rank==1:
      Xw = X
    elif rank==2:
      Xw = np.vstack([X,np.ones(len(X))]).T
    Yw = Y

  if len(Xw.shape) < 2:
    Xw = Xw[:,np.newaxis]

  N_data = len(Y);

  fit_jk = np.zeros((N_data,2));
  mask_all = (~np.isnan(Y)) & (~np.isnan(X))

  for i in range(N_data):
    mask = np.copy(mask_all)
    mask[i] = False;
    try:
      if rank==1:
        fit_jk[i,0] = np.linalg.lstsq(Xw[mask,:],Yw[mask])[0]
      elif rank==2:
        fit_jk[i,:] = np.linalg.lstsq(Xw[mask,:],Yw[mask])[0]
      #fit_jk[i,1] = 0
    except:
      fit_jk[i,:] = np.NaN

  return np.nanmean(fit_jk,0)

### -------------- lognorm distribution ---------------------
def lognorm_paras(mean,sd):
  shape = np.sqrt(np.log(sd/mean**2+1))
  mu = np.log(mean/np.sqrt(sd/mean**2 + 1))
  return mu, shape

### -------------- Gamma distribution -----------------------
def gamma_paras(mean,SD):
  alpha = (mean/SD)**2
  beta = mean/SD**2
  return alpha, beta


def pathcat(strings):
  return '/'.join(strings)
