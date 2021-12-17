#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic demo for the CaImAn Online algorithm (OnACID) using CNMF initialization.
It demonstrates the construction of the params and online_cnmf objects and
the fit function that is used to run the algorithm.
For a more complete demo check the script demo_OnACID_mesoscope.py

@author: jfriedrich & epnev
"""

import logging
import numpy as np
import scipy as sp
import os, sys, hdf5storage, time
from scipy.io import savemat

import matplotlib.pyplot as plt
from utils import pathcat

sys.path.append('/home/wollex/Data/Science/PhD/Programs/CaImAn')
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.paths import caiman_datadir
from caiman.motion_correction import MotionCorrect


try:
    if __IPYTHON__:
        print("Detected iPython")
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

#%%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
    "[%(process)d] %(message)s",
    level=logging.ERROR)

# set up CNMF parameters
params_dict ={

        #general data
        'fnames': [fname],
        'fr': 15,
        'decay_time': 0.47,
        'gSig': [6, 6],                     # expected half size of neurons

        #model/analysis
        'rf': 64//2,                        # size of patch
        'K': 200,                           # max number of components
        'nb': 2,                            # number of background components per patch
        'p': 0,                             # order of AR indicator dynamics
        'stride': 8,
        #'simultaneously': True,

        # init
        'ssub': 2,                          # spatial subsampling during initialization
        'tsub': 5,                          # temporal subsampling during initialization

        #motion
        'motion_correct': False,
        'pw_rigid': False,
        'strides': 96,
        'max_shifts': 12,                   # maximum allowed rigid shift in pixels
        'overlaps': 24,                     # overlap between patches (size of patch in pixels: strides+overlaps)
        'max_deviation_rigid': 12,          # maximum deviation allowed for patch with respect to rigid shifts
        #'only_init': False,                # whether to run only the initialization

        #online
        'init_batch': 300,                  # number of frames for initialization
        'init_method': 'bare',              # initialization method
        'update_freq': 2000,                # update every shape at least once every update_freq steps
        'use_dense': False,
        #'dist_shape_update': True,

        #make things more memory efficient
        'memory_efficient': False,
        'block_size_temp': 5000,
        'num_blocks_per_run_temp': 20,
        'block_size_spat': 5000,
        'num_blocks_per_run_spat': 20,

        #quality
        'min_SNR': 2.5,                     # minimum SNR for accepting candidate components
        'rval_thr': 0.85,                   # space correlation threshold for accepting a component
        'rval_lowest': 0,
        'sniper_mode': True,                # flag for using CNN for detecting new components
        #'test_both': True,                 # use CNN and correlation to test for new components
        'use_cnn': True,

        'thresh_CNN_noisy': 0.6,            # CNN threshold for candidate components
        'min_cnn_thr': 0.8,                 # threshold for CNN based classifier
        'cnn_lowest': 0.3,                  # neurons with cnn probability lower than this value are rejected

        #display
        'show_movie': False,
        'save_online_movie': False,
        'movie_name_online': "test_mp4v.avi"
}


def run_CaImAn_mouse(pathMouse,params_dict,sessions=None,fname_start=["thy","shank"],use_parallel=True,reprocess=False):
    """
        wrapper function for running neuron detection on a specified subset of sessions
        of a mouse

        :param string pathMouse
            absolute path to the mouse folder
        :param dict params_dict
            dictionary, containing parameters for the CaImAn algorithm
        :param list(int) sessions
            contains first and last session to process
        :param list(string) fname_start (default=["thy","shank"] (e.g. mouse-model names))
            defines strings that recording files are allowed to begin with
        :param bool use_parallel (default=True)
            defines, whether parallel computing should be used
        :param bool reprocess (default=False)
            defines, whether existing analyzed data should be overwritten, or skipped

        This function assumes that data is stored in a specific folder structure:
        |
        | - pathMouse
        | |
        | | - pathSession(1) (formatted as "Session01")
        | | | - recordingData
        | |
        | | - pathSession(2) (formatted as "Session02")
        | | | - recordingData
        | | .
        | | .
        | | .
        |
    """
    plt.ion()
    if not sessions==None:
        for s in range(sessions[0],sessions[-1]+1):
            pathSession = pathcat([pathMouse,"Session%02d/"%s])
            print("\t Session: "+pathSession)
            run_CaImAn_session(pathSession,params_dict,use_parallel,reprocess)
      # l_Ses = os.listdir(pathMouse)
      # l_Ses.sort()
      # for f in l_Ses:
      #   if f.startswith("Session"):
      #     pathSession = pathMouse + f + '/'
      #     print("\t Session: "+pathSession)
      #     run_CaImAn_session(pathSession,use_parallel=use_parallel)
          ##return cnm, Cn, opts

def run_CaImAn_session(pathSession,params_dict,fname_start=None,use_parallel=True,reprocess=False):
    """
        TODO:
            [ ] check, what kind of motion correction is run and how well it performs vs external one
            [ ] move all parameters to input
            [ ] allow to receive memmapped files
            [ ] check, how well the memmapping actually works (RAM usage)
            [ ] allow selection of wished for steps
            [ ] enable p=1 when skipping last step
            [ ] remove "c" from output of parallel initiation?

        wrapper function for running neuron detection on the recording of a session

        :param string pathSession
            absolute path to the session folder
        :param dict params_dict
            dictionary, containing parameters for the CaImAn algorithm
        :param list(string) fname_start
            defines strings that recording files are allowed to begin with
        :param bool use_parallel (default=True)
            defines, whether parallel computing should be used
        :param bool reprocess (default=False)
            defines, whether existing analyzed data should be overwritten, or skipped

        This function assumes that data is stored in a specific folder structure:
        |
        | - pathSession
        | | - recordingData
        |

        This function assumes that sessions have been subject to rigid and non-rigid
        motion correction before (I'm not happy with the built-in one of CaImAn)

        The algorithm works in 3 steps:
            1: prepare data for analysis by rewriting it into a memmapped file and
                performing motion correction if necessary
            2: passing over data using OnACID for neuron detection
            3 (optional, not recommended when using large data): passing over data again,
                inputting previously detected neurons for refinement and detection of
                activity prior to declaration as neuron
    """
    pass  # For compatibility between running under Spyder and the CLI

    plt.close('all')

    ###### general parameters (maybe as input?)
    border_thr = 5         # minimal distance of centroid to border
    run_motion_correct = False #not fname.endswith('.h5')  # defines, whether motion correction is run throughout OnACID

    ## set path for memmapped file (this only requires temporary storage, best on
    ## an internal hard disk to allow quick access)
    sv_dir = "/home/wollex/Data/Science/PhD/Data/tmp/"

    ## set path for saving results and check, whether it's already present
    svname = "results_OnACID"
    svname_mat = pathSession + svname + '.mat'
    svname_h5 = pathSession + svname + '.hdf5'
    if os.path.exists(svname_mat):
        print("Processed file already present - skipping")
        return

    ## find path of recording session within pathSession
    fname = None
    for f in os.listdir(pathSession):
        if any([f.startswith(start) for start in fname_start]):
            fname = pathSession + f
            if f.endswith('.h5'):
                break
    if not fname or not os.path.exists(fname):
        print("No file here to process :(")
        return
    # params_dict['fnames'] = [fname]

    t_start = time.time()   # start time measurement from here

    ## initialize parameters and settings for running OnACID
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    if use_parallel:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=4, single_thread=False)
    else:
        dview=None
        n_processes=1

### ------------ preparation for running neuron detection ------------------ ###
    ## process recording file into memmapped file and run motion correction, if required
    ## CAREFUL! requires loading the whole file into memory - might go beyond your RAM
    print("now running step 1: preparing data for analysis @t = " +  time.ctime())
    if run_motion_correct:
        ## requires to be run prior to neuron detection as this function passes
        ## over the recordings twice, but should only need to correct for motion once
        print('\tperform motion correction')

        ## create a motion correction object with the specified parameters and run it
        mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)  ## is this only piecewise-rigid motion correction? (using NoRMCorre)
        fname_memmap = cm.save_memmap(mc.mmap_file, base_name='memmap_', save_dir=sv_dir, n_chunks=100, order='C', dview=dview)
        os.remove(mc.mmap_file[0])  # clean up afterwards
        # opts.change_params({'motion_correct':True,'pw_rigid':True})
    else:
        fname_memmap = cm.save_memmap([fname], base_name='memmap_', save_dir=sv_dir, n_chunks=100, order='C', dview=dview)
    if use_parallel:
        cm.stop_server(dview=dview)      ## restart server to clean up memory

    print("\tpreparation done @t = %s, (time passed: %s)" % (time.ctime(),str(time.time()-t_start)))
    #fname_memmap = sv_dir + "memmap__d1_512_d2_512_d3_1_order_C_frames_8989_.mmap"
    opts.change_params({'fnames': [fname_memmap]})

    ## create background image from local-correlation analysis (reduction along t-dimension) for display
    Cn = cm.load(fname_memmap, subindices=slice(0,None,5)).local_correlations(swap_dim=False)

    ## open memmap file to prepare for processing
    Yr, dims, T = cm.load_memmap(fname_memmap)
    Y = np.reshape(Yr.T, [T] + list(dims), order='F')


### -------------------------- 1st run (OnACID) ---------------------------- ###
    print("now running step 2: OnACID... ")
    ## fit with online object on memmapped data
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()

    print('\tNumber of components found:' + str(cnm.estimates.A.shape[-1]))

    ### %% evaluate components (CNN, SNR, correlation, border-proximity)
    if use_parallel:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=4, single_thread=False)
    else:
        dview=None
        n_processes=1

    ## evaluate components according to SNR, CNN, rval
    cnm.estimates.evaluate_components(Y,opts,dview) # does this work with a memmapped file?
    #cnm.estimates.view_components(img=Cn)
    #plt.close('all')
    cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components, crd=None)   ## plot contours, need that one to get the coordinates
    plt.draw()
    plt.pause(1)

    ## find and remove neurons which are too close to the border
    idx_border = []
    for n in cnm.estimates.idx_components:
        if (cnm.estimates.coordinates[n]['CoM'] < border_thr).any() or (cnm.estimates.coordinates[n]['CoM'] > (cnm.estimates.dims[0]-border_thr)).any():
            idx_border.append(n)
    cnm.estimates.idx_components = np.setdiff1d(cnm.estimates.idx_components,idx_border)
    cnm.estimates.idx_components_bad = np.union1d(cnm.estimates.idx_components_bad,idx_border)

    # update object with selected components
    cnm.estimates.select_components(use_object=True, save_discarded_components=False)


    print('\tNumber of components left after evaluation:' + str(cnm.estimates.A.shape[-1]))

    ### %% save file to allow loading as CNMF- (instead of OnACID-) file
    # cnm.estimates = clear_cnm(cnm.estimates,remove=['shifts','discarded_components']) # why? retain should remove them anyway
    cnm.estimates = clear_cnm(cnm.estimates,retain=['A','C','S','b','f','YrA','dims','coordinates','sn'])
    cnm.save(svname_h5)


### ------------------------ 2nd run (CaImAn batch) ------------------------ ###
    print("now running step 3: refinement... ")
    ## run a refit on the whole data
    if use_parallel:
        cm.stop_server(dview=dview)      ## restart server to clean up memory
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=4, single_thread=False)
    else:
        dview=None
        n_processes=1

    cnm = cnmf.cnmf.load_CNMF(svname_h5,n_processes,dview)
    cnm.params.change_params({'p':1})
    cnm.estimates.dims = Cn.shape # gets lost for some reason

    print("\tmerge & update spatial + temporal & deconvolve @t = %s, (time passed: %s)" % (time.ctime(),str(time.time()-t_start)))
    cnm.update_temporal(Yr)   # need this to calculate noise per component for merging purposes
    cnm.merge_comps(Yr,mx=1000,fast_merge=False)
    cnm.estimates.C[np.where(np.isnan(cnm.estimates.C))] = 0    ## for some reason, there are NaNs in it -> cant process this
    cnm.update_temporal(Yr)   # update temporal trace after merging

    cnm.params.change_params({'n_pixels_per_process':1000})     ## for some reason this one gets lost
    cnm.estimates.sn, _ = cnmf.pre_processing.get_noise_fft(Yr[:,:2000].astype(np.float32))
    cnm.update_spatial(Yr)    # update shapes a last time
    cnm.update_temporal(Yr)   # update temporal trace a last time
    cnm.deconvolve()

    if use_parallel:
        cm.stop_server(dview=dview)

    print('\tNumber of components left after merging:' + str(cnm.estimates.A.shape[-1]))
    print("Done @t = %s, (time passed: %s)" % (time.ctime(),str(time.time()-t_start)))


### ------------------- store results ------------------- ###
###%% store results in matlab array for further processing
    print('final step: storing results...')
    results = dict(A=cnm.estimates.A,
                   C=cnm.estimates.C,
                   S=cnm.estimates.S,
                   #Cn=Cn,
                   b=cnm.estimates.b,
                   f=cnm.estimates.f)
    savemat(svname_mat,results)
    #hdf5storage.write(results, '.', svname_mat, matlab_compatible=True)

    #cnm.estimates.coordinates = None
    #cnm.estimates.plot_contours(img=Cn, crd=None)
    #cnm.estimates.view_components(img=Cn)
    #plt.draw()
    #plt.pause(1)
    ### %% save only items that are needed to save disk-space
    #cnm.estimates = clear_cnm(cnm.estimates,retain=['A','C','S','b','f','YrA'])
    #cnm.save(svname_h5)

    print("Total time taken: " +  str(time.time()-t_start))

    os.remove(fname_memmap)
    #return cnm, Cn, opts



def clear_cnm(dic,retain=None,remove=None):

    if retain is None and remove is None:
        print('Please provide a list of keys to obtain in the structure')
        return

    if not (remove is None):
        keys = list(dic.__dict__.keys())
        for key in keys:
            if key in remove:
                dic.__dict__.pop(key)

    if not (retain is None):
        keys = list(dic.__dict__.keys())
        for key in keys:
            if key not in retain:
                dic.__dict__.pop(key)

    return dic

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
#if __name__ == "__main__":
    #[cnm, Cn, opts] = main()
    #main()
