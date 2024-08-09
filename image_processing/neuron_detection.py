import time, os
# import h5py, hdf5storage
from scipy.io import savemat
import numpy as np
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import *


def neuron_detection(fname,params,use_parallel=True,n_processes=None,border_thr=5,suffix='',save_type='hdf5'):

    """
        Runs the neuron detection algorithm OnACID and returns the path to the output file
    """

    fileName = os.path.basename(fname).split('_els')[0]
    saveName = os.path.join(os.path.split(fname)[0],f'results_CaImAn_{fileName}{suffix}.')

    print('in the end, saving results to: ',saveName, ' with data type: ',save_type)

    print(f"\tNow running neuron detection @t = {time.ctime()}")
    t_start = time.time()   # start time measurement from here

    params['fnames'] = [fname]

    opts = cnmf.params.CNMFParams(params_dict=params)
    if use_parallel:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes, single_thread=False)
    else:
        dview=None
        n_processes=1
    
    try:
        cnmf_obj = cnmf.online_cnmf.OnACID(params=opts,dview=dview)
        cnmf_obj.fit_online()
    except:
        print('online processing failed - try batch processing')
        opts.set('init',{'K':100})
        cnmf_obj = cnmf.CNMF(n_processes,dview=dview,params=opts)

        fname_new = cm.save_memmap([fname],
            base_name='Yr',
            order='C')
        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        cnmf_obj = cnmf_obj.fit(images)

    if use_parallel:
        ## restart server (otherwise crashes)
        cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes, single_thread=False)

    N = cnmf_obj.estimates.A.shape[-1]
    # print(f'\tNumber of components found: {N}')

    ### %% evaluate components (CNN, SNR, correlation, border-proximity)
    Yr, dims, T = cm.load_memmap(opts.get("data","fnames")[0])
    Y = np.reshape(Yr.T, [T] + list(dims), order='F')

    cnmf_obj.estimates.evaluate_components(Y,opts,dview) # does this work with a memmapped file?
    cnmf_obj.estimates.Cn = cm.load(fname, subindices=slice(0,None,10)).local_correlations(swap_dim=False)

    plot_stuff=False
    if plot_stuff:
        cnmf_obj.estimates.plot_contours(idx=cnmf_obj.estimates.idx_components)   ## plot contours, need that one to get the coordinates
        # plt.draw()
        # plt.pause(1)
    else:
        cnmf_obj.estimates.coordinates = cm.utils.visualization.get_contours(cnmf_obj.estimates.A, dims, thr=0.2, thr_method='max')

    ## find and remove neurons which are too close to the border
    idx_border = []
    for n in range(N):
        if (cnmf_obj.estimates.coordinates[n]['CoM'] < border_thr).any() or (cnmf_obj.estimates.coordinates[n]['CoM'] > (cnmf_obj.estimates.dims[0]-border_thr)).any():
            idx_border.append(n)
    cnmf_obj.estimates.idx_components = np.setdiff1d(cnmf_obj.estimates.idx_components,idx_border)
    cnmf_obj.estimates.idx_components_bad = np.union1d(cnmf_obj.estimates.idx_components_bad,idx_border)

    # update object with selected components
    cnmf_obj.estimates.select_components(use_object=True, save_discarded_components=False)
    print(f'\tNumber of components left after evaluation: {cnmf_obj.estimates.A.shape[-1]}')


    # print(cnmf_obj.estimates.Cn)
    # print(cnmf_obj.estimates.Cn.shape)
    retain_keys = ['A','C','S','b','f','Cn','dims','coordinates','SNR_comp','r_values','cnn_preds']

    ## for some reason, Cn has issues in 1% of cases
    cnmf_obj.estimates.Cn[np.isnan(cnmf_obj.estimates.Cn)] = np.nanmean(cnmf_obj.estimates.Cn)
    
    # = cnmf_obj.estimates.Cn.T
    # if cnmf_obj.estimates.Cn.shape[0] != cnmf_obj.estimates.dims[0]:
        # retain_keys = ['A','C','S','b','f','dims','coordinates','SNR_comp','r_values','cnn_preds']

    cnmf_obj.estimates = clear_cnm(cnmf_obj.estimates,retain=retain_keys)
    cnmf_obj.fname = fname

    # for tp in save_type:
    out_file = saveName + save_type
    print('out',out_file,save_type)
    if save_type=='hdf5':
        save_dict_to_hdf5(cnmf_obj.estimates.__dict__, out_file)
    if save_type=='mat':
        savemat(out_file,cnmf_obj.estimates.__dict__)
    # else:
        # assert False, f'save_type {save_type} not recognized'

    print("\tNeuron detection done @t = %s, (time passed: %s)" % (time.ctime(),str(time.time()-t_start)))

    return out_file

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