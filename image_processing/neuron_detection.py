import time, pickle, os
import numpy as np
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import *


def neuron_detection(fname,params,use_parallel=True,n_processes=None,border_thr=5):

    """
        Runs the neuron detection algorithm OnACID and returns the path to the output file
    """

    print(f"\tNow running neuron detection @t = {time.ctime()}")
    t_start = time.time()   # start time measurement from here

    params['fnames'] = [fname]

    opts = cnmf.params.CNMFParams(params_dict=params)
    if use_parallel:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes, single_thread=False)
    else:
        dview=None
        n_processes=1

    # print(opts.get_group('online'))
    onacid = cnmf.online_cnmf.OnACID(params=opts,dview=dview)
    onacid.fit_online()

    if use_parallel:
        ## restart server (otherwise crashes)
        cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes, single_thread=False)

    N = onacid.estimates.A.shape[-1]
    # print(f'\tNumber of components found: {N}')

    ### %% evaluate components (CNN, SNR, correlation, border-proximity)
    Yr, dims, T = cm.load_memmap(opts.get("data","fnames")[0])
    Y = np.reshape(Yr.T, [T] + list(dims), order='F')

    ### it crashes here!!!
    onacid.estimates.evaluate_components(Y,opts,dview) # does this work with a memmapped file?

    onacid.estimates.Cn = cm.load(fname, subindices=slice(0,None,10)).local_correlations(swap_dim=False)

    ## find and remove neurons which are too close to the border
    plot_stuff=False
    if plot_stuff:
        onacid.estimates.plot_contours(idx=onacid.estimates.idx_components)   ## plot contours, need that one to get the coordinates
        # plt.draw()
        # plt.pause(1)
    else:
        onacid.estimates.coordinates = cm.utils.visualization.get_contours(onacid.estimates.A, dims, thr=0.2, thr_method='max')

    # return onacid
    ## find and remove neurons which are too close to the border
    # try:
    idx_border = []
    for n in range(N):
        if (onacid.estimates.coordinates[n]['CoM'] < border_thr).any() or (onacid.estimates.coordinates[n]['CoM'] > (onacid.estimates.dims[0]-border_thr)).any():
            idx_border.append(n)
    onacid.estimates.idx_components = np.setdiff1d(onacid.estimates.idx_components,idx_border)
    onacid.estimates.idx_components_bad = np.union1d(onacid.estimates.idx_components_bad,idx_border)
    # except:
    #     print('border thresholding failed')
    # CoM = np.zeros((len(onacid.estimates.idx_components),2))
    # for i,n in enumerate(onacid.estimates.idx_components):
        # CoM[i,:] = onacid.estimates.coordinates[n]['CoM']

    # update object with selected components
    onacid.estimates.select_components(use_object=True, save_discarded_components=False)
    print(f'\tNumber of components left after evaluation: {onacid.estimates.A.shape[-1]}')



    # print('final step: storing results...')
    # results = dict(A=onacid.estimates.A,
    #     C=onacid.estimates.C,
    #     S=onacid.estimates.S,
    #     CoM=CoM,
    #     SNR=onacid.estimates.SNR_comp,
    #     r_val=onacid.estimates.r_values,
    #     CNN=onacid.estimates.cnn_preds,
    #     b=onacid.estimates.b,
    #     f=onacid.estimates.f,
    # )

    out_file = os.path.join(os.path.split(fname)[0],'OnACID_results.hdf5')
    onacid.estimates = clear_cnm(onacid.estimates,retain=['A','C','S','b','f','Cn','dims','coordinates','SNR_comp','r_values','cnn_preds'])

    save_dict_to_hdf5(onacid.estimates.__dict__, out_file)

    # onacid.save(out_file)
    # pickle.dump(results, open(out_file, "wb"))
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
