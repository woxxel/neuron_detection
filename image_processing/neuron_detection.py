import time, pickle, os

import numpy as np
import caiman as cm
from caiman.source_extraction import cnmf as cnmf


def neuron_detection(fname,params,use_parallel=True,n_processes=None):

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

    N = onacid.estimates.A.shape[-1]
    # print(f'\tNumber of components found: {N}')

    ### %% evaluate components (CNN, SNR, correlation, border-proximity)
    Yr, dims, T = cm.load_memmap(opts.get("data","fnames")[0])
    Y = np.reshape(Yr.T, [T] + list(dims), order='F')
    onacid.estimates.evaluate_components(Y,opts,dview) # does this work with a memmapped file?

    ## find and remove neurons which are too close to the border
    plot_stuff=False
    if plot_stuff:
        onacid.estimates.plot_contours(idx=onacid.estimates.idx_components)   ## plot contours, need that one to get the coordinates
        # plt.draw()
        # plt.pause(1)
    else:
        onacid.estimates.coordinates = cm.utils.visualization.get_contours(onacid.estimates.A, dims, thr=0.2, thr_method='max')

    # return onacid
    CoM = np.zeros((len(onacid.estimates.idx_components),2))
    for i,n in enumerate(onacid.estimates.idx_components):
        CoM[i,:] = onacid.estimates.coordinates[n]['CoM']

    # update object with selected components
    onacid.estimates.select_components(use_object=True, save_discarded_components=False)
    print(f'\tNumber of components left after evaluation: {onacid.estimates.A.shape[-1]}')

    # print('final step: storing results...')
    results = dict(A=onacid.estimates.A,
        C=onacid.estimates.C,
        S=onacid.estimates.S,
        CoM=CoM,
        SNR=onacid.estimates.SNR_comp,
        r_val=onacid.estimates.r_values,
        CNN=onacid.estimates.cnn_preds,
        b=onacid.estimates.b,
        f=onacid.estimates.f,
    )

    out_file = os.path.join(os.path.split(fname)[0],'OnACID_results.pkl')
    pickle.dump(results, open(out_file, "wb"))
    print("\tNeuron detection done @t = %s, (time passed: %s)" % (time.ctime(),str(time.time()-t_start)))

    return out_file
