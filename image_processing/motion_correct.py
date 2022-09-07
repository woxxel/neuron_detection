import time

import CaImAn.caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.motion_correction import MotionCorrect

def motion_correct(fname,params,use_parallel=True,n_processes=None):

    """
        Runs the motion correction algorithm NoRMCorr and returns the path to the output file
    """

    print(f"\tNow running motion correction @t = {time.ctime()}")
    print(f"\t\tNeed to add: save statistics of motion correction, such as framewise rigid shift (what else?)")

    t_start = time.time()   # start time measurement from here

    ## initialize parameters and settings for running OnACID
    opts = cnmf.params.CNMFParams(params_dict=params)
    if use_parallel:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes, single_thread=False)
    else:
        dview=None
        n_processes=1

    ## run the actual motion correction algorithm
    mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)

    if use_parallel:
        cm.stop_server(dview=dview)      ## restart server to clean up memory

    print("\tMotion correction done @t = %s, (time passed: %s)" % (time.ctime(),str(time.time()-t_start)))

    return mc.mmap_file[0]
