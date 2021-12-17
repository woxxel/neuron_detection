import caiman as cm
import numpy as np
from caiman.motion_correction import MotionCorrect
import time, os
import cv2
from tifffile import *

def NormCorr(fname,use_parallel=False,mode='other'):

    tic = time.time()
    if mode=="NormCorr":
        print('running norm corr')
        if use_parallel:
            c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=10, single_thread=False)
        else:
            dview=None
            n_processes=1

        mc = MotionCorrect(fname,dview=dview)
        mc.motion_correct(save_movie=True)

    # return mc
    else:
        print('running farneback')
        pathOut = './nonRigidCorr.tif'

        ext = os.path.splitext(fname)[-1]

        if ext=='.tif':
            tif = TiffFile(fname)
            d = tif.series[0].shape
            data_type = tif.series[0].dtype

            memmap_image = memmap(pathOut,shape=d+(2,),dtype=np.float32)

            A0 = tif.pages[0].asarray()

            for i in range(1000):
                A1 = tif.pages[i].asarray()

                memmap_image[i,:,:,:] = cv2.calcOpticalFlowFarneback(A0,A1,None,0.5,5,128,3,7,1.5,0)

                if not (i % 100):
                    print('i=%d, time spent: %.2fs'%(i,time.time()-tic))
                    memmap_image.flush
                A0 = A1

        del memmap_image
        print('time spent: %.2fs'%(time.time()-tic))
