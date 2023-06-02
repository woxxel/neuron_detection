from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div

import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.INFO)

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo



fnames = ['./data/Test/Session01/test_stack.tif']
# fnames = 'Sue_2x_3000_40_-46.tif'
# fnames = [download_demo(fnames)]     # the file will be downloaded if it doesn't already exist
m_orig = cm.load_movie_chain(fnames)
downsample_ratio = .2  # motion can be perceived better when downsampling in time
m_orig.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=30, magnification=1)   # play movie (press q to exit)

max_shifts = (10, 10)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)


#%% start the cluster (if a cluster already exists terminate it)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# create a motion correction object
mc = MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid,
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan)

# capture
# correct for rigid motion correction and save the file (in memory mapped form)
mc.motion_correct(save_movie=True)

# load motion corrected movie
m_rig = cm.load(mc.mmap_file)
bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
# # %% visualize templates
# plt.figure(figsize = (20,10))
# plt.imshow(mc.total_template_rig, cmap = 'gray')
# plt.show(block=False)
#
# #%% inspect movie
# m_rig.resize(1, 1, downsample_ratio).play(
#     q_max=99.5, fr=30, magnification=2, bord_px = 0*bord_px_rig) # press q to exit
#
# #%% plot rigid shifts
# # plt.close()
# plt.figure(figsize = (20,10))
# plt.plot(mc.shifts_rig)
# plt.legend(['x shifts','y shifts'])
# plt.xlabel('frames')
# plt.ylabel('pixels')
# plt.show(block=False)




#%% motion correct piecewise rigid
mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)

mc.motion_correct(save_movie=True, template=mc.total_template_rig)
m_els = cm.load(mc.fname_tot_els)
# m_els.resize(1, 1, downsample_ratio).play(
#     q_max=99.5, fr=30, magnification=2,bord_px = bord_px_rig)


cm.concatenate([m_orig.resize(1, 1, downsample_ratio) - mc.min_mov*mc.nonneg_movie,
                m_rig.resize(1, 1, downsample_ratio), m_els.resize(
            1, 1, downsample_ratio)], axis=2).play(fr=60, q_max=99.5, magnification=2, bord_px=bord_px_rig)


# #%% visualize elastic shifts
# plt.close()
# plt.figure(figsize = (20,10))
# plt.subplot(2, 1, 1)
# plt.plot(mc.x_shifts_els)
# plt.ylabel('x shifts (pixels)')
# plt.subplot(2, 1, 2)
# plt.plot(mc.y_shifts_els)
# plt.ylabel('y_shifts (pixels)')
# plt.xlabel('frames')
# #%% compute borders to exclude
# bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
#                                  np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
