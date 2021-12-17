"""
    Class for aligning frames in Tiff files.

    INSERT AND ADJUST DESCRIPTION

    Written by: chepe@nld.ds.mpg.de (matlab version)
    Date: 03Feb2017

    translated to python by alexander.schmidt@ds.mpg.de
    Date: 15.Dec.2021

"""

import os, time
import numpy as np
from tifffile import *
from scipy import signal

def make_file_structure(pathIn):

    """
        general function for reading tiff-files from "pathIn", being either
            - a path to the tiff file
            - a path to a folder containing tiff-files
            - a list of lists to several tiff-files (inner list = file_num, outer list = channel)

        TODO:
            recurse doesnt work properly, when only a string is given

    """

    assertMsg = 'The input to tiff_reader is either a string of the path for the GUI or a list of lists of file paths {rows = file_num, columns = channel}.'
    if type(pathIn) is str:
        # input is either path for a single file or for image folder (or for GUI)
        if pathIn[-4:] == '.tif':
            return [pathIn]
        elif os.path.isdir(pathIn):
            # pathIn specifies a folder, of which all files are read into a single channel
            return [os.path.join(pathIn,path) for path in os.listdir(pathIn) if path[-4:]=='.tif']
        else:
            assert False, assertMsg
    elif type(pathIn) is list:
        # file_names are given as list of lists
        if np.all([type(sublist)==list for sublist in pathIn]):
            return pathIn
        else:
            return [make_file_structure(path) for path in pathIn]
    else:
        # everything else
        assert False, assertMsg

def check_files(tif_info):
    """
        checks the provided files for having the same pixel size and datatype
        raises error, when comparison fails for any

        TODO:
            type checking needed? or should it just be casted to same?
    """

    # test for equal dimensions of frames in all files
    # tif_ref = TiffFile(file_names[0][0]).series[0]
    tifs = [[tif.series[0] for tif in channel] for channel in tif_info]
    assert np.all([
        np.all([((tif.shape[-2:]==tifs[0][0].shape[-2:]) and (tif.dtype==tifs[0][0].dtype)) for tif in channel])
        for channel in tifs]), 'provided files are not of the same size or datatype'

    # test for equal total number of frames in all channels
    tif_length = [np.sum([tif.shape[0] for tif in channel]) for channel in tifs]
    assert np.all([length==tif_length[0] for length in tif_length]), 'channels have a different number of total frames'


def optical_flow(I1g, I2g, window_size, tau=1e-2):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / np.iinfo(I1g.dtype).max # normalize pixels
    I2g = I2g / np.iinfo(I2g.dtype).max # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    tic = time.time()
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + \
         signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    print('time spent on convolving:%.2fs'%(time.time()-tic))

    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    print(w)
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(It, (It.shape[0],1)) # get b here
            A = np.vstack((Ix, Iy)).T # get A here

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
                u[i,j]=nu[0]
                v[i,j]=nu[1]

            #b = ... # get b here
            #A = ... # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            # nu = ... # get velocity here
            # u[i,j]=nu[0]
            # v[i,j]=nu[1]

    return (u,v)




def tiff_align(pathIn,pathOut):

    file_names = make_file_structure(pathIn)

    tif_info = [[TiffFile(tif) for tif in channel] for channel in file_names]

    check_files(tif_info)



    return tif_info
