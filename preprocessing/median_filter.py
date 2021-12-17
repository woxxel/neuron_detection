"""
    Created by Alexander Schmidt on 14.Dec.2021
    last changed on 15.Dec.2021
"""

from tifffile import *
from scipy.ndimage import median_filter
import multiprocessing as mp
import itertools, tqdm

def median_filter_slice(input):

    """
        Runs a median_filter on a slice of data read from a tiff stack and
        written into a memmapped tiffstack

        receives:
            tuple (int) input
                4 indices controlling the section of the memmapped array to
                write to (start, end) and the section of the input file needed
                to run the median_filter (slightly larger, to account for the
                z-size of the filter)

        access to (via global variables set in init function):
            dict lock:
                containing different types of locks to allow save reading
                and writing
            dict pars:
                containing parameters ('filterSz','chunkSz'), the name of the file
                to read from ('pathIn') and the memmapped tiff structure to write
                to ('memmap')

    """

    # read and define indexes from input
    start, end, startSlice, endSlice = input
    startOffset = start-startSlice
    endOffset = endSlice-end

    # read in the tiff file
    lock['read'].acquire()
    im = imread(pars['pathIn'],key=range(startSlice,endSlice))
    lock['read'].release()

    # run median filter
    im_med = median_filter(im,size=pars['filterSz'])

    # write results to according slice in memmap
    lock['write'].acquire()
    pars['memmap'][start:end,...] = im_med[startOffset:startOffset + pars['chunkSz'] - endOffset + 1,...]
    pars['memmap'].flush()
    lock['write'].release()


def init(l,inPars):
    """
        initializer function for setting global variables of multiprocessing

        receives
            dict l
                dictionary containing locks
            dict inPars
                dictionary containing parameters
    """
    global lock
    lock = {}
    for key in l:
        lock[key] = l[key]

    global pars
    pars = {}
    for key in inPars:
        pars[key] = inPars[key]


def chunk(dataSz,chunksize,offset):

    """
        creates the start- and end-indices to define read- and write-sections
        of the tiff files

        receives:
            int dataSz:
                the overall length of the dataset (number of frames)
            int chunksize:
                the number of frames to be processed by a single process
            int offset:
                the size of the median_filter footprint into z-direction

        returns:
            iterator tuple(int)
                indices for reading and writing tiff files (see median_filter_slice)
    """
    intervals = list(range(0,dataSz,chunksize)) + [dataSz]
    for start, end in zip(intervals[:-1],intervals[1:]):
        startSlice = max(0,start-offset)
        endSlice = min(dataSz,end+offset)
        yield (start,end,startSlice,endSlice)


def medfilt(pathIn,pathOut,pars):

    """
        Function for running a median filter on the input image in a somewhat
        memory efficient way

        receives:
            str pathIn
                path to the input image (in tiff format)
            str pathOut
                path to the output image (also tiff format)
            dict parameter
                parameter object containing
                    tuple filterSz: tuple of dimension sizes (n,m,p)
                    int chunkSz: number of timesteps to be processed at a time

    """

    # obtain general information on the tiff-file
    tif = TiffFile(pathIn)
    d = tif.series[0].shape
    data_type = tif.series[0].dtype

    filterRange_t = (pars['filterSz'][0]-1)//2

    memmap_image = memmap(pathOut,shape=d,dtype=data_type)

    pool = mp.Pool(initializer=init, initargs=({'read':mp.Lock(),'write':mp.Lock()},{'filterSz': pars['filterSz'],'chunkSz':pars['chunkSz'],'pathIn':pathIn,'memmap':memmap_image}))

    # wrapper for displaying the progress of the program
    for _ in tqdm.tqdm(pool.imap_unordered(median_filter_slice, chunk(d[0],chunksize=pars['chunkSz'],offset=filterRange_t))):
        pass

    del memmap_image
