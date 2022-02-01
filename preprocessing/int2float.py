from tifffile import *
import numpy as np
import tqdm, os


def tiff_int2float(path_in,to_type='float32'):

    tif = TiffFile(path_in)
    d =  tif.series[0].shape
    data_type = tif.series[0].dtype

    max_val = np.iinfo(data_type).max
    min_val = np.iinfo(data_type).min

    print(min_val,max_val)
    print(data_type)
    print(d)
    fileparts = os.path.split(path_in)
    path_out = os.path.join(fileparts[0],to_type+'_'+fileparts[1])
    print(path_out)
    memmap_image = memmap(path_out,shape=d,dtype=to_type)
    for t in tqdm.tqdm(range(d[0])):
        memmap_image[t,...] = (tif.pages[t].asarray()/max_val

    memmap_image.flush()
    del memmap_image
    return tif
