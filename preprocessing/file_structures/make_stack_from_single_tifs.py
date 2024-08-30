from tifffile import *
import os, tqdm
import shutil
import numpy as np

import h5py

def make_stack_from_single_tifs(in_folder,out_folder,file_name=None,T_max=np.inf,data_type=None,normalize=True,clean_after_stacking=False):

    """
        Creates a stack of tif-files at path 'out_path' from all tif_files found in 'in_folder'.

        TODO:
            [ ] extend to accepting tiff-files with several layers
            [ ] extend to accepting a list of in_folders

        input:
            in_folder (string)
                path of the folder from which to process tiff-files
            out_path (string)
                path where to put the created tiff-stack

        returns:
            nothing
    """
    
    if isinstance(in_folder,list):
        assert (file_name is not None), "Please provide a file_name for the resulting stack, when using a list of folders"
        fnames = []
        for folder in in_folder:
            fnames_tmp = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith('.tif') and not f.startswith('.')]
            fnames_tmp.sort()
            fnames.extend(fnames_tmp)
        fname = file_name
    else:
        fnames = [os.path.join(in_folder,f) for f in os.listdir(in_folder) if f.lower().endswith('.tif') and not f.startswith('.')]
        fnames.sort()

        fname = file_name if file_name else os.path.splitext(fnames[0])[0][:-4]     ## assuming single recording images, where the last 4 digits indicate framenumber

    os.makedirs(out_folder,exist_ok=True)
    out_path = os.path.join(out_folder,os.path.split(fname)[-1]+'.tif')

    ## get information on size of resulting stack and check for data consistency
    tif = TiffFile(fnames[0])
    d = tif.pages[0].shape
    data_type_in = tif.pages[0].dtype

    T_total = 0
    for fname in tqdm.tqdm(fnames):
        tif = TiffFile(fname)
        assert tif.pages[0].shape==d, f"Dimensions of the imaging data do not agree! {d} vs {tif.pages[0].shape}"
        assert tif.pages[0].dtype==data_type_in, f"Data type of the imaging data do not agree! {data_type_in} vs {tif.pages[0].dtype}"

        T_total += len(tif.pages)

    T = min(T_max,T_total)

    # if file exists already, remove first to avoid conflicts (shouldn't be necessary)
    if os.path.exists(out_path):
        os.remove(out_path)

    # prepare parameters of memmapped file
    d = (T,) + d
    data_type = data_type if data_type else data_type_in
    print(f"Now merging files into a single stack {out_path} with d={d} frames ({T_total} found) and datatype {data_type}")

    if normalize:
        if str(data_type_in).startswith('float'):
            normalize = False
        else:
            max_val = np.iinfo(data_type_in).max
    print(data_type,max_val)

    memmap_image = memmap(out_path,shape=d,dtype=data_type)

    # writing data to memmapped tif-file
    t = 0
    for fname in tqdm.tqdm(fnames):
        tif = TiffFile(fname)
        for page in tif.pages:
            if normalize:
                memmap_image[t,...] = page.asarray()/max_val
            else:
                memmap_image[t,...] = page.asarray()
            t += 1

            # commit changes to file every so often to avoid memory overuse
            if (not ((t+1)%2000) or (t+1)==T):
                memmap_image.flush()

            if ((t+1)==T):
                break
        if ((t+1)==T):
            break

    del memmap_image

    if clean_after_stacking:
        print("Removing tmp files...")
        try:
            shutil.rmtree(in_folder)
        except:
            pass
    return out_path




def make_stack_from_h5(in_file,out_folder,T_max=np.inf,data_type=None,normalize=True,clean_after_stacking=False):

    """
        Creates a stack of tif-files at path 'out_path' from all tif_files found in 'in_folder'.

        TODO:
            [ ] extend to accepting tiff-files with several layers
            [ ] extend to accepting a list of in_folders

        input:
            in_folder (string)
                path of the folder from which to process tiff-files
            out_path (string)
                path where to put the created tiff-stack

        returns:
            nothing
    """
    
    f = h5py.File(in_file,'r')
    dataset_key = list(f.keys())[0]
    data = f[dataset_key]

    basename = os.path.basename(in_file)
    fname = os.path.splitext(basename)[0] + '.tif'
    out_path = os.path.join(out_folder,fname)

    dims = data.shape
    T_total = dims[0]
    # d = dims[1:]

    data_type_in = data.dtype

    os.makedirs(out_folder,exist_ok=True)
    
    T = min(T_max,T_total)
    
    data_type = data_type if data_type else data_type_in
    print(f"Now writing input .h5 file into a tiff stack {out_path} with d={dims} frames ({T_total} found) and datatype {data_type}")

    if normalize:
        if str(data_type_in).startswith('float'):
            normalize = False
        else:
            max_val = np.iinfo(data_type_in).max
    print(data_type,max_val)

    memmap_image = memmap(out_path,shape=dims,dtype=data_type)

    # writing data to memmapped tif-file
    for t in tqdm.tqdm(range(T)):
        if normalize:
            memmap_image[t,...] = data[t,...]/max_val
        else:
            memmap_image[t,...] = data[t,...]

        # commit changes to file every so often to avoid memory overuse
        if (not ((t+1)%2000) or (t+1)==T):
            memmap_image.flush()


    del memmap_image

    if clean_after_stacking:
        print("Removing tmp files...")
        os.remove(in_file)
        # try:
        #     shutil.rmtree(in_folder)
        # except:
        #     pass
    return out_path