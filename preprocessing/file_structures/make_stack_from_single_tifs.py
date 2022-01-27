from tifffile import *
import os, tqdm

def make_stack_from_single_tifs(in_folder,out_path):

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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print('now merging files into a single stack...')
    files = os.listdir(in_folder)
    fnames = [os.path.join(in_folder,f) for f in files if f.lower().endswith('.tif')]
    fnames.sort()

    tif = TiffFile(fnames[0])
    d = (len(fnames),) + tif.series[0].shape
    data_type = tif.series[0].dtype

    if os.path.exists(out_path):
        os.remove(out_path)

    memmap_image = memmap(out_path,shape=d,dtype=data_type)

    for f,fname in tqdm.tqdm(enumerate(fnames)):
        tif = TiffFile(fname)
        assert tif.series[0].shape==d[1:], f"Dimensions of images do not match: Occured @ {fname}"
        memmap_image[f,...] = tif.pages[0].asarray()

        # commit changes to file every so often to avoid memory overuse
        if (not ((f+1)%2000) or (f+1)==d[-1]):
            memmap_image.flush()

    del memmap_image

    print("Removing single files...")
    for fname in tqdm.tqdm(fnames):
        os.remove(fname)
