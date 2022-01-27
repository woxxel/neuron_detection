
from tifffile import *
import os, tqdm

conn = 'login-gwdg'                 # name of the connection
dataset = 'AlzheimerMice_Hayashi'   # name of the dataset
mouse = '555wt'                     # mouse name
session = 3                         # session number

tmp_folder = './tmp'

obtain = False
if obtain:
    cp_cmd = f"rsync -r \
        --info=progress2 \
        -e ssh login-gwdg:/usr/users/cidbn1/neurodyn/{dataset}/{mouse}/{'Session%.2d'%session}/images/ ./tmp/"

    if not os.path.isdir(tmp_folder):
        os.mkdir(tmp_folder)
    # print(cp_cmd)
    print("Obtaining data from the server... (this may take ~5-10mins depending on your connection)")
    os.system(cp_cmd)


make_stack = True
if make_stack:
    print('now merging files into a single stack...')
    files = os.listdir(tmp_folder)
    fnames = [os.path.join(tmp_folder,f) for f in files if f.lower().endswith('.tif') and not f.lower().startswith('stack')]
    fnames.sort()

    tif = TiffFile(fnames[0])
    d = (len(fnames),) + tif.series[0].shape
    data_type = tif.series[0].dtype

    fileOut = f"stack_m{mouse}_s{'%.02d'%session}.tif"
    pathOut = os.path.join(tmp_folder,fileOut)

    if os.path.exists(pathOut):
        os.remove(pathOut)

    memmap_image = memmap(pathOut,shape=d,dtype=data_type)

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
    
