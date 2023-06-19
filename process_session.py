import os, sys, shutil

from preprocessing.file_structures.make_stack_from_single_tifs import *
from image_processing import *
import parameters as para

### dont forget putting caiman_data/model/* in home folder!

## obtain input parameters
_, dataset, mouse, path_to_session_on_cloud, cpus = sys.argv
n_processes = int(cpus)
print(sys.argv)

session_name = os.path.split(path_to_session_on_cloud)[-1]
path_to_session_on_home = os.path.join('/scratch/users',os.environ['USER'],'data',dataset,mouse,session_name)
if os.path.isfile(os.path.join(path_to_session_on_home,'OnACID_results.hdf5')):
    print(f"\n\t +++ {session_name} of mouse {mouse} has already been processed - skipping +++ \n\n")
    exit()

print(f"\n\t +++ Now processing session {session_name} of mouse {mouse} +++ \n\n")

## run processing algorithms
path_to_stacks = make_stack_from_single_tifs(
    os.path.join(path_to_session_on_cloud,'images/'),
    os.environ['TMP_LOCAL'],
    data_type='float16',clean_after_stacking=False
)

path_to_motion_correct = motion_correct(path_to_stacks,para.CaImAn,n_processes=n_processes)

path_to_neuron_detection = neuron_detection(path_to_motion_correct,para.CaImAn,n_processes=n_processes)


## copying detection results over to home folder
resultFile_name = os.path.split(path_to_neuron_detection)[-1]

print(f"Copied results-file {resultFile_name} to {path_to_session_on_home} and finish!")
os.makedirs(path_to_session_on_home,exist_ok=True)
shutil.copyfile(path_to_neuron_detection,os.path.join(path_to_session_on_home,resultFile_name))
