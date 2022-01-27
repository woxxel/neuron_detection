import os

from file_structures.get_data_from_server import *
from file_structures.make_stack_from_single_tifs import *

def preprocess(dataset='AlzheimerMice_Hayashi',mouse='555wt',sessions=None,target_folder='./data',ssh_alias='login-gwdg'):

    """
        Calls all functions for preprocessing data

        input:
            dataset (string)
                name of the dataset
            mouse (string)
                mouse name
            sessions (list(int)) | None
                specifies which sessions should be preprocessed. if 'None', all Sessions found will be processed
            ssh_alias (string)
                name of the connection

    """

    tmp_folder = './data/tmp'

    if not sessions:
        folder = f"{ssh_alias}:/usr/users/cidbn1/neurodyn/{dataset}/{mouse}"
        fnames = os.listdir(folder)
        fnames = [f for f in fnames if f.startswith('Session') and os.path.isdir(f)]

    path_mouse = os.path.join(target_folder,mouse)

    for session in sessions:
        fname = get_data_from_server(dataset,mouse,session,tmp_folder,ssh_alias)

        # fname = "thy1g7#555_hp_16x1.5x_146um_95v93v_58p_res_lave2_pm"
        # fileOut = f"stack_m{mouse}_s{'%.02d'%session}.tif"
        # pathOut = os.path.join(target_folder,'stack.tif')
        path_session = os.path.join(path_mouse,'Session%.02d'%session)
        path_stack = os.path.join(path_session,'%s.tif'%fname)
        print(path_stack)
        make_stack_from_single_tifs(tmp_folder,path_stack)
