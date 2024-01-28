import os

from .utils.session_info import *
from preprocessing import *
from image_processing import *

from .utils.parameters import CaImAn as CaImAn_params
from .utils.parameters import system as system_params


def run_pipeline(dataset='AlzheimerMice_Hayashi',mouse='556wt',sessions=None,n_processes=4):

    """
        Calls all functions for preprocessing data

        input:
            dataset (string)
                name of the dataset
            mouse (string)
                mouse name
            sessions (list(int)) | None
                specifies which sessions should be preprocessed. if 'None', all sessions found will be processed
    """

    folder = f"{system_params['source_folder']}/{dataset}/{mouse}"
    if not sessions:
        cmd = f"ssh {system_params['ssh_alias']} 'find {folder}/Session* -type d -maxdepth 0'"
        stdout = os.popen(cmd).read()
        session_names = stdout.split('\n')[:-1] # parse output and remove last linebreak
    else:
        session_names = [f"{folder}/Session%.02d"%s for s in sessions]

    path_to_mouse = os.path.join(system_params['target_folder'],mouse)
    # mouseInfo = MouseInfo(path_to_mouse)

    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    # os.environ['CAIMAN_DATA'] = para.system['caiman_datadir']

    for session_name in session_names:

        path_to_session = os.path.join(path_to_mouse,os.path.split(session_name)[-1])
        os.makedirs(path_to_session, exist_ok=True)
        sessionInfo = SessionInfo(path_to_session)

        print(f"Processing {os.path.split(session_name)[-1]}...")

        if not sessionInfo.status("stacks"):
            ## this should be merged to one function, testing whether stacking is needed
            ## further catch when no data is present
            ## further get behavior data from server
            get_data_from_server(session_name,system_params['tmp_folder'],system_params['ssh_alias'])
            path_to_stacks = make_stack_from_single_tifs(system_params['tmp_folder'],path_to_session,data_type='float16',clean_after_stacking=True)
            sessionInfo.register_new("stacks",path_to_stacks)
        else:
            print(f"\tStack already present: {sessionInfo.get('stacks')}")

        if not sessionInfo.status("motion_correct"):
            path_to_motion_correct = motion_correct(sessionInfo.get("stacks"),para.CaImAn,n_processes=n_processes)
            sessionInfo.register_new("motion_correct",path_to_motion_correct)
        else:
            print(f"\tMotion correction already done: {sessionInfo.get('motion_correct')}")

        if not sessionInfo.status("neuron_detection"):
            # path_out = neuron_detection(sessionInfo.get("motion_correct"),para.CaImAn)
            path_to_neuron_detection = neuron_detection(sessionInfo.get("motion_correct"),para.CaImAn,n_processes=n_processes)
            sessionInfo.register_new("neuron_detection",path_to_neuron_detection)
        else:
            print(f"\tNeuron detection already done: {sessionInfo.get('neuron_detection')}")
