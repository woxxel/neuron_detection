# set up CNMF parameters
params_motion = {
    "max_shifts": (
        12,
        12,
    ),  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    "strides": (96, 96),  # create a new patch every x pixels for pw-rigid correction
    "overlaps": (24, 24),  # overlap between patches (size of patch strides+overlaps)
    "max_deviation_rigid": 6,  # maximum deviation allowed for patch with respect to rigid shifts
    "pw_rigid": True,
    # "shifts_opencv": True,
}

param_init = {
    "nb": 2,
    "gSig": [10, 10],  # radius of average neurons (in pixels)
    "gSiz": (21, 21),  # half-size of bounding box for each neuron
    # "ssub": 2,  # spatial subsampling during initialization
    # "ssub_B": 2,  # spatial subsampling for background during intialization
    # "tsub": 5,  # temporal subsampling during initialization
}

param_quality = {
    "min_SNR": 2.5,
    "SNR_lowest": 1.0,
    "rval_thr": 0.8,
    "rval_lowest": -1,
    "min_cnn_thr": 0.9,
    "cnn_lowest": 0.1,
    "use_cnn": True,
}

param_online = {
    "motion_correct": False,
    "epochs": 1,  # number of times to go over the data
    "normalize": True,
    "init_batch": 1000,  # number of frames for initialization
    "init_method": "cnmf",  # initialization method
    "dist_shape_update": True,
    "update_freq": 1000,  # update shapes at least once every update_freq steps
    "use_dense": False,
    "merge_thr": 0.65,
    # "p": 2,
    # "ds_factor": 1,  # spatial downsampling
    # "stop_detection": True,
    # "expected_comps": 1000,
    # "n_refit": 0,  # additional iterations for computing traces
    # "simultaneously": False,
    # "sniper_mode": False,  # flag for using CNN for detecting new components
    # "test_both": False,  # use CNN and correlation to test for new components
    # "thresh_CNN_noisy": 0.6,  # CNN threshold for candidate components
    # further parameters
}

CaImAn = {
    ### data
    "fr": 30.05,
    "decay_time": 1.5,
    "dxy": 512 / 530.68,
    ### PATCH PARAMS
    "rf": 48,  # half size of each patch
    "stride": 8,
    "border_pix": 0,
    ### FURTHER PARAMS
    **param_init,
    **param_quality,
    **param_online,
    **params_motion,
    "show_movie": False,
    "save_online_movie": False,
    # 'n_processes': 4, #?
    # "remove_very_bad_comps": True,
    # "only_init": True,  # (what exactly is this?) # whether to run only the initialization
    ### INIT PARAMS
    # "K": 100,  # max number of components
    # expected half size of neurons
    ### SPATIAL PARAMS
    # "nb": 2,  # number of background components per patch
    ### temporal params
    # "p": 2,  # order of AR indicator dynamics
    ### online
    # "use_corr_img": True,
    # 'path_to_model': '~/caiman_data/model/cnn_model_online.h5',
    ### display during online
    # 'movie_name_online': "test_mp4v.avi"
    ### quality
    ### merging
}
