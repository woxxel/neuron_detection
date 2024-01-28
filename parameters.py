system = {
    'ssh_alias': 'transfer-gwdg',    # (string) name of the connection
    'source_folder': '/usr/users/cidbn1/neurodyn/',
    'target_folder': './../data',
    'tmp_folder': './../data/tmp',
    # 'caiman_datadir': '/home/wollex/Data/Science/WolfGroup/PlaceFields/data_pipeline/CaImAn'
}

# preprocess = {
# }

# set up CNMF parameters
CaImAn = {

    ### data
    'fr': 15,
    'decay_time': 0.47,
    'dxy': 512/530.68,

    ### PATCH PARAMS
    'rf': 64,                           # size of patch
    'stride': 16,
    # 'n_processes': 4, #?
    'border_pix': 0,
    'only_init': True, # (what exactly is this?) # whether to run only the initialization

    ### INIT PARAMS
    'K': 100,                           # max number of components
    'gSig': [10, 10],
    'ssub': 2,                          # spatial subsampling during initialization
    'tsub': 5,                          # temporal subsampling during initialization
                                              # expected half size of neurons
    ### SPATIAL PARAMS
    'nb': 2,                            # number of background components per patch

    ### temporal params
    'p': 2,                             # order of AR indicator dynamics

    ### online
    'motion_correct': False,
    'ds_factor': 1,                     # spatial downsampling
    'epochs': 2,                        # number of times to go over the data
    'expected_comps': 1000,
    'init_batch': 200,                  # number of frames for initialization
    'init_method': 'bare',              # initialization method
    'n_refit': 0,                       # additional iterations for computing traces
    'simultaneously': True,
    'sniper_mode': True,                # flag for using CNN for detecting new components
    'test_both': True,                  # use CNN and correlation to test for new components
    'thresh_CNN_noisy': 0.6,            # CNN threshold for candidate components
    'update_freq': 500,                 # update shapes at least once every update_freq steps
    'use_dense': False,
    # 'path_to_model': '~/caiman_data/model/cnn_model_online.h5',

    ### display during online
    'show_movie': False,
    'save_online_movie': False,
    # 'movie_name_online': "test_mp4v.avi"

    ### quality
    'min_SNR': 2.5,
    'SNR_lowest': 1.0,
    'rval_thr': 0.8,
    'rval_lowest': -1,
    'min_cnn_thr': 0.9,
    'cnn_lowest': 0.1,
    'use_cnn': True,

    ### merging

    ### motion
    'pw_rigid': True,
    'shifts_opencv': True,
    'strides': (96,96),
    'overlaps': (48,48),       # overlap between patches (size of patch in pixels: strides+overlaps)
    'num_frames_split': 200,
    'max_deviation_rigid': 12, # maximum deviation allowed for patch with respect to rigid shifts
    'max_shifts': (12,12),     # maximum allowed rigid shift in pixels

    ### ring_CNN
}

matching = {

}
