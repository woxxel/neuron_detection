import os
import numpy as np
from scipy.ndimage import binary_closing


def get_file_path(ssh_alias,session_path):

    host_cmd = f"ssh {ssh_alias} " if ssh_alias else ""
    try:
        bh_path = os.popen(f"{host_cmd}find {session_path}/a* -maxdepth 0 -type f").read()
        assert len(bh_path)>0
    except:
        bh_path = os.popen(f"{host_cmd}find {session_path}/crop* -maxdepth 0 -type f").read()
    if not len(bh_path):
        raise Exception('No behavior file found')
    bh_path = bh_path[:-1]
    _, ext = os.path.splitext(bh_path)

    if ssh_alias:
        tmp_path = f"./tmp/behavior{ext}"
        cmd = f"scp {ssh_alias}:{bh_path} {tmp_path} > /dev/null"
        os.system(cmd)
        bh_path = tmp_path

    return bh_path,ext


def is_rw_col(data,time):
    """
        behavior data is not stored homogeneously across all experiments, but may contain
        reward data in different columns (3 or 8). This function attempts to identify the
        reward column by 3 criteria:
            - number of reward events (maximum of one per trial):
                assuming around 10 secs per trial, the number of reward events shouldnt surpass T/10
            - time between rewards:
                this should not go much beneath 10 secs (here min 5 secs)
            - reward deliveries per trial
                there should be at most 3 reward deliveries per reward event
    """

    rw_max_num = (time[-1]/10.)*1.1

    rw_tmp = np.array(data)>0.5
    rw_tmp_closed = binary_closing(rw_tmp,np.ones(100))
    rw_start = np.where(np.diff(rw_tmp.astype('int'))==1)[0]
    rw_start_closed = np.where(np.diff(rw_tmp_closed.astype('int'))==1)[0]

    dTime = np.array(time)[rw_start_closed]
    ratio = len(rw_start)/len(rw_start_closed)

    return (len(rw_start_closed) < rw_max_num) and (np.min(dTime)>5) and ratio < 4


def apply_to_stretched_out(fun,loc_in,loc_buffer=2):
    """
        method to apply some function to location data which is not interrupted by teleports.
        this avoid weird behavior when averaging, taking median, etc
    """

    loc = loc_in.copy()
    max_val = np.max(loc)
    min_val = np.min(loc)

    total_distance = max_val - min_val

    ## identify and remove teleports
    idx_teleport = np.where(np.diff(loc)<-total_distance/2)[0]
    for idx_tp in idx_teleport:
        loc[idx_tp+1:] = loc[idx_tp+1:] + total_distance + loc_buffer

    ## apply function
    loc = fun(loc)

    ## reintroduce teleports
    loc = np.mod(loc-min_val+loc_buffer/2,total_distance+loc_buffer)+min_val

    return loc
