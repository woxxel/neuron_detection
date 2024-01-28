import os

def get_data_from_server(path_source,path_target,ssh_conn,show_progress=True):

    """
        Pulls data from the GWDG servers using rsync

        input:
            path_source (int)
                path to the session folder to be accessed
            path_target (string)
                path to a folder in which to store the pulled data
            ssh_alias (string)
                an alias for an ssh connection to the GWDG server - make sure to have it properly set up with an account with appropriate rights to access the given filestructures

        returns:
            nothing
    """

    cp_cmd = f"rsync -r "
    if show_progress:
        cp_cmd += f"--info=progress2 "
    cp_cmd += f"-e ssh {ssh_conn}:{path_source}/images/ {path_target}"

    if not os.path.isdir(path_target):
        os.mkdir(path_target)
    
    print(f"Obtaining data from {ssh_conn}:{path_source}... (this may take ~5-10mins depending on your connection)")
    os.system(cp_cmd)
