import os

def get_data_from_server(data_path,out_folder,ssh_conn,show_progress=True):

    """
        Pulls data from the GWDG servers using rsync

        input:
            data_path (int)
                path to the session folder to be accessed
            out_folder (string)
                path to a folder in which to store the pulled data
            ssh_alias (string)
                an alias for an ssh connection to the GWDG server - make sure to have it properly set up with an account with appropriate rights to access the given filestructures

        returns:
            nothing
    """

    cp_cmd = f"rsync -r "
    if show_progress:
        cp_cmd += f"--info=progress2 "
    cp_cmd += f"-e ssh {ssh_conn}:{data_path}/images/ {out_folder}"

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    print(f"Obtaining data from {ssh_conn}:{data_path}... (this may take ~5-10mins depending on your connection)")
    os.system(cp_cmd)
