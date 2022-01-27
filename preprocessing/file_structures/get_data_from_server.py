import os

def get_data_from_server(dataset,mouse,session,out_folder,ssh_alias):

    """
        Pulls data from the GWDG servers using rsync

        input:
            dataset (string)
                the name of the dataset you want to access. currently only two are present: "AlzheimerMice_Hayashi" and "Shank2Mice_Hayashi"
            mouse (string)
                name of the mouse in the dataset
            session (int)
                number of the session to be accessed
            out_folder (string)
                path to a folder in which to store the pulled data
            ssh_alias (string)
                an alias for an ssh connection to the GWDG server - make sure to have it properly set up with an account with appropriate rights to access the given filestructures

        returns:
            nothing
    """
    cp_cmd = f"rsync -r \
        --info=progress2 \
        -e ssh {ssh_alias}:/usr/users/cidbn1/neurodyn/{dataset}/{mouse}/{'Session%.2d'%session}/images/ ./tmp/"

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    print("Obtaining data from the server... (this may take ~5-10mins depending on your connection)")
    os.system(cp_cmd)

    fnames = os.listdir(out_folder)
    fnames = [f for f in fnames if not f.startswith('.')]
    fnames.sort()
    return fnames[0][:-8]   # cuts off extension + last 4 digits, indicating framenumber
