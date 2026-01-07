import os, json, glob

class SessionInfo:
    """
        Class to help monitor the status of processing and links to according
        files, etc for each session
    """

    steps = ['stacks','motion_correct','neuron_detection','neuron_detection_refinement','PC_detection']

    def __init__(self,path_to_session):

        path_to_info = os.path.join(path_to_session,'session_info.json')

        if os.path.exists(path_to_info):
            self.get_info(path_to_info)
        else:
            self.create_new(path_to_session)


    def create_new(self,path_to_session):

        self.info = {
            'general': {
                #information such as reward/gate position, wait time, track length, ...
                'reward': [50.,70.],
                'gate': [],
                'fname': None
            },
            'paths': {
                'to_session': os.path.abspath(path_to_session),
                'to_info': os.path.abspath(os.path.join(path_to_session,'session_info.json')),
            },
            'progress': {},
        }

        self.get_behavior()

        self.update_status()

    def get_behavior(self):

        fnames = os.listdir(self.info['paths']['to_session'])
        fnames = [f for f in fnames if os.path.splitext(f) in ['.mat']]

        fname = choose_file(fnames)
        self.info['paths']['to_behavior'] = fname


    def update_status(self):
        for step in self.steps:
            self.info['paths'][f"to_{step}"], self.info['progress'][step] = self.check_status(step)
        self.commit_info()

    def check_status(self,step):

        status = False
        path = None

        if f"to_{step}" in self.info['paths'] and self.info['paths'][f"to_{step}"]:
            return self.info['paths'][f"to_{step}"], self.info['progress'][step]
        else:
            if step=='stacks':

                fnames = os.listdir(self.info['paths']['to_session'])
                fnames = [f for f in fnames if f.endswith('.tif')]

                fname = choose_file(fnames,f'### {step} ###')

                if fname:
                    self.info['general']['fname'] = os.path.splitext(fname)[0]
                    path = os.path.join(self.info['paths']['to_session'],fname)
                    status = True
            else:
                if self.info['general']['fname']:
                    path_tmp = self.get_step_path(step,default=True)
                    fnames = glob.glob(os.path.join(self.info['paths']['to_session'],path_tmp+'*'))
                    fname = choose_file(fnames,f'### {step} ###')
                    if fname:
                        path = fname
                        status = True

            return path, status

    def get_step_path(self,step,default=False):

        """
            returns the path to the process-step file according to naming convention.
            if 'default', the path is returned without an extension or dir
        """
        if default:
            return self.info['general']['fname'] + '_' + step
        else:
            return self.info['paths'][f"to_{step}"]


    def register_new(self,step,path=None):

        # if path:
            # fname,ext = os.path.splitext(os.path.split(path)[-1])
        if not path:
            fnames = os.listdir(self.info['paths']['to_session'])
            step_fnames = [os.path.split(fn)[-1] for fn in self.info['paths'].values() if fn]
            fnames = [f for f in fnames if not f in step_fnames]
            fname = choose_file(fnames,f'### {step} ###')

            path = os.path.join(self.info['paths']['to_session'],fname)
            # fname,ext = os.path.splitext(fname)

        self.info['paths'][f"to_{step}"] = os.path.abspath(path)
        self.info['progress'][step] = True

        self.commit_info()

    def remove_step(self,step):

        path = self.info['paths'][f'to_{step}']
        print(path)

        if path and os.path.exists(path):
            os.remove(path)
        self.info['paths'][f'to_{step}'] = None
        self.info['progress'][step] = False
        self.commit_info()

    def commit_info(self):
        with open(self.info['paths']['to_info'], 'w') as outfile:
            json.dump(self.info,outfile)

    def get_info(self,path_to_info):
        with open(path_to_info, 'r') as infile:
            self.info = json.load(infile)

    def get(self,step):
        return self.info['paths'][f"to_{step}"]

    def status(self,step):
        return self.info['progress'][step]



def choose_file(fnames,title=''):

    """
        function to let user choose a file from a list of files
    """

    if not len(fnames):
        return None
    elif len(fnames) == 1:
        return(fnames[0])
    elif len(fnames) > 1:

        print(title)
        print(f"\t0: None")
        [print(f"\t{i+1}: {f}") for (i,f) in enumerate(fnames)]
        idx = int(input(f"Choose which file is the proper stack containing recordings (0-{len(fnames)}): "))-1
        return fnames[idx]
