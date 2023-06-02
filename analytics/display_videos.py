"""
    Created by Alexander Schmidt on 15.Dec.2021
    last changed on 15.Dec.2021
"""

from tifffile import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np
import os, cv2

## interaction plot taken from https://stackoverflow.com/questions/46325447/animated-interactive-plot-using-matplotlib on 15.Dec.2021

class tif_mov:

    shape = None
    dtype = None
    file = None
    getSlice = None

    def __init__(self,path,dim=None,dtype=None):

        ext = os.path.splitext(path)[-1]
        print(path)
        print(ext)

        if ext=='.tif':
            self.file = TiffFile(path)
            self.shape = self.file.series[0].shape
            self.dtype = self.file.series[0].dtype
            self.getSlice = lambda t: self.file.pages[t].asarray().astype('float32') ## for some reason, cv2 doesn't like float16 images...
        elif ext=='.mmap':

            assert dim, 'dim and dtype needs to be specified for mmap'
            self.shape = dim
            self.dtype = np.float32
            self.file = np.memmap(path,mode='r',shape=(self.shape[1]*self.shape[2],self.shape[0]),dtype=self.dtype,order='F')   # for files created by caimans NormCorr
            # self.file = self.file.copy()
            # self.file = cv2.cvtColor(self.file, cv2.COLOR_BGR2RGB)
            self.getSlice = lambda t: np.reshape(self.file[:,t],(512,512),'F').copy()
        else:
            print('not yet available')

def display_videos(paths,f=15):
    """
        function for displaying up to 4 videos simultaneously (meant for comparing
        different stages of preprocessing)

        receives:
            list(str) paths
                paths to videos to be displayed
            float f
                frequency at which to play the video
    """

    ## reading in video(s) metadata
    nVideos = len(paths)

    vids = [tif_mov(path,(8989,512,512)) for path in paths]

    # return vids
    ## brief sanity checks
    dims = vids[0].shape
    for vid in vids:
        assert vid.shape == dims, "videos do not have the same size";

    # Animation controls
    global is_manual
    is_manual = False       # True if user has taken control of the animation
    global interval
    interval = 1./f*1000    # ms, time between animation frames

    windowName = "Video"
    cv2.namedWindow(windowName)

    ## definition of interaction functions
    def update_slider(val):
        create_frame(vids,val)

    def toggle_play(event,*args):
        if event == cv2.EVENT_LBUTTONDOWN:
            global is_manual
            is_manual = not is_manual

    def update_frequency(val):
        global interval
        interval = 1./val*1000

    def create_frame(vids,val):
        frame = np.concatenate([vid.getSlice(val) for vid in vids],axis=1)
        val_freq = cv2.getTrackbarPos('frequency',windowName)
        cv2.putText(frame, 't=%.2fs @%dHz'%(val/15.,val_freq), (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,0,0), thickness=2) ## color not working...
        cv2.imshow(windowName,frame)

    cv2.createTrackbar('slider', windowName, 0, dims[0]-1, update_slider)
    cv2.createTrackbar('frequency', windowName, 0, 100, update_frequency)
    cv2.setMouseCallback(windowName, toggle_play)
    cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)

    while True:

        if not is_manual:
            val = cv2.getTrackbarPos('slider',windowName)
            val = (val + 1) % dims[0]
            create_frame(vids,val)

            cv2.setTrackbarPos('slider',windowName,val)

        cv2.waitKey(int(interval))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) <1:
            break

    cv2.destroyAllWindows()
