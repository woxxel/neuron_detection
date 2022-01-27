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

        if ext=='.tif':
            self.file = TiffFile(path)
            self.shape = self.file.series[0].shape
            self.dtype = self.file.series[0].dtype
            self.getSlice = lambda t: self.file.pages[t].asarray()
        elif ext=='.mmap':

            assert dim, 'dim and dtype needs to be specified for mmap'
            self.shape = dim
            self.dtype = np.float32
            self.file = np.memmap(path,mode='r',shape=(self.shape[1]*self.shape[2],self.shape[0]),dtype=self.dtype,order='F')   # for files created by caimans NormCorr
            self.getSlice = lambda t: np.reshape(self.file[:,t],(512,512),'F')
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
    interval = 1./f*1000    # ms, time between animation frames

    ## definition of interaction functions
    def update_slider(val):
        global is_manual
        is_manual=True
        update(int(val))

    def update(val):
        # update curve
        plt.suptitle('t=%.2fs'%(val/f))
        for i,(vid,img) in enumerate(zip(vids,imgs)):
            img.set_data(vid.getSlice(val))
        if nVideos>1:
            imgs[-1].set_data(vids[0].getSlice(val)-vids[1].getSlice(val))
        # redraw canvas while idle
        fig.canvas.draw_idle()

    def update_plot(num):
        global is_manual
        if is_manual:
            return imgs, # don't change

        val = (samp.val + 1) % samp.valmax
        samp.set_val(val)
        is_manual = False # the above line called update_slider, so we need to reset this
        return imgs,

    def on_click(event):
        # Check where the click happened
        (xm,ym),(xM,yM) = samp.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # user clicked somewhere else on canvas = unpause
            global is_manual
            is_manual=False

    ## preparing figure
    fig, axes = plt.subplots(1,nVideos+1,figsize=(12,6))
    # if nVideos==1:
        # axes = [axes]

    axamp = plt.axes([0.25, .03, 0.50, 0.02])
    samp = Slider(axamp, 'time', 0, dims[0], valinit=0)

    # call update function on slider value change
    samp.on_changed(update_slider)

    imgs = []
    for vid,ax in zip(vids,axes):
        imgs.append(ax.imshow(vid.getSlice(0)))

    if nVideos>1:
        imgs.append(axes[-1].imshow(vids[0].getSlice(0)-vids[1].getSlice(0)))

    fig.canvas.mpl_connect('button_press_event', on_click)

    anim = FuncAnimation(fig, update_plot, frames=dims[0], blit=False, interval=interval)
    plt.show()
