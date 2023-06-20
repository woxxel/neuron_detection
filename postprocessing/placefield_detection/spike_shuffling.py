### function to shuffle spike trains according to Gansel, 2012
###
### inputs:
###         mode  - specify mode for shuffling
###               'shift'       - shift spike train by a fixed offset (default)
###                       provide values as (mode,shuffle_peaks,spike_train)
###               'dither'      - dither each spike by an independently drawn random value (max = w)
###                       provide values as (mode,shuffle_peaks,spike_times,spikes,T,ISI,w,shuffle_spikes)
###               'dithershift' - combination of 'shift' and 'dither' method
###                       provide values as (mode,shuffle_peaks,spike_times,spikes,T,ISI,w,shuffle_spikes)
###               'dsr'         - (dither-shift-reorder), same as 'dithershift' but with random reordering of consecutive ISIs < w (?)
###                       provide values as (mode,shuffle_peaks,spike_times,spikes,T,ISI,w,shuffle_spikes)
###
###         shuffle_peaks  - boolean: should assignment "spikes" to "spike_times" be shuffled?
###
###         spike_train - spike train as binary array (should be replaced by ISI & T
###
###         spike_times - frames at which spikes happen
###
###         spikes      - number of spikes happening at times "spike_times"
###
###         T     - length of the overall recording (= maximum value for new spike time)
###
###         ISI   - InterSpike Intervalls of the spike train
###
###         w     - maximum dithering (~1/(2*rate)?)
###
### ouputs:
###         new_spike_train - shuffled spike train
###
###   written by A.Schmidt, last reviewed on January, 22nd, 2020

import numpy as np
import matplotlib.pyplot as plt
#from numba import jit

#@jit
def shuffling(mode,shuffle_peaks,**varin):
  
  if mode == 'shift':
    
    [new_spike_train,tmp] = shift_spikes(varin['spike_train'])
    if shuffle_peaks:
      spike_times = find(new_spike_train)
      spikes = new_spike_train(spike_times)
      new_spike_train[spike_times] = spikes[np.random.permutation(len(spike_times))]        ## shuffle spike numbers
    
  elif mode == 'dither':
    
    assert len(args)>=2, "You did not provide enough input. Please check the function description for further information."
    [spike_times,spikes,T,ISI,w] = get_input_dither(varin);
    
    new_spike_train = dither_spikes(spike_times,spikes,T,ISI,w,shuffle_peaks);
    
  elif mode == 'dithershift':
    
    assert len(args)>=4, "You did not provide enough input. Please check the function description for further information."
    [spike_times,spikes,T,ISI,w] = get_input_dither(varin);
    
    new_spike_train = dither_spikes(spike_times,spikes,T,ISI,w,shuffle_peaks);
    [new_spike_train,shift] = shift_spikes(new_spike_train);
    
  elif mode == 'dsr':
  
    print('not yet implemented')
    new_spike_train = NaN;
    
  
  #plt = false;
  #if plt && strcmp(mode,'dithershift')
    
    #if ~exist('spike_train','var')
      #spike_train = zeros(1,T);
      #spike_train(spike_times) = spikes;
    #end
    #ISI = get_ISI(spike_train);
    #newISI = get_ISI(new_spike_train);
    
    #figure('position',[500 500 1200 900])
    #subplot(3,1,1)
    #plot(spike_train)
    #subplot(3,1,2)
    #plot(new_spike_train)
    #title('new spike train')
    
    #subplot(3,2,5)
    #hold on
    #histogram(log10(ISI),linspace(-2,2,51),'FaceColor','b')
    #histogram(log10(newISI),linspace(-2,2,51),'FaceColor','r')
    #hold off
    
    #waitforbuttonpress;
  #end
  return new_spike_train



def shift_spikes(spike_train):
  
  shift = np.random.randint(np.max([1,len(spike_train)]))
  new_spike_train = np.concatenate([spike_train[shift:],spike_train[:shift]])    ## shift spike train
  return new_spike_train,shift


def get_input_dither(argin):
  
  if len(argin['w']) == 1:
    spike_train = argin['spike_train']
    spike_times = np.where(spike_train)[0]
    spikes = spike_train[spike_times]
    T = len(spike_train)
    ISI = np.diff(spike_times)
    
  else:
    spike_times = argin['spike_times']
    spikes = argin['spikes']
    T = argin['T']
    ISI = argin['ISI']
    
  return spike_times,spikes,T,ISI,argin['w']


def dither_spikes(spike_times,spikes,T,ISI,w,shuffle_peaks):
  
  nspike_times = len(spike_times);
  
  dither = np.min([ISI-1,2*w])/2;
  
  r = 2*(rand(1,len(ISI)-1)-0.5);
  
  for i in range(1,len(ISI)):   ## probability of being left or right of initial spike should be equal! (otherwise, it destroys bursts!)
    print('i: %d',i)
    spike_times[i] = spike_times[i] + min(0,r[i-1])*ISI[i-1] + max(0,r[i-1])*ISI[i];
  
  spike_times = round(spike_times)
  
  if shuffle_peaks:
    print(nspike_times)
    print(nspike_times.shape)
    print('watch out: permutation only works along first dimension ... proper shape?')
    spikes = spikes[np.random.permutation(nspike_times)]
  
  new_spike_train = np.zeros(T)
  for i in range(nspike_times):
    t = spike_times[i]
    new_spike_train[t] = new_spike_train[t] + spikes[i]
  
  return new_spike_train



def get_ISI(spike_train):
  
  ## this part effectively splits up spike bursts (single event with multiple spikes to multiple events with single spikes)
  spike_times = find(spike_train);
  idx_old = 1;
  new_spike_times = [];
  print(np.where(spike_train>1))
  print(np.where(spike_train>1)[0])
  for t in np.where(spike_train>1)[0]:
    idx_new = np.where(spike_times==t)[0]
    nspikes = spike_train[t]
    
    new_spike_times = np.append([new_spike_times,spike_times[idx_old:idx_new],t+np.linspace(0,1-1/nspikes,nspikes)]);
    #idx_old = idx_new+1;
  
  new_spike_times = np.append([new_spike_times,spike_times[idx_old:]]);
  return np.diff(new_spike_times);
  
