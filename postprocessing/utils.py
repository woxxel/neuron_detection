''' contains various useful program snippets:

  get_nFolder   get number of folders in path
  pathcat       attach strings to create proper paths
  _hsm          half sampling mode to obtain baseline


'''

import pickle, cmath, time, cv2, h5py
import scipy as sp
import scipy.stats as sstats
from scipy import signal, cluster
import numpy as np
import matplotlib.pyplot as plt
from fastcluster import linkage
from scipy.spatial.distance import squareform

def find_modes(data,axis=None,sort_it=True):

  if axis is not None:
    def fnc(x): return find_modes(x,sort_it=sort_it)
    dataMode = np.apply_along_axis(fnc, axis, data)
  else:
    data = data[np.isfinite(data)]
    if sort_it:
      data = np.sort(data)

    dataMode = _hsm(data)

  return dataMode



def _hsm(data):
  ### adapted from caiman
  ### Robust estimator of the mode of a data set using the half-sample mode.
  ### versionadded: 1.0.3

  ### Create the function that we can use for the half-sample mode
  ### sorting done as first step, if not specified else

  if data.size == 0:
    return np.NaN
  if np.all(data == data[0]):
    return data[0]

  if data.size == 1:
    return data[0]
  elif data.size == 2:
    return data.mean()
  elif data.size == 3:
    i1 = data[1] - data[0]
    i2 = data[2] - data[1]
    if i1 < i2:
      return data[:2].mean()
    elif i2 > i1:
      return data[1:].mean()
    else:
      return data[1]
  else:
    wMin = np.inf
    N = data.size//2 + data.size % 2
    for i in range(N):
      w = data[i + N - 1] - data[i]
      if w < wMin:
        wMin = w
        j = i
    return _hsm(data[j:j + N])


def periodic_distr_distance(p1,p2,nbin,L_track,mu1=None,mu2=None,N_bs=1000,mode='wasserstein'):

  if mode=='wasserstein':
    ### test, whether any distribution is cut off by "periodic bounds"
    d_raw = mu2-mu1
    shift_sign = 1 if (d_raw>=0) & (abs(d_raw)<(L_track/2)) else -1
    if abs(d_raw) > L_track/2:
      shift = int((mu1+mu2)/2)
      d_out = periodic_wasserstein_distance(np.roll(p1,shift),np.roll(p2,shift),(mu1+shift)%L_track,(mu2+shift)%L_track,L_track)
    else:

      idx_p1 = p1>p1.max()*10**(-2)
      idx_p2 = p2>p2.max()*10**(-2)
      x = np.arange(nbin)
      if (idx_p1[0] & idx_p1[-1]) | (idx_p2[0] & idx_p2[-1]) | (abs(d_raw)>L_track/3):
        d = np.zeros(nbin)
        for shift in np.linspace(0,L_track,nbin):
          #d[shift] = sstats.wasserstein_distance(x,x,np.roll(p1,shift),np.roll(p2,shift))
          d[shift] = wasserstein_distance(x,x,np.roll(p1,shift),np.roll(p2,shift))
        d_out = d.min()

      else:
        #print('calc direct')
        #d_out = sstats.wasserstein_distance(x,x,p1,p2)
        d_out = wasserstein_distance(x,x,p1,p2)
      d_out *= shift_sign
      #print('wasserstein: %5.3f: '%d_out)
      #print('raw: %5.3f'%d_raw)
    return d_out
  else:
    ## get distance via bootstrapping

    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)

    rnd1 = np.random.rand(N_bs,1)
    rnd2 = np.random.rand(N_bs,1)
    x1 = np.argmin(abs(rnd1-cdf1[np.newaxis,:]),1)
    x2 = np.argmin(abs(rnd2-cdf2[np.newaxis,:]),1)

    ## correct for very low probabilities, where very first entry is taken - should rather be entry, closest to distribution (>0, but <distr)
    for i in np.where(x1==0)[0]:
        tmp = abs(rnd1[i]-cdf1)
        x1[i] = np.where(tmp==tmp.min())[0][-1]
    for i in np.where(x2==0)[0]:
        tmp = abs(rnd2[i]-cdf2)
        x2[i] = np.where(tmp==tmp.min())[0][-1]

    shift_distr = np.zeros(N_bs)

    ## generate two samples from distribution from cdfs
    shift_distr = (x2-x1 + L_track/2)%L_track -L_track/2
    d_out = get_average(shift_distr,1,periodic=True,bounds=[-L_track/2,L_track/2])
    p_out = np.histogram(shift_distr,np.linspace(-L_track/2,L_track/2,nbin+1),density=True)[0]
    p_out[p_out<10**(-10)] = 0
    #CI = np.percentile(shift_distr,[5,95])
    #print('CI: %5.3f,%5.3f'%(CI[0],CI[1]))

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(p1)
    # plt.plot(p2,'r')
    # plt.subplot(212)
    # plt.hist(shift_distr,np.linspace(-50,50,101),density=True)
    # plt.plot(np.linspace(-49.5,49.5,100),p_out)
    # plt.plot(d_out,0,'rx')
    # #d_raw = abs((d_raw+nbin/2)%nbin-nbin/2) * shift_sign
    # #plt.plot(d_raw,0,'kx')
    # plt.xlim([-50,50])
    # plt.show(block=False)
    return d_out, p_out



  #### test, whether extrema of distributions are < nbin/2 apart
  #plt.figure()
  ##plt.subplot(211)
  #plt.plot(p1)
  #plt.plot(p2,'r')
  ##plt.subplot(212)
  ##plt.plot(d)
  ##plt.plot(d_raw,d_raw,'rx');
  #plt.show(block=False)



  return d_out


def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):

    deltas = np.ones(u_values.shape)

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_cdf = np.cumsum(u_weights)

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_cdf = np.cumsum(v_weights)

    return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))


def bootstrap_data(fun,data,N_bs):
  ## data:    data to be bootstrapped over
  ## fun:     function to be applied to data "f(data)". needs to return calculated parameters in first return statement
  ## N_bs:    number of bootstrap-samples
  N_data = data.shape[0]
  single=False
  try:
      pars,_ = fun(data)
      par = np.zeros(np.append(N_bs,np.array(pars).shape))*np.NaN
  except:
      pars = fun(data)
      single = True
      par = np.zeros((N_bs,1))*np.NaN

  samples = np.random.randint(0,N_data,(N_bs,N_data))

  for i in range(N_bs):
    data_bs = data[samples[i,:],...]    ### get bootstrap sample
    if single:
        par[i,...] = fun(data_bs)     ### obtain parameters from function "fun"
    else:
        par[i,...],p_cov = fun(data_bs)     ### obtain parameters from function "fun"

  return par.mean(0), par.std(0)

def pickleData(dat,path,mode='load',prnt=True):

  if mode=='save':
    f = open(path,'wb')
    pickle.dump(dat,f)
    f.close()
    if prnt:
        print('Data saved in %s'%path)
  else:
    f = open(path,'rb')
    dat = pickle.load(f)
    f.close()
    if prnt:
        print('Data loaded from %s'%path)
    return dat


def get_average(x,p,periodic=False,bounds=None):

  #assert abs(1-p.sum()) < 10**(-2), 'probability not normalized, sum(p) = %5.3g'%p.sum()
  if not np.isscalar(p) | (abs(1-np.sum(p)) < 10**(-2)):
      p /= p.sum()
  if periodic:
    assert bounds, 'bounds not specified'
    L = bounds[1]-bounds[0]
    scale = L/(2*np.pi)
    avg = (cmath.phase((p*np.exp(+complex(0,1)*(x-bounds[0])/scale)).sum())*scale) % L + bounds[0]
  else:
    avg = (x*p).sum()
  return avg


def fdr_control(x,alpha):

  if alpha < 1:
    x[x==0.001] = 10**(-10)
    x_mask = np.isfinite(x)
    N = x_mask.sum()
    FDR_thr = range(1,N+1)/N*alpha
    x_masked = x[x_mask]
    idxes = np.where(x_mask)[0];
    idx_sorted = np.argsort(x_masked);
    x_sorted = x_masked[idx_sorted]
    FDR_arr = x_sorted<FDR_thr;
    idx_cut = np.where(FDR_arr==False)[0][0]
    FDR_arr[idx_cut:] = False

    classified = np.zeros(len(x)).astype('bool')
    classified[idxes[idx_sorted[FDR_arr]]] = True
  else:
    classified = np.ones(len(x)).astype('bool')
  return classified

def fit_plane(data,anchor=None):
  ## Constructs a plane from a collection of points
  ## so that the summed squared distance to all points is minimzized
  ### obtained from https://www.ilikebigbits.com/2015_03_04_plane_from_points.html

  assert len(data.shape) == 2, 'Please provide data as 2D-array'
  assert data.shape[-1] == 3, 'Please provide the values of x,y and z-direction for each data point'
  assert data.shape[0]>2, 'More points are required to fit'

  ## calculate centroid of data as anchor point for plane
  if anchor is None:
    p = data.mean(0)
  else:
    p = np.array(anchor)

  Cov = np.cov(data-p,rowvar=False)

  det = np.zeros(3)
  det[0] = Cov[1,1]*Cov[2,2] - Cov[1,2]*Cov[2,1]
  det[1] = Cov[0,0]*Cov[2,2] - Cov[0,2]*Cov[2,0]
  det[2] = Cov[0,0]*Cov[1,1] - Cov[0,1]*Cov[1,0]
  det_max = np.max(det)

  n = np.zeros(3)   ## normal vector
  if det_max == det[0]:
    n[0] = det[0]
    n[1] = Cov[0,2]*Cov[1,2] - Cov[0,1]*Cov[2,2]
    n[2] = Cov[0,1]*Cov[1,2] - Cov[0,2]*Cov[1,1]
  elif det_max == det[1]:
    n[0] = Cov[0,2]*Cov[1,2] - Cov[0,1]*Cov[2,2]
    n[1] = det[1]
    n[2] = Cov[0,1]*Cov[0,2] - Cov[1,2]*Cov[0,0]
  else:
    n[0] = Cov[0,1]*Cov[1,2] - Cov[0,2]*Cov[1,1]
    n[1] = Cov[0,1]*Cov[0,2] - Cov[1,2]*Cov[0,0]
    n[2] = det[2]
  n /= np.linalg.norm(n)  ## normalize!
  return (p,n), np.NaN


def z_from_point_normal_plane(x,y,p,n):
  ## p: some point on plane
  ## n: normal vector of plane
  return - n[0]/n[2]*(x-p[0]) - n[1]/n[2]*(y-p[1]) + p[2]


def rotation_matrix(u,theta,degree=True):

  d = 3
  R = np.zeros((d,d))
  I = np.eye(d)

  if degree:
    theta *= 2*np.pi/360

  u /= np.array(u).sum()

  return np.cos(theta) * I + np.sin(theta) * np.cross(I,u) + (1 - np.cos(theta)) * np.outer(u,u)


def KS_test(dat1,dat2):

  ## p1 & p2 are probability distributions defined on the same kernel

  ### normalize distributions
  #p1 /= p1.sum()
  #p2 /= p2.sum()

  ## generate cumulative density functions from data:
  N1 = dat1.shape[0]
  dat1.sort()

  N2 = dat2.shape[0]
  dat2.sort()

  all_values = np.concatenate((dat1,dat2))
  all_values.sort()

  d1 = np.zeros(N1+N2)
  d2 = np.zeros(N1+N2)

  d1[all_values.searchsorted(dat1)] = 1/N1
  d2[all_values.searchsorted(dat2)] = 1/N2

  plt.figure()
  plt.plot(d1,'k')
  plt.plot(d2,'r')
  plt.show(block=False)

  ## add 0 entries at other data points



  return np.abs(d1-d2).max()

def occupation_measure(data,x_ext,y_ext,nA=[10,10]):

  ## nA:      number zones per row / column (2 entries)
  N = data.shape[0]
  NA_exp = N/(nA[0]*nA[1])
  print(NA_exp)
  A = np.histogram2d(data[:,0],data[:,1],[np.linspace(x_ext[0],x_ext[1],nA[0]+1),np.linspace(y_ext[0],y_ext[1],nA[1]+1)])[0]
  rA = A/NA_exp

  return 1-np.sqrt(np.sum((rA-1)**2))/(nA[0]*nA[1])




def E_stat_test(dat1,dat2):

  n = dat1.shape[0]
  m = dat2.shape[0]

  A = 1/(n*m)*sp.spatial.distance.cdist(dat1,dat2).sum()
  B = 1/n**2*sp.spatial.distance.cdist(dat1,dat1).sum()
  C = 1/m**2*sp.spatial.distance.cdist(dat2,dat2).sum()

  print(A)
  print(B)
  print(C)
  E = 2*A - B - C

  T = n*m/(n+m)*E
  print('test stat: %5.3g'%T)



def com(A, d1, d2, d3=None):

  if 'csc_matrix' not in str(type(A)):
      A = sp.sparse.csc_matrix(A)

  if d3 is None:
      Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                        np.outer(np.arange(d2), np.ones(d1)).ravel()], dtype=A.dtype)
  else:
      Coor = np.matrix([
          np.outer(np.ones(d3), np.outer(np.ones(d2), np.arange(d1)).ravel()).ravel(),
          np.outer(np.ones(d3), np.outer(np.arange(d2), np.ones(d1)).ravel()).ravel(),
          np.outer(np.arange(d3), np.outer(np.ones(d2), np.ones(d1)).ravel()).ravel()],
          dtype=A.dtype)

  Anorm = sp.sparse.vstack([a.multiply(a>0.001*a.max())/a[a>0.001*a.max()].sum() if (a>0).sum()>0 else sp.sparse.csc_matrix(a.shape) for a in A.T]).T;
  cm = (Coor * Anorm).T
  cm[np.squeeze(np.array((Anorm>0).sum(0)))==0,:] = np.NaN
  return np.array(cm)


def calculate_img_correlation(A1,A2,dims=(512,512),crop=False,cm_crop=None,binary=False,shift=True,plot_bool=False):

  if shift:

    ## try with binary and continuous
    if binary == 'half':
      A1 = (A1>np.median(A1.data)).multiply(A1)
      A2 = (A2>np.median(A2.data)).multiply(A2)
    elif binary:
      A1 = A1>np.median(A1.data)
      A2 = A2>np.median(A2.data)

    #t_start = time.time()
    if not np.all(A1.shape == dims):
      A1 = A1.reshape(dims)
    if not np.all(A2.shape == dims):
      A2 = A2.reshape(dims)
    #t_end = time.time()
    #print('reshaping --- time taken: %5.3g'%(t_end-t_start))

    if crop:
      #t_start = time.time()
      row,col,tmp = sp.sparse.find(A1)
      A1 = A1.toarray()[row.min():row.max()+1,col.min():col.max()+1]
      row,col,tmp = sp.sparse.find(A2)
      A2 = A2.toarray()[row.min():row.max()+1,col.min():col.max()+1]
      #t_end = time.time()
      #print('cropping 1 --- time taken: %5.3g'%(t_end-t_start))

      #t_start = time.time()
      padding = np.subtract(A2.shape,A1.shape)
      if padding[0] > 0:
        A1 = np.pad(A1,[[padding[0],0],[0,0]],mode='constant',constant_values=0)
      else:
        A2 = np.pad(A2,[[-padding[0],0],[0,0]],mode='constant',constant_values=0)

      if padding[1] > 0:
        A1 = np.pad(A1,[[0,0],[padding[1],0]],mode='constant',constant_values=0)
      else:
        A2 = np.pad(A2,[[0,0],[-padding[1],0]],mode='constant',constant_values=0)
      #t_end = time.time()
      #print('cropping 2 --- time taken: %5.3g'%(t_end-t_start))
    else:
      if not (type(A1) is np.ndarray):
        A1 = np.array(A1)
        A2 = np.array(A2)

    dims = A1.shape

    #t_start = time.time()
    C = signal.convolve(A1-A1.mean(),A2[::-1,::-1]-A2.mean(),mode='same')/(np.prod(dims)*A1.std()*A2.std())
    #t_end = time.time()
    #print('corr-computation --- time taken: %5.3g'%(t_end-t_start))
    C_max = C.max()
    if np.isnan(C_max) | (C_max == 0):
      return np.NaN, np.ones(2)*np.NaN

    #if not crop:
    crop_half = ((dims[0]-np.mod(dims[0],2))/2,(dims[1]-np.mod(dims[1],2))/2)#tuple(int(d/2-1) for d in dims)
    idx_max = np.unravel_index(np.argmax(C),C.shape)
    img_shift = np.subtract(idx_max,crop_half)

    if (plot_bool):# | ((C_max>0.95)&(C_max<0.9999)):
      #idx_max = np.where(C.real==C_max)
      plt.figure()
      ax1 = plt.subplot(221)
      im = ax1.imshow(A1,origin='lower')
      plt.colorbar(im)
      plt.subplot(222,sharex=ax1,sharey=ax1)
      plt.imshow(A2,origin='lower')
      plt.colorbar()
      plt.subplot(223)
      plt.imshow(C,origin='lower')
      plt.plot(crop_half[1],crop_half[0],'ro')
      plt.colorbar()
      plt.suptitle('corr: %5.3g'%C_max)
      plt.show(block=True)

    return C_max, img_shift # C[crop_half],
  else:
    #if not (type(A1) is np.ndarray):
      #A1 = A1.toarray()
    #if not (type(A2) is np.ndarray):
      #A2 = A2.toarray()

    if not (cm_crop is None):

      cr = 20
      extent = np.array([cm_crop-cr,cm_crop+cr+1]).astype('int')
      extent = np.maximum(extent,0)
      extent = np.minimum(extent,dims)
      A1 = A1.reshape(dims)[extent[0,0]:extent[1,0],extent[0,1]:extent[1,1]]
      A2 = A2.reshape(dims)[extent[0,0]:extent[1,0],extent[0,1]:extent[1,1]]
    #else:
      #extent = [[0,0],[dims[0],dims[1]]]
    if plot_bool:
      #idx_max = np.where(C.real==C_max)
      plt.figure()
      plt.subplot(221)
      plt.imshow(A1.reshape(extent[1,0]-extent[0,0],extent[1,1]-extent[0,1]),origin='lower')
      plt.colorbar()
      plt.subplot(222)
      plt.imshow(A2.reshape(extent[1,0]-extent[0,0],extent[1,1]-extent[0,1]),origin='lower')
      plt.colorbar()
      #plt.subplot(223)
      #plt.imshow(C,origin='lower')
      #plt.plot(crop_half[1],crop_half[0],'ro')
      #plt.colorbar()
      plt.suptitle('corr: %5.3g'%np.corrcoef(A1.flat,A2.flat)[0,1])
      plt.show(block=False)
    return A1.multiply(A2).sum()/np.sqrt(A1.power(2).sum()*A2.power(2).sum()), None
    #return (A1*A2).sum()/np.sqrt((A1**2).sum()*(A2**2).sum()), None
    #return np.corrcoef(A1.flat,A2.flat)[0,1], None


def get_shift_and_flow(A1,A2,dims=(512,512),projection=-1,transpose_it=False,plot_bool=False):

  ## dims:          shape of the (projected) image
  ## projection:    axis, along which to project. If None, no projection needed

  if not (projection is None):
    A1 = np.array(A1.sum(projection))
    A2 = np.array(A2.sum(projection))
  A1 = A1.reshape(dims)
  A2 = A2.reshape(dims)

  if transpose_it:
      A2 = A2.T

  A1 = normalize_array(A1,'uint',8)
  A2 = normalize_array(A2,'uint',8)

  c,(y_shift,x_shift) = calculate_img_correlation(A1,A2,plot_bool=plot_bool)

  x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32), np.arange(0., dims[1]).astype(np.float32))
  x_remap = (x_grid - x_shift).astype(np.float32)
  y_remap = (y_grid - y_shift).astype(np.float32)

  A2 = cv2.remap(A2, x_remap, y_remap, interpolation=cv2.INTER_CUBIC)
  A2 = normalize_array(A2,'uint',8)

  flow = cv2.calcOpticalFlowFarneback(A1,A2,None,0.5,5,128,3,7,1.5,0)

  if plot_bool:

    print('shift:',[x_shift,y_shift])
    idxes = 15
    plt.figure()
    plt.quiver(x_grid[::idxes,::idxes], y_grid[::idxes,::idxes], flow[::idxes,::idxes,0], flow[::idxes,::idxes,1], angles='xy', scale_units='xy', scale=1, headwidth=4,headlength=4, width=0.002, units='width')
    plt.show(block=False)

  return (x_shift,y_shift), flow, (x_grid,y_grid), c

def normalize_array(A,a_type='uint',a_bits=8,axis=None):
  A -= A.min()
  A = A/A.max()

  return (A*(A>A.mean(axis))*(2**a_bits-1)).astype('%s%d'%(a_type,a_bits))

def normalize_sparse_array(A):
  #A = sp.sparse.vstack([a-a.min() for a in A.T]).T
  return sp.sparse.vstack([a/a.max() for a in A.T]).T

#def display_projected_movie(basePath,mouse,s):

  #f1 = h5py.File(file_name,'r+')

def fun_wrapper(fun,x,p):
  if np.isscalar(p):
    return fun(x,p)
  if p.shape[-1] == 2:
    return fun(x,p[...,0],p[...,1])
  if p.shape[-1] == 3:
    return fun(x,p[...,0],p[...,1],p[...,2])
  if p.shape[-1] == 4:
    return fun(x,p[...,0],p[...,1],p[...,2],p[...,3])
  if p.shape[-1] == 5:
    return fun(x,p[...,0],p[...,1],p[...,2],p[...,3],p[...,4])
  if p.shape[-1] == 6:
    return fun(x,p[...,0],p[...,1],p[...,2],p[...,3],p[...,4],p[...,5])
  if p.shape[-1] == 7:
    return fun(x,p[...,0],p[...,1],p[...,2],p[...,3],p[...,4],p[...,5],p[...,6])


def gmean(X,axis=1,nanflag=False):

  if nanflag:
    return np.exp(np.nansum(np.log(X),axis)/(~np.isnan(X)).sum(axis))
  else:
    return np.exp(np.sum(np.log(X),axis)/X.shape[axis])

def corr0(X,Y=None):

  Y = X if Y is None else Y

  X -= np.nanpercentile(X,20)
  Y -= np.nanpercentile(X,20)

  c_xy = np.zeros((len(X),len(X)))
  for i,x in enumerate(X):
    for j,y in enumerate(Y):
      c_xy[i,j] = (x*y).sum()/np.sqrt((x**2).sum()*(y**2).sum())

  return c_xy

def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))

def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat,checks=False)

    #res_linkage = linkage(flat_dist_mat, method=method,preserve_input=False)
    res_linkage = sp.cluster.hierarchy.linkage(flat_dist_mat,method=method,optimal_ordering=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]

    return seriated_dist, res_order, res_linkage

def gauss_smooth(X,smooth=None,mode='wrap'):
  if (smooth is None) or not np.any(np.array(smooth)>0):
    return X
  else:
    V = X.copy()
    V[np.isnan(X)] = 0
    VV = sp.ndimage.gaussian_filter(V,smooth,mode=mode)

    W = 0*X.copy()+1
    W[np.isnan(X)] = 0
    WW = sp.ndimage.gaussian_filter(W,smooth,mode=mode)

  return VV/WW


def get_reliability(trial_map,map,field,t):

    ### need: obtain reliability for single (or batch of?) neuron
    ### ability to test / plot

    ## trial_map:       firing maps of single trials in (t,bin)-format
    ## map:             overall firing map of session
    ## field:           parameters of place field

    ## obtain noise level + threshold
    sd_fmap = 2
    nbin = 100
    base = np.nanmedian(map)

    if np.all(np.isnan(field[t,...])):
        return

    ## find threshold value of firing map as significant fluctuations over baseline
    fmap_noise = map - np.nanmedian(map)
    fmap_noise = -1*fmap_noise*(fmap_noise<0)
    N_noise = (fmap_noise>0).sum()
    noise = np.sqrt((fmap_noise**2).sum()/(N_noise*(1-2/np.pi)))
    fmap_thr = np.maximum(4,base+sd_fmap*noise)

    ## find bins belonging to place field
    field_bin = int(field[t,3,0])
    field_bin_l = int(field[t,3,0]-field[t,2,0]) % nbin
    field_bin_r = int(field[t,3,0]+field[t,2,0]+1) % nbin
    ## obtain average firing rate within field (slight smoothing)
    fmap = gauss_smooth(trial_map,(0,4))
    if field_bin_l < field_bin_r:
        field_rate_bins = fmap[:,field_bin_l:field_bin_r]
    else:
        field_rate_bins = np.hstack([fmap[:,field_bin_l:],fmap[:,:field_bin_r]])
    # field_rate = np.mean(field_rate_bins,1)
    field_rate = np.max(field_rate_bins,1)

    ## field_trials are such, where average field firing rate is above threshold
    trial_field = field_rate > fmap_thr
    field_max = np.mean(np.max(field_rate_bins[trial_field],1))
    rel = (field_rate > fmap_thr).mean()

    testing = False
    if testing:
        print('max fr: %.2g'%field_max)
        print('reliability: %.2f'%rel)

        plt.figure()
        plt.subplot(211)

        plt.plot(map,'k')
        plt.plot(gauss_smooth(trial_map[-1,:],0),'b--')
        plt.plot(gauss_smooth(trial_map[-1,:],2),'b')

        plt.plot(field[t,3,0],5,'rx')

        plt.plot([0,100],[fmap_thr,fmap_thr],'r--')
        # print(field_rate)
        plt.subplot(212)
        plt.plot(field_rate)
        plt.plot([0,field_rate.shape[0]],[fmap_thr,fmap_thr],'r--')
        plt.legend()
        plt.show(block=False)

        plt.figure()
        for tt in range(trial_map.shape[0]):
            plt.subplot(5,6,tt+1)
            plt.plot([field_bin_l,field_bin_l],[0,10],'k--')
            plt.plot([field_bin_r,field_bin_r],[0,10],'k--')
            for i in [1,3]:
                plt.plot(gauss_smooth(trial_map[tt,:],i))
            plt.plot([0,100],[fmap_thr,fmap_thr],'r:')
            plt.ylim([0,20])
        plt.show(block=False)

    return rel, field_max, trial_field

def get_firingrate(S,f=15,sd_r=1):

    S[S<0.0001*S.max()]=0
    Ns = (S>0).sum()
    if Ns==0:
      return 0,np.NaN,np.NaN
    else:
      trace = S[S>0]
      baseline = np.median(trace)
      trace -= baseline
      trace *= -1*(trace <= 0)

      Ns_baseline = (trace>0).sum()
      noise = np.sqrt((trace**2).sum()/(Ns_baseline*(1-2/np.pi)))

      sd_r = sstats.norm.ppf((1-0.1)**(1/Ns)) if (sd_r==-1) else sd_r
      firing_threshold_adapt = baseline + sd_r*noise

      N_spikes = np.floor(S / firing_threshold_adapt).sum()
      return N_spikes/(S.shape[0]/f),firing_threshold_adapt,np.floor(S / firing_threshold_adapt)#S > firing_threshold_adapt#

def add_number(fig,ax,order=1,offset=None):

    # offset = [-175,50] if offset is None else offset
    offset = [-150,50] if offset is None else offset

    pos = fig.transFigure.transform(plt.get(ax,'position'))
    x = pos[0,0]+offset[0]
    y = pos[1,1]+offset[1]
    ax.text(x=x,y=y,s='%s)'%chr(96+order),ha='center',va='center',transform=None,weight='bold',fontsize=14)


def get_status_arr(cluster,SD=1):

    nSes = cluster.meta['nSes']
    nC = cluster.meta['nC']
    nbin = cluster.para['nbin']
    sig_theta = cluster.stability['all']['mean'][0,2]

    status_arr = ['act','code','stable']

    ds_max = nSes
    status = {}
    status['stable'] = np.zeros((nC,nSes,nSes),'bool')
    for c in np.where(cluster.stats['cluster_bool'])[0]:
        for s in np.where(cluster.sessions['bool'])[0]:
            if cluster.status[c,s,2]:
                for f in np.where(cluster.status_fields[c,s,:])[0]:

                    loc_compare = cluster.fields['location'][c,:s,:,0]
                    loc_compare[~cluster.status_fields[c,:s,:]] = np.NaN
                    dLoc = np.abs(np.mod(cluster.fields['location'][c,s,f,0] - loc_compare +nbin/2,nbin)-nbin/2)

                    stable_s = np.where(dLoc<(SD*sig_theta))[0]
                    if len(stable_s)>0:
                        ds = s - stable_s[-1]
                        status['stable'][c,s,np.unique(s-stable_s)] = True
                        # status['stable'][c,s] = ds

    status['act'] = np.pad(cluster.status[...,1][...,np.newaxis],((0,0),(0,0),(0,nSes-1)),mode='edge')
    status['code'] = np.pad(cluster.status[...,2][...,np.newaxis],((0,0),(0,0),(0,nSes-1)),mode='edge')
    # status['stable'] = status['stable']
    # status['stable'] = status['stable']==1

    status_dep = {}
    status_dep['act'] = np.ones((nC,nSes),'bool')
    status_dep['act'][:,~cluster.sessions['bool']] = False
    status_dep['code'] = np.copy(status['act'][...,0])
    status_dep['stable'] = np.copy(status['code'][...,0])

    return status,status_dep


def get_CI(p,X,Y,alpha=0.05):
    n,k = X.shape

    sigma2 = np.sum((Y-np.dot(X,p))**2) / (n-k)
    C = sigma2 * np.linalg.inv(np.dot(X.T,X))
    se = np.sqrt(np.diag(C))

    sT = sstats.distributions.t.ppf(1.0 - alpha/2.0, n-k)
    CI = sT * se
    return CI


def get_recurr(status,status_dep):

    nC,nSes = status.shape
    recurr = np.zeros((nSes,nSes))*np.NaN
    for s in range(nSes):#np.where(cluster.sessions['bool'])[0]:
        overlap = status[status[:,s],:].sum(0).astype('float')
        N_ref = status_dep[status[:,s],:].sum(0)
        recurr[s,1:nSes-s] = (overlap/N_ref)[s+1:]

    return recurr

def get_mean_SD(SDs):

    mask = np.isfinite(SDs)
    n = mask.sum()
    vars = SDs[mask]**2
    return np.sqrt(1/n**2 * np.sum(vars))
