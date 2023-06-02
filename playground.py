import matplotlib.pyplot as plt

git status
git add .
git commit -m 'blabla'
git push
git pull


### run neuron detection manually
from run_pipeline import *
sI = SessionInfo('data/555wt/Session02/')
sI.info['paths']['to_neuron_detection']

para.CaImAn['ds_factor'] = 2
para.CaImAn['fnames'] = [sI.info['paths']['to_motion_correct']]

opts = cnmf.params.CNMFParams(params_dict=para.CaImAn)
c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=2, single_thread=False)
onacid = cnmf.online_cnmf.OnACID(params=opts,dview=dview)
onacid.fit_online()
dview.terminate()

onacid.estimates.plot_contours(idx=onacid.estimates.idx_components)


### run neuron detection automatically
neuron_detection(sI.info['paths']['to_motion_correct'],para.CaImAn,n_processes=2)


## casting component n to dense matrix and displaying it
n = 140 ### neuron number
A = onacid.estimates.A[:,n].reshape(512,512).todense()
plt.figure(); plt.imshow(A); plt.show()


T = np.linspace(0,600,8989)
plt.figure(); plt.plot(T,onacid.estimates.S[n,:]*5,'r'); plt.plot(T,onacid.estimates.C[n,:],'k'); plt.show()

b = 0 ### background number
plt.figure(); plt.imshow(onacid.estimates.b[:,b].reshape(512,512)); plt.show()
plt.figure(); plt.plot(T,onacid.estimates.f[0,:],'r'); plt.plot(T,onacid.estimates.f[1,:],'k'); plt.show()

onacid.estimates.__dict__.keys()
onacid.estimates.SNR_comp

plt.figure();plt.scatter(onacid.estimates.SNR_comp,onacid.estimates.r_values);plt.show()

np.where(onacid.estimates.SNR_comp<5)



### load result data
opts = cnmf.params.CNMFParams(params_dict=para.CaImAn)
c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=2, single_thread=False)
onacid = cnmf.online_cnmf.OnACID(params=opts,dview=dview)

svPath = sI.info['paths']['to_neuron_detection']
for key, val in load_dict_from_hdf5(svPath).items():
    setattr(onacid.estimates,key,val)
