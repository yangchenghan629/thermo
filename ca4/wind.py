import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.colors as mcolors , matplotlib.cm as cm
from PIL.Image import *

ck_cd=0.9 #exchange coefficient
cp=1004  #isobaric heat capacity
Lv=2.5e6  #latent heat constant
sst=300  #sea surface temp

zc=nc.Dataset('./TC/axmean-000000.nc').variables['zc'][:]/1000  #height [km]
radius=nc.Dataset('./TC/axmean-000000.nc').variables['radius'][:]/1000  #[km]

pres=np.loadtxt('./TC/fort.98',skiprows=237,usecols=3,max_rows=34)/100
pres=np.broadcast_to(pres[None,:,None],(201,34,192))

th=np.zeros((201,34,192))
rv=np.zeros((201,34,192))
rvs=np.zeros((201,34,192)) #saturated vapor mixing ratio
qi=np.zeros((201,34,192))  #cloud ice mixing ratio
qc=np.zeros((201,34,192))  #cloud water
qr=np.zeros((201,34,192))  #cloud rain
radi=np.zeros((201,34,192))
tang=np.zeros((201,34,192))


for i in range(0,601,3):
    var=nc.Dataset(f'TC/axmean-{i:06d}.nc')
    th[int(i/3),:,:]=var.variables['th'][0,0,:,:]
    rv[int(i/3),:,:]=var.variables['qv'][0,0,:,:]
    rvs[int(i/3),:,:]=var.variables['qvs'][0,0,:,:]
    qi[int(i/3),:,:]=var.variables['qi'][0,0,:,:]
    qc[int(i/3),:,:]=var.variables['qc'][0,0,:,:]
    qr[int(i/3),:,:]=var.variables['qr'][0,0,:,:]
    radi[int(i/3),:,:]=var.variables['radi_wind'][:,0,:,:]
    tang[int(i/3),:,:]=var.variables['tang_wind'][:,0,:,:]

qi=np.where(np.abs(qi)>=3e-5,qi,0)
qc=np.where(np.abs(qc)>=3e-5,qc,0)
qr=np.where(np.abs(qr)>=3e-5,qr,0)


qv=rv/(1+rv)
qvs=rvs/(1+rvs)

qv_mean=np.mean(qv[:,0,:],axis=1)
qvs_mean=np.mean(qvs[:,0,:],axis=1)

temp=th/((1000/pres))**0.287

cloud=(qi+qc+qr)*1000
cloud_value=cloud.copy()
cloud=np.isnan(np.where(np.abs(cloud)<5e-5,cloud,np.nan))  #bool : is cloud? (201,34,192)
no_cloud_mask=np.all(cloud==False,axis=1)  #Whole lev is clean? (201,192)

cloud_height_flip=np.nanargmax(np.flip(cloud,axis=1),axis=1)
cloud_height_flip[no_cloud_mask]=-1  #No any cloud , marked -1
cloud_height_index=np.where(cloud_height_flip==-1,0,cloud.shape[1]-1-cloud_height_flip)  #cloud height index

total=(radi**2+tang**2)**0.5

rr,hh=np.meshgrid(radius,zc)
fig,ax=plt.subplots(2,layout='constrained',sharex='row')
plt.suptitle('200 hours')
a0=ax[0].pcolormesh(rr,hh,cloud_value[200,:,:],cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=6,vmin=0))
plt.colorbar(a0,ax=ax[0],extend='max',ticks=[0,0.05,0.1,1,2,4,6],label='mixing ratio [g/kg]')
ax[0].plot(radius,zc[np.argmax(radi[200,:,:],axis=0)],'k')
ax[0].set_title('Cross-section and height of max wind speed')
ax[1].plot(radius,np.amax(radi[200,:,:],axis=0))
ax[1].set_title('max wind speed')
ax[0].grid()
ax[1].grid()
ax[0].set_xlim([0,radius[-1]])
ax[1].set_xlim([0,radius[-1]])
ax[1].set_xlabel('radius [km]')
ax[0].set_ylabel('height [km]')
ax[1].set_ylabel('speed [m/s]')
plt.savefig('wind.png')