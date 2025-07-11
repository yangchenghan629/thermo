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

th=np.zeros((201,34,192))
rv=np.zeros((201,34,192))
rvs=np.zeros((201,34,192)) #saturated vapor mixing ratio
qi=np.zeros((201,34,192))  #cloud ice mixing ratio
qc=np.zeros((201,34,192))  #cloud water
qr=np.zeros((201,34,192))  #cloud rain
radi=np.zeros((201,192))
tang=np.zeros((201,192))

sh=np.zeros((201,192))  #sensible heat flux
lh=np.zeros((201,192))  #latent heat flux

for i in range(0,601,3):
    var=nc.Dataset(f'TC/axmean-{i:06d}.nc')
    th[int(i/3),:,:]=var.variables['th'][0,0,:,:]
    rv[int(i/3),:,:]=var.variables['qv'][0,0,:,:]
    rvs[int(i/3),:,:]=var.variables['qvs'][0,0,:,:]
    qi[int(i/3),:,:]=var.variables['qi'][0,0,:,:]
    qc[int(i/3),:,:]=var.variables['qc'][0,0,:,:]
    qr[int(i/3),:,:]=var.variables['qr'][0,0,:,:]
    sh[int(i/3),:]=var.variables['sh'][0,0,:]
    lh[int(i/3),:]=var.variables['lh'][0,0,:]
    radi[int(i/3),:]=var.variables['radi_wind'][:,0,0,:]
    tang[int(i/3),:]=var.variables['tang_wind'][:,0,0,:]

qv=rv/(1+rv)
qvs=rvs/(1+rvs)

qv_mean=np.mean(qv[:,0,:],axis=1)
qvs_mean=np.mean(qvs[:,0,:],axis=1)


cloud=qi+qc+qr
cloud_value=cloud.copy()
cloud=np.isnan(np.where(np.abs(cloud)<1e-5,cloud,np.nan))  #bool : is cloud?
no_cloud_mask=np.all(cloud==False,axis=1)

cloud_height_flip=np.nanargmax(np.flip(cloud,axis=1),axis=1)
cloud_height_flip[no_cloud_mask]=-1  #No any cloud , marked -1
cloud_height=np.where(cloud_height_flip==-1,0,cloud.shape[1]-1-cloud_height_flip)  #cloud height index
max_cloud_height=np.amax(cloud_height,axis=1)  #highest cloud height

rr,hh=np.meshgrid(radius,zc)

# cloud profile hourly intergral 
# for i in range(201):
#     plt.clf()
#     plt.pcolormesh(rr,hh,cloud_value[i,:,:]*1000,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=6,vmin=0),cmap=cm.jet)
#     plt.colorbar(ticks=[0,0.05,0.1,2,4,6],extend='max')
#     plt.plot(radius,zc[cloud_height[i,:]],'r')
#     plt.savefig(f'./hourly_cloud/cloud_height{i:02d}.png')
# gif=[]
# for i in range(201):
#     img=open(f'./hourly_cloud/cloud_height{i:02d}.png')
#     gif.append(img)
# gif[0].save('./hourly_cloud/hourly.gif',save_all=True,append_images=gif[1:],duration=300)


h=(sh+lh) #enthalpy flux [kg m2 s-2] **if unit mass=>m2 s-2 = cp T **
dT=h/cp

lapse_rate=np.full((201,34,192),-9.8)
lapse_rate=np.where(qv/qvs<1,lapse_rate,-6.5)

temp=np.zeros((201,34,192))
temp[:,0,:]=sst+dT


for i in range(1,34):
    del_h=zc[i]-zc[i-1]
    temp[:,i,:]=temp[:,i-1,:]+del_h*lapse_rate[:,i,:]

Tout=np.full((201),np.nan)

for i in range(201):
    Tout[i]=np.amax(temp[i,max_cloud_height[i],:])


Tin=sst+np.mean(h,axis=1)/cp
eta=(Tin-Tout)/Tin
eta=np.where(eta>0,eta,np.nan)

v_mpi=((eta/(1-eta))*(ck_cd)*(cp*(sst-Tin)+Lv*(qvs_mean-qv_mean)))**0.5
speed=np.amax((radi**2+tang**2)**0.5,axis=1)


plt.clf()
plt.plot(np.arange(201),v_mpi,label='MPI')
plt.plot(np.arange(201),speed,label='max speed')
plt.title('Maximum Potential Intensity and \nNear Surface Maximum Wind Speed',fontsize=14)
plt.xticks(np.arange(0,202,24))
plt.xlim([2,201])
plt.ylim([0,110])
plt.xlabel('Time [hrs]',fontsize=12)
plt.ylabel('Speed [m/s]',fontsize=12)
plt.legend()
plt.grid()
plt.savefig('speed.png',dpi=500)
