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


temp=th/(1000/pres)**0.287

maxtangloc=np.argmax(tang,axis=1)
tc_range=np.zeros((201))
for t in range(tang.shape[0]):
    max_loc = maxtangloc[t]
    for r in range(max_loc, tang.shape[1]):
        if tang[t, r] <= 5:
            tc_range[t] = r
            break
        if tc_range[t] == 0:
            tc_range[t] = tang.shape[1] - 1

tc_range=list(map(int,tc_range))

cloud=qi+qc+qr
cloud_value=cloud.copy()
cloud=np.isnan(np.where(np.abs(cloud)<5e-5,cloud,np.nan))  #bool : is cloud? (201,34,192)
no_cloud_mask=np.all(cloud==False,axis=1)  #Whole lev is clean? (201,192)

cloud_height_flip=np.nanargmax(np.flip(cloud,axis=1),axis=1)
cloud_height_flip[no_cloud_mask]=-1  #No any cloud , marked -1
cloud_height_index=np.where(cloud_height_flip==-1,0,cloud.shape[1]-1-cloud_height_flip)  #cloud height index



Tout=np.full((201),np.nan)

cloud_height=zc[cloud_height_index]
cloud_height[no_cloud_mask]=np.nan
mean_cloud_height=[]
for i in range(201):
    mean_cloud_height.append(np.nanmean(cloud_height[i,0:tc_range[i]]))
    Tout[i]=np.nanmean(temp[i,np.where(np.abs(zc-mean_cloud_height[i])<0.5),:])

Tin=sst
eta=(Tin-Tout)/Tin
eta=np.where(eta>0,eta,np.nan)

Tsfc=[]
for i in range(201):
    Tsfc.append(np.nanmean(temp[i,0,0:tc_range[i]]))
Tsfc=np.array(Tsfc)

qv_mean=np.full((201),np.nan)
qvs_mean=np.full((201),np.nan)
for i in range(201):
    qv_mean[i]=np.mean(qv[i,0,0:tc_range[i]])
    qvs_mean[i]=np.mean(qvs[i,0,0:tc_range[i]])


v_mpi=((eta/(1-eta))*(ck_cd)*(cp*(sst-Tsfc)+Lv*(qvs_mean-qv_mean)))**0.5
speed=np.amax((radi**2+tang**2)**0.5,axis=1)


rr,hh=np.meshgrid(radius,zc)

plt.clf()
plt.plot(np.arange(201),v_mpi,label='MPI')
plt.plot(np.arange(201),speed,label='max speed')
plt.title('Maximum Potential Intensity and \nNear Surface Maximum Wind Speed',fontsize=14)
plt.xticks(np.arange(0,202,24))
plt.xlim([2,201])
plt.xlabel('Time [hrs]',fontsize=12)
plt.ylabel('Speed [m/s]',fontsize=12)
plt.legend(loc='upper left')
plt.grid()
plt.ylim([0,100])
plt.savefig('mpifinal.png',dpi=500)
