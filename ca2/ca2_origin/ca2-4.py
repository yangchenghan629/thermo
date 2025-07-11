import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image

loc=nc.Dataset('./ne_20191030/TOPO.nc')
lon=loc.variables['lon'][:]
lat=loc.variables['lat'][:]
lev=loc.variables['lev'][:]

lon_c_index=np.where(np.abs(lon-120.6289)<=1e-5)[0][0]
lat_c_index=np.where(np.abs(lat-23.49798)<=1e-5)[0][0]

lon_c=lon[lon_c_index]
lat_c=lat[lat_c_index]

rnum=1

lon_range=lon[lon_c_index-rnum:lon_c_index+rnum+1]
lat_range=lat[lat_c_index-rnum:lat_c_index+rnum+1]

mask=loc.variables['mask'][:,lat_c_index-rnum:lat_c_index+rnum+1,lon_c_index-rnum:lon_c_index+rnum+1]

pres=np.loadtxt('./ne_20191030/fort.98',skiprows=2,usecols=3)
pres=np.broadcast_to(pres[None,:,None,None],(73,60,2*rnum+1,2*rnum+1))

rv=np.zeros((73,60,2*rnum+1,2*rnum+1))
th=np.zeros((73,60,2*rnum+1,2*rnum+1))

for i in range(73):
    path=f'./ne_20191030/ne_20191030.L.Thermodynamic-{i:06d}.nc'
    var=nc.Dataset(path)
    rv[i,:,:,:]=np.where(mask,var.variables['qv'][:,:,lat_c_index-rnum:lat_c_index+rnum+1,lon_c_index-rnum:lon_c_index+rnum+1],np.nan)
    th[i,:,:,:]=np.where(mask,var.variables['th'][:,:,lat_c_index-rnum:lat_c_index+rnum+1,lon_c_index-rnum:lon_c_index+rnum+1],np.nan)

qv=rv/(1+rv)
temp=th/((1000/(pres/100))**0.286)
temp[:,:,rnum,rnum]+=10
Tv=temp*(1+0.608*qv)

Tvenv=np.copy(Tv)
Tvenv[:,:,rnum,rnum]=np.nan

avgTv=np.nanmean(Tvenv,axis=(2,3))

fb=(Tv[:,:,rnum,rnum]-avgTv)*9.8/avgTv

time=np.arange(0,73)
time2,lev2=np.meshgrid(time,lev)


fig,ax=plt.subplots(2,1)
plt.suptitle('Vertical Diurnal Profile of Buoyancy [$m/s^2$]\nHeated',fontsize=14)
a1=ax[0].contourf(time2,lev2,fb.T,cmap=cm.coolwarm,extend='both',levels=np.linspace(0.32,0.38,11))
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
ax[0].set_xticks(xticks,xtexts)
ax[0].set_xlabel('Time (LST)')
yticks=np.arange(1,13,2)
ytexts=[f'{y:2d}'for y in yticks]
ax[0].set_yticks(yticks,ytexts)
ax[0].set_ylabel('Altitude (km)')
ax[0].set_xlim([0,72])
ax[0].set_ylim([1,12])
plt.colorbar(a1,ax=ax[0],orientation='vertical')

a2=ax[1].contourf(time2,lev2,fb.T,cmap=cm.coolwarm,extend='both',levels=np.linspace(0.32,0.38,11))
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
ax[1].set_xticks(xticks,xtexts)
ax[1].set_xlabel('Time (LST)')
yticks=np.arange(1,4.1,0.5)
ytexts=[f'{y:.2f}'for y in yticks]
ax[1].set_yticks(yticks,ytexts)
ax[1].set_ylabel('Altitude (km)')
ax[1].set_xlim([0,72])
ax[1].set_ylim([1,4])
plt.colorbar(a2,ax=ax[1],orientation='vertical')

plt.tight_layout()
plt.savefig('buoyancy10K_contourf.png',dpi=600)