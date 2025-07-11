import matplotlib.pyplot as plt,matplotlib.cm as cm
import numpy as np
import netCDF4 as nc
import PIL.Image as Image

loc=nc.Dataset('./ne_20191030/TOPO.nc')
lon=loc.variables['lon'][:]
lat=loc.variables['lat'][:]
lev=loc.variables['lev'][:]

lon_c_index=np.where(np.abs(lon-120.6289)<=1e-5)[0][0]
lat_c_index=np.where(np.abs(lat-23.49798)<=1e-5)[0][0]


grange=2

lat_range=lat[lat_c_index-grange:lat_c_index+grange+1]
lon_range=lat[lon_c_index-grange:lon_c_index+grange+1]

mask=loc.variables['mask'][:,lat_c_index-grange:lat_c_index+grange+1,lon_c_index-grange:lon_c_index+grange+1]
mask=np.broadcast_to(mask[None,:,:,:],(73,60,2*grange+1,2*grange+1))
nannum=np.isnan(mask[0,:,grange,grange]).sum()

pres=np.loadtxt('./ne_20191030/fort.98',skiprows=2,usecols=3)
pres=np.broadcast_to(pres[None,:,None,None],(73,60,2*grange+1,2*grange+1))

den=np.loadtxt('./ne_20191030/fort.98',skiprows=2,usecols=1)
den=np.broadcast_to(den[None,:,None,None],(73,60,2*grange+1,2*grange+1))


rv=np.zeros((73,60,2*grange+1,2*grange+1))
th=np.zeros((73,60,2*grange+1,2*grange+1))

for i in range(73):
    path=f'./ne_20191030/ne_20191030.L.Thermodynamic-{i:06d}.nc'
    var=nc.Dataset(path)
    rv[i,:,:,:]=var.variables['qv'][:,:,lat_c_index-grange:lat_c_index+grange+1,lon_c_index-grange:lon_c_index+grange+1]
    th[i,:,:,:]=var.variables['th'][:,:,lat_c_index-grange:lat_c_index+grange+1,lon_c_index-grange:lon_c_index+grange+1]

qv=rv/(1+rv)
temp=th/((1000/(pres/100))**0.286)
temp=np.where(mask,temp,np.nan)

#Relativity Humidity
Rv=461
e=qv*den*Rv*temp #Pa

A=2.53e11 #Pa
B=5.42e3 #K

es=A*np.exp(-B/temp)
RH=e/es

time=np.arange(0,73)
time2,lev2=np.meshgrid(time,lev)
plt.title('Parcel Relative Humidity Profile')
plt.contourf(time2,lev2,RH[:,:,grange,grange].T)
plt.colorbar()
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
plt.xticks(xticks,xtexts)
yticks=np.arange(1,13,2)
ytexts=[f'{y:2d}'for y in yticks]
plt.yticks(yticks,ytexts)
plt.xlim([0,72])
plt.ylim([1,12])
plt.xlabel('Time [LST]')
plt.ylabel('Altitude [km]')
plt.savefig('./test/RH_test.png')

RH=np.where(mask,RH,np.nan)

#air parcel lapse rate
lapse=np.zeros((73,60))
lapse[:]=-9.8
lapse=np.where(RH[:,:,grange,grange]<1,lapse,-6.5)
lapse=np.where(mask[:,:,grange,grange],lapse,np.nan)

#surface temperature (initial state)
temp_parcel=np.zeros((73,60))
temp_parcel=temp[:,:,grange,grange]
temp10=temp_parcel+10

#from ground to top calulate temp.
for i in range(nannum+2,len(mask[0,:,grange,grange])-1): 
    del_h=lev[i]-lev[i-1]
    temp_parcel[:,i]+=lapse[:,i]*del_h
    temp10[:,i]+=lapse[:,i]*del_h

Tvenv=temp*(1+0.608*qv)
Tv_parcel=temp_parcel*(1+0.608*qv[:,:,grange,grange])
Tv10=temp10*(1+0.608*qv[:,:,grange,grange])

Tvenv=np.where(mask,Tvenv,np.nan)
Tvenv[:,:,grange,grange]=np.nan

avgTv=np.nanmean(Tvenv,axis=(2,3))

fb=(Tv_parcel-avgTv)*9.8/avgTv
fb10=(Tv10-avgTv)*9.8/avgTv

time=np.arange(0,73)
time2,lev2=np.meshgrid(time,lev)

plt.clf()
plt.contourf(time2,lev2,avgTv.T,cmap=cm.RdYlBu_r,extend='both')
plt.colorbar()
plt.title('Enviromental Tv')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
plt.xticks(xticks,xtexts)
yticks=np.arange(1,13,2)
ytexts=[f'{y:2d}'for y in yticks]
plt.yticks(yticks,ytexts)
plt.xlim([0,72])
plt.ylim([1,12])
plt.savefig('./test/tv_test.png')


plt.clf()
plt.contourf(time2,lev2,fb.T,cmap=cm.RdYlBu_r,extend='both',levels=np.linspace(-0.15,0.15,11))
plt.colorbar()
plt.title('buoyancy [m/$s^2$]')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
plt.xticks(xticks,xtexts)
yticks=np.arange(1,13,2)
ytexts=[f'{y:2d}'for y in yticks]
plt.yticks(yticks,ytexts)
plt.xlim([0,72])
plt.ylim([1,12])
plt.xlabel('Time [LST]')
plt.ylabel('Altitude [km]')
plt.savefig('./test/buoyancy_test.png')

plt.clf()
plt.contourf(time2,lev2,fb10.T,cmap=cm.RdYlBu_r,extend='both',levels=np.arange(0.2,0.41,0.03))
plt.colorbar()
plt.title('buoyancy [m/$s^2$] after heating 10K')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
plt.xticks(xticks,xtexts)
yticks=np.arange(1,13,2)
ytexts=[f'{y:2d}'for y in yticks]
plt.yticks(yticks,ytexts)
plt.xlim([0,72])
plt.ylim([1,12])
plt.xlabel('Time [LST]')
plt.ylabel('Altitude [km]')
plt.savefig('./test/buoyancy_10K_test.png')
