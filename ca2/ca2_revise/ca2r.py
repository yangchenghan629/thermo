import matplotlib.pyplot as plt,matplotlib.cm as cm,matplotlib.colors as mcolors
import numpy as np
import netCDF4 as nc
import PIL.Image as Image

Rv=461
A=2.53e11 #Pa
B=5.42e3 #K

def lapse(qv,temp,den): #all of var are array-like
    e=qv*den*Rv*temp #Pa
    es=A*np.exp(-B/temp)
    RH=e/es
    rate=np.full((73),9.8) #initialize lapse rate with dry lapse rate
    rate=np.where(RH<1,rate,6.5) #alter pase rate to moist lapse rate when saturated
    return rate,RH


loc=nc.Dataset('../ne_20191030/TOPO.nc')
lon=loc.variables['lon'][:]
lat=loc.variables['lat'][:]
lev=loc.variables['lev'][:]

lon_c_index=np.where(np.abs(lon-120.6289)<=1e-5)[0][0]
lat_c_index=np.where(np.abs(lat-23.49798)<=1e-5)[0][0]

#enviroment range
grange=4

lat_range=lat[lat_c_index-grange:lat_c_index+grange+1]
lon_range=lat[lon_c_index-grange:lon_c_index+grange+1]

mask=loc.variables['mask'][:,lat_c_index-grange:lat_c_index+grange+1,lon_c_index-grange:lon_c_index+grange+1]
mask=np.broadcast_to(mask[None,:,:,:],(73,60,2*grange+1,2*grange+1))
nannum=np.isnan(np.where(mask[0,:,grange,grange]!=0,mask[0,:,grange,grange],np.nan)).sum()
pres=np.loadtxt('../ne_20191030/fort.98',skiprows=2,usecols=3)
pres=np.broadcast_to(pres[None,:,None,None],(73,60,2*grange+1,2*grange+1))

den=np.loadtxt('../ne_20191030/fort.98',skiprows=2,usecols=1)
den=np.broadcast_to(den[None,:,None,None],(73,60,2*grange+1,2*grange+1))


rv=np.zeros((73,60,2*grange+1,2*grange+1))
th=np.zeros((73,60,2*grange+1,2*grange+1))

for i in range(73):
    path=f'../ne_20191030/ne_20191030.L.Thermodynamic-{i:06d}.nc'
    var=nc.Dataset(path)
    rv[i,:,:,:]=var.variables['qv'][:,:,lat_c_index-grange:lat_c_index+grange+1,lon_c_index-grange:lon_c_index+grange+1]
    th[i,:,:,:]=var.variables['th'][:,:,lat_c_index-grange:lat_c_index+grange+1,lon_c_index-grange:lon_c_index+grange+1]

qv=rv/(1+rv)
temp=th/((1000/(pres/100))**0.286)
temp=np.where(mask,temp,np.nan)

#parcel
temp_parcel=np.zeros((73,len(mask[0,:,grange,grange])))
qv_parcel=np.zeros(73)
temp_parcel[:,nannum]=temp[:,nannum+2,grange,grange]
qv_parcel[:]=qv[:,nannum,grange,grange]
qv_parcel=np.broadcast_to(qv_parcel[:,None],(73,60))
temp_parcel_10=temp_parcel+10
RH_parcel=np.zeros((73,len(mask[0,:,grange,grange])))
RH_parcel_10=np.zeros((73,len(mask[0,:,grange,grange])))

#air parcel iteration
for i in range(nannum+1,len(mask[0,:,grange,grange])):
    del_h=lev[i]-lev[i-1]
    #origin
    rate,RH=lapse(qv_parcel[:,i-1],temp_parcel[:,i-1],den[:,i-1,grange,grange])
    RH_parcel[:,i]=RH
    temp_parcel[:,i]=temp_parcel[:,i-1]-rate*del_h
    
    #heating 10K
    rate_10,RH_10=lapse(qv_parcel[:,i-1],temp_parcel_10[:,i-1],den[:,i-1,grange,grange])
    RH_parcel_10[:,i]=RH_10
    temp_parcel_10[:,i]=temp_parcel_10[:,i-1]-rate_10*del_h


temp_parcel=np.where(mask[0,:,grange,grange],temp_parcel,np.nan)
temp_parcel_10=np.where(mask[0,:,grange,grange],temp_parcel_10,np.nan)
RH_parcel=np.where(RH_parcel<1,RH_parcel,1)
RH_parcel=np.where(mask[0,:,grange,grange],RH_parcel,np.nan)
RH_parcel_10=np.where(RH_parcel_10<1,RH_parcel_10,1)
RH_parcel_10=np.where(mask[0,:,grange,grange],RH_parcel_10,np.nan)


Tv_parcel=temp_parcel*(1+0.608*qv_parcel)
Tv_parcel_10=temp_parcel_10*(1+0.608*qv_parcel)

Tv_env=temp*(1+0.608*qv)
Tv_env=np.nanmean(Tv_env,axis=(2,3))
fb=(Tv_parcel-Tv_env)*9.8/Tv_env
fb_10=(Tv_parcel_10-Tv_env)*9.8/Tv_env

time=np.arange(0,73)
time2,lev2=np.meshgrid(time,lev)

#graphing
normalize = mcolors.TwoSlopeNorm(vcenter=0,vmin=-0.3,vmax=0.2)
plt.clf()
fig,ax=plt.subplots()
ax.contourf(time2,lev2,fb.T,cmap=cm.RdBu_r,norm=normalize)
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cm.RdBu_r)
scalarmappaple.set_array(fb.T)
fig.colorbar(scalarmappaple,ax=ax,extend='both',ticks=np.linspace(-0.3,0.2,11))
plt.title('buoyancy [m/$s^2$]',fontsize=14)
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
plt.xticks(xticks,xtexts)
yticks=np.arange(1,13,2)
ytexts=[f'{y:2d}'for y in yticks]
plt.yticks(yticks,ytexts)
plt.xlim([0,72])
plt.ylim([1,12])
plt.xlabel('Time [LST]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.savefig('./result/buoyancy.png')

normalize = mcolors.TwoSlopeNorm(vcenter=0,vmin=-0.1,vmax=0.35)
plt.clf()
fig,ax=plt.subplots()
ax.contourf(time2,lev2,fb_10.T,cmap=cm.RdBu_r,norm=normalize)
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cm.RdBu_r)
scalarmappaple.set_array(fb_10.T)
fig.colorbar(scalarmappaple,ax=ax,extend='both',ticks=np.arange(-0.1,0.36,0.05))
plt.title('buoyancy after heating 10K [m/$s^2$]',fontsize=14)
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
plt.xticks(xticks,xtexts)
yticks=np.arange(1,13,2)
ytexts=[f'{y:2d}'for y in yticks]
plt.yticks(yticks,ytexts)
plt.xlim([0,72])
plt.ylim([1,12])
plt.xlabel('Time [LST]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.savefig('./result/buoyancy_10.png')

plt.clf()
plt.contourf(time2,lev2,RH_parcel.T,extend='both',levels=np.arange(0,1.1,0.1))
plt.colorbar()
plt.title('RH',fontsize=14)
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
plt.xticks(xticks,xtexts)
yticks=np.arange(1,13,2)
ytexts=[f'{y:2d}'for y in yticks]
plt.yticks(yticks,ytexts)
plt.xlim([0,72])
plt.ylim([1,12])
plt.xlabel('Time [LST]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.savefig('./result/RH.png')

plt.clf()
plt.contourf(time2,lev2,RH_parcel_10.T,extend='both',levels=np.arange(0,1.1,0.1))
plt.colorbar()
plt.title('RH after heating 10K',fontsize=14)
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
plt.xticks(xticks,xtexts)
yticks=np.arange(1,13,2)
ytexts=[f'{y:2d}'for y in yticks]
plt.yticks(yticks,ytexts)
plt.xlim([0,72])
plt.ylim([1,12])
plt.xlabel('Time [LST]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.savefig('./result/RH_10.png')

# plt.clf()
# for i in range(73):
#     plt.clf()
#     plt.plot(temp_parcel[i,:],lev)
#     plt.title('Temperature Profile')
#     xticks=np.arange(220,301,10)
#     xtexts=[f'{x:d}'for x in xticks]
#     plt.xticks(xticks,xtexts)
#     yticks=np.arange(0,12.1,2)
#     ytexts=[f'{y:.1f}'for y in yticks]
#     plt.yticks(yticks,ytexts)
#     plt.xlabel('$T$ [K]')
#     plt.ylabel('Altitude [km]')
#     plt.xlim([220,300])
#     plt.ylim([0,12])
#     plt.grid()
#     plt.savefig(f'./result/temp/t{i:02d}.png')

# gif=[]
# for i in range(73):
#     img=Image.open(f'./result/temp/t{i:02d}.png')
#     gif.append(img)
# gif[0].save('./result/temp/profile.gif',save_all=True,append_images=gif[1:],duration=300)


# plt.clf()
# for i in range(73):
#     plt.clf()
#     plt.plot(temp_parcel_10[i,:],lev)
#     plt.title('Temperature Profile after heating 10K')
#     xticks=np.arange(220,311,10)
#     xtexts=[f'{x:d}'for x in xticks]
#     plt.xticks(xticks,xtexts)
#     yticks=np.arange(0,12.1,2)
#     ytexts=[f'{y:.1f}'for y in yticks]
#     plt.yticks(yticks,ytexts)
#     plt.xlabel('$T$ [K]')
#     plt.ylabel('Altitude [km]')
#     plt.xlim([220,310])
#     plt.ylim([0,12])
#     plt.grid()
#     plt.savefig(f'./result/temp/t_10_{i:02d}.png')

# gif=[]
# for i in range(73):
#     img=Image.open(f'./result/temp/t_10_{i:02d}.png')
#     gif.append(img)
# gif[0].save('./result/temp/profile_10.gif',save_all=True,append_images=gif[1:],duration=300)
