import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

loc=nc.Dataset('./ne_20191030/TOPO.nc')
lon=loc.variables['lon'][:]
lat=loc.variables['lat'][:]
lev=loc.variables['lev'][:]

lon_c_index=np.where(np.abs(lon-120.6289)<=1e-5)[0][0]
lat_c_index=np.where(np.abs(lat-23.49798)<=1e-5)[0][0]

lon_c=lon[lon_c_index]
lat_c=lat[lat_c_index]

lon_range=lon[lon_c_index-2:lon_c_index+3]
lat_range=lat[lat_c_index-2:lat_c_index+3]

mask=loc.variables['mask'][:,lat_c_index-2:lat_c_index+3,lon_c_index-2:lon_c_index+3]

pres=np.loadtxt('./ne_20191030/fort.98',skiprows=2,usecols=3)
pres=np.broadcast_to(pres[None,:,None,None],(73,60,5,5))

rv=np.zeros((73,60,5,5))
th=np.zeros((73,60,5,5))
for i in range(73):
    path=f'./ne_20191030/ne_20191030.L.Thermodynamic-{i:06d}.nc'
    var=nc.Dataset(path)
    rv[i,:,:,:]=np.where(mask,var.variables['qv'][:,:,lat_c_index-2:lat_c_index+3,lon_c_index-2:lon_c_index+3],np.nan)
    th[i,:,:,:]=np.where(mask,var.variables['th'][:,:,lat_c_index-2:lat_c_index+3,lon_c_index-2:lon_c_index+3],np.nan)

qv=rv/(1+rv)
temp=th/((1000/(pres/100))**0.286)
Tv=temp*(1+0.608*qv)

diff=Tv-temp

diff0=[]
for i in range(73):
    diff0.append(np.amin(np.where(diff[i,:,2,2]<0.1)))

for i in range(73):
    plt.clf()
    plt.plot(diff[i,:,2,2],lev,'b-')
    plt.plot(np.linspace(0,20,1000),np.tile(lev[np.nanmin(np.where(mask[:,2,2]!=0))],(1000)),'k--')
    plt.scatter([diff[i,diff0[i],2,2]],[lev[diff0[i]]],color='k',s=[10])
    plt.text(diff[i,diff0[i],2,2]+0.1,lev[diff0[i]],f'{lev[diff0[i]]:.2f} km')
    plt.text(0,lev[np.nanmin(np.where(mask[:,2,2]!=0))]+0.1,f'{np.nanmin(lev[np.nanmin(np.where(mask[:,2,2]!=0))]*1000):.0f} m')
    plt.title(f'Difference between Virtual Temperature and Temperature\nprofile-{i:02d}')
    xticks=np.arange(0,5,1)
    xtexts=[f'{x:d}'for x in xticks]
    plt.xticks(xticks,xtexts)
    yticks=np.arange(0,12.1,2)
    ytexts=[f'{y:.1f}'for y in yticks]
    plt.yticks(yticks,ytexts)
    plt.xlim([0,4])
    plt.ylim([0,12])
    plt.xlabel('difference [K]')
    plt.ylabel('Altitude [km]')
    plt.savefig(f'./prob2/diff{i:02d}.png',dpi=600)



gif=[]
for i in range(73):
    img=Image.open(f'./prob2/diff{i:02d}.png')
    gif.append(img)

gif[0].save('diff.gif',save_all=True,append_images=gif[1:],duration=300,loop=1,disposol=2)


plt.clf()
time=np.arange(0,73,1)
lev2,time2=np.meshgrid(lev,time)
# fig,ax=plt.subplots(1)
plt.title('Vertical Diurnal Profile of Difference between $T_v$ and T [K]',size=14)
plt.contourf(time2,lev2,diff[:,:,2,2],cmap=cm.RdYlBu_r,levels=[0,0.1,0.3,0.5,0.7,0.9,1,2],extend='max')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
plt.xticks(xticks,xtexts)
plt.xlabel('Time (LST)')
yticks=np.arange(0,13,2)
ytexts=[f'{y:2d}'for y in yticks]
plt.yticks(yticks,ytexts)
plt.ylabel('Altitude (km)')
plt.xlim([0,72])
plt.ylim([1,12])
plt.colorbar(orientation='vertical')

plt.tight_layout()
plt.savefig('diff_contourf.png',dpi=600)