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

for i in range(73):
    plt.clf()
    fig,ax=plt.subplots(1,2,sharey='row')
    plt.suptitle(f'Vertical Profile-{i:02d}',fontsize=14)
    ax[0].set_title(f'Temperature T\nVirtual Temperature $T_v$',fontsize=12)
    ax[0].plot(temp[i,:,2,2],lev,'g-')
    ax[0].plot(Tv[i,:,2,2],lev,'b-')
    ax[0].plot(np.linspace(220,300,1000),np.tile(lev[np.nanmin(np.where(mask[:,2,2]!=0))],(1000)),'k--')
    ax[0].text(220,lev[np.nanmin(np.where(mask[:,2,2]!=0))]+0.1,f'{np.nanmin(lev[np.nanmin(np.where(mask[:,2,2]!=0))]*1000):.0f} m')
    xticks=np.arange(220,301,20)
    xtexts=[f'{x:d}'for x in xticks]
    ax[0].set_xticks(xticks,xtexts)
    yticks=np.arange(0,12.1,2)
    ytexts=[f'{y:.1f}'for y in yticks]
    ax[0].set_yticks(yticks,ytexts)
    ax[0].set_xlabel('$T_{v}$ [K]')
    ax[0].set_ylabel('Altitude [km]')
    ax[0].set_xlim([220,300])
    ax[0].set_ylim([0,12])
    ax[0].legend(['T','$T_v$'],loc='upper right')
    ax[0].grid()

    ax[1].set_title(f'Spesific Humidity $q_v$',fontsize=12)
    ax[1].plot(qv[i,:,2,2]*1000,lev,'r-')
    ax[1].plot(np.linspace(0,20,1000),np.tile(lev[np.nanmin(np.where(mask[:,2,2]!=0))],(1000)),'k--')
    ax[1].text(0,lev[np.nanmin(np.where(mask[:,2,2]!=0))]+0.1,f'{np.nanmin(lev[np.nanmin(np.where(mask[:,2,2]!=0))]*1000):.0f} m')
    xticks=np.arange(0,21,5)
    xtexts=[f'{x:d}'for x in xticks]
    ax[1].set_xticks(xticks,xtexts)
    yticks=np.arange(0,12.1,2)
    ytexts=[f'{y:.1f}'for y in yticks]
    ax[1].set_yticks(yticks,ytexts)
    ax[1].set_xlabel('$q_{v}$ [g/kg]')
    ax[1].set_xlim([0,20])
    ax[1].set_ylim([0,12])
    ax[1].grid()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.1, wspace=0.2, hspace=0.9)
    plt.savefig(f'./prob1/profile{i:02d}.png',dpi=500)
    plt.close()

gif=[]
for i in range(73):
    img=Image.open(f'./prob1/profile{i:02d}.png')
    gif.append(img)

gif[0].save('Profile.gif',save_all=True,append_images=gif[1:],duration=300,loop=1,disposol=2)


plt.clf()
time=np.arange(0,73,1)
lev2,time2=np.meshgrid(lev,time)
fig,ax=plt.subplots(2)
plt.suptitle('Vertical Diurnal Profile of Virtual Temperature ($T_v$ [K])',fontsize=14)
a1=ax[0].contourf(time2,lev2,Tv[:,:,2,2],cmap=cm.RdYlBu_r,levels=np.linspace(220,300,11),extend='both')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
ax[0].set_xticks(xticks,xtexts)
ax[0].set_xlabel('Time (LST)')
yticks=np.arange(0,13,2)
ytexts=[f'{y:2d}'for y in yticks]
ax[0].set_yticks(yticks,ytexts)
ax[0].set_ylabel('Altitude (km)')
ax[0].set_xlim([0,72])
ax[0].set_ylim([0,12])
plt.colorbar(a1,ax=ax[0],orientation='vertical')

a2=ax[1].contourf(time2,lev2,Tv[:,:,2,2],cmap=cm.RdYlBu_r,levels=np.linspace(280,300,11),extend='both')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
ax[1].set_xticks(xticks,xtexts)
ax[1].set_xlabel('Time (LST)')
yticks=np.arange(1,3.1,0.5)
ytexts=[f'{y:.1f}'for y in yticks]
ax[1].set_yticks(yticks,ytexts)
ax[1].set_ylabel('Altitude (km)')
ax[1].set_xlim([0,72])
ax[1].set_ylim([1,3])
plt.colorbar(a2,ax=ax[1],orientation='vertical')

plt.tight_layout()
plt.savefig('Tv_contourf.png',dpi=500)

plt.clf()
fig,ax=plt.subplots(2,1)
plt.suptitle('Vertical Diurnal Profile of Temperature [K]',fontsize=14)
a1=ax[0].contourf(time2,lev2,temp[:,:,2,2],cmap=cm.RdYlBu_r,levels=np.linspace(220,300,9),extend='both')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
ax[0].set_xticks(xticks,xtexts)
ax[0].set_xlabel('Time (LST)')
yticks=np.arange(0,13,2)
ytexts=[f'{y:2d}'for y in yticks]
ax[0].set_yticks(yticks,ytexts)
ax[0].set_ylabel('Altitude (km)')
ax[0].set_xlim([0,72])
ax[0].set_ylim([0,12])
plt.colorbar(a1,ax=ax[0],orientation='vertical')

a2=ax[1].contourf(time2,lev2,temp[:,:,2,2],cmap=cm.RdYlBu_r,levels=np.linspace(280,300,11),extend='both')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
ax[1].set_xticks(xticks,xtexts)
ax[1].set_xlabel('Time (LST)')
yticks=np.arange(1,3.1,0.5)
ytexts=[f'{y:.1f}'for y in yticks]
ax[1].set_yticks(yticks,ytexts)
ax[1].set_ylabel('Altitude (km)')
ax[1].set_xlim([0,72])
ax[1].set_ylim([1,3])
plt.colorbar(a2,ax=ax[1],orientation='vertical')

plt.tight_layout()
plt.savefig('temp_contourf.png',dpi=500)

plt.clf()
fig,ax=plt.subplots(2,1)
plt.suptitle('Vertical Diurnal Profile of Specific Humidity [g/kg]',fontsize=14)
a1=ax[0].contourf(time2,lev2,qv[:,:,2,2]*1000,cmap=cm.RdYlBu_r,levels=np.linspace(0,20,10),extend='both')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
ax[0].set_xticks(xticks,xtexts)
ax[0].set_xlabel('Time (LST)')
yticks=np.arange(0,13,2)
ytexts=[f'{y:2d}'for y in yticks]
ax[0].set_yticks(yticks,ytexts)
ax[0].set_ylabel('Altitude (km)')
ax[0].set_xlim([0,72])
ax[0].set_ylim([0,12])
plt.colorbar(a1,ax=ax[0],orientation='vertical')

a2=ax[1].contourf(time2,lev2,temp[:,:,2,2],cmap=cm.RdYlBu_r,levels=np.linspace(280,300,11),extend='both')
xticks=np.arange(0,73,6)
xtexts=[f'{x/3:02.0f}'for x in xticks]
ax[1].set_xticks(xticks,xtexts)
ax[1].set_xlabel('Time (LST)')
yticks=np.arange(1,3.1,0.5)
ytexts=[f'{y:.1f}'for y in yticks]
ax[1].set_yticks(yticks,ytexts)
ax[1].set_ylabel('Altitude (km)')
ax[1].set_xlim([0,72])
ax[1].set_ylim([1,3])
plt.colorbar(a2,ax=ax[1],orientation='vertical')

plt.tight_layout()
plt.savefig('qv_contourf.png',dpi=500)
