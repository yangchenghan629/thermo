import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors
import PIL as pil
g=9.8
cp=1004

loc=nc.Dataset('/home/B13/b13209015/thermo/ca3/ne_20191030/TOPO.nc')
lon=loc.variables['lon'][:]
lat=loc.variables['lat'][:]
lev=loc.variables['lev'][:] #km

lon_cen_index=np.where(np.abs(lon-120.6289)<=1e-5)[0][0]
lat_cen_index=np.where(np.abs(lat-23.49798)<=1e-5)[0][0]

mask=loc.variables['mask'][:,lat_cen_index,lon_cen_index]
mask=np.broadcast_to(mask[None,:],(73,60))

sfc_height=lev[np.amin(np.where(mask[0,:]==1),axis=1)][0]

pres=np.loadtxt('/home/B13/b13209015/thermo/ca3/ne_20191030/fort.98',skiprows=2,usecols=3)/100 #hPa
pres=np.broadcast_to(pres[None,:],(73,60))
pres=np.where(mask,pres,np.nan)

th=np.zeros((73,60))

for i in range(73):
    path=f'./ne_20191030/ne_20191030.L.Thermodynamic-{i:06d}.nc'
    var=nc.Dataset(path)
    th[i,:]=var.variables['th'][:,:,lat_cen_index,lon_cen_index]

th=np.where(mask,th,np.nan)

z=(lev[1:]+lev[:-1])/2

temp=th/(1000/pres)**0.287
sd=cp*temp+g*lev*1000 

dth_dz=np.gradient(th,lev,axis=1)
dsd_cp_dz=np.gradient(sd/cp,lev,axis=1)
dt_dz=np.gradient(temp,lev,axis=1)

time=np.arange(0,73,1)
dthmax=np.nanargmax(dth_dz,axis=1)

#filled contour of TH PROFILE
# time2,lev2=np.meshgrid(time,lev)
# plt.title('Potential Temperature Profile (shaded)\nand Height at which max gradient (line)',fontsize=12)
# plt.contourf(time2,lev2,th.T,cmap=cm.Reds,extend='both',levels=np.linspace(300,330,9))
# plt.colorbar(label=r'$\theta$[K]')
# plt.plot(time,lev[dthmax],'k-')
# xticks=np.arange(0,73,3)
# xtexts=[f'{x/3:02.0f}'for x in xticks]
# plt.xticks(xticks,xtexts,fontsize=10)
# yticks=np.arange(0,5.1,0.5)
# ytexts=[f'{y:2.1f}'for y in yticks]
# plt.yticks(yticks,ytexts,fontsize=10)
# plt.xlim([18,42])
# plt.ylim([0,5])
# plt.xlabel('Time [LST]',fontsize=12)
# plt.ylabel('Altitude [km]',fontsize=12)
# plt.hlines(y=sfc_height,xmin=18,xmax=42,linestyles='dashed',color='k')
# plt.text(18,sfc_height-0.2,f'{sfc_height*1000:.0f}m')
# plt.savefig('thprofile.png',dpi=500)


# line of TH PROFILE and generate gif
# for i in range(18,43,3):
#     plt.clf()
#     plt.title(f'Profile of Potential Temperature \nand Height at which max gradient {i/3:02.0f}Z',fontsize=14)
#     plt.plot(th[i,:],lev,'b-',label='Potential Temperature')
#     plt.scatter(th[i,dthmax[i]],lev[dthmax][i],color='k',s=[14])
#     plt.text(th[i,dthmax[i]]+0.1,lev[dthmax][i]-0.2,f'{lev[dthmax][i]*1000:.0f}m',fontsize=12)
#     xticks=np.arange(300,331,5)
#     xtexts=[f'{x}'for x in xticks]
#     plt.xticks(xticks,xtexts,fontsize=10)
#     yticks=np.arange(0,5.1,0.5)
#     ytexts=[f'{y:2.1f}'for y in yticks]
#     plt.yticks(yticks,ytexts,fontsize=10)
#     plt.xlim([300,330])
#     plt.ylim([0,5])
#     plt.xlabel('Potential Temperature [K]')
#     plt.ylabel('Altitude [km]')
#     plt.grid()
#     plt.hlines(y=sfc_height,xmin=300,xmax=330,linestyles='dashed',color='k')
#     plt.text(300,sfc_height-0.2,f'{sfc_height*1000:.0f}m')
#     plt.savefig(f'./thprofile/th{i:02d}.png',dpi=500)
# gif=[]
# for i in range(18,43,3):
#     img=pil.Image.open(f'./thprofile/th{i:02d}.png')
#     gif.append(img)
# gif[0].save('thProfile.gif',save_all=True,append_images=gif[1:],duration=700,loop=0)


# filled contour of th,sd,t gradient
# time2,lev2=np.meshgrid(time,lev)
# fig,ax=plt.subplots(nrows=1,ncols=3,sharey='row',figsize=(12,3))
# ax[0].set_title(r'$\dfrac{d\theta}{dz}\ [K/km]$',fontsize=14)
# a0=ax[0].contourf(time2,lev2,dth_dz.T,cmap=cm.RdBu_r)
# xticks=np.arange(0,73,12)
# xtexts=[f'{x/3:02.0f}'for x in xticks]
# ax[0].set_xticks(xticks,xtexts)
# yticks=np.arange(0,12.1,2)
# ytexts=[f'{y:.0f}'for y in yticks]
# ax[0].set_yticks(yticks,ytexts)
# ax[0].set_ylim([0,12])
# ax[0].set_xlabel('Time [LST]',fontsize=12)
# ax[0].set_ylabel('Altitude [km]',fontsize=12)
# plt.colorbar(a0,ax=ax[0])

# ax[1].set_title(r'$\dfrac{d(s_d\ /\ c_p)}{dz}\ [K/km]$',fontsize=14)
# a1=ax[1].contourf(time2,lev2,dsd_cp_dz.T,cmap=cm.RdBu_r)
# xticks=np.arange(0,73,12)
# xtexts=[f'{x/3:02.0f}'for x in xticks]
# ax[1].set_xticks(xticks,xtexts)
# ax[1].set_xlabel('Time [LST]',fontsize=12)
# ax[1].set_ylim([0,12])
# plt.colorbar(a1,ax=ax[1])

# ax[2].set_title(r'$\dfrac{dT}{dz}\ [K/km]$',fontsize=14)
# a2=ax[2].contourf(time2,lev2,dt_dz.T,cmap=cm.RdBu_r)
# xticks=np.arange(0,73,12)
# xtexts=[f'{x/3:02.0f}'for x in xticks]
# ax[2].set_xticks(xticks,xtexts)
# ax[2].set_xlabel('Time [LST]',fontsize=12)
# ax[2].set_ylim([0,12])
# plt.colorbar(a2,ax=ax[2],orientation='vertical')
# plt.savefig('diurnal.png',bbox_inches='tight')


# line of th,t,sd gradient and generate gif
# for i in range(73):
#     plt.clf()
#     plt.title('Vertical Gradient',fontsize=14)
#     plt.plot(dth_dz[i,:],z,'b-',label=r'$\dfrac{d\theta}{dz}$')
#     plt.plot(dsd_cp_dz[i,:],z,'k-',label=r'$\dfrac{d(s_d\ /\ c_p)}{dz}$')
#     plt.plot(dt_dz[i,:],z,'g-',label=r'$\dfrac{dT}{dz}$')
#     plt.plot(-9.8+dsd_cp_dz[i,:],z,'r--',label=r'$\Gamma_d+\dfrac{d(s_d\ /\ c_p)}{dz}$')
#     plt.legend(fontsize=8,bbox_to_anchor=(1.01,1),loc='upper left')
#     plt.xlim([-10,15])
#     plt.ylim([0,12])
#     plt.grid()
#     plt.xlabel('Gradient [K/km]',fontsize=12)
#     plt.ylabel('Altitude [km]',fontsize=12)
#     plt.savefig(f'./gradient/dthdz{i:02d}.png',bbox_inches='tight',dpi=500)
# gif=[]
# for i in range(73):
#     img=pil.Image.open(f'./gradient/dthdz{i:02d}.png')
#     gif.append(img)
# gif[0].save('gradient.gif',save_all=True,append_images=gif[1:],duration=300,loop=1)


# difference of sd and th gradient
# diff=dth_dz-dsd_cp_dz
# for i in range(73):
#     plt.clf()
#     plt.title(r'$Vertical\ Gradient\ Difference\ between\ \theta\ and\ s_d$',fontsize=14)
#     plt.plot(diff[i,:],z,'b-',label=r'$\dfrac{d\theta}{dz}-\dfrac{d(s_d/c_p)}{dz}$')
#     plt.legend(fontsize=10,loc='upper left')
#     plt.xlim([-2,2])
#     plt.ylim([0,12])
#     plt.grid()
#     plt.xlabel('Difference [K/km]',fontsize=12)
#     plt.ylabel('Altitude [km]',fontsize=12)
#     plt.savefig(f'./difference/diff{i:02d}.png',bbox_inches='tight',dpi=500)
# gif=[]
# for i in range(73):
#     img=pil.Image.open(f'./difference/diff{i:02d}.png')
#     gif.append(img)
# gif[0].save('Difference.gif',save_all=True,append_images=gif[1:],duration=300,loop=1)


# hourly profile of th
plt.title('Hourly Potential Temperature Profile')
for i in range(18,43,3):
    plt.plot(th[i,:],lev,label=f'{i/3:02.0f}Z')
plt.hlines(y=sfc_height,xmin=300,xmax=320,linestyles='dashed',color='k')
plt.legend()
plt.xlim([300,320])
plt.ylim([1,5])
plt.grid()
plt.xlabel('Potential Temperature [K]')
plt.ylabel('Altitude [km]')
plt.savefig('hourly_th_profile.png',dpi=500)

# PROFILE of dT/dz and max dTH/dz
# time2,z2=np.meshgrid(time,z)
# plt.title('Temperature Gradient Profile (shaded)\nHeight at which MAX Potential Temperature Gradient (line)',fontsize=12)
# plt.contourf(time2,z2,dth_dz.T,cmap=cm.RdBu_r,extend='both',levels=np.linspace(-10,20,11))
# plt.colorbar()
# plt.plot(time,lev[dthmax],'k-')
# xticks=np.arange(0,73,3)
# xtexts=[f'{x/3:02.0f}'for x in xticks]
# plt.xticks(xticks,xtexts,fontsize=10)
# yticks=np.arange(0,5.1,0.5)
# ytexts=[f'{y:2.1f}'for y in yticks]
# plt.yticks(yticks,ytexts,fontsize=10)
# plt.xlim([18,42])
# plt.ylim([0,5])
# plt.xlabel('Time [LST]',fontsize=12)
# plt.ylabel('Altitude [km]',fontsize=12)
# plt.hlines(y=sfc_height,xmin=18,xmax=42,linestyles='dashed',color='k')
# plt.text(18,sfc_height-0.2,f'{sfc_height*1000:.0f}m')
# plt.savefig('dthdz_max_dTdz.png',dpi=500)
