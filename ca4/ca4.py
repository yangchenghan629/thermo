import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors

# 10 hour
var=nc.Dataset('/home/B13/b13209015/thermo/ca4/TC/axmean-000030.nc')
radius=var.variables['radius'][:]/1000
height=var.variables['zc'][:]/1000
radi_wind=var.variables['radi_wind'][0,0,:,:]
tang_wind=var.variables['tang_wind'][0,0,:,:]


qi=var.variables['qi'][0,0,:,:]
qc=var.variables['qc'][0,0,:,:]
qr=var.variables['qr'][0,0,:,:]

qi=np.where(np.abs(qi)>=5e-5,qi,0)
qc=np.where(np.abs(qc)>=5e-5,qc,0)
qr=np.where(np.abs(qr)>=5e-5,qr,0)

cloud=(qi+qc+qr)*1000

rr,hh=np.meshgrid(radius,height)

fig,ax=plt.subplots(ncols=1,nrows=2,sharex=True,layout='constrained')
plt.suptitle('TC structure at 10 Hours Intergration',fontsize=14)
ax[0].set_title('Maximum Speed along the Radius',fontsize=12)
ax[0].plot(radius,np.amax(np.abs(radi_wind),axis=0),label='radial')
ax[0].plot(radius,np.amax(np.abs(tang_wind),axis=0),label='tangential')
ax[0].set_xlim([0,750])
ax[0].set_ylabel('Wind Speed [m/s]',fontsize=10)
ax[0].legend()
ax[0].grid()

ax[1].set_title('Cross-Sections of Clouds\nand Height of Max Speed',fontsize=12)
a1=ax[1].pcolormesh(rr,hh,cloud,cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=6,vmin=0))
ax[1].set_xlim([0,750])
ax[1].set_xlabel('Radius [km]',fontsize=10)
ax[1].set_ylabel('Height [km]',fontsize=10)
fig.colorbar(a1,ax=ax[1],extend='max',label='Mixing Ratio [g/kg]',ticks=[0,0.05,0.1,2,4,6])
ax[1].plot(radius,height[np.argmax(np.abs(radi_wind),axis=0)],'k-',label='radial')
ax[1].plot(radius,height[np.argmax(np.abs(tang_wind),axis=0)],'k--',label='tangential')
ax[1].legend(bbox_to_anchor=(0.5,-0.5),ncols=2,loc='center')

plt.savefig('10hr_MaxSpeed_CrossSection.png',dpi=500,bbox_inches='tight')


plt.clf()
fig,ax=plt.subplots(ncols=1,nrows=3,sharex=True,layout='constrained')
plt.suptitle('10 hrs cloud cross-section',fontsize=14)
a0=ax[0].pcolormesh(rr,hh,qc*1000,cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=3,vmin=0))
ax[0].set_title('$q_c$ (cloud water mixing ratio)')
ax[0].set_ylabel('Height [km]')
a1=ax[1].pcolormesh(rr,hh,qi*1000,cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=3,vmin=0))
ax[1].set_title('$q_i$ (cloud ice mixing ratio)')
ax[1].set_ylabel('Height [km]')
a2=ax[2].pcolormesh(rr,hh,qr*1000,cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=3,vmin=0))
ax[2].set_title('$q_r$ (cloud rain mixing ratio)')
ax[2].set_ylabel('Height [km]')
ax[2].set_xlabel('Radius [km]')
fig.colorbar(a2,ax=ax[:],extend='max',label='g/kg',ticks=[0,0.05,0.1,1,1.5,2,2.5,3])
plt.savefig('qiqcqr10.png')


# 200 hour
var=nc.Dataset('/home/B13/b13209015/thermo/ca4/TC/axmean-000600.nc')
radius=var.variables['radius'][:]/1000
height=var.variables['zc'][:]/1000
radi_wind=var.variables['radi_wind'][0,0,:,:]
tang_wind=var.variables['tang_wind'][0,0,:,:]
qi=var.variables['qi'][0,0,:,:]
qc=var.variables['qc'][0,0,:,:]
qr=var.variables['qr'][0,0,:,:]

qi=np.where(np.abs(qi)>=5e-5,qi,0)
qc=np.where(np.abs(qc)>=5e-5,qc,0)
qr=np.where(np.abs(qr)>=5e-5,qr,0)

cloud=(qi+qc+qr)*1000

rr,hh=np.meshgrid(radius,height)

fig,ax=plt.subplots(ncols=1,nrows=2,sharex=True,layout='constrained')
plt.suptitle('TC structure at 200 Hours Intergration',fontsize=14)
ax[0].set_title('Maximum Speed along the Radius',fontsize=12)
ax[0].plot(radius,np.amax(np.abs(radi_wind),axis=0),label='radial')
ax[0].plot(radius,np.amax(np.abs(tang_wind),axis=0),label='tangential')
ax[0].set_xlim([0,750])
ax[0].set_ylabel('Wind Speed [m/s]',fontsize=10)
ax[0].legend()
ax[0].grid()

ax[1].set_title('Cross-Sections of Clouds\nand Height of Max Speed',fontsize=12)
a1=ax[1].pcolormesh(rr,hh,cloud,cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=6,vmin=0))
ax[1].set_xlim([0,750])
ax[1].set_xlabel('Radius [km]',fontsize=10)
ax[1].set_ylabel('Height [km]',fontsize=10)
fig.colorbar(a1,ax=ax[1],extend='max',label='Mixing Ratio [g/kg]',ticks=[0,0.05,0.1,2,4,6])
ax[1].plot(radius,height[np.argmax(np.abs(radi_wind),axis=0)],'k-',label='radial')
ax[1].plot(radius,height[np.argmax(np.abs(tang_wind),axis=0)],'k--',label='tangential')
ax[1].legend(bbox_to_anchor=(0.5,-0.5),ncols=2,loc='center')

plt.savefig('200hr_MaxSpeed_CrossSection.png',dpi=500,bbox_inches='tight')

plt.clf()
fig,ax=plt.subplots(ncols=1,nrows=3,sharex=True,layout='constrained')
plt.suptitle('200 hrs cloud cross-section',fontsize=14)
a0=ax[0].pcolormesh(rr,hh,qc*1000,cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=3,vmin=0))
ax[0].set_title('$q_c$ (cloud water mixing ratio)')
ax[0].set_ylabel('Height [km]')
a1=ax[1].pcolormesh(rr,hh,qi*1000,cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=3,vmin=0))
ax[1].set_title('$q_i$ (cloud ice mixing ratio)')
ax[1].set_ylabel('Height [km]')
a2=ax[2].pcolormesh(rr,hh,qr*1000,cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=3,vmin=0))
ax[2].set_title('$q_r$ (cloud rain mixing ratio)')
ax[2].set_ylabel('Height [km]')
ax[2].set_xlabel('Radius [km]')
fig.colorbar(a2,ax=ax[:],extend='max',label='g/kg',ticks=[0,0.05,0.1,1,1.5,2,2.5,3])
plt.savefig('qiqcqr200.png')


