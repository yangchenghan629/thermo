import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors
from tools import Thermo
#############################
# READ DATA
#############################
radius=nc.Dataset('./TC/axmean-000000.nc').variables['radius'][:]
zc=nc.Dataset('./TC/axmean-000000.nc').variables['zc'][:]

th,qv,qvs,qr,radi_wind,tang_wind=[np.full((201,34,192),np.nan) for _ in range(6)]
for i in range(601):
    var=nc.Dataset(f'./TC/axmean-{i:06d}.nc')
    th[int(i//3),:,:]=var.variables['th'][0,0,:,:]
    qv[int(i//3),:,:]=var.variables['qv'][0,0,:,:]
    qvs[int(i//3),:,:]=var.variables['qvs'][0,0,:,:]
    qr[int(i//3),:,:]=var.variables['qr'][0,0,:,:]
    radi_wind[int(i//3),:,:]=var.variables['radi_wind'][0,0,:,:]
    tang_wind[int(i//3),:,:]=var.variables['tang_wind'][0,0,:,:]
pres=np.broadcast_to(np.loadtxt('./TC/fort.98',skiprows=237,usecols=3,max_rows=34)[None,:,None],(201,34,192))

#############################
# CALCULATION
#############################
temp=Thermo.temp_from_th(th,pres)
Tc,_=Thermo.Tc_Zc(temp,pres,qv,dim=3)
th_e=Thermo.th_equivalent(th,Tc,qv)
th_es=Thermo.th_equivalent(th,Tc,qvs)

max_radi_index=np.argmax(np.abs(radi_wind[:,0,:]),axis=1)
r150_index=np.where(np.abs(radius-150000)<10)[0][0]

#############################
# PLOTTING
#############################

# TH_E(S) @20 HRS
fig,ax=plt.subplots(nrows=1,ncols=2,sharey='row')
ax[0].plot(th_e[20,:,max_radi_index[20]],zc/1000,label=r'$\theta_{e}$',color='blue')
ax[0].plot(th_es[20,:,max_radi_index[20]],zc/1000,label=r'$\theta_{es}$',color='red')
ax[0].set_xticks(np.arange(310,391,20))
ax[0].set_xlim([310,390])
ax[0].set_ylim([0,18])
ax[0].set_title('@RMW',fontsize=12)
ax[0].set_xlabel(r'$\theta_{e},\theta_{es}\ [K]$ ',fontsize=12)
ax[0].set_ylabel('Altitude [km]',fontsize=12)
ax[0].legend()
ax[0].grid()

ax[1].plot(th_e[20,:,r150_index],zc/1000,label=r'$\theta_{e}$',color='blue')
ax[1].plot(th_es[20,:,r150_index],zc/1000,label=r'$\theta_{es}$',color='red')
ax[1].set_xticks(np.arange(310,391,20))
ax[1].set_xlim([310,390])
ax[1].set_ylim([0,18])
ax[1].set_title('@150km',fontsize=12)
ax[1].set_xlabel(r'$\theta_{e},\theta_{es}\ [K]$ ',fontsize=12)
ax[1].grid()
fig.suptitle(r'$\theta_{e}\ and\ \theta_{es}$ 20 hrs',fontsize=14)
plt.savefig('./result/TC20_th_e(s).png',dpi=450)

# TH_E(S) @ 180 HRS
plt.clf()
fig,ax=plt.subplots(nrows=1,ncols=2,sharey='row')
ax[0].plot(th_e[180,:,max_radi_index[180]],zc/1000,label=r'$\theta_{e}$',color='blue')
ax[0].plot(th_es[180,:,max_radi_index[180]],zc/1000,label=r'$\theta_{es}$',color='red')
ax[0].set_xticks(np.arange(300,391,20))
ax[0].set_xlim([300,390])
ax[0].set_ylim([0,18])
ax[0].set_title('@RMW',fontsize=12)
ax[0].set_xlabel(r'$\theta_{e},\theta_{es}\ [K]$ ',fontsize=12)
ax[0].set_ylabel('Altitude [km]',fontsize=12)
ax[0].legend()
ax[0].grid()

ax[1].plot(th_e[180,:,r150_index],zc/1000,label=r'$\theta_{e}$',color='blue')
ax[1].plot(th_es[180,:,r150_index],zc/1000,label=r'$\theta_{es}$',color='red')
ax[1].set_xticks(np.arange(300,391,20))
ax[1].set_xlim([300,390])
ax[1].set_ylim([0,18])
ax[1].set_title('@150km',fontsize=12)
ax[1].set_xlabel(r'$\theta_{e},\theta_{es}\ [K]$ ',fontsize=12)
ax[1].grid()
fig.suptitle(r'$\theta_{e}\ and\ \theta_{es}$ 180 hrs',fontsize=14)
plt.savefig('./result/TC180_th_e(s).png',dpi=450)

# QR PROFILE
# 20 HRS
plt.clf()
rr,hh=np.meshgrid(radius,zc)
plt.pcolormesh(rr/1000,hh/1000,qr[20,:,:]*1000,cmap=cm.Blues)
plt.vlines(radius[max_radi_index[20]]/1000,0,18,colors='r',linewidths=1)
plt.text(radius[max_radi_index[20]]/1000+10,12,'RMW',fontsize=10,color='r')
plt.ylim([0,zc[-1]/1000])
plt.title('$q_r$ Profile - 20 hrs',fontsize=14)
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.colorbar(extend='max',label='mixing ratio [g/kg]')
plt.savefig('./result/qr20.png',dpi=450)

# 180 HRS
plt.clf()
plt.pcolormesh(rr/1000,hh/1000,qr[180,:,:]*1000,cmap=cm.Blues)
plt.vlines(radius[max_radi_index[180]]/1000,0,18,colors='r',linewidths=1)
plt.text(radius[max_radi_index[180]]/1000+10,12,'RMW',fontsize=10,color='r')
plt.ylim([0,zc[-1]/1000])
plt.title('$q_r$ Profile - 180 hrs',fontsize=14)
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.colorbar(extend='max',label='mixing ratio [g/kg]')
plt.savefig('./result/qr180.png',dpi=450)