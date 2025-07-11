import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors , matplotlib.lines as mlines
import netCDF4 as nc
from tools import Thermo


#################################
# READ DATA
#################################

# topography
loc_file=nc.Dataset('./ne_20191030/TOPO.nc')
lat=loc_file.variables['lat'][:]
lon=loc_file.variables['lon'][:]
lev=loc_file.variables['lev'][:]
lat_index=np.where(np.abs(lat-23.49798)<=1e-5)[0][0]
lon_index=np.where(np.abs(lon-120.6289)<=1e-5)[0][0]
mask=loc_file.variables['mask'][:,lat_index,lon_index]
mask_2d=np.broadcast_to(mask,(73,60))

surface_index=np.argmax(mask)

# thermodynamics
th,rv,qi,qc=[np.full((73,60),np.nan) for _ in range(4)] 
for i in range(73):
    var=nc.Dataset(f'./ne_20191030/ne_20191030.L.Thermodynamic-{i:06d}.nc')
    th[i,:]=np.where(mask,var.variables['th'][0,:,lat_index,lon_index],np.nan) # K
    rv[i,:]=np.where(mask,var.variables['qv'][0,:,lat_index,lon_index],np.nan) # kg/kg
    qi[i,:]=var.variables['qi'][0,:,lat_index,lon_index]*1000 # g/kg
    qc[i,:]=var.variables['qc'][0,:,lat_index,lon_index]*1000 # g/kg
qv=rv/(1+rv) # kg/kg
# qi=np.where(mask_2d,np.where(qi>0.1,qi,0),np.nan)
# qc=np.where(mask_2d,np.where(qc>0.1,qc,0),np.nan)

pres=np.broadcast_to(np.loadtxt('./ne_20191030/fort.98',usecols=3,skiprows=2)/100,(73,60)) #hPa
dens=np.broadcast_to(np.loadtxt('./ne_20191030/fort.98',usecols=1,skiprows=2),(73,60))

#################################
# CALCULATION
#################################

temp=Thermo.temp_from_th(th,pres) # K
Tc,Zc=Thermo.Tc_Zc(temp,pres,qv,lev,dim=1) # K , km
Tc_diurnal,_=Thermo.Tc_Zc(temp,pres,qv,lev,dim=2)
Tc_diurnal=np.where(mask_2d,Tc_diurnal,np.nan)
Te=Thermo.Te(temp,qv) # K
hm=Thermo.hm(temp,lev*1000,qv) # J/kg
th_e=Thermo.th_equivalent(th,Tc_diurnal,qv) # K

#################################
# PLOTTING
#################################
# T&TE , TH&THE&HM , QV&QC PROFILE
for i in range(73):
    plt.clf()
    fig,ax=plt.subplots(nrows=1,ncols=3,sharey='row',figsize=(6,5))
    ax[0].plot(temp[i,:],lev,'b',label='T')
    ax[0].plot(Te[i,:],lev,'g',label='$T_e$')
    ax[0].hlines(Zc[i],0,400,linestyles='dashed',linewidths=1,colors='k')
    ax[0].hlines(lev[surface_index],0,400,linestyles='dashed',linewidths=1,colors='k')
    ax[0].text(222,Zc[i]+0.2,f'Zc:{Zc[i]:.2f}km',fontsize=10,color='k')
    ax[0].text(222,lev[surface_index]-0.5,f'SFC:{lev[surface_index]:.2f}km')
    ax[0].set_xticks(np.arange(200,401,40))
    ax[0].set_xlim(220,340)
    ax[0].set_ylim(0,12)
    ax[0].set_title('$T\ and\ T_e$',fontsize=12,pad=10)
    ax[0].set_xlabel('$T,T_e$ [K]',fontsize=12)
    ax[0].set_ylabel('Height [km]',fontsize=12)
    ax[0].grid()
    ax[0].legend(fontsize=8)

    ax[1].plot(th[i,:],lev,'b',label=r'$\theta$')
    ax[1].plot(th_e[i,:],lev,'g',label=r'$\theta_e$')
    ax[1].plot(hm[i,:]/Thermo.cp,lev,'r--',label='$h_m\ /\ cp$')
    ax[1].hlines(Zc[i],0,400,linestyles='dashed',linewidths=1,colors='k')
    ax[1].hlines(lev[surface_index],0,400,linestyles='dashed',linewidths=1,colors='k')
    ax[1].set_xticks(np.arange(0,401,20))
    ax[1].set_xlim(290,360)
    ax[1].set_title(r'$\theta,\theta_e\ and\ \frac{h_m}{c_p}$',fontsize=12,pad=10)
    ax[1].set_xlabel(r'$\theta,\theta_e,\frac{h_m}{c_p}$ [K]',fontsize=12)
    ax[1].legend(fontsize=8)
    ax[1].grid()

    line_qv,=ax[2].plot(qv[i,:]*1000,lev,'b',label='qv')
    ax[2].set_xlabel('$q_v$ [g/kg]',fontsize=12,color='b')
    ax[2].tick_params(axis='x',labelsize=8,colors='b')
    ax[2].set_xlim([-0.2,20])
    ax[2].set_ylim([0,12])
    
    a22=ax[2].twiny()
    line_qc,=a22.plot(qc[i,:],lev,'g',label='qc')
    a22.set_xlabel('$q_c$ [g/kg]',fontsize=12,color='g')
    a22.tick_params(axis='x',labelsize=8,colors='g')
    a22.set_xlim([0,1.5])
    a22.set_ylim([0,12])

    ax[2].hlines(Zc[i],0,20,linestyle='dashed',color='k',linewidth=1)
    ax[2].hlines(lev[surface_index],0,20,linestyle='dashed',color='k',linewidth=1)
    a22.set_title('$q_v\ and\ q_c$',fontsize=12,pad=10)
    ax[2].grid()
    ax[2].legend(handles=[line_qv, line_qc],fontsize=8)

    plt.suptitle(f'Time={i//3} hr {i*20%60} min | Frame-{i:02d}',fontsize=14)
    plt.tight_layout()
    plt.savefig(f'./1.1/profile{i:02d}.png',dpi=450)
    plt.close()


time=np.arange(0,73)
time2,lev2=np.meshgrid(time,lev)


# Base:Tc ,Contour:qv ,color contour:qc,qi
## REVISED : BASE:qv plot ZC and marked qc,qi
plt.clf()
plt.contourf(time2,lev2,qv.T*1000,cmap=cm.coolwarm,extend='max',levels=np.arange(0,17,2))
plt.colorbar(label='$q_v$ [g/kg]')
plt.contour(time2,lev2,qi.T,levels=[0.1],colors='yellow')
plt.contour(time2,lev2,qc.T,levels=[0.1],colors='green')
plt.plot(time,Zc,'k--',label='Zc')
legend_element=[mlines.Line2D([0],[0],color='green',label='$q_c$'),mlines.Line2D([0],[0],color='yellow',label='$q_i$'),mlines.Line2D([0],[0],color='k',linestyle='dashed',label='$Z_c$')]
plt.legend(handles=legend_element,loc='lower right',ncols=3,fontsize=9)
plt.xticks(np.arange(0,73,9),[f'{h//3:02d}'for h in range(0,73,9)])
plt.ylim([0,12])
plt.xlabel('Time [hr]',fontsize=12)
plt.ylabel('Height [km]',fontsize=12)
plt.title('$q_v,q_c,q_i$ Profile',fontsize=14)
plt.savefig('./result/diurnal.2R.png',dpi=450)
