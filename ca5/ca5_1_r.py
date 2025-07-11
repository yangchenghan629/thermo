import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors , matplotlib.lines as mline

# PARAMETER
#############################
Rv=461.5 # J/K.kg
es0=6.11 # hPa
T0=273.15 # K
Lv=2.5e6 # J/kg
Lf=3.34e5 # J/kg
Ls=Lv+Lf # J/kg
epsilon=0.622 

# READ DATA
################################
loc=nc.Dataset('./ne_20191030/TOPO.nc')
lat=loc.variables['lat'][:]
lon=loc.variables['lon'][:]
lev=loc.variables['lev'][:]
lat_index=np.where(np.abs(lat-23.49798)<=1e-5)[0][0]
lon_index=np.where(np.abs(lon-120.6289)<=1e-5)[0][0]
mask=loc.variables['mask'][:,lat_index,lon_index]

th=np.full((73,60),np.nan)
rv=np.full((73,60),np.nan)
qi=np.full((73,60),np.nan)
qc=np.full((73,60),np.nan)
for i in range(73):
    var=nc.Dataset(f'./ne_20191030/ne_20191030.L.Thermodynamic-{i:06d}.nc')
    th[i,:]=np.where(mask,var.variables['th'][0,:,lat_index,lon_index],np.nan) # K
    rv[i,:]=np.where(mask,var.variables['qv'][0,:,lat_index,lon_index],np.nan) # kg/kg
    qi[i,:]=var.variables['qi'][0,:,lat_index,lon_index]*1000 # g/kg
    qc[i,:]=var.variables['qc'][0,:,lat_index,lon_index]*1000 # g/kg
qv=rv/(1+rv) # kg/kg

qi=np.where(mask,np.where(qi>0.1,qi,0),np.nan)
qc=np.where(mask,np.where(qc>0.1,qc,0),np.nan)

pres=np.broadcast_to(np.loadtxt('./ne_20191030/fort.98',usecols=3,skiprows=2)/100,(73,60)) #hPa
dens=np.broadcast_to(np.loadtxt('./ne_20191030/fort.98',usecols=1,skiprows=2),(73,60))

w=np.full((73,60),np.nan)
for i in range(73):
    var=nc.Dataset(f'./ne_20191030/ne_20191030.L.Dynamic-{i:06d}.nc')
    w[i,:]=np.where(mask,var.variables['w'][0,:,lat_index,lon_index],np.nan)

# CALCULATION
################################
temp=th/(1000/pres)**0.287 # K

es=es0*np.exp(Lv*(1/T0-1/temp)/Rv)
esi=es0*np.exp(Ls*(1/T0-1/temp)/Rv)

e=qv*pres/epsilon

RH=(e/es)*100

# when temp>=0,es=es ; temp<0,es=esi
es_alter=np.where(temp>=T0,es.copy(),0)+np.where(temp<T0,esi.copy(),0)
RH_alter=(e/es_alter)*100

# dew point
Td=1/(1/T0-Rv*np.log(e/es0)/Lv)

# GRAPHING
###############################################
time=np.arange(73)
time2,lev2=np.meshgrid(time,lev)

cmap=cm.viridis.copy()
cmap.set_over('m')

# Original RH (Just liquid-vapor)
plt.contourf(time2,lev2,RH[:,:].T,levels=np.arange(0,101,10),extend='max',cmap=cmap)
plt.colorbar(label='RH %]')
plt.title('Duinal RH Profile',fontsize=14)
plt.xlabel('Time [hr]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.xticks(np.arange(0,74,6),[f'{h//3:02d}'for h in np.arange(0,74,6)])
plt.savefig('RH_r.png',dpi=500)
plt.clf()

# Altered RH (Consider vapor-ice and vapor-liquid)
plt.contourf(time2,lev2,RH_alter[:,:].T,levels=np.arange(0,101,10),extend='max',cmap=cmap)
plt.colorbar(label='RH %]')
plt.title('Duinal RH Profile',fontsize=14)
plt.xlabel('Time [hr]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.xticks(np.arange(0,74,6),[f'{h//3:02d}'for h in np.arange(0,74,6)])
plt.savefig('RH_altered_r.png',dpi=500)
plt.clf()


# T - Td
cmap=cm.viridis_r.copy()
cmap.set_under('m')

plt.contourf(time2,lev2,(temp-Td).T,extend='both',cmap=cmap,levels=np.arange(0,15,2))
plt.colorbar(label='T-Td [K]')
plt.contour(time2,lev2,qi.T,colors='Red',levels=[0.1])
plt.contour(time2,lev2,qc.T,colors='Blue',levels=[0.1])
legend_element=[mline.Line2D([0],[0],color='blue',linewidth=2,label='qc'),mline.Line2D([0],[0],color='red',label='qi')]
plt.legend(handles=legend_element,loc='lower right',ncols=2)
plt.title('Profile of $T-T_d$ , $q_i$ and $q_c$',fontsize=14)
plt.xlabel('Time [hr]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.xticks(np.arange(0,74,6),[f'{h//3:02d}'for h in np.arange(0,74,6)])
plt.savefig('T_Td_qi_qc.png',dpi=500)


# convection and qi qc
cmap=cm.magma_r.copy()
cmap.set_under('g')

plt.pcolormesh(time2,lev2,(temp-Td).T,cmap=cmap)
plt.colorbar(label='T-Td [K]',extend='both')

plt.pcolormesh(time2,lev2,w.T,norm=mcolors.CenteredNorm(vcenter=0),cmap=cm.coolwarm,alpha=0.8)
plt.colorbar(label='w [m/s]')

plt.contour(time2,lev2,qi.T,colors='k',levels=[0.1])
plt.contour(time2,lev2,qc.T,colors='c',levels=[0.1])
legend_element=[mline.Line2D([0],[0],color='k',linewidth=2,label='qc'),mline.Line2D([0],[0],color='c',label='qi')]
plt.legend(handles=legend_element,loc='lower right',ncols=2)

plt.xlabel('Time [hr]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.xticks(np.arange(0,74,6),[f'{h//3:02d}'for h in np.arange(0,74,6)])
