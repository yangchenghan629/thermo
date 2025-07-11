import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

#open files
loc=nc.Dataset("TOPO.nc")
var=nc.Dataset("ne_20191030.L.Thermodynamic-000024.nc")

#read location and variables
lon=loc.variables['lon'][:]
lat=loc.variables['lat'][:]
lev=loc.variables['lev'][:]

#mountain area (23.508N,120.813E)
latm=np.where(np.abs(lat-23.508)<=2e-3)[0]
lonm=np.where(np.abs(lon-120.813)<=2e-3)[0]
qv_mt=np.zeros(60)
for i in range(60):
  qv_mt[i]=var.variables['qv'][:,i,latm,lonm]
sfc_mt=np.min(np.where(qv_mt!=0))
lev_sfc_mt=np.zeros(50)
lev_sfc_mt[:]=lev[sfc_mt]

#plain area (22.730N,120.312E)
latp=np.where(np.abs(lat-22.730)<=2e-3)[0]
lonp=np.where(np.abs(lon-120.312)<=2e-3)[0]
qv_pn=np.zeros(60)
for i in range(60):
  qv_pn[i]=var.variables['qv'][:,i,latp,lonp]
sfc_pn=np.min(np.where(qv_pn!=0))
lev_sfc_pn=np.zeros(50)
lev_sfc_pn[:]=lev[sfc_pn]

#plot profile
plt.plot(qv_mt[:],lev[:],'k')
plt.plot(qv_pn[:],lev[:],'b')
plt.plot(np.linspace(0,0.015),lev_sfc_mt,'r--',linewidth=0.5)
plt.plot(np.linspace(0,0.015),lev_sfc_pn,'r--',linewidth=0.5)
plt.grid(True,'major','both')
plt.text(0.003,lev[sfc_mt]+0.1,f'Moutain Surface\nHeight:{lev[sfc_mt]:.3f} km',fontsize=10,color='r')
plt.text(0.003,lev[sfc_pn]+0.1,f'Plain Surface\nHeight:{lev[sfc_pn]:.3f} km',fontsize=10,color='r')
plt.legend(['mountain area','plain area'])
plt.xlabel('$q_{v}$ [kg/kg]')
plt.ylabel('altitude [km]')
plt.xlim([0,0.015])
plt.ylim([0,12])
plt.title('Vertical Profile of Specific Humidity ($q_{v}$) \n Mountain (23.508N,120.813E) and Plain (22.730N,120.312E)',size=12)
plt.savefig('qv_profile_marked.png')
plt.show()
