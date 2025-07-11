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
qv=rv/(1+rv) # kg/kg

pres=np.broadcast_to(np.loadtxt('./ne_20191030/fort.98',usecols=3,skiprows=2)/100,(73,60)) #hPa
dens=np.broadcast_to(np.loadtxt('./ne_20191030/fort.98',usecols=1,skiprows=2),(73,60))

# surface
tg=np.full((73),np.nan)
for i in range(73):
    tg[i]=nc.Dataset(f'./ne_20191030/ne_20191030.C.Surface-{i:06d}.nc').variables['tg'][:,lat_index,lon_index]

#################################
# CALCULATION
#################################
class state:
    def __init__(self,temp,qv,pres,tg):
        self.temp=temp
        self.pres=pres
        self.qv=qv
        self.e=Thermo.e_from_qv_p(self.qv,self.pres)
        self.es=Thermo.cc_equation(self.temp)
        self.rh=self.e/self.es
        self.tg=tg-273.15
        self.tw=Thermo.Tw(self.temp,self.pres,self.qv)-273.15
        self.wbgt=state.WBGT(self.temp-273.15,self.tw,self.tg)
    def WBGT(temp,Tw,Tg):
        return 0.7*Tw+0.2*Tg+0.1*temp

temp=Thermo.temp_from_th(th,pres)
temp_sfc=temp[:,surface_index]
state1=state(temp_sfc,qv[:,surface_index],pres[:,surface_index],tg)
state2=state(temp_sfc+2,qv[:,surface_index],pres[:,surface_index],tg+2)


time=np.arange(0,73)

#################################
# PLOTTING
#################################


plt.plot(time,state1.wbgt,'b',label='original')
plt.plot(time,state2.wbgt,'r',label='warming')
plt.xticks(np.arange(0,73,3),[f'{h//3:02d}'for h in range(0,73,3)])
plt.xlim([0,72])
plt.grid()
plt.legend()
plt.xlabel('Time [hr]',fontsize=12)
plt.ylabel('WBGT [deg C]',fontsize=12)
plt.title('Diurnal WBGT',fontsize=14)
plt.savefig('./result/wbgt.png',dpi=450)