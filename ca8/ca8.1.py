import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import metpy.calc as metcal
from metpy.units import units
from tools import Thermo
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
###############################################
# READ DATA
###############################################
# topographic
loc=nc.Dataset('ne_20191030/TOPO.nc')
lon=np.where(np.abs(loc.variables['lon'][:]-120.6289)<=1e-5)[0][0]
lat=np.where(np.abs(loc.variables['lat'][:]-23.49798)<=1e-5)[0][0]
lev=loc.variables['lev'][:] #km
mask=loc.variables['mask'][:,lat,lon]
sfc=np.argmin(mask!=1)

# thermodynamic variables
th,rv=[np.full((73,60),np.nan) for _ in range(2)]
for time in range(73):
    file=nc.Dataset(f'ne_20191030/ne_20191030.L.Thermodynamic-{time:06d}.nc')
    th[time,:]=np.where(mask,file.variables['th'][:,:,lat,lon],np.nan) # k
    rv[time,:]=np.where(mask,file.variables['qv'][:,:,lat,lon],np.nan) # kg/kg

p_ref=np.loadtxt('./ne_20191030/fort.98',usecols=3,skiprows=2)/100 # hPa
pres=np.broadcast_to(np.where(mask,p_ref,np.nan),(73,60))

# surface variables
prec_rate=np.full((73),np.nan)
for time in range(73):
    prec_rate[time]=nc.Dataset(f'ne_20191030/ne_20191030.C.Surface-{time:06d}.nc').variables['sprec'][0,lat,lon] # kg m-2 s-1 = mm/s

###############################################
# CALCULATION
###############################################
interp_func = interp1d(p_ref, lev, kind='cubic', bounds_error=False, fill_value='extrapolate')

qv=rv/(1+rv) # kg/kg
temp=Thermo.temp_from_th(th,pres) # K
td=metcal.dewpoint_from_specific_humidity(pres*units.hPa,temp*units.K,qv*units('kg/kg')).to('K')
prec=prec_rate*1200*3 # accumulation prec : mm

# Mixed Layer
th_gradient=np.gradient(th,axis=1)
mlh=lev[np.nanargmax(th_gradient[:,lev<3],axis=1)]


# LCL
lcl=np.full((73),np.nan)
for time in range(73):
    lcl_p,_=metcal.lcl(pres[time,sfc]*units.hPa,temp[time,sfc]*units.K,td[time,sfc])
    lcl[time]=interp_func(lcl_p.m) #km


# cape cin
Td=metcal.dewpoint_from_specific_humidity(pres*units.hPa,temp*units.K,qv*units('kg/kg')).to('K')
parcel=np.full((73,60-sfc),np.nan)
for time in range(73):
    parcel[time,:]=metcal.parcel_profile(pres[time,sfc:]*units.hPa,temp[time,sfc]*units.K,Td[time,sfc])
cape,cin=[np.full((73),np.nan) for _ in range(2)]
for time in range(73):
    ca,ci=metcal.cape_cin(pres[time,sfc:]*units.hPa,temp[time,sfc:]*units.K,Td[time,sfc:],parcel[time,:]*units.K)
    cape[time]=ca.magnitude
    cin[time]=ci.magnitude

    

###############################################
# GRAPHING
###############################################

fig,ax=plt.subplots(layout='constrained')
ax2=ax.twinx()
ax3=ax.twinx()
fig.suptitle('Diurnal Cycle',fontsize=14)
ax.plot(np.arange(73),cape,'red',label='CAPE')
ax.plot(np.arange(73),np.abs(cin),'blue',label='CIN')
ax.set_xticks(np.arange(0,73,3))
ax.set_xticklabels([f'{h//3:02d}'for h in range(0,73,3)])
ax.set_xlim(0,72)
ax.set_ylim(0,3000)
ax.grid(True)
ax.set_xlabel('Time [hr]',fontsize=12)
ax.set_ylabel('CAPE/CIN [J/kg]',fontsize=12)
ax2.plot(np.arange(73),lcl,'green',label='LCL')
ax2.plot(np.arange(73),mlh,'orange',label='MLH')
ax2.set_ylabel('LCL/MLH [km]',fontsize=12)
ax2.grid(False)
ax2.set_yticks(np.arange(0,3.1,0.5))
ax2.set_ylim(0,3)
ax3.spines["right"].set_position(("axes", 1.15))
ax3.bar(np.arange(73),prec,alpha=0.5,color='k',label='Precipitation')
ax3.set_ylabel('Precipitation [mm/hr]',fontsize=12)
ax3.set_ylim(0,10)
fig.legend(ncols=5,bbox_to_anchor=(0.5,-0.02),loc='center')
plt.savefig('diurnal_cycle.png',dpi=450,bbox_inches='tight')

plt.clf()
fig,ax=plt.subplots(layout='constrained')
ax2=ax.twinx()
ax3=ax.twinx()
ax4=ax.twinx()
fig.suptitle('CAPE LCL QV and Precipitation',fontsize=14)
ax.plot(np.arange(73),cape,'red',label='CAPE')
ax.set_xticks(np.arange(0,73,6))
ax.set_xticklabels([f'{h//3:02d}'for h in range(0,73,6)])
ax.set_xlim(0,72)
ax.set_ylim(0,3000)
ax.grid(True)
ax.set_xlabel('Time [hr]',fontsize=12)
ax.set_ylabel('CAPE [J/kg]',fontsize=12)
ax2.plot(np.arange(73),qv[:,sfc]*1000,'blue',label='qv')
ax2.set_ylim(0,20)
# ax2.spines["right"].set_position(("axes", 1.15))
ax2.grid(False)
ax2.set_ylabel('qv [g/kg]',fontsize=12)
ax3.plot(np.arange(73),lcl,'green',label='LCL')
ax3.set_ylabel('LCL [km]',fontsize=12)
ax3.grid(False)
ax3.set_yticks(np.arange(0,3.1,0.5))
ax3.set_ylim(0,3)
ax3.spines["right"].set_position(("axes", 1.2))
ax4.bar(np.arange(73),prec,alpha=0.5,color='k',label='Precipitation')
ax4.set_ylabel('Precipitation [mm/hr]',fontsize=12)
ax4.set_ylim(0,10)
ax4.spines["right"].set_position(("axes", 1.4))
fig.legend(ncols=5,bbox_to_anchor=(0.5,-0.02),loc='center')
plt.savefig('cape_lcl_qv_prec.png',dpi=450,bbox_inches='tight')
