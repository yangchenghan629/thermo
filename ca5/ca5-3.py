import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors
import PIL as pil

# PARAMETER
#########################
Rv=461.5 # K/kg.K
es0=6.11 # hPa
T0=273.15 # K
Lv=2.5e6 # J/kg
Lf=3.34e5 # J/kg
Ls=Lv+Lf
epsilon=0.622

# READ DATA
#################################
var=nc.Dataset('./TC/axmean-000090.nc')
zc=var.variables['zc'][:] # m
radius=var.variables['radius'][:] # m
th=var.variables['th'][0,0,:,:] # K
rv=var.variables['qv'][0,0,:,:] # kg/kg
qi=var.variables['qi'][0,0,:,:]*1000 # g/kg
qc=var.variables['qc'][0,0,:,:]*1000 # g/kg
pres=np.tile(np.loadtxt('./TC/fort.98',skiprows=237,max_rows=34,usecols=3),(192,1)).T/100

# CALCULATION
#######################################
qv=rv/(1+rv) # kg/kg
temp=th/(1000/pres)**0.287 # K
e=qv*pres/epsilon
es=es0*np.exp(Lv*(1/T0-1/temp)/Rv) #hPa
esi=es0*np.exp(Ls*(1/T0-1/temp)/Rv) #hPa
# consider vaporization and sublimation determined by temperature
es_alter=np.where(temp>=T0,es,0)+np.where(temp<T0,esi,0)
RH=(e/es_alter)*100


# GRAPHING
########################################
temp_0=np.argmax(temp<=T0,axis=0)
rr,hh=np.meshgrid(radius,zc)

# RELATIVE HUMIDITY PROFILE
plt.contourf(rr/1000,hh/1000,RH,levels=np.arange(0,101,10),extend='max')
plt.colorbar(label='RH [%]')
plt.contour(rr/1000,hh/1000,temp,levels=[T0],colors='r')
plt.text(10,zc[temp_0[0]]/1000+0.1,'freezing level',fontsize=10,color='r')
plt.title('Cross-section of RH',fontsize=14)
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.savefig('TC_RH.png',dpi=500)
plt.clf()

# QI , QC PROFILE
plt.contourf(rr/1000,hh/1000,qi,cmap=cm.Blues,extend='max',norm=mcolors.Normalize())
plt.colorbar(label='qi [g/kg]')
plt.contourf(rr/1000,hh/1000,qc,cmap=cm.Reds,extend='max',alpha=0.5,norm=mcolors.Normalize())
plt.colorbar(label='qc [g/kg]')
plt.contour(rr/1000,hh/1000,temp,levels=[T0],colors='k')
plt.text(400,4,'freezing level',fontsize=10,color='k')
plt.title('Cross-section of Cloud-liquid and Cloud-ice',fontsize=14)
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.xticks(np.arange(0,radius[-1]/1000,100))
plt.savefig('TC_qiqc.png',dpi=500)
plt.clf()

# QI , QC AND QV PROFILE
plt.contourf(rr/1000,hh/1000,qi,cmap=cm.Blues,extend='max',norm=mcolors.Normalize())
plt.colorbar(label='qi [g/kg]')
plt.contourf(rr/1000,hh/1000,qc,cmap=cm.Reds,extend='max',alpha=0.6,norm=mcolors.Normalize())
plt.colorbar(label='qc [g/kg]')
plt.contourf(rr/1000,hh/1000,qv*1000,cmap=cm.Greens,extend='max',alpha=0.45,norm=mcolors.Normalize())
plt.colorbar(label='qv [g/kg]')
plt.contour(rr/1000,hh/1000,temp,levels=[T0],colors='k')
plt.text(400,4,'freezing level',fontsize=10,color='k')
plt.title('Cross-section of qi,qc,qv',fontsize=14)
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Altitude [km]',fontsize=12)
plt.xticks(np.arange(0,radius[-1]/1000,100))
plt.clf()
