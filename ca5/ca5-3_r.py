import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors , matplotlib.lines as mlines

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

cmap=cm.viridis.copy()
cmap.set_over('m')

# RELATIVE HUMIDITY , QI AND QC PROFILE
plt.contourf(rr/1000,hh/1000,RH,levels=np.arange(0,101,10),extend='max',cmap=cmap,alpha=0.9)
plt.colorbar(label='RH [%]')
plt.contour(rr/1000,hh/1000,temp,levels=[T0],colors='k',linestyles='dashed')
plt.text(400,zc[temp_0[-1]]/1000+0.1,'freezing level',fontsize=11,color='k')
plt.contour(rr/1000,hh/1000,qi,levels=[0.1],colors='b')
plt.contour(rr/1000,hh/1000,qc,levels=[0.01],colors='r')
legend_element=[mlines.Line2D([0],[0],color='b',lw=2,label='qi'),mlines.Line2D([0],[0],color='r',lw=2,label='qc')]
plt.legend(handles=legend_element)
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Height [km]',fontsize=12)
plt.title('Profile of RH , $q_i$ and $q_c$ at 30 hrs',fontsize=14)
plt.tight_layout()
plt.savefig('TC_RH_qi_qc.png',dpi=500)