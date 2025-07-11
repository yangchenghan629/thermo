import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt 

Rd=287
g=9.8
dt=2
dt2=0.5

loc=nc.Dataset('/home/B13/b13209015/thermo/ca3/ne_20191030/TOPO.nc')

lon=loc.variables['lon'][:]
lat=loc.variables['lat'][:]
lev=loc.variables['lev'][:]*1000  #m

lon_cen_index=np.where(np.abs(lon-120.6289)<=1e-5)[0][0]
lat_cen_index=np.where(np.abs(lat-23.49798)<=1e-5)[0][0]

mask=loc.variables['mask'][:,lat_cen_index,lon_cen_index]
sfc_height=lev[np.amin(np.where(mask==1))]

rv=np.zeros((73,60))
th=np.zeros((73,60))

for i in range(73):
    path=f'/home/B13/b13209015/thermo/ca3/ne_20191030/ne_20191030.L.Thermodynamic-{i:06d}.nc'
    var=nc.Dataset(path)
    rv[i,:]=var.variables['qv'][:,:,lat_cen_index,lon_cen_index]
    th[i,:]=var.variables['th'][:,:,lat_cen_index,lon_cen_index]

qv=rv/(1+rv)

pres=np.loadtxt('/home/B13/b13209015/thermo/ca3/ne_20191030/fort.98',skiprows=2,usecols=3) #Pa
pres=np.where(mask,pres,np.nan)
pres=np.broadcast_to(pres[None,:],(73,60))


#initial Prrssure , Temperature and qv at 10000m 
p0=np.interp(10000,lev,pres[0,:])/100 #hPa
th0=np.interp(10000,lev[:],th[24,:])
qv0=np.interp(10000,lev[:],qv[24,:])
temp0=th0/(1000/p0)**0.287
Tv0=temp0*(1+0.608*qv0)

#dt=2
p=[p0] #hPa
temp=[temp0]
Tv=[Tv0]

z=[10000] #m
h=z[0]
t=0
v=0
i=1
while(h>=sfc_height):
    v+=g*dt
    h-=0.5*g*dt**2+v*dt
    if h>=sfc_height:
        z.append(h)
    t+=dt
    i+=1
z.append(sfc_height)
for i in range(1,len(z)):
    dz=z[i]-z[i-1]
    p.append(p[i-1]/np.exp(dz*g/(Rd*Tv[i-1])))  #hPa
    th_i=np.interp(z[i],lev[:],th[24,:])  #K
    temp.append(th_i/(1000/(p[i]))**0.287)  #K
    qv_i=np.interp(z[i],lev[:],qv[24,:])  #kg/kg
    Tv.append(temp[i]*(1+0.608*qv_i))  #K


#dt=0.5
p2=[p0] #hPa
temp2=[temp0]
Tv2=[Tv0]
z2=[10000] #m
h=z2[0]
t=0
v=0
i=1
while(h>=sfc_height):
    v+=g*dt2
    h-=0.5*g*dt2**2+v*dt2
    if h>=sfc_height:
        z2.append(h)
    t+=dt2
    i+=1
z2.append(sfc_height)
for i in range(1,len(z2)):
    dz=z2[i]-z2[i-1]
    p2.append(p2[i-1]/np.exp(dz*g/(Rd*Tv2[i-1]))) #hPa
    th_i=np.interp(z2[i],lev[:],th[24,:])    #K
    temp2.append(th_i/(1000/(p2[i]))**0.287) #K
    qv_i=np.interp(z2[i],lev[:],qv[24,:])    #kg/kg
    Tv2.append(temp2[i]*(1+0.608*qv_i))          #K


fig,ax=plt.subplots(1,2)
plt.suptitle('Pressure Profile',fontsize=14)
ax[0].plot(p,z,'b-',linewidth=1)
ax[0].plot(p2,z2,'g-',linewidth=1)
ax[0].plot(pres[0,:]/100,lev,'r--',linewidth=1)
ax[0].set_xlim([200,1000])
ax[0].set_ylim([0,12500])
ax[0].set_xlabel('Pressure [hPa]',fontsize=12)
ax[0].set_ylabel('Altitude [m]',fontsize=12)
ax[0].plot(np.linspace(0,10000),np.full((50),sfc_height),'k--')
ax[0].text(200,sfc_height,f'{sfc_height:.0f}m')
ax[0].grid()

ax[1].plot(p,z,'b-',linewidth=1)
ax[1].plot(p2,z2,'g-',linewidth=1)
ax[1].plot(pres[0,:]/100,lev,'r--',linewidth=1)
ax[1].set_xlim([800,950])
ax[1].set_ylim([1000,2000])
ax[1].set_xlabel('Pressure [hPa]',fontsize=12)
ax[1].plot(np.linspace(0,10000),np.full((50),sfc_height),'k--')
ax[1].text(800,sfc_height,f'{sfc_height:.0f}m')
ax[1].grid()

fig.legend(['dt=2 sec','dt=0.5 sec','VVM'],loc='lower center',ncol=3,bbox_to_anchor=(0.5,-0.06),fontsize=12)
plt.tight_layout()
plt.savefig('P-Z.png',dpi=600,bbox_inches='tight')