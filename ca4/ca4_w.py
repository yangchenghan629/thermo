import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors

# 10 hour
var=nc.Dataset('/home/B13/b13209015/thermo/ca4/TC/axmean-000030.nc')
radius=var.variables['radius'][:]/1000
height=var.variables['zc'][:]/1000
radi_wind=var.variables['radi_wind'][0,0,:,:]
tang_wind=var.variables['tang_wind'][0,0,:,:]
w=var.variables['w'][0,0,:,:]

qi=var.variables['qi'][0,0,:,:]
qc=var.variables['qc'][0,0,:,:]
qr=var.variables['qr'][0,0,:,:]

qi=np.where(np.abs(qi)>=5e-5,qi,0)
qc=np.where(np.abs(qc)>=5e-5,qc,0)
qr=np.where(np.abs(qr)>=5e-5,qr,0)

cloud=(qi+qc+qr)*1000
cloud_mask=np.where(cloud>0,1,-1)

rr,hh=np.meshgrid(radius,height)

plt.title('10 hours\nradial wind (shaded) cloud (line)',fontsize=14)
plt.pcolormesh(rr,hh,radi_wind,cmap=cm.RdBu_r,norm=mcolors.CenteredNorm())
plt.colorbar(extend='both',label='velocity [m/s]')
c=plt.contour(rr,hh,cloud_mask,levels=[0],colors='black')
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Height [km]',fontsize=12)
plt.savefig('w10.png')
plt.clf()

plt.title('10 hours\nvertical wind (shaded) cloud (line)',fontsize=14)
plt.pcolormesh(rr,hh,w,cmap=cm.RdBu_r,norm=mcolors.CenteredNorm())
plt.colorbar(extend='both',label='velocity [m/s]')
c=plt.contour(rr,hh,cloud_mask,levels=[0],colors='black')
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Height [km]',fontsize=12)
plt.savefig('w_vertical10.png')
plt.clf()


# 200 hour
var=nc.Dataset('/home/B13/b13209015/thermo/ca4/TC/axmean-000600.nc')
radius=var.variables['radius'][:]/1000
height=var.variables['zc'][:]/1000
radi_wind=var.variables['radi_wind'][0,0,:,:]
tang_wind=var.variables['tang_wind'][0,0,:,:]
w=var.variables['w'][0,0,:,:]


qi=var.variables['qi'][0,0,:,:]
qc=var.variables['qc'][0,0,:,:]
qr=var.variables['qr'][0,0,:,:]

qi=np.where(np.abs(qi)>=5e-5,qi,0)
qc=np.where(np.abs(qc)>=5e-5,qc,0)
qr=np.where(np.abs(qr)>=5e-5,qr,0)

cloud=(qi+qc+qr)*1000
cloud_mask=np.where(cloud>0,1,-1)

rr,hh=np.meshgrid(radius,height)
plt.title('200 hours\nradial wind (shaded) cloud (line)',fontsize=14)
plt.pcolormesh(rr,hh,radi_wind,cmap=cm.RdBu_r,norm=mcolors.CenteredNorm())
plt.colorbar(extend='both',label='velocity [m/s]')
c=plt.contour(rr,hh,cloud_mask,levels=[0],colors='black')
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Height [km]',fontsize=12)
plt.savefig('w200.png',dpi=500)
plt.clf()

plt.title('200 hours\nvertical wind (shaded) cloud (line)',fontsize=14)
plt.pcolormesh(rr,hh,w,cmap=cm.RdBu_r,norm=mcolors.CenteredNorm())
plt.colorbar(extend='both',label='velocity [m/s]')
c=plt.contour(rr,hh,cloud_mask,levels=[0],colors='black')
plt.xlabel('Radius [km]',fontsize=12)
plt.ylabel('Height [km]',fontsize=12)
plt.savefig('w_vertical200.png')
plt.clf()
