import PIL.Image
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors
import PIL as PIL

radius=nc.Dataset('./TC/axmean-000000.nc').variables['radius'][:]/1000
zc=nc.Dataset('./TC/axmean-000000.nc').variables['zc'][:]/1000

radi_wind=np.zeros((201,34,192))
qi=np.zeros((201,34,192))
qc=np.zeros((201,34,192))
qr=np.zeros((201,34,192))

for i in range(601):
    radi_wind[int(i/3),:,:]=nc.Dataset(f'./TC/axmean-{i:06d}.nc').variables['radi_wind'][0,0,:,:]
    qi[int(i/3),:,:]=nc.Dataset(f'./TC/axmean-{i:06d}.nc').variables['qi'][0,0,:,:]
    qc[int(i/3),:,:]=nc.Dataset(f'./TC/axmean-{i:06d}.nc').variables['qc'][0,0,:,:]
    qr[int(i/3),:,:]=nc.Dataset(f'./TC/axmean-{i:06d}.nc').variables['qr'][0,0,:,:]

qi=np.where(np.abs(qi)>=1e-4,qi,0)
qc=np.where(np.abs(qc)>=1e-4,qc,0)
qr=np.where(np.abs(qr)>=1e-4,qr,0)

cloud=(qi+qc+qr)*1000
cloud_mask=np.where(cloud>0,1,-1)

rr,hh=np.meshgrid(radius,zc)
for i in range(201):
    plt.clf()
    plt.title(f'Radial wind (shaded) Cloud (line) {i:02d} hrs',fontsize=14)
    plt.pcolormesh(rr,hh,radi_wind[i,:,:],norm=mcolors.Normalize(vmax=20,vmin=-20),cmap=cm.RdBu_r)
    plt.colorbar(extend='both',label='velocity [m/s]',ticks=[-20,-15,-10,-5,0,5,10,15,20])
    c=plt.contour(rr,hh,cloud_mask[i,:,:],levels=[0],colors='black')
    plt.xlabel('Radius [km]',fontsize=12)
    plt.ylabel('Height [km]',fontsize=12)
    plt.xlim([0,radius[-1]])
    plt.ylim([0,zc[-1]])
    plt.savefig(f'./cloud_wind/cw{i:02d}.png',dpi=300)
    plt.close()

gif=[]
for i in range(201):
    img=PIL.Image.open(f'./cloud_wind/cw{i:02d}.png')
    gif.append(img)
gif[0].save('cloud_radial.gif',save_all=True,append_images=gif[1:],duration=200)