import PIL.Image
import netCDF4 as nc
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors
import numpy as np
import PIL as PIL

radius=nc.Dataset('./TC/axmean-000000.nc').variables['radius'][:]/1000
zc=nc.Dataset('./TC/axmean-000000.nc').variables['zc'][:]/1000

qi=np.zeros((201,34,192))  #cloud ice mixing ratio
qc=np.zeros((201,34,192))  #cloud water
qr=np.zeros((201,34,192))  #cloud rain

for i in range(0,601,3):
    var=nc.Dataset(f'TC/axmean-{i:06d}.nc')
    qi[int(i/3),:,:]=var.variables['qi'][0,0,:,:]
    qc[int(i/3),:,:]=var.variables['qc'][0,0,:,:]
    qr[int(i/3),:,:]=var.variables['qr'][0,0,:,:]

cloud=(qi+qc+qr)*1000

mask_qi=np.where(np.abs(qi)>=5e-5,qi,0)
mask_qc=np.where(np.abs(qc)>=5e-5,qc,0)
mask_qr=np.where(np.abs(qr)>=5e-5,qr,0)

mask_cloud=(mask_qc+mask_qr+mask_qi)*1000

rr,hh=np.meshgrid(radius,zc)

plt.pcolormesh(rr,hh,mask_cloud[200,:,:],cmap=cm.coolwarm,norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmin=0,vmax=6))
plt.savefig('cloud200.png')
for i in range(201):
    plt.clf()
    fig,ax=plt.subplots(ncols=1,nrows=2,sharex='col')
    plt.suptitle(f'Cloud {i:02d} hours',fontsize=14)
    p0=ax[0].pcolormesh(rr,hh,cloud[i,:,:],norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=6,vmin=0),cmap=cm.coolwarm)
    ax[0].set_title('No mask',fontsize=12)
    ax[0].set_ylim([0,zc[-1]])
    ax[0].set_ylabel('Height [km]',fontsize=10)
    p1=ax[1].pcolormesh(rr,hh,mask_cloud[i,:,:],norm=mcolors.TwoSlopeNorm(vcenter=0.1,vmax=6,vmin=0),cmap=cm.coolwarm)
    ax[1].set_title('Masked',fontsize=12)
    ax[1].set_xlim([0,radius[-1]])
    ax[1].set_ylim([0,zc[-1]])
    ax[1].set_xlabel('Radius [km]',fontsize=10)
    ax[1].set_ylabel('Height [km]',fontsize=10)
    fig.colorbar(p1,ax=ax[:],ticks=[0,0.05,0.1,2,4,6],label='Mixing Ratio [g/kg]')
    plt.savefig(f'./cloud/combined_cloud{i:02d}.png',dpi=300)
    plt.close()

gif=[]
for i in range(201):
    img=PIL.Image.open(f'./cloud/combined_cloud{i:02d}.png')
    gif.append(img)
gif[0].save('./combined_cloud.gif',save_all=True,append_images=gif[1:],duration=200)
