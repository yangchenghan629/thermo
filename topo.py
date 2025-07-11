import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.colors as mcolors
import netCDF4 as nc

loc_file=nc.Dataset('./ne_20191030/TOPO.nc')
lat=loc_file.variables['lat'][:]
lon=loc_file.variables['lon'][:]
lev=loc_file.variables['lev'][:]
lat_index=np.where(np.abs(lat-23.49798)<=1e-5)[0][0]
lon_index=np.where(np.abs(lon-120.6289)<=1e-5)[0][0]


r=10
lon_range=lon[lon_index-r:lon_index+r+1]
lat_range=lat[lat_index-r:lat_index+r+1]
altitude=loc_file.variables['height'][lat_index-r:lat_index+r+1,lon_index-r:lon_index+r+1]
lon2,lat2=np.meshgrid(lon_range,lat_range)

colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colors_land = plt.cm.terrain(np.linspace(0.25, 0.7, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = mcolors.LinearSegmentedColormap.from_list(
    'terrain_map', all_colors)

plt.contourf(lon2,lat2,altitude,cmap=terrain_map,levels=np.arange(0,2.1,0.1),norm=mcolors.TwoSlopeNorm(vmin=-1,vcenter=0,vmax=2))
plt.colorbar(ticks=np.arange(0,2.1,0.5),label='Altitude [km]')
c=plt.contour(lon2,lat2,altitude,colors='k',linewidths=0.5,levels=np.arange(0,2.1,0.1))
plt.clabel(c,inline=True,colors='k',fontsize=8)
plt.scatter(lon[lon_index],lat[lat_index],marker='x',s=100,c='r')
plt.xlim(lon[lon_index-r],lon[lon_index+r])
plt.ylim(lat[lat_index-r],lat[lat_index+r])
plt.title('Topographic Map',fontsize=14)
plt.xlabel('Longitude',fontsize=12)
plt.ylabel('Latitude',fontsize=12)
plt.savefig('topo.png',dpi=500)
