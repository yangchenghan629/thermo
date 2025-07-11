import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt , matplotlib.cm as cm , matplotlib.lines as mlines
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as metcal
from tools import Thermo
from scipy.interpolate import interp1d
from matplotlib.transforms import blended_transform_factory

###############################################
# topography
###############################################
loc=nc.Dataset('ne_20191030/TOPO.nc')
lon=np.where(np.abs(loc.variables['lon'][:]-120.6289)<=1e-5)[0][0]
lat=np.where(np.abs(loc.variables['lat'][:]-23.49798)<=1e-5)[0][0]
lev=loc.variables['lev'][:]
mask=loc.variables['mask'][:,lat,lon]
sfc=np.argmin(mask!=1)

# thermodynamic variables
th,rv=[np.full((73,60),np.nan) for _ in range(2)]
for i in range(73):
    file=nc.Dataset(f'ne_20191030/ne_20191030.L.Thermodynamic-{i:06d}.nc')
    th[i,:]=np.where(mask,file.variables['th'][:,:,lat,lon],np.nan)
    rv[i,:]=np.where(mask,file.variables['qv'][:,:,lat,lon],np.nan)

p_ref=np.loadtxt('./ne_20191030/fort.98',usecols=3,skiprows=2)/100
pres=np.broadcast_to(np.where(mask,p_ref,np.nan),(73,60))*units.hPa

# dynamic variables
u,v=[np.full((73,60),np.nan) for _ in range(2)]
for i in range(73):
    file=nc.Dataset(f'ne_20191030/ne_20191030.L.Dynamic-{i:06d}.nc')
    u[i,:]=np.where(mask,file.variables['u'][:,:,lat,lon],np.nan)
    v[i,:]=np.where(mask,file.variables['v'][:,:,lat,lon],np.nan)
u=u*units('m/s')
v=v*units('m/s')

###############################################
# calculation
###############################################
qv=rv/(1+rv)
temp=(Thermo.temp_from_th(th,pres.magnitude)-273.15)*units.degC
Td=metcal.dewpoint_from_specific_humidity(pres,temp,qv).to('degC')
parcel=np.full((73,60-sfc),np.nan)
for time in range(73):
    parcel[time,:]=metcal.parcel_profile(pres[time,sfc:],temp[time,sfc],Td[time,sfc]).to('degC')
cape,cin,lcl_p,lcl_t,lfc_p,lfc_t,el_p,el_t=[np.full((73),np.nan) for _ in range(8)]
for time in range(73):
    ca,ci=metcal.cape_cin(pres[time,sfc:],temp[time,sfc:],Td[time,sfc:],parcel[time,:]*units.degC)
    cape[time]=ca.magnitude
    cin[time]=ci.magnitude
    lclp,lclt=metcal.lcl(pres[time,sfc],temp[time,sfc],Td[time,sfc])
    lcl_p[time]=lclp.magnitude
    lcl_t[time]=lclt.magnitude
    lfcp,lfct=metcal.lfc(pres[time,sfc:],temp[time,sfc:],Td[time,sfc:],parcel[time,:]*units.degC)
    lfc_p[time]=lfcp.magnitude
    lfc_t[time]=lfct.magnitude
    elp,elt=metcal.el(pres[time,sfc:],temp[time,sfc:],Td[time,sfc:],parcel[time,:]*units.degC)
    el_p[time]=elp.magnitude
    el_t[time]=elt.magnitude

interp_func = interp1d(p_ref, lev, kind='cubic', bounds_error=False, fill_value='extrapolate')
h_ref=interp_func(np.arange(1000,199,-100))

###############################################
# plot
###############################################
for time in range(0,73,1):
    plt.clf()
    fig = plt.figure(figsize=(6, 5))
    fig.suptitle(f'Skew-T Log-P Diagram | {time//3:02d} hr {time*20%60:02d} min',fontsize=14,x=0.5)
    skew = SkewT(fig, rotation=45)
    ax = skew.ax

    # Shade every other section between isotherms
    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90, 50, 8)
    y = [1100, 200]
    for i in range(0, 8):
        skew.shade_area(y=y, x1=x1[i], x2=x2[i], color='papayawhip', alpha=0.4, zorder=1)

    # dry adiabats lines
    t0 = units.K * np.arange(243.15, 444.15, 10)
    skew.plot_dry_adiabats(t0=t0, linestyles='solid', colors='gray', linewidths=1)

    # moist adiabats lines
    t0 = units.K * np.arange(283.15, 304.15, 5)
    p = units.hPa * np.linspace(1000, 201, 10)
    skew.plot_moist_adiabats(t0=t0, pressure=p, linestyles='solid', colors='green', linewidth=1.5)

    # mixing ratio lines
    w = (np.hstack([np.arange(0,0.0010,0.0007),np.arange(0.0010,0.0031,0.0005),np.arange(0.004,0.0081,0.002),np.arange(0.010,0.0310,0.005)])).reshape(-1, 1)
    p = units.hPa * np.linspace(1000, 200, 10)
    skew.plot_mixing_lines(mixing_ratio=w, pressure=p, linestyle='dashed', colors='k', linewidths=1)
    mixratio=w*units('kg/kg')
    p_1000=1000*units.hPa
    Td_1000=metcal.dewpoint_from_specific_humidity(p_1000,mixratio).to('degC').magnitude
    for td, wval in zip(Td_1000, mixratio[:,0].to('g/kg').magnitude):
        ax.text(td+1, 1050, f'{wval:.1f}'.rstrip('0').rstrip('.'), fontsize=8, color='k',ha='center')
    ax.text(35,1050,'g/kg',fontsize=7,color='k')


    # set pressure scale and height scale
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.set_yticks(np.arange(1000,199,-100))
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xlabel('Temperature ($\degree$C)',fontsize=12)
    ax.set_ylabel('')
    ax.text(-0.05, 190, 'Pressure (hPa)', transform=transform,va='bottom', ha='left', fontsize=12)
    ax.set_xlim(-20,40)
    ax.set_ylim(1100,200)
    for p, h in zip(np.arange(1000,199,-100), h_ref):
        ax.text(1.01, p, f'{h:.2f}', transform=transform,va='center', ha='left', fontsize=10, color='black')
    ax.text(0.8, 190, 'Height (km)', transform=transform,va='bottom', ha='left', fontsize=12)
    plt.grid(True, which='major', axis='both', color='dimgray', linewidth=1.5, alpha=0.5)

    # Plot the data
    skew.plot(pres[time,:],temp[time,:],'blue',linewidth=2)
    skew.plot(pres[time,:],Td[time,:],'red',linewidth=2)
    skew.plot(pres[time,sfc:],parcel[time,:],'k',linewidth=1.5)

    # wind barbs
    skew.plot_barbs(pres[time,::3],u[time,::3],v[time,::3],xloc=1.25,barb_increments=dict(half=5,full=10,flag=20),sizes=dict(emptybarb=0.1))
    line = mlines.Line2D(xdata=[1.25],ydata=[0, 1],color='gray',linewidth=0.5,transform=ax.transAxes,clip_on=False,zorder=1)
    ax.add_line(line)
    ax.text(1.15,190,'Wind (m/s)',transform=transform,va='bottom', ha='left', fontsize=12,color='k')

    # legend
    legend_elements=[mlines.Line2D([0],[0],color='blue',label='T',linewidth=1),mlines.Line2D([0],[0],color='red',label='$T_d$',linewidth=1),mlines.Line2D([0],[0],color='k',label='parcel',linewidth=1)]
    plt.legend(handles=legend_elements,loc='upper left',fontsize=10)

    # calculation results
    info_text=f'CAPE={cape[time]:6.1f} J/kg\n'\
              f'CIN ={cin[time]:6.1f} J/kg\n'\
              f'LCL ={interp_func(lcl_p[time])*1000:6.0f} m\n'\
              f'LFC ={interp_func(lfc_p[time])*1000:6.0f} m\n'\
              f'EL  ={interp_func(el_p[time])*1000:6.0f} m'
    ax.text(0.04, 0.55, info_text, transform=ax.transAxes, fontsize=9, color='black',bbox=dict(facecolor='white',alpha=0.8,edgecolor='gray'),fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f'./skewT/skewT-{time//3:02d}{time*20%60:02d}.png',dpi=450)
    plt.close(fig)
