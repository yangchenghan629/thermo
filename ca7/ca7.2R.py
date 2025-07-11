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
# read data
###############################################
zc=nc.Dataset('TC/axmean-000000.nc').variables['zc'][:]/1000
radius=nc.Dataset('TC/axmean-000000.nc').variables['radius'][:]
th,rv,u,v=[np.full((601,34,192),np.nan) for _ in range(4)]
sfc_radi_wind,sfc_tang_wind=[np.full((601,192),np.nan) for _ in range(2)]
for time in range(601):
    file=nc.Dataset(f'TC/axmean-{time:06d}.nc')
    th[time,:,:]=file.variables['th'][:,0,:,:]
    rv[time,:,:]=file.variables['qv'][:,0,:,:]
    u[time,:,:]=file.variables['u'][:,0,:,:]
    v[time,:,:]=file.variables['v'][:,0,:,:]
    sfc_radi_wind[time,:]=file.variables['radi_wind'][:,0,0,:]
    sfc_tang_wind[time,:]=file.variables['tang_wind'][:,0,0,:]

pres=np.broadcast_to(np.loadtxt('./TC/fort.98',skiprows=237,usecols=3,max_rows=34)[None,:,None]/100,(601,34,192))*units.hPa

###############################################
# calculation
###############################################
sfc_speed=np.sqrt(sfc_radi_wind**2+sfc_tang_wind**2)
rmw=np.argmax(sfc_speed,axis=1)
outer=np.tile(np.where(np.abs(radius-200000)<=5000)[0][0],len(rmw))

qv=rv/(1+rv)
temp=(Thermo.temp_from_th(th,pres.magnitude)-273.15)*units.degC
Td=metcal.dewpoint_from_specific_humidity(pres,temp,qv).to('degC')
parcel,parcel_mixing_ratio,Tv_parcel=[np.full((601,34,2),np.nan) for _ in range(3)]
for time in range(601):
    parcel[time,:,0]=metcal.parcel_profile(pres[time,:,int(rmw[time])],temp[time,0,int(rmw[time])],Td[time,0,int(rmw[time])]).to('degC')
    parcel[time,:,1]=metcal.parcel_profile(pres[time,:,outer[time]],temp[time,0,outer[time]],Td[time,0,outer[time]]).to('degC')

    pressure_lcl, _ = metcal.lcl(pres[time,:,int(rmw[time])], temp[time,0,int(rmw[time])], Td[time,0,int(rmw[time])])
    below_lcl = pres[time,:,int(rmw[time])] > pressure_lcl
    parcel_mixing_ratio[time,:,0] = np.where(below_lcl,rv[time,0,int(rmw[time])],metcal.saturation_mixing_ratio(pres[time,:,int(rmw[time])], parcel[time,:,0]*units.degC))
    Tv_parcel[time,:,0]=metcal.virtual_temperature(parcel[time,:,0]*units.degC,parcel_mixing_ratio[time,:,0])
    
    pressure_lcl, _ = metcal.lcl(pres[time,:,outer[time]], temp[time,0,outer[time]], Td[time,0,outer[time]])
    below_lcl = pres[time,:,outer[time]] > pressure_lcl
    parcel_mixing_ratio[time,:,1] = np.where(below_lcl,rv[time,0,outer[time]],metcal.saturation_mixing_ratio(pres[time,:,outer[time]], parcel[time,:,1]*units.degC))
    Tv_parcel[time,:,1]=metcal.virtual_temperature(parcel[time,:,1]*units.degC,parcel_mixing_ratio[time,:,1])

cape,cin,lcl_p,lcl_t,lfc_p,lfc_t,el_p,el_t=[np.full((601,2),np.nan) for _ in range(8)]

"""
NOTE: enumerate() returns a tuple of (index, value) pairs.
in second loop, i=0 for rmw ,i=1 for outer . indx is index of rmw or outer.
"""
for time,(index_rmw,index_outer) in enumerate(zip(rmw,outer)):
    for i,indx in enumerate([index_rmw,index_outer]):
        ca,ci=metcal.cape_cin(pres[time,:,indx],temp[time,:,indx],Td[time,:,indx],parcel[time,:,i]*units.degC)
        cape[time,i]=ca.magnitude
        cin[time,i]=ci.magnitude
        lclp,lclt=metcal.lcl(pres[time,0,indx],temp[time,0,indx],Td[time,0,indx])
        lcl_p[time,i]=lclp.magnitude
        lcl_t[time,i]=lclt.magnitude
        lfcp,lfct=metcal.lfc(pres[time,:,indx],metcal.virtual_temperature(temp[time,:,indx],rv[time,:,indx]),Td[time,:,indx],Tv_parcel[time,:,i]*units.degC)
        lfc_p[time,i]=lfcp.magnitude
        lfc_t[time,i]=lfct.magnitude
        elp,elt=metcal.el(pres[time,:,indx],metcal.virtual_temperature(temp[time,:,indx],rv[time,:,indx]),Td[time,:,indx],Tv_parcel[time,:,i]*units.degC)
        el_p[time,i]=elp.magnitude
        el_t[time,i]=elt.magnitude

interp_func = interp1d(pres[0,:,0],zc, kind='cubic', bounds_error=False, fill_value='extrapolate')
h_ref=interp_func(np.arange(1000,0,-100))

###############################################
# plot
###############################################

# RMW temperature
for time in range(0,601,3):
    plt.clf()
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'Skew-T Log-P Diagram | {time//3:02d} hr @ RMW',fontsize=14,x=0.5)
    skew = SkewT(fig, rotation=45)
    ax = skew.ax

    # Shade every other section between isotherms
    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90, 50, 8)
    y = [1100, 50]
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
    ax.set_yticks(np.arange(1000,99,-100))
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xlabel('Temperature ($\degree$C)',fontsize=12)
    ax.set_ylabel('')
    ax.text(-0.05, 98, 'Pressure (hPa)', transform=transform,va='bottom', ha='left', fontsize=12)
    ax.set_xlim(-20,40)
    ax.set_ylim(1100,100)
    for p, h in zip(np.arange(1000,99,-100), h_ref):
        ax.text(1.01, p, f'{h:.2f}', transform=transform,va='center', ha='left', fontsize=10, color='black')
    ax.text(0.8, 98, 'Height (km)', transform=transform,va='bottom', ha='left', fontsize=12)
    plt.grid(True, which='major', axis='both', color='dimgray', linewidth=1.5, alpha=0.5)

    # Plot temp,Td,parcel
    skew.plot(pres[time,:,rmw[time]],temp[time,:,rmw[time]],'blue',linewidth=2)
    skew.plot(pres[time,:,rmw[time]],Td[time,:,rmw[time]],'red',linewidth=2)
    skew.plot(pres[time,:,rmw[time]],parcel[time,:,0],'k',linewidth=1.5)

    # wind barbs
    skew.plot_barbs(pres[time,:-2:2,rmw[time]],u[time,:-2:2,rmw[time]],v[time,:-2:2,rmw[time]],xloc=1.25,barb_increments=dict(half=5,full=10,flag=50),sizes=dict(emptybarb=0.1))
    line = mlines.Line2D(xdata=[1.25],ydata=[0, 1],color='gray',linewidth=0.5,transform=ax.transAxes,clip_on=False,zorder=1)
    ax.add_line(line)
    ax.text(1.15,98,'Wind (m/s)',transform=transform,va='bottom', ha='left', fontsize=12,color='k')

    # legend
    legend_elements=[mlines.Line2D([0],[0],color='blue',label='T',linewidth=1),mlines.Line2D([0],[0],color='red',label='$T_d$',linewidth=1),mlines.Line2D([0],[0],color='k',label='parcel',linewidth=1)]
    plt.legend(handles=legend_elements,loc='upper right',fontsize=10)

    # calculation results
    info_text=f'CAPE={cape[time,0]:6.1f} J/kg\n'\
              f'CIN ={cin[time,0]:6.1f} J/kg\n'\
              f'LCL ={interp_func(lcl_p[time,0])*1000:6.0f} m\n'\
              f'LFC ={interp_func(lfc_p[time,0])*1000:6.0f} m\n'\
              f'EL  ={interp_func( el_p[time,0])*1000:6.0f} m'
    ax.text(0.61, 0.67,info_text,fontfamily='monospace', transform=ax.transAxes, fontsize=9, color='black',bbox=dict(facecolor='white',alpha=0.8,edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(f'./TC_skewT/skewT-{time//3:03d}{time*20%60:02d}.png',dpi=450)
    plt.close(fig)
# RMW virtual temperature
for time in range(0,601,3):
    plt.clf()
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'Skew-T Log-P Diagram | {time//3:02d} hr @ RMW',fontsize=14,x=0.5)
    skew = SkewT(fig, rotation=45)
    ax = skew.ax

    # Shade every other section between isotherms
    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90, 50, 8)
    y = [1100, 50]
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
    ax.set_yticks(np.arange(1000,99,-100))
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xlabel('Temperature ($\degree$C)',fontsize=12)
    ax.set_ylabel('')
    ax.text(-0.05, 98, 'Pressure (hPa)', transform=transform,va='bottom', ha='left', fontsize=12)
    ax.set_xlim(-20,40)
    ax.set_ylim(1100,100)
    for p, h in zip(np.arange(1000,99,-100), h_ref):
        ax.text(1.01, p, f'{h:.2f}', transform=transform,va='center', ha='left', fontsize=10, color='black')
    ax.text(0.8, 98, 'Height (km)', transform=transform,va='bottom', ha='left', fontsize=12)
    plt.grid(True, which='major', axis='both', color='dimgray', linewidth=1.5, alpha=0.5)

    # Plot temp,Td,parcel
    skew.plot(pres[time,:,rmw[time]],metcal.virtual_temperature(temp[time,:,rmw[time]],rv[time,:,rmw[time]]),'blue',linewidth=2)
    skew.plot(pres[time,:,rmw[time]],Td[time,:,rmw[time]],'red',linewidth=2)
    skew.plot(pres[time,:,rmw[time]],Tv_parcel[time,:,0],'k',linewidth=1.5)

    # wind barbs
    skew.plot_barbs(pres[time,:-2:2,rmw[time]],u[time,:-2:2,rmw[time]],v[time,:-2:2,rmw[time]],xloc=1.25,barb_increments=dict(half=5,full=10,flag=50),sizes=dict(emptybarb=0.1))
    line = mlines.Line2D(xdata=[1.25],ydata=[0, 1],color='gray',linewidth=0.5,transform=ax.transAxes,clip_on=False,zorder=1)
    ax.add_line(line)
    ax.text(1.15,98,'Wind (m/s)',transform=transform,va='bottom', ha='left', fontsize=12,color='k')

    # legend
    legend_elements=[mlines.Line2D([0],[0],color='blue',label='Tv',linewidth=1),mlines.Line2D([0],[0],color='red',label='$T_d$',linewidth=1),mlines.Line2D([0],[0],color='k',label='parcel Tv',linewidth=1)]
    plt.legend(handles=legend_elements,loc='upper right',fontsize=10)

    # calculation results
    info_text=f'CAPE={cape[time,0]:6.1f} J/kg\n'\
              f'CIN ={cin[time,0]:6.1f} J/kg\n'\
              f'LCL ={interp_func(lcl_p[time,0])*1000:6.0f} m\n'\
              f'LFC ={interp_func(lfc_p[time,0])*1000:6.0f} m\n'\
              f'EL  ={interp_func( el_p[time,0])*1000:6.0f} m'
    ax.text(0.61, 0.67,info_text,fontfamily='monospace', transform=ax.transAxes, fontsize=9, color='black',bbox=dict(facecolor='white',alpha=0.8,edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(f'./TC_skewTR/skewT-{time//3:03d}{time*20%60:02d}.png',dpi=450)
    plt.close(fig)


# OUTER temperature
for time in range(0,601,3):
    # print(rmw[time],outer[time])

    plt.clf()
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'Skew-T Log-P Diagram | {time//3:02d} hr @ 200km',fontsize=14,x=0.5)
    skew = SkewT(fig, rotation=45)
    ax = skew.ax

    # Shade every other section between isotherms
    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90, 50, 8)
    y = [1100, 50]
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
    ax.set_yticks(np.arange(1000,99,-100))
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xlabel('Temperature ($\degree$C)',fontsize=12)
    ax.set_ylabel('')
    ax.text(-0.05, 98, 'Pressure (hPa)', transform=transform,va='bottom', ha='left', fontsize=12)
    ax.set_xlim(-20,40)
    ax.set_ylim(1100,100)
    for p, h in zip(np.arange(1000,99,-100), h_ref):
        ax.text(1.01, p, f'{h:.2f}', transform=transform,va='center', ha='left', fontsize=10, color='black')
    ax.text(0.8, 98, 'Height (km)', transform=transform,va='bottom', ha='left', fontsize=12)
    plt.grid(True, which='major', axis='both', color='dimgray', linewidth=1.5, alpha=0.5)

    # Plot temp,Td,parcel
    skew.plot(pres[time,:,outer[time]],temp[time,:,outer[time]],'blue',linewidth=2)
    skew.plot(pres[time,:,outer[time]],Td[time,:,outer[time]],'red',linewidth=2)
    skew.plot(pres[time,:,outer[time]],parcel[time,:,1],'k',linewidth=1.5)

    # wind barbs
    skew.plot_barbs(pres[time,:-2:2,outer[time]],u[time,:-2:2,outer[time]],v[time,:-2:2,outer[time]],xloc=1.25,barb_increments=dict(half=5,full=10,flag=50),sizes=dict(emptybarb=0.1))
    line = mlines.Line2D(xdata=[1.25],ydata=[0, 1],color='gray',linewidth=0.5,transform=ax.transAxes,clip_on=False,zorder=1)
    ax.add_line(line)
    ax.text(1.15,98,'Wind (m/s)',transform=transform,va='bottom', ha='left', fontsize=12,color='k')

    # legend
    legend_elements=[mlines.Line2D([0],[0],color='blue',label='T',linewidth=1),mlines.Line2D([0],[0],color='red',label='$T_d$',linewidth=1),mlines.Line2D([0],[0],color='k',label='parcel',linewidth=1)]
    plt.legend(handles=legend_elements,loc='upper right',fontsize=10)

    # calculation results
    info_text=f'CAPE={cape[time,1]:6.1f} J/kg\n'\
              f'CIN ={cin[time,1]:6.1f} J/kg\n'\
              f'LCL ={interp_func(lcl_p[time,1])*1000:6.0f} m\n'\
              f'LFC ={interp_func(lfc_p[time,1])*1000:6.0f} m\n'\
              f'EL  ={interp_func( el_p[time,1])*1000:6.0f} m'
    ax.text(0.61, 0.67,info_text,fontfamily='monospace', transform=ax.transAxes, fontsize=9, color='black',bbox=dict(facecolor='white',alpha=0.8,edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(f'./TC_skewT2/skewT-{time//3:03d}{time*20%60:02d}.png',dpi=450)
    plt.close(fig)
    
# OUTER virtual temperature
for time in range(0,601,3):

    plt.clf()
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'Skew-T Log-P Diagram | {time//3:02d} hr @ 200km',fontsize=14,x=0.5)
    skew = SkewT(fig, rotation=45)
    ax = skew.ax

    # Shade every other section between isotherms
    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90, 50, 8)
    y = [1100, 50]
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
    ax.set_yticks(np.arange(1000,99,-100))
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xlabel('Temperature ($\degree$C)',fontsize=12)
    ax.set_ylabel('')
    ax.text(-0.05, 98, 'Pressure (hPa)', transform=transform,va='bottom', ha='left', fontsize=12)
    ax.set_xlim(-20,40)
    ax.set_ylim(1100,100)
    for p, h in zip(np.arange(1000,99,-100), h_ref):
        ax.text(1.01, p, f'{h:.2f}', transform=transform,va='center', ha='left', fontsize=10, color='black')
    ax.text(0.8, 98, 'Height (km)', transform=transform,va='bottom', ha='left', fontsize=12)
    plt.grid(True, which='major', axis='both', color='dimgray', linewidth=1.5, alpha=0.5)

    # Plot temp,Td,parcel
    skew.plot(pres[time,:,outer[time]],metcal.virtual_temperature(temp[time,:,outer[time]],rv[time,:,outer[time]]),'blue',linewidth=2)
    skew.plot(pres[time,:,outer[time]],Td[time,:,outer[time]],'red',linewidth=2)
    skew.plot(pres[time,:,outer[time]],Tv_parcel[time,:,1],'k',linewidth=1.5)

    # wind barbs
    skew.plot_barbs(pres[time,:-2:2,outer[time]],u[time,:-2:2,outer[time]],v[time,:-2:2,outer[time]],xloc=1.25,barb_increments=dict(half=5,full=10,flag=50),sizes=dict(emptybarb=0.1))
    line = mlines.Line2D(xdata=[1.25],ydata=[0, 1],color='gray',linewidth=0.5,transform=ax.transAxes,clip_on=False,zorder=1)
    ax.add_line(line)
    ax.text(1.15,98,'Wind (m/s)',transform=transform,va='bottom', ha='left', fontsize=12,color='k')

    # legend
    legend_elements=[mlines.Line2D([0],[0],color='blue',label='Tv',linewidth=1),mlines.Line2D([0],[0],color='red',label='$T_d$',linewidth=1),mlines.Line2D([0],[0],color='k',label='parcel Tv',linewidth=1)]
    plt.legend(handles=legend_elements,loc='upper right',fontsize=10)

    # calculation results
    info_text=f'CAPE={cape[time,1]:6.1f} J/kg\n'\
              f'CIN ={cin[time,1]:6.1f} J/kg\n'\
              f'LCL ={interp_func(lcl_p[time,1])*1000:6.0f} m\n'\
              f'LFC ={interp_func(lfc_p[time,1])*1000:6.0f} m\n'\
              f'EL  ={interp_func( el_p[time,1])*1000:6.0f} m'
    ax.text(0.61, 0.67,info_text,fontfamily='monospace', transform=ax.transAxes, fontsize=9, color='black',bbox=dict(facecolor='white',alpha=0.8,edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(f'./TC_skewTR2/skewT-{time//3:03d}{time*20%60:02d}.png',dpi=450)
    plt.close(fig)



