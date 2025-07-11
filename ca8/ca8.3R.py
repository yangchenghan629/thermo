import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import metpy.calc as metcal
from metpy.units import units
from tools import Thermo
from scipy.ndimage import uniform_filter1d

###############################################
# READ DATA
###############################################

zc=nc.Dataset('./TC/axmean-000000.nc').variables['zc'][:]
radius=nc.Dataset('./TC/axmean-000000.nc').variables['radius'][:]

th,rv=[np.full((601,34,192),np.nan) for _ in range(2)]
for time in range(601):
    file=nc.Dataset(f'./TC/axmean-{time:06d}.nc')
    th[time,:]=file.variables['th'][:,0,:,:]
    rv[time,:]=file.variables['qv'][:,0,:,:]
radi,tang=[np.full((601,192),np.nan) for _ in range(2)]
for time in range(601):
    file=nc.Dataset(f'./TC/axmean-{time:06d}.nc')
    radi[time,:]=file.variables['radi_wind'][0,0,0,:]
    tang[time,:]=file.variables['tang_wind'][0,0,0,:]

pres=np.broadcast_to(np.loadtxt('./TC/fort.98',skiprows=237,usecols=3,max_rows=34)[None,:,None],(601,34,192))/100 # hPa

###############################################
# CALCULATION
###############################################
ws=np.sqrt(radi**2+tang**2) # m/s
rmw=np.argmax(ws,axis=1)
maxspeed=np.zeros((601))
for time in range(601):
    maxspeed[time]=ws[time,rmw[time]]

## ORIGINAL

temp=Thermo.temp_from_th(th,pres)
qv=rv/(1+rv)
rh=qv/(Thermo.cc_equation(temp)*Thermo.epsilon/(pres-Thermo.cc_equation(temp)))

mse=Thermo.hm(temp,np.broadcast_to(zc[None,:,None],(601,34,192)),qv) /Thermo.cp
qvs=Thermo.cc_equation(temp)*Thermo.epsilon/(pres-Thermo.cc_equation(temp)) 
mses=Thermo.hm(temp,np.broadcast_to(zc[None,:,None],(601,34,192)),qvs) /Thermo.cp

temp_avg=[]
mse_avg=[]
mses_avg=[]
avgrange=2
for time in range(601):
    mse_avg.append(np.nanmean(mse[time,:,rmw[time]-avgrange:rmw[time]+avgrange+1],axis=1))
    mses_avg.append(np.nanmean(mses[time,:,rmw[time]-avgrange:rmw[time]+avgrange+1],axis=1))
    temp_avg.append(np.nanmean(temp[time,:,rmw[time]-avgrange:rmw[time]+avgrange+1],axis=1))
mse_avg=np.array(mse_avg)
mses_avg=np.array(mses_avg)
temp_avg=np.array(temp_avg)

parcel_mse=np.empty((601,34))
diff=[]
for time in range(601):
    parcel_mse[time,:]=np.tile(mse_avg[time,0],(1,34))
    diff.append(parcel_mse[time]-mses_avg[time,:])
diff=np.array(diff) 

all_region=[]
mse_cape=[]
el=[]
Tin=[]
Tout=[]
lnp=np.log(pres[0,:,0])
for time in range(601):
    d=diff[time,:]>0
    edges = np.diff(d.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if d[0]:
        starts = np.r_[0, starts]
    if d[-1]:
        ends = np.r_[ends, len(d)]
    regions = list(zip(starts, ends))
    all_region.append(regions)

    # print(f'Time={time:3d} ,regions#={len(all_region[time]):2d}','*'if len(all_region[time])>1 else '')

    total=0
    if len(all_region[time])==0:
        mse_cape.append(0)
    else:
        for start,end in all_region[time]:
            # print(f'Time {i}, region {start}-{end}')
            dth=diff[time,start:end-1]
            # print(f'dth={dth}')
            dlnp=lnp[start:end-1]
            # print(f'dlnp={dlnp}')
            total+=np.trapz(dth[::-1],dlnp[::-1])
            # print(f'area={np.trapz(dth[::-1],dlnp[::-1])}\n','='*30)
        mse_cape.append(total)

    el.append(all_region[time][-1][1]) if len(all_region[time])>0 else el.append(0)
    Tout.append(temp_avg[time,el[time]])
    Tin.append(temp_avg[time,0])


Tin=np.array(Tin)
Tout=np.array(Tout)

mse_cape=np.array(mse_cape)*Thermo.Rd


def mpi(tin,tout,tsfc,qvsfc):
    cp=1004
    lv=2.5e6
    sst=300
    eta=(tin-tout)/tin
    es=Thermo.cc_equation(tsfc)
    qvs_sfc=es*Thermo.epsilon/(pres[0,0,0]-es)
    dk=cp*(sst-tsfc)+lv*(qvs_sfc-qvsfc)
    return (eta*0.9*dk/(1-eta))**0.5,eta

tsfc=[]
qvsfc=[]
for i in range(601):
    tsfc.append(np.nanmean(temp[i,0,rmw[i]-avgrange:rmw[i]+avgrange+1]))
    qvsfc.append(np.nanmean(qv[i,0,rmw[i]-avgrange:rmw[i]+avgrange+1]))
tsfc=np.array(tsfc)
qvsfc=np.array(qvsfc)

## WARMING

temp2=temp+3
qvs2=Thermo.cc_equation(temp2)*Thermo.epsilon/(pres-Thermo.cc_equation(temp2))
qv2=rh*qvs2

mse2=Thermo.hm(temp2,np.broadcast_to(zc[None,:,None],(601,34,192)),qv2) /Thermo.cp
mses2=Thermo.hm(temp2,np.broadcast_to(zc[None,:,None],(601,34,192)),qvs2) /Thermo.cp

temp2_avg=[]
mse2_avg=[]
mses2_avg=[]
avgrange=2
for time in range(601):
    mse2_avg.append(np.nanmean(mse2[time,:,rmw[time]-avgrange:rmw[time]+avgrange+1],axis=1))
    mses2_avg.append(np.nanmean(mses2[time,:,rmw[time]-avgrange:rmw[time]+avgrange+1],axis=1))
    temp2_avg.append(np.nanmean(temp2[time,:,rmw[time]-avgrange:rmw[time]+avgrange+1],axis=1))
mse2_avg=np.array(mse2_avg)
mses2_avg=np.array(mses2_avg)
temp2_avg=np.array(temp2_avg)


parcel_mse2=np.empty((601,34))
diff=[]
for time in range(601):
    parcel_mse2[time,:]=np.tile(mse2_avg[time,0],(1,34))
    diff.append(parcel_mse2[time]-mses2_avg[time,:])
diff=np.array(diff) 

all_region=[]
mse_cape2=[]
el2=[]
Tin2=[]
Tout2=[]
lnp=np.log(pres[0,:,0])
for time in range(601):
    d=diff[time,:]>0
    edges = np.diff(d.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if d[0]:
        starts = np.r_[0, starts]
    if d[-1]:
        ends = np.r_[ends, len(d)]
    regions = list(zip(starts, ends))
    all_region.append(regions)

    # print(f'Time={time:3d} ,regions#={len(all_region[time]):2d}','*'if len(all_region[time])>1 else '')

    total=0
    if len(all_region[time])==0:
        mse_cape2.append(0)
        if not np.all(d):#parcel always cooler than environment
            mse_cape2.append(0)
        else:
            np.trapz(diff[time,::-1],lnp[::-1])
            mse_cape2.append()
    else:
        for start,end in all_region[time]:
            # print(f'Time {i}, region {start}-{end}')
            dth=diff[time,start:end-1]
            # print(f'dth={dth}')
            dlnp=lnp[start:end-1]
            # print(f'dlnp={dlnp}')
            total+=np.trapz(dth[::-1],dlnp[::-1])
            # print(f'area={np.trapz(dth[::-1],dlnp[::-1])}\n','='*30)
        mse_cape2.append(total)
    if len(all_region[time])>0 :
        el2.append(all_region[time][-1][1]) if all_region[time][-1][1]<=33 else el2.append(33)
    else:
        el2.append(0)
    # print(f'time={time:3d},all_regions={all_region[time]},el={el2[time]}')

    Tout2.append(temp2_avg[time,el2[time]])
    Tin2.append(temp2_avg[time,0])

Tin2=np.array(Tin2)
Tout2=np.array(Tout2)

mse_cape2=np.array(mse_cape2)*Thermo.Rd


tsfc2_avg=[]
qvsfc2_avg=[]
for i in range(601):
    tsfc2_avg.append(np.nanmean(temp2[i,0,rmw[i]-avgrange:rmw[i]+avgrange+1]))
    qvsfc2_avg.append(np.nanmean(qv2[i,0,rmw[i]-avgrange:rmw[i]+avgrange+1]))
tsfc2=np.array(tsfc2_avg)
qvsfc2=np.array(qvsfc2_avg)


#calc mpi
mpi_origin,eta_origin=mpi(Tin,Tout,tsfc,qvsfc)
# mpi_origin=np.where(eta_origin!=0,mpi_origin,np.nan)
mpi_warm,eta_warm=mpi(Tin2,Tout2,tsfc2,qvsfc2)
# mpi_warm=np.where(eta_warm!=0,mpi_warm,np.nan)


mpi_origin_smooth = uniform_filter1d(mpi_origin, size=5)
mpi_warm_smooth = uniform_filter1d(mpi_warm, size=5)

###############################################
# GRAPHING
###############################################

plt.plot(np.arange(601),np.where(eta_origin!=0,mpi_origin,np.nan),'blue',label='Original')
plt.plot(np.arange(601),mpi_warm,'red',label='Warming')
plt.plot(np.arange(601),maxspeed,'orange',label='Real Speed')

plt.xticks(np.arange(0,601,30),[f'{t//3}'for t in np.arange(0,601,30)],rotation=45)
plt.xlim(0,601)
plt.xlabel('Time (hr)')
plt.ylabel('Speed (m/s)')
plt.title('MPI')
plt.grid()
plt.legend()
plt.savefig('mpi_R.png',dpi=450,bbox_inches='tight')

plt.clf()
plt.plot(np.arange(601),mpi_origin_smooth,'blue',label='Original')
plt.plot(np.arange(601),mpi_warm_smooth,'red',label='Warming')
plt.plot(np.arange(601),maxspeed,'orange',label='Real Speed')
plt.xticks(np.arange(0,601,30),[f'{t//3}'for t in np.arange(0,601,30)],rotation=45)
plt.xlim(0,601)
plt.xlabel('Time (hr)')
plt.ylabel('Speed (m/s)')
plt.title('Smooth MPI')
plt.grid()
plt.legend()
plt.savefig('mpi_smooth_R.png',dpi=450,bbox_inches='tight')


# plt.clf()
# plt.title('Efficiency ($\eta$) of TC',fontsize=14)
# plt.plot(np.arange(601),eta_origin,'b-',label='Original')
# plt.plot(np.arange(601),eta_warm,'r-',label='Warming')
# plt.xticks(np.arange(0,601,30),[f'{t//3}'for t in np.arange(0,601,30)],rotation=45)
# plt.xlim(0,601)
# plt.ylim(0,0.5)
# plt.xlabel('Time (hr)')
# plt.ylabel('$\eta$')
# plt.grid()
# plt.legend()
# plt.savefig('eta.png',dpi=450,bbox_inches='tight')


plt.clf()
plt.title('CAPE Evolution',fontsize=14)
plt.plot(np.arange(601),mse_cape,'b',label='Original')
plt.plot(np.arange(601),mse_cape2,'r',label='Warming')
plt.xticks(np.arange(0,601,30),[f'{t//3}'for t in np.arange(0,601,30)],rotation=45)
plt.xlim(0,600)
plt.xlabel('Time (hr)',fontsize=12)
plt.ylabel('CAPE [J/kg]',fontsize=12)
plt.grid()
plt.legend()
plt.savefig('TC_cape_R.png',dpi=450,bbox_inches='tight')


plt.clf()
plt.title('Changing of CAPE (Warming-Presnt)',fontsize=14)
plt.plot(np.arange(601),mse_cape2-mse_cape,'b')
plt.xticks(np.arange(0,601,30),[f'{t//3}'for t in np.arange(0,601,30)],rotation=45)
plt.xlim(0,600)
plt.xlabel('Time (hr)',fontsize=12)
plt.ylabel('Difference [J/kg]',fontsize=12)
plt.grid()
plt.savefig('diff_cape_R.png',dpi=450,bbox_inches='tight')
