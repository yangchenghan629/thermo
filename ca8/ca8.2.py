import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import metpy.calc as metcal
from metpy.units import units
from tools import Thermo
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
###############################################
# READ DATA
###############################################
# topographic
loc=nc.Dataset('ne_20191030/TOPO.nc')
lon=np.where(np.abs(loc.variables['lon'][:]-120.6289)<=1e-5)[0][0]
lat=np.where(np.abs(loc.variables['lat'][:]-23.49798)<=1e-5)[0][0]
lev=loc.variables['lev'][:] #km
mask=loc.variables['mask'][:,lat,lon]
sfc=np.argmin(mask!=1)

# thermodynamic variables
th,rv=[np.full((73,60),np.nan) for _ in range(2)]
for time in range(73):
    file=nc.Dataset(f'ne_20191030/ne_20191030.L.Thermodynamic-{time:06d}.nc')
    th[time,:]=np.where(mask,file.variables['th'][:,:,lat,lon],np.nan) # k
    rv[time,:]=np.where(mask,file.variables['qv'][:,:,lat,lon],np.nan) # kg/kg

p_ref=np.loadtxt('./ne_20191030/fort.98',usecols=3,skiprows=2)/100 # hPa
pres=np.broadcast_to(np.where(mask,p_ref,np.nan),(73,60))


###############################################
# CALCULATION
###############################################
interp_func = interp1d(p_ref, lev, kind='cubic', bounds_error=False, fill_value='extrapolate')

qv=rv/(1+rv) # kg/kg
temp=Thermo.temp_from_th(th,pres) # K
td=metcal.dewpoint_from_specific_humidity(pres*units.hPa,temp*units.K,qv*units('kg/kg')).to('K')


# cape cin
Td=metcal.dewpoint_from_specific_humidity(pres*units.hPa,temp*units.K,qv*units('kg/kg')).to('K')
cape,cin=[np.full((73),np.nan) for _ in range(2)]
Tv=metcal.virtual_temperature(temp[:,sfc:]*units.K,rv[:,sfc:]*units('kg/kg'))
parcel,parcel_mixing_ratio,Tv_parcel=[np.full((73,60-sfc),np.nan) for _ in range(3)]
for time in range(73):
    parcel[time,:]=metcal.parcel_profile(pres[time,sfc:]*units.hPa,temp[time,sfc]*units.K,Td[time,sfc]).to('degC')
    pressure_lcl, _ = metcal.lcl(pres[time,sfc]*units.hPa, temp[time,sfc]*units.K, Td[time,sfc])
    below_lcl = pres[time,sfc:] > pressure_lcl.magnitude
    parcel_mixing_ratio[time,:] = np.where(below_lcl,rv[time,sfc],metcal.saturation_mixing_ratio(pres[time,sfc:]*units.hPa, parcel[time,:]*units.degC))
    Tv_parcel[time,:]=metcal.virtual_temperature(parcel[time,:]*units.degC,parcel_mixing_ratio[time]).to('K')
# metpy
for time in range(73):
    ca,ci=metcal.cape_cin(pres[time,sfc:]*units.hPa,temp[time,sfc:]*units.K,Td[time,sfc:],parcel[time,:]*units.degC)
    cape[time]=ca.magnitude
    cin[time]=ci.magnitude
# intergration by self
diff=Tv_parcel-Tv.m
all_region=[]
Tv_cape=[]
lnp=np.log(pres[0,sfc:])
for i in range(73):
    
    d=diff[i,:]>0
    edges = np.diff(d.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if d[0]:
        starts = np.r_[0, starts]
    if d[-1]:
        ends = np.r_[ends, len(d)]
    regions = list(zip(starts, ends))
    all_region.append(regions)
    total=0
    if len(all_region[i])==0:
        Tv_cape.append(0)
    else:
        for start,end in all_region[i]:
            # print(f'Time {i}, region {start}-{end}')
            dth=diff[i,start:end]
            # print(f'dth={dth}')
            dlnp=lnp[start:end]
            # print(f'dlnp={dlnp}')
            total+=np.trapz(dth[::-1],dlnp[::-1])
            # print(f'area={np.trapz(dth[::-1],dlnp[::-1])}\n','='*30)
        Tv_cape.append(total)
Tv_cape=np.array(Tv_cape)*Thermo.Rd



# theta_e and theta_es of environment and use first layer th_e as parcel
th_e=Thermo.th_equivalent(th,temp,qv)
qvs=Thermo.cc_equation(temp)*Thermo.epsilon/(pres-Thermo.cc_equation(temp)) 
th_es=Thermo.th_equivalent(th,temp,qvs)
parcel_th_e=np.tile(th_e[:,sfc],(60-sfc,1)).T # 73 x 60-sfc

diff=parcel_th_e-th_es[:,sfc:]

all_region=[]
th_cape=[]
lnp=np.log(pres[0,sfc:])
for i in range(73):
    d=diff[i,:]>0
    edges = np.diff(d.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if d[0]:
        starts = np.r_[0, starts]
    if d[-1]:
        ends = np.r_[ends, len(d)]
    regions = list(zip(starts, ends))
    all_region.append(regions)
    total=0
    if len(all_region[i])==0:
        th_cape.append(0)
    else:
        for start,end in all_region[i]:
            # print(f'Time {i}, region {start}-{end}')
            dth=diff[i,start:end]
            # print(f'dth={dth}')
            dlnp=lnp[start:end]
            # print(f'dlnp={dlnp}')
            total+=np.trapz(dth[::-1],dlnp[::-1])
            # print(f'area={np.trapz(dth[::-1],dlnp[::-1])}\n','='*30)
        th_cape.append(total)
th_cape=np.array(th_cape)*Thermo.Rd

# use MSE calc cape
hm=Thermo.hm(temp,lev*1000,qv) /Thermo.cp
hms=Thermo.hm(temp,lev*1000,qvs) /Thermo.cp
parcel_hm=np.tile(hm[:,sfc],(60-sfc,1)).T 

diff=parcel_hm-hms[:,sfc:]

all_region=[]
hm_cape=[]
lnp=np.log(pres[0,sfc:])
for i in range(73):
    d=diff[i,:]>0
    edges = np.diff(d.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if d[0]:
        starts = np.r_[0, starts]
    if d[-1]:
        ends = np.r_[ends, len(d)]
    regions = list(zip(starts, ends))
    all_region.append(regions)
    # print(i,regions)
    total=0
    if len(all_region[i])==0:
        hm_cape.append(0)
    else:
        for start,end in all_region[i]:
            # print(f'Time {i}, region {start}-{end}')
            # print(f'dhm.shape={diff[i,start:end-1]}')
            # print(f'dlnp.shape={lnp[start:end-1]}')
            dhm=diff[i,start:end]
            dlnp=lnp[start:end]
            area = np.trapz(dhm[::-1], dlnp[::-1])
            if end-start==1:
                area=0
            total+=area
            # print(f"i={i}, area={area}, is masked: {np.ma.is_masked(area)}")

            # print(f'area={np.trapz(dth[::-1],dlnp[::-1])}\n','='*30)
        hm_cape.append(total)
hm_cape=np.array(hm_cape)*Thermo.Rd

###############################################
# GRAPHING
###############################################

plt.title('CAPE from different methods',fontsize=14)
plt.plot(np.arange(73),th_cape,'r-',label=r'from $\theta_{e}$')
plt.plot(np.arange(73),hm_cape,'g-',label='from MSE')
# plt.plot(np.arange(73),cape,'m-',label='from $T_v$ with metpy')
plt.plot(np.arange(73),Tv_cape,'b-',label='from Tv')
plt.xticks(np.arange(0,73,3),[f'{h//3:02d}'for h in range(0,73,3)])
plt.xlim(0,72)
plt.xlabel('Time [hr]',fontsize=12)
plt.ylabel('CAPE [J/kg]',fontsize=12)
plt.grid()
plt.legend()
plt.savefig('compare_cape.png',dpi=450)

plt.clf()
plt.title(r'Difference between CAPE from $\theta_e$  MSE and $T_v$',fontsize=14)
plt.plot(np.arange(73),th_cape-cape,'b',label=r'$\theta_e$_CAPE-Tv_CAPE')
plt.plot(np.arange(73),hm_cape-cape,'g',label=r'$h_m$_CAPE-Tv_CAPE')
plt.hlines(0,0,72,colors='black')
plt.xticks(np.arange(0,73,3),[f'{h//3:02d}'for h in range(0,73,3)])
plt.xlim(0,72)
plt.xlabel('Time [hr]',fontsize=12)
plt.ylabel('Difference [J/kg]',fontsize=12)
plt.grid()
plt.legend()
plt.savefig('cape_diff.png',dpi=450)
