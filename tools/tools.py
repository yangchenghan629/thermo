class Wind:
    @staticmethod
    def wswd_to_uv(ws,wd):
        import numpy as np
        wd_rad = np.deg2rad(wd)
        u = -ws * np.sin(wd_rad)
        v = -ws * np.cos(wd_rad)
        return u, v
    
    @staticmethod
    def uv_to_wswd(u,v):
        import numpy as np
        ws = (u**2 + v**2)**0.5
        wd = np.degrees(np.arctan2(-u, -v) + 360) % 360
        return ws, wd

class Thermo:

    """
    CONSTANTS
    """
    Rd=287.    # J/(kg*K)
    Rv=461.    # J/(kg*K)    
    cp=1004.   # J/(kg*K)
    cv=cp+Rd   # J/(kg*K)
    g=9.8      # m/s^2
    Lv=2.5e6   # J/kg
    Lf=3.34e5  # J/kg
    Ls=Lv+Lf   # J/kg
    epsilon=Rd/Rv

    """
    REFERENCE
    """
    P0=1000.   # hPa
    T0=273.15  # K
    es0=6.11  # hPa

    """
    CONVERTING
    """

    @staticmethod
    def th_from_temp(temp,pres):
        return temp*(Thermo.P0/pres)**(Thermo.Rd/Thermo.cp)
    
    @staticmethod
    def temp_from_th(th,pres):
        return th/(Thermo.P0/pres)**(Thermo.Rd/Thermo.cp)

    @staticmethod
    def cc_equation(temp):
        from numpy import exp
        return Thermo.es0*exp(Thermo.Lv*(1/Thermo.T0-1/temp)/Thermo.Rv)
    
    @staticmethod
    def e_from_qv_p(qv,pres):
        return qv*pres/Thermo.epsilon
   
    @staticmethod
    def Tc_Zc(temp,pres,qv,lev_ref=None,dim=0):
        """
        Calculate the condensation temperature and condensation level
        units:
            temp: K
            pres: hPa
            qv: kg/kg
            lev_ref: km
            Tc: K
            Zc: km
        """
        import numpy as np
        from scipy.optimize import root_scalar
        A=2.53e9 # hPa
        B=5420 # K

        def find_Tc(temp,pres,qv):
            f=lambda tc:tc-B/(np.log((A*Thermo.epsilon*(temp/tc)**(Thermo.cp/Thermo.Rd))/(pres*qv)))
            if f(100)*f(400)>0:
                print(f'[WARN] Skipping root finding: f(100)={f(100):.2f}, f(400)={f(400):.2f}\nfor T={temp:.2f},P={pres:.2f},qv={qv:.2e}')
                return np.nan
            else:
                return root_scalar(f,bracket=[100,400]).root
        
        if np.ndim(temp)==0 and np.ndim(pres)==0 and np.ndim(qv)==0:
            Tc=find_Tc(temp,pres,qv)
            Zc=np.min(lev_ref[np.nanargmin(np.abs(temp-Tc))])
            return Tc,Zc
        else:
            if dim==1:
                surface_index=np.nanargmax(~np.isnan(temp),axis=1)[0]
                Tc = np.full((temp.shape[0]), np.nan)
                for i in range(temp.shape[0]):
                    Tc[i]=find_Tc(temp[i,surface_index],pres[i,surface_index],qv[i,surface_index])
                Zc=np.full((temp.shape[0]),np.nan)
                for i in range(temp.shape[0]):
                    if (temp[i,:]-Zc[i]<=1e-2).sum()>0:
                        Zc[i]=lev_ref[np.where(temp[i,:]-Tc[i]<=1e-2)]
                    else:
                        Zc[i]=lev_ref[np.nanargmin(np.abs(temp[i,:]-Tc[i]))]
                return Tc,Zc
            elif dim==2:
                Tc_array=np.full((temp.shape[0],temp.shape[1]),np.nan)
                for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        t,p,q=temp[i,j],pres[i,j],qv[i,j]
                        Tc_array[i,j]=find_Tc(t,p,q)
                return Tc_array,None
            elif dim==3:
                Tc_array=np.full((temp.shape[0],temp.shape[1],temp.shape[2]),np.nan)
                for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        for k in range(temp.shape[2]):
                            t,p,q=temp[i,j,k],pres[i,j,k],qv[i,j,k]
                            Tc_array[i,j,k]=find_Tc(t,p,q)
                return Tc_array,None




    @staticmethod
    def Tw(temp,pres,qv):
        """
        Calculate the wet bulb temperature
        units: 
            temp: K
            pres: hPa
            qv: kg/kg
            Tw: K
        """
        import numpy as np
        from scipy.optimize import root_scalar
        A=2.53e9 # hPa
        B=5420 # K

        def find_Tw(temp,pres,qv):
            return root_scalar(lambda tw:tw-temp+(Thermo.Lv*(Thermo.epsilon*A*np.exp(-B/tw)/pres-qv)/Thermo.cp),bracket=[150,400]).root
        
        if np.ndim(temp)==0 and np.ndim(pres)==0 and np.ndim(qv)==0:
            Tw=find_Tw(temp,pres,qv)
            return Tw
        else:
            Tw=np.vectorize(find_Tw)(temp,pres,qv)
            return np.array(Tw)

    @staticmethod
    def Te(temp,qv):
        """
        Calculate the equivalent temperature
        units: 
            temp: K
            qv: kg/kg
            Te: K
        """
        return temp+Thermo.Lv*qv/Thermo.cp
    
    @staticmethod
    def th_equivalent(th,temp,qv):
        """
        Calculate the equivalent potential temperature
        units: 
            th: K
            temp (or Tc,Td) : K
            qv: kg/kg
            theta_e: K
        """
        import numpy as np
        return th*np.exp((Thermo.Lv*qv)/(Thermo.cp*temp))

    @staticmethod
    def Sd(temp,z):
        """
        Calculate the dry static energy
        units: 
            temp: K
            z: m
            Sd: J/kg
        """
        return Thermo.cp*temp+Thermo.g*z

    @staticmethod
    def hm(temp,z,qv):
        """
        Calculate the moist static energy
        units: 
            temp: K
            z: m
            qv: kg/kg
            hm: J/kg
        """
        return Thermo.Sd(temp,z)+Thermo.Lv*qv
    
