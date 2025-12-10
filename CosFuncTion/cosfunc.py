import numpy as np
from astropy import units as u
from astropy.constants import m_p
from functools import wraps

cosmo_params = {}
_cosmo_initialized = False

def Set_Cosmology(h0=0.674, om0=0.315):
    global cosmo_params, _cosmo_initialized
    
    cosmo_params['h'] = h0
    cosmo_params['om0'] = om0
    cosmo_params['ob0'] = 0.0493 * h0**-2
    cosmo_params['ol0'] = 1 - om0
    cosmo_params['rhoc_with_u'] = 2.775366e11 * h0**2 * u.Msun / u.Mpc**3  
    cosmo_params['H0u'] = 100 * h0 * (u.km * u.s**-1 * u.Mpc**-1)
    cosmo_params['H0'] = 100 * h0  # km/s/Mpc
    cosmo_params['rhoc'] = cosmo_params['rhoc_with_u'].value
    cosmo_params['rhom'] = cosmo_params['rhoc'] * om0
    cosmo_params['omegak'] = 0.0
    cosmo_params['omegar'] = 0.0
    
    _cosmo_initialized = True
    print(f"-----[Cosmological Parameters Have Been Set] h={h0}, Ωm={om0}------")


def ensure_cosmology(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _cosmo_initialized
        if not _cosmo_initialized:
            print("-----[Automatically Initializing Cosmological Parameters]-----")
            Set_Cosmology()  # 使用默认参数
        return func(*args, **kwargs)
    return wrapper


@ensure_cosmology
def E(z):
    return np.sqrt(
        cosmo_params['omegar'] * (1.0 + z) ** 4.0 + 
        cosmo_params['om0'] * (1.0 + z) ** 3 + 
        cosmo_params['omegak'] * (1.0 + z) ** 2 + 
        cosmo_params['ol0']
    )

@ensure_cosmology
def H(z):
    return cosmo_params['H0u'] * E(z)

@ensure_cosmology
def dtdz(z):
    return -1.0 / ( (1.0 + z) * H(z) ) 

@ensure_cosmology
def omz(z):
    return cosmo_params['om0'] * (1 + z) ** 3 / E(z) ** 2

@ensure_cosmology
def olz(z):
    return cosmo_params['ol0'] / E(z) ** 2

@ensure_cosmology
def Dz(z):
    def gz(z):
        return 2.5 * omz(z) / (omz(z) ** (4. / 7.) - olz(z) + (1. + omz(z) / 2.) * (1. + olz(z) / 70.))
    return gz(z) / (gz(0.0) * (1.0 + z))

@ensure_cosmology
def n_H(deltaV=0.0):
    X = 0.76
    return (X * cosmo_params['ob0'] * cosmo_params['rhoc_with_u']*(1+deltaV)/m_p).to(1/u.cm**3)

@ensure_cosmology
def Delta_cc(z):
    d = omz(z) - 1.0
    return 18 * np.pi**2 + 82.0 * d - 39.0 * d**2

@ensure_cosmology
def M_vir(mu, Tvir, z):
    a1 = (cosmo_params['om0'] / omz(z) * Delta_cc(z) / (18 * np.pi**2))**(-1.0 / 3.0)
    a2 = a1 * (mu / 0.6)**(-1.0) * ((1.0 + z) / 10)**(-1.0) / 1.98e4 * Tvir
    return a2**(3.0 / 2.0) * 1e8 / cosmo_params['h']

@ensure_cosmology
def M_min(z):
    return M_vir(0.61, 1e4, z)

@ensure_cosmology
def M_jeans(z):
    return 5.73e3*(cosmo_params['om0']*cosmo_params['h']**2/0.15)**(-1/2) * (cosmo_params['ob0']*cosmo_params['h']**2/0.0224)**(-3/5) * ((1+z)/10)**(3/2)

@ensure_cosmology
def fstar(Mh):
    f0 = .14
    ylo = .46
    yhi = .82
    Mp = 10**12.3  # M_sun solmass
    fup = 2 * f0
    fdown = ((Mh / Mp)**-ylo + (Mh / Mp)**yhi)
    return fup / fdown

@ensure_cosmology
def fduty(Mh):
    al = 1.5
    Mc = 6e7
    return (1 + (2.**(al / 3.) - 1) * (Mh / Mc)**-al)**(-3. / al)

@ensure_cosmology
def dMdt(Mh, z):
    return 24.1 * (Mh / (1e12))**1.094 * (1 + 1.75 * z) * E(z)

@ensure_cosmology
def deltac(z):
    return 1.686 / Dz(z)


def z2f(z, nu_rest=1420 * u.MHz):
    if z < 0:
        raise ValueError("Redshift z must be >= 0")
    nu_obs = nu_rest / (1 + z)
    return nu_obs.to(u.MHz)