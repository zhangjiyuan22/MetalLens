import sys
#from functions_gpu import *
# from functions_cpu import *
from getconfig import *
import os
import time
# from scipy.interpolate import griddata
import numpy as np
# from astropy.coordinates import SkyCoord
# import astropy.units as u
from multiprocessing import get_context
import h5py


def mkdir(path):
    Exist = os.path.exists(path)
    if Exist:
        print(path, '  already exists!')
    else:
        os.makedirs(path)
        print(path,'  created!')

def Gaussian1D(x,mean,sigma):
    A = 1/np.sqrt(2*np.pi)/sigma
    cx = (x-mean)/sigma
    #chi = cx**2
    return A*np.exp(-0.5*cx**2)

def Gaussian2D(x1,x2,mean1,mean2,sigma1,sigma2,rho12):
    A = 1/(2*np.pi*sigma1*sigma2*np.sqrt(1-rho12**2))
    cx1 = (x1-mean1)/sigma1
    cx2 = (x2-mean2)/sigma2
    chi = (1/(1-rho12**2)) * ( cx1**2 -2*rho12*cx1*cx2 + cx2**2 )
    return A*np.exp(-0.5*chi)



######################################
## Look up for lens mass range & #####
## Interpolate one or more filters' magnitudes at (MH, Age[Gyr], Mass) triplets from PARSEC npz file ##
######################################
# ---- global PARSEC cache (one per worker process) ----
PARSEC_DB = None

def _init_parsec_db(npz_path, fields_needed):
    """
    Runs once in each worker process.
    Loads only what we need into memory (dict of ndarrays/scalars).
    """
    global PARSEC_DB
    z = np.load(npz_path, allow_pickle=False)

    db = {}
    # scalars / axes / packing
    for k in ["mh_min", "dmh", "n_mh", "age_min", "dage", "n_age", "MH_axis", "Age_axis",
              "offset", "length", "Mass", "mass_min", "mass_max"]:
        db[k] = z[k]

    # requested photometric fields
    for f in fields_needed:
        db[f] = z[f]

    z.close()

    PARSEC_DB = db



def _nearest_index_uniform(x, x0, dx, n):
    #############################################################################################
    #     """Nearest integer index on a uniform axis; robust to tiny float jitter."""
    #############################################################################################
    idx = np.rint((x - x0) / dx).astype(np.int64)
    return np.clip(idx, 0, n - 1)

def mass_bounds_for_samples(DB, Age_lens, MH_lens):
    #     """
    #     Parameters
    #     ----------
    #     npz_path : str
    #         Path to your packed PARSEC NPZ (MH-outer, Age-inner).
    #     Age_lens, MH_lens : array-like (same shape)
    #         Sampled ages [Gyr] and metallicities [M/H].

    #     Returns
    #     -------
    #     mass_max_arr, mass_min_arr : ndarray
    #         Arrays of max/min allowable mass for each sample, same shape as inputs.
    #     """
    # DB = np.load(npz_path, allow_pickle=False)

    # axes / spacing
    mh_min, dmh, n_mh   = float(DB["mh_min"]),  float(DB["dmh"]),  int(DB["n_mh"])
    age_min, dage, n_age = float(DB["age_min"]), float(DB["dage"]), int(DB["n_age"])

    # nearest grid indices (MH outer, Age inner)
    i_mh  = _nearest_index_uniform(MH_lens,  mh_min,  dmh,  n_mh)
    j_age = _nearest_index_uniform(Age_lens, age_min, dage, n_age)
    iso_idx = i_mh * n_age + j_age  # linear index in your NPZ packing

    # lookup per-iso bounds and gather
    mass_min_tbl = np.asarray(DB["mass_min"], float)  # shape (n_mh*n_age,)
    mass_max_tbl = np.asarray(DB["mass_max"], float)
    
    mass_min_arr = mass_min_tbl[iso_idx]
    mass_max_arr = mass_max_tbl[iso_idx]

    return mass_min_arr, mass_max_arr



def _interp_strict_increasing(m_grid, y_grid, q):
    #########################################################################
    #     Vectorized linear interpolation on a strictly increasing 1D grid.
    #     Returns (y(q) #, in_range_mask). Out-of-range -> NaN.
    #########################################################################
#     m_grid = np.asarray(m_grid, float)
#     y_grid = np.asarray(y_grid, float)
#     q      = np.asarray(q, float)

    n = m_grid.size
    out = np.full(q.shape, np.nan, dtype=float)
#     if n < 2:
#         return out, np.zeros_like(q, bool)

    # indices of left bin edge
    # a = np.array([1, 3, 5, 7, 9])
    # indices_left = np.searchsorted(a, 5, side='left')   # output 2
    # indices_right = np.searchsorted(a, 5, side='right') # output 3
    #
    # left:  a[i-1] < v <= a[i]
    # right: a[i-1] <= v < a[i]
    k = np.searchsorted(m_grid, q, side="right") - 1
    # m_grid = [0.094, 0.1, 0.5, 1.0, 1.5]
    # q = [0.094, 0.095, 0.1, 0.2, 1.4, 1.5]
    # k = [1, 1, 2, 2, 4, 5] - 1 = [0, 0, 1, 1, 3, 4]
    
#     # in-range if q in [m_grid[0], m_grid[-1]]
#     inrng = (q >= m_grid[0]) & (q <= m_grid[-1])

    # clamp k to [0, n-2] for valid pairs (k, k+1)
    # k = [0, 0, 1, 1, 3, 4] to k = [0, 0, 1, 1, 3, 3]
    # only affect the points == m_grid[-1]
    k = np.clip(k, 0, n - 2)

    m0 = m_grid[k]
    m1 = m_grid[k + 1]
    y0 = y_grid[k]
    y1 = y_grid[k + 1]

    # linear fraction; safe because m1>m0 for strictly increasing grid
    t = (q - m0) / (m1 - m0)
#     out[inrng] = y0[inrng] + t[inrng] * (y1[inrng] - y0[inrng])
    out = y0 + t * (y1 - y0)
    
    return out #, inrng



NPZ_PATH = "PARSEC_isochrone/final_isochrone_label012_vista_roman_euclid_ogle2_csst_merged_no_repeating_mass.npz"

FIELDS = [
    "Mass","logL","logTe","logg","label",
    "mbolmag",
    "Zmag_VISTA","Ymag_VISTA","Jmag_VISTA","Hmag_VISTA","Ksmag_VISTA",
    "F062mag_ROMAN","F087mag_ROMAN","F106mag_ROMAN","F129mag_ROMAN", #F087 is 12
    "F158mag_ROMAN","F184mag_ROMAN","F146mag_ROMAN","F213mag_ROMAN", #F146 is 17
    "VISmag_EUCLID","Ymag_EUCLID","Bluemag_EUCLID","Jmag_EUCLID","Redmag_EUCLID","Hmag_EUCLID", #VIS is 19
    "Umag_OGLE2","Bmag_OGLE2","Vmag_OGLE2","Imag_OGLE2", # V is 27
    "NUVmag_CSST", "umag_CSST", "gmag_CSST", "rmag_CSST", "imag_CSST", "zmag_CSST", "ymag_CSST" #gmag_CSST is 31
]

###########################
### change filters here ###
### if you change the filters used, 
### remember to also recalculate the lookup table LOOKUP_CCM89_MDWARF_F062_F087_F213 
### using the code provided alongside. 
###########################
# F062, F087, F213
fields = (FIELDS[11], FIELDS[12], FIELDS[18])



def interpolate_from_triplets(DB, MH_input_array, Age_input_array, Mass_input_array, fields=("Ksmag_VISTA",),
                         return_snapped_axes=True):
    #############################################################################################
    #     Interpolate one or more fields from a PARSEC NPZ DB for arrays of (MH, Age[Gyr], Mass).

    #     Parameters
    #     ----------
    #     npz_path : str
    #         Path to your compact NPZ (MH-outer, Age-inner).
    #     MH, Age, Mass : array-like, same shape
    #         Query values per simulated event.
    #     fields : tuple/list of str
    #         Keys inside NPZ to interpolate (e.g., "Ksmag_VISTA", "Jmag_VISTA", "logg", "logTe", "logL").
    #     return_snapped_axes : bool
    #         If True, also return the snapped (grid) MH and Age used for each event.

    #     Returns
    #     -------
    #     out : dict
    #         out[field] -> interpolated array (same shape as inputs, NaN if out-of-range in mass).
    #     meta : dict
    #         "iso_idx"  : linear isochrone index (i_mh * n_age + j_age)
    #         "inrange"  : boolean mask where mass was within isochrone bounds
    #         "MH_grid"  : snapped MH grid values (optional)
    #         "Age_grid" : snapped Age[Gyr] grid values (optional)
    #############################################################################################
    
    #############################################################################################
    ## notice: here _array suffix only refer to the input simulated events array
    #############################################################################################
    
    # DB = np.load(npz_path, allow_pickle=False)

#     MH      = np.asarray(MH, float)
#     Age     = np.asarray(Age, float)
#     Mass    = np.asarray(Mass, float)
#     if not (MH.shape == Age.shape == Mass.shape):
#         raise ValueError("MH, Age, Mass must have the same shape.")

    # axes / packing
    mh_min, dmh, n_mh  = float(DB["mh_min"]),  float(DB["dmh"]),  int(DB["n_mh"])
    age_min, dage, n_age = float(DB["age_min"]), float(DB["dage"]), int(DB["n_age"])
    MH_axis  = DB["MH_axis"].astype(float)
    Age_axis = DB["Age_axis"].astype(float)
#     print('mh_min, dmh, n_mh', mh_min, dmh, n_mh)
#     print('age_min, dage, n_age', age_min, dage, n_age)
#     print('MH_axis', MH_axis)
#     print('Age_axis', Age_axis)

    i_mh_array  = _nearest_index_uniform(MH_input_array,  mh_min,  dmh,  n_mh)
    j_age_array = _nearest_index_uniform(Age_input_array, age_min, dage, n_age)
    iso_idx_array = i_mh_array * n_age + j_age_array  # MH-outer, Age-inner 

    offset = DB["offset"].astype(np.int64)
    length = DB["length"].astype(np.int64)
    Mass_cat = DB["Mass"]  # concatenated per-iso grids (strictly increasing)

    # prepare outputs
    out = {f: np.full(MH_input_array.shape, np.nan, dtype=float) for f in fields}
#     inrange = np.zeros(MH_input_array.shape, dtype=bool)

    # do work per unique isochrone
    # 
    # arr = np.array([1, 20, 20, 300, 1, 400, 300, 5000])
    # unique_elements, inverse = np.unique(arr, return_inverse=True)
    #
    # Unique elements: [1 20 300 400 5000]
    # Inverse indices (to reconstruct original array): [0 1 1 2 0 3 2 4]
    #
    # but as out sample size is huge, the k==u in most cases
    uniq, inverse_array = np.unique(iso_idx_array, return_inverse=True)
    
    for k, u in enumerate(uniq):
        # loop through all isochrones used
        # for the kth isochrone used in sample, its iso_index is u
        select_array = (inverse_array == k) # index array of all simulated events using the kth isochrone
        
        off = int(offset[u]); L = int(length[u]) # readout the kth isochrone used, which has iso_index = u
#         if L <= 1:
#             # no interpolation possible
#             continue
        m_grid = Mass_cat[off:off+L] # the mass grid of the kth isochrone used; strickly increasing mass, no repeating value
        
        
        # interpolate each requested field on the same mass grid slice
        for f in fields:
            
            y_grid = DB[f][off:off+L] # the magnitude/parameter grid of the kth isochrone used
            
            # interpolate 'all simulated events using the kth isochrone'
#             y, ok = _interp_strict_increasing(m_grid, y_grid, Mass_input_array[select_array])
            y = _interp_strict_increasing(m_grid, y_grid, Mass_input_array[select_array])
            
            out[f][select_array] = y
            
            # accumulate in-range info: if any field interpolated OK, mark OK
#             inrange[select_array] |= ok

    meta = {"iso_idx": iso_idx_array}#, "inrange": inrange}
    
    if return_snapped_axes:
        meta["MH_grid"]  = MH_axis[i_mh_array]
        meta["Age_grid"] = Age_axis[j_age_array]
        
    return out, meta



def getpsi(phi,ecc):
    ''' compute the eccentric anomaly for given phase value phi (mean anomaly) and eccentricity '''
    psi = phi
    fun = psi-ecc*np.sin(psi)
    while np.abs(phi-fun)>1e-5:
        dpsi = 1.-ecc*np.cos(psi)
        psi += (phi-fun)/dpsi
        fun = psi-ecc*np.sin(psi)
    return psi

def xrotation(vec,angle):
    angle *= np.pi/180.
    vec_new = np.ones_like(vec)
    vec_new[0] = vec[0]
    vec_new[1] = vec[1]*np.cos(angle)-vec[2]*np.sin(angle)
    vec_new[2] = vec[1]*np.sin(angle)+vec[2]*np.cos(angle)
    return vec_new

def zrotation(vec,angle):
    angle *= np.pi/180.
    vec_new = np.ones_like(vec)
    vec_new[0] = vec[0]*np.cos(angle)-vec[1]*np.sin(angle)
    vec_new[1] = vec[0]*np.sin(angle)+vec[1]*np.cos(angle)
    vec_new[2] = vec[2]
    return vec_new

def getpos(hjd,tperi,prd,ecc,f0):
    phi = (hjd-tperi)/prd*2*np.pi
    psi = getpsi(phi,ecc)
    pos_earth = np.ones(3)
    sma = 1.0
    pos_earth[0] = sma*np.cos(psi)-ecc
    pos_earth[1] = sma*np.sqrt(1-ecc**2)*np.sin(psi)
    pos_earth[2] = 0.
    pos_earth = zrotation(pos_earth,180-f0)
    pos_earth = xrotation(pos_earth,23.43)
    return pos_earth

def getdir(alpha,delta):
    alpha *= np.pi/180.
    delta *= np.pi/180.
    ## unit vector to the target
    pos_tar = np.array([np.cos(alpha)*np.cos(delta),np.sin(alpha)*np.cos(delta),np.sin(delta)])
    north = np.array([0.,0.,1.])
    east = np.zeros_like(north)
    ## determine the unit vectors of (east,north) directions
    east[0] = north[1]*pos_tar[2]-north[2]*pos_tar[1]
    east[1] = north[2]*pos_tar[0]-north[0]*pos_tar[2]
    east[2] = north[0]*pos_tar[1]-north[1]*pos_tar[0]
    east /= np.sqrt(east[0]**2+east[1]**2+east[2]**2)
    north[0] = pos_tar[1]*east[2]-pos_tar[2]*east[1]
    north[1] = pos_tar[2]*east[0]-pos_tar[0]*east[2]
    north[2] = pos_tar[0]*east[1]-pos_tar[1]*east[0]
    return pos_tar,east,north

### currently use murel_geo and murel_hel, 
### later should be changed to murel_Roman and murel_hel. 
def getEarthVelocity(t0par,alpha,delta):
    tperi = 2644.55 #it's 75 days before 2000 vertal equinox
    prd = 365.25
    ecc = 0.0167
    ## first find the angular offset from t_peri to t_vernal
    phi0 = 75./prd*2*np.pi
    psi0 = getpsi(phi0,ecc)
    f0 = np.arccos((np.cos(psi0)-ecc)/(1-ecc*np.cos(psi0))) # true anomaly at vernal equinox
    f0 *= 180./np.pi

    pos_tar,east,north = getdir(alpha,delta)
    pos1 = getpos(t0par-1,tperi,prd,ecc,f0) #Earth position relative to Sun at t0par-1
    pos2 = getpos(t0par+1,tperi,prd,ecc,f0) #Earth position relative to Sun at t0par+1
    vne  = np.sum((pos2-pos1)*north)*0.5 #Earth velocity along north direction at t0par
    vee  = np.sum((pos2-pos1)*east)*0.5  #Earth velocity along east direction at t0par
    vre  = np.sum((pos2-pos1)*pos_tar)*0.5 #Earth velocity along line of sight at t0par
    
    return vne * prd, vee * prd, vre * prd # unit: AU/yr



# lookup : 2D array, shape (N_RV, 4)
#     Column 0: R_V grid (monotonic increasing)
#     Column 1: A_F062 / A_V
#     Column 2: A_F087 / A_V
#     Column 3: A_F213 / A_V

# LOOKUP_CCM89_MDWARF_F062_F087_F213 = np.array([
#     [2.0, 0.7815, 0.4083, 0.0927],
#     [2.1, 0.7879, 0.4224, 0.0965],
#     [2.2, 0.7937, 0.4353, 0.0999],
#     [2.3, 0.7989, 0.4470, 0.1030],
#     [2.4, 0.8037, 0.4577, 0.1058],
#     [2.5, 0.8082, 0.4676, 0.1084],
#     [2.6, 0.8122, 0.4767, 0.1109],
#     [2.7, 0.8160, 0.4852, 0.1131],
#     [2.8, 0.8195, 0.4930, 0.1152],
#     [2.9, 0.8227, 0.5003, 0.1171],
#     [3.0, 0.8258, 0.5071, 0.1189], 
#     [3.5, 0.8382, 0.5353, 0.1264], 
#     [4.0, 0.8475, 0.5565, 0.1320], 
#     [4.5, 0.8548, 0.5730, 0.1364],
#     [5.0, 0.8605, 0.5861, 0.1399], 
#     [5.5, 0.8652, 0.5969, 0.1428], 
#     [6.0, 0.8691, 0.6059, 0.1451]
# ])

LOOKUP_CCM89_MDWARF_F062_F087_F213 = np.array([
    [2.0, 0.7641, 0.4034, 0.0924],
    [2.1, 0.7709, 0.4176, 0.0961],
    [2.2, 0.7772, 0.4304, 0.0995],
    [2.3, 0.7828, 0.4421, 0.1026],
    [2.4, 0.7880, 0.4529, 0.1054],
    [2.5, 0.7928, 0.4627, 0.1081],
    [2.6, 0.7971, 0.4719, 0.1105],
    [2.7, 0.8012, 0.4803, 0.1127],
    [2.8, 0.8049, 0.4882, 0.1148],
    [2.9, 0.8084, 0.4955, 0.1167],
    [3.0, 0.8117, 0.5023, 0.1185],
    [3.5, 0.8252, 0.5305, 0.1260],
    [4.0, 0.8352, 0.5517, 0.1316],
    [4.5, 0.8430, 0.5681, 0.1359],
    [5.0, 0.8492, 0.5813, 0.1394],
    [5.5, 0.8543, 0.5921, 0.1423],
    [6.0, 0.8585, 0.6011, 0.1446]
])

def interp_A_F062_F087_F213(Av, Rv, lookup = LOOKUP_CCM89_MDWARF_F062_F087_F213):
    
    ### Interpolate A_F062, A_F087, A_F213 from CCM89 (A_V, R_V) using a 2D lookup table.

    ### if you change the filters used, remember to also recalculate the lookup table using the code provided alongside. 

    # lookup : 2D array, shape (N_RV, 4)
    #     Column 0: R_V grid (monotonic increasing)
    #     Column 1: A_F062 / A_V
    #     Column 2: A_F087 / A_V
    #     Column 3: A_F213 / A_V
    
    rv_grid   = lookup[:, 0]
    a062_grid = lookup[:, 1]
    a087_grid = lookup[:, 2]
    a213_grid = lookup[:, 3]

    a062_over_Av = np.interp(Rv, rv_grid, a062_grid)
    a087_over_Av = np.interp(Rv, rv_grid, a087_grid)
    a213_over_Av = np.interp(Rv, rv_grid, a213_grid)

    A_F062 = Av * a062_over_Av
    A_F087 = Av * a087_over_Av
    A_F213 = Av * a213_over_Av

    E_F062_F087 = A_F062 - A_F087
    E_F087_F213 = A_F087 - A_F213

    return E_F062_F087, E_F087_F213, A_F213





def CalculateWeights(args): #MCfile,config,SimulatedEvents=False,savename=False):
    
    # args_iter = [(ni, f"output/{config.name}/MockEvent_{config.name}_batch_{ni}.h5", config) for ni in ni_list]
    ni, readname, config = args

    # Per-process seed for reproducibility and independence
    np.random.seed( 7654321 + int(ni) )

    time_begin = time.time()
   
    # k_trans = 0.0005775483273639937   # km/s/kpc -> mas/day: 0.0005775483273639937
    Year2Day = 365.25
    # RsunOverkpc2mas = 0.004650467260962158 # R_sun/kpc -> mas


    # if SimulatedEvents==False:
    #     SimulatedEvents = np.loadtxt(MCfile,dtype='str')

    print('%dth run...'%ni, flush=True)

    print('  Reading data of %s ...'%(readname), flush=True)

    with h5py.File(readname, "r") as f:
        g = f["events"]  

        Ds          = np.asarray(g["Ds"],          dtype=np.float64)
        Dl          = np.asarray(g["Dl"],          dtype=np.float64)

        Age_lens    = np.asarray(g["Age_lens"],    dtype=np.float64)
        MH_lens     = np.asarray(g["MH_lens"],     dtype=np.float64)
        Ml          = np.asarray(g["Ml"],          dtype=np.float64)

        murel_hel_E = np.asarray(g["murel_hel_E"], dtype=np.float64)
        murel_hel_N = np.asarray(g["murel_hel_N"], dtype=np.float64)

        AV_lens     = np.asarray(g["AV_lens"],     dtype=np.float64)
        RV_lens     = np.asarray(g["RV_lens"],     dtype=np.float64)

    # Ds = SimulatedEvents[:,0].astype('float')
    # Dl = SimulatedEvents[:,1].astype('float')
    physical = (Dl < Ds)
    print('all(physical)', all(physical))
    # physical = (Dl <= Ds)
    # print('all(physical) allow equal', all(physical))
    Ds = Ds[physical]
    Dl = Dl[physical]

    Age_lens = Age_lens[physical]
    MH_lens  = MH_lens[physical]
    Ml       = Ml[physical]

    murel_hel_E = murel_hel_E[physical]
    murel_hel_N = murel_hel_N[physical]

    AV_lens = AV_lens[physical]
    RV_lens = RV_lens[physical]

    del physical

    # Ds = SimulatedEvents[physical,0].astype('float')
    # Dl = SimulatedEvents[physical,1].astype('float')
    # # Vsl = SimulatedEvents[physical,2].astype('float')
    # # Vsb = SimulatedEvents[physical,3].astype('float')
    # # Vll = SimulatedEvents[physical,4].astype('float')
    # # Vlb = SimulatedEvents[physical,5].astype('float')
    # Ml = SimulatedEvents[physical,2].astype('float')
    # # Lens_type  = SimulatedEvents[physical,7]
    # # Source_loc = SimulatedEvents[physical,8]
    # # Lens_loc   = SimulatedEvents[physical,9]
    # # murel_hel_E = SimulatedEvents[physical,3].astype('float')
    # # murel_hel_N = SimulatedEvents[physical,4].astype('float')


    # Earth_Vl = config.Earth_Vl
    # Earth_Vb = config.Earth_Vb
    # Solar_Vl = config.Solar_Vl
    # Solar_Vb = config.Solar_Vb
    # v_earth_E = config.v_earth_E
    # v_earth_N = config.v_earth_N

    ### currently use murel_geo and murel_hel, 
    ### later should be changed to murel_Roman and murel_hel. 
    ### in AU/yr
    v_earth_N, v_earth_E, v_earth_r = getEarthVelocity(config.t0_for_velocity, config.ra, config.dec)
    print('v_earth_N, v_earth_E', v_earth_N, v_earth_E)

    # vsl,vsb = np.copy(Vsl), np.copy(Vsb)
    # vll,vlb = np.copy(Vll), np.copy(Vlb)
    # if config.SolarMotion == False: # the MCevent output do not consider SolarMotion
    #     vsl -= Solar_Vl
    #     vll -= Solar_Vl
    #     vsb -= Solar_Vb
    #     vlb -= Solar_Vb
    # #musl = k_trans* (vsl / Ds)
    # #musb = k_trans* (vsb / Ds)
    # murell_2sun = k_trans* (vll / Dl - vsl / Ds )*Year2Day #unit: mas/yr
    # murelb_2sun = k_trans* (vlb / Dl - vsb / Ds )*Year2Day
    # ######################### newly added #########################
    # murel_2sun  = np.sqrt(murell_2sun**2 + murelb_2sun**2) #unit: mas/yr
    # ######################### newly added #########################

    if config.binary_lens == True : 
        
        q = np.random.normal(config.q_mean, config.q_err, size=len(Ml))
        bad = (q <= 0) | (q > 1)
        while np.any(bad):
            q[bad] = np.random.normal(config.q_mean, config.q_err, size=bad.sum())
            bad = (q <= 0) | (q > 1)

        Mprimary   = Ml
        Msecondary = Mprimary * q   # in solar mass
        ### below Ml means binary total mass
        Ml = Mprimary + Msecondary
    
        del q, bad

    # if config.EarthMotion == False: # the MCevent output do not consider EarthMotion
    #     vsl -= Earth_Vl
    #     vll -= Earth_Vl
    #     vsb -= Earth_Vb
    #     vlb -= Earth_Vb

    print('  Calculating lensing observables for process %s'%ni, flush=True)

    pirel = 1./Dl-1./Ds       # au*(1/Dl-1/Ds), unit: mas
    thetaE = np.sqrt(8.144*Ml*pirel) # unit:mas


    # murel_hel = np.sqrt(murel_hel_E**2 + murel_hel_N**2) # unit: mas/yr

    # murel_hel = murel_geo + v_earth(AU/yr) * pirel(mas)/au
    ### v_earth_E, v_earth_N: AU/yr
    murel_geo_E = murel_hel_E - v_earth_E * pirel # unit: mas/yr
    murel_geo_N = murel_hel_N - v_earth_N * pirel # unit: mas/yr

    murel_geo = np.sqrt(murel_geo_E**2 + murel_geo_N**2) # unit: mas/yr
    # murell = k_trans* (vll / Dl - vsl / Ds )  #unit: mas/day
    # murelb = k_trans* (vlb / Dl - vsb / Ds )  #unit: mas/day
    # murel = np.sqrt( murell**2+murelb**2 )

    piE_E = pirel * murel_geo_E / (thetaE * murel_geo) # unit: unity
    piE_N = pirel * murel_geo_N / (thetaE * murel_geo) # unit: unity
    # pil = pirel*murell/(thetaE*murel)
    # pib = pirel*murelb/(thetaE*murel)
    # piEE,piEN = gm.YZ2EN(pil,pib)


    tE = thetaE / murel_geo * Year2Day  # unit: day   
    # Dl_profile = config.Dl_profile
    # Ds_profile = config.Ds_profile
    # Gamma = config.gamma
    # EventRate = murel*thetaE *Dl**(2-Dl_profile)*Ds**(-Ds_profile)
    # EventRate = murel * thetaE 
    
    ###########################
    ### Define weights here ###
    good0 = np.ones_like(Ds,dtype='bool')
    if config.use_tE==True:
        tEmean, tEerr = config.tEmean, config.tEerr
        #print('Use tE weight: tE = %.3f+-%.3f'%(tEmean,tEerr))
        good0 = ( good0 & ( tE > (tEmean-10*tEerr) ) & ( tE < (tEmean+10*tEerr) ) )
#    if config.use_piE==True:
#        piEEmean,piEEerr = config.piEEmean,config.piEEerr
#        piENmean,piENerr = config.piENmean,config.piENerr
#        piEcor = config.piEcor
#        #print('Use piE weight: piEN = %.4f+-%.4f'%(piENmean,piENerr))
#        #print('                piEE = %.4f+-%.4f'%(piEEmean,piEEerr))
#        #print('                piEcor = %.4f'%(piEcor))
#        good0 = ( good0 & (piEE>piEEmean-10*piEEerr) & (piEE<piEEmean+10*piEEerr) )
#    if config.use_thetaE==True:
#        rhomean,rhoerr = config.rhomean,config.rhoerr
#        thetasmean,thetaserr = config.thetasmean*1e-3,config.thetaserr*1e-3 # µas -> mas
#        thetaEmean = thetasmean/rhomean
#        thetaEerr = np.sqrt( (thetasmean**2)*(rhoerr**2)+((thetaserr**2)*(rhomean**2)) )/(rhomean**2)
#        #print('Use thetaE weight: thetaE = %.3f+-%.3f'%(thetaEmean,thetaEerr))
#        #Rstar,Rstarerr = config.Rstarmean,config.Rstarerr
#        #thetaEmean = Rstar/(rhomean*Ds) * RsunOverkpc2mas
#        #thetaEerr = np.sqrt((Rstar**2)*(rhoerr**2)+((Rstarerr**2)*(rhomean**2)) )/(rhomean**2) /Ds*RsunOverkpc2mas
#        good0 = ( good0 & (thetaE>thetaEmean-10*thetaEerr) & (thetaE<thetaEmean+10*thetaEerr) )

    #good0 = (tE>tEmean-10*tEerr) & (tE<tEmean+10*tEerr) \
    #        & (piEE>piEEmean-10*piEEerr) & (piEE<piEEmean+10*piEEerr) \
    #        & (thetaE>thetaEmean-10*thetaEerr) & (thetaE<thetaEmean+10*thetaEerr)
    Ds, Dl                              = Ds[good0], Dl[good0]
    Age_lens, MH_lens, Ml               = Age_lens[good0], MH_lens[good0], Ml[good0]
    murel_hel_E, murel_hel_N            = murel_hel_E[good0], murel_hel_N[good0]
    AV_lens, RV_lens                    = AV_lens[good0], RV_lens[good0]

    if config.binary_lens == True : 
        Mprimary, Msecondary = Mprimary[good0], Msecondary[good0]

    pirel, thetaE                       = pirel[good0], thetaE[good0]
    murel_geo_E, murel_geo_N, murel_geo = murel_geo_E[good0], murel_geo_N[good0], murel_geo[good0]
    piE_E, piE_N                        = piE_E[good0], piE_N[good0]
    tE                                  = tE[good0]

    del good0

    if config.binary_lens == True : 

        rE = thetaE * Dl # mas*kpc=au

        s = np.random.normal(config.s_mean, config.s_err, size=len(rE))
        bad = (s <= 0) 
        while np.any(bad):
            s[bad] = np.random.normal(config.s_mean, config.s_err, size=bad.sum())
            bad = (s <= 0) 

        a_prep = rE * s # au
        
        del s, bad, rE



    
    print('  Interpolating for lens absolute magnitudes at ([M/H], Age, Mass) triplets for process %s'%ni, flush=True)

    #################################################################################################################
    ###### Interpolate one or more filters' magnitudes at (MH, Age[Gyr], Mass) triplets from PARSEC npz file ########
    #################################################################################################################

    


    def combine_mags(m1, m2):
        # """
        # Combine two magnitudes (same band) by summing fluxes:
        # m = -2.5 log10(10^{-0.4 m1} + 10^{-0.4 m2})
        # If m2 is nan => treated as zero flux.
        # """
        f1 = 10.0**(-0.4 * m1)
        f2 = np.where(np.isfinite(m2), 10.0**(-0.4 * m2), 0.0)
        return -2.5 * np.log10(f1 + f2)

    if config.binary_lens == True : 
        mass_min_arr, mass_max_arr = mass_bounds_for_samples(PARSEC_DB, Age_lens, MH_lens)
        
        out1, meta = interpolate_from_triplets(PARSEC_DB, MH_lens, Age_lens, Mprimary, fields=fields)

        index_stellar_Msecondary = (Msecondary > mass_min_arr)

        # Prepare container for secondary outputs (np.nan means "no flux")
        out2 = {}
        for f in fields:
            out2[f] = np.full_like(out1[f], np.nan, dtype=float)

        if index_stellar_Msecondary.sum() > 0 : 

            out2_sub, _ = interpolate_from_triplets(PARSEC_DB, MH_lens[index_stellar_Msecondary], Age_lens[index_stellar_Msecondary], Msecondary[index_stellar_Msecondary], fields=fields)
            
            for f in fields:
                out2[f][index_stellar_Msecondary] = out2_sub[f]
            
            del out2_sub 

        # For fields in magnitudes, combine in flux space
        mag_fields = [f for f in fields if ("mag" in f.lower())]

        out = out1  # start with primary
        for f in mag_fields:
            out[f] = combine_mags(out1[f], out2[f])
        
        del mass_min_arr, mass_max_arr, out1, index_stellar_Msecondary, out2

    else :
        out, meta = interpolate_from_triplets(PARSEC_DB, MH_lens, Age_lens, Ml, fields=fields)


    ### absolute magnitude and intrinsic color
    M_F062_lens, M_F087_lens, M_F213_lens = out[ fields[0] ], out[ fields[1] ], out[ fields[2] ]

    Intrinsic_Color_F062_F087_lens = M_F062_lens - M_F087_lens
    Intrinsic_Color_F087_F213_lens = M_F087_lens - M_F213_lens

    MH_lens_snapped = meta["MH_grid"]
    Age_lens_snapped = meta["Age_grid"]

    del out, meta
    del M_F062_lens, M_F087_lens
    ### large arrays: Intrinsic_Color_F062_F087_lens, Intrinsic_Color_F087_F213_lens, M_F213_lens, MH_lens_snapped, Age_lens_snapped




    print('  Calculating lens Alambda(AV, RV) for process %s'%ni, flush=True)

    E_F062_F087, E_F087_F213, A_F213 = interp_A_F062_F087_F213(AV_lens, RV_lens, lookup = LOOKUP_CCM89_MDWARF_F062_F087_F213)
    


    print('  Calculating lens apparent magnitude and colors for process %s'%ni, flush=True)

    Color_F062_F087_lens = Intrinsic_Color_F062_F087_lens + E_F062_F087
    Color_F087_F213_lens = Intrinsic_Color_F087_F213_lens + E_F087_F213
    ## distance modulus, Dl in kpc
    mu_lens = 5.0 * (np.log10(Dl) + 2.0)
    F213_lens = M_F213_lens + mu_lens + A_F213

    del mu_lens
    ### large arrays: Intrinsic_Color_F062_F087_lens, Intrinsic_Color_F087_F213_lens, M_F213_lens, 
    ###               MH_lens_snapped, Age_lens_snapped,
    ###               E_F062_F087, E_F087_F213, A_F213, 
    ###               Color_F062_F087_lens, Color_F087_F213_lens, F213_lens


    


    if config.use_tE == True :
        ### tEmean, tEerr have been read from config on above
        tEweight = Gaussian1D(tE, tEmean, tEerr) 
        print('  Weighting with tE = %.3f+-%.3f for process %s'%(tEmean, tEerr, ni))

        ### max weight = 1/np.sqrt(2*np.pi)/tEerr
        max_weight = 1./np.sqrt(2*np.pi)/tEerr

    else:
        tEweight = 1.
        max_weight = 1. 
        
#    if config.use_piE==True:
#        #piEweight = Gaussian2D(piEE[good0],piEN[good0],piEEmean,piENmean,piEEerr,piENerr,piEcor#)
#        piE_weight_data = np.loadtxt('data/kb193289/weight_piE.dat#')
#        pie_model = piE_weight_data[:,:2]
#        pie_weight_model = piE_weight_data[:,2]
#        #pieangle = np.arctan(piEN[good0]/piEE[good0])*180/np.pi
#        pieangle_weight = 1.0 #Gaussian1D(pieangle,-80.0,0.6)
#        piEweight = griddata(pie_model,pie_weight_model,xi=(np.vstack((piEE[good0],piEN[good0])).T),method='cubic',fill_value=0)
#        piEweight = piEweight * pieangle_weight
#    else:
#        piEweight = 1.

    if config.use_piE == True :

        piE_E_mean, piE_E_err = config.piE_E_mean, config.piE_E_err
        piE_N_mean, piE_N_err = config.piE_N_mean, config.piE_N_err
        piE_rhoEN = config.piE_rhoEN

        piEweight = Gaussian2D(piE_E, piE_N, piE_E_mean,piE_N_mean,piE_E_err,piE_N_err,piE_rhoEN)
        print('  Weighting with piE_E = %.4f+-%.4f, piE_N = %.4f+-%.4f, piE_rhoEN = %.4f for process %s'%(piE_E_mean, piE_E_err, piE_N_mean, piE_N_err, piE_rhoEN, ni))

        ### max weight = 1/(2*np.pi*sigma1*sigma2*np.sqrt(1-rho12**2))
        max_weight *= 1./(2*np.pi * piE_E_err * piE_N_err * np.sqrt(1-piE_rhoEN**2) )

    else:
        piEweight = 1.
        max_weight *= 1.


    if config.use_Color_F062_F087_lens == True :

        Color_F062_F087_lens_mean, Color_F062_F087_lens_err = config.Color_F062_F087_lens_mean, config.Color_F062_F087_lens_err
        Color_F062_F087_isochrone_uncertainty = config.Color_F062_F087_isochrone_uncertainty

        Color_F062_F087_lens_weight = Gaussian1D(Color_F062_F087_lens, Color_F062_F087_lens_mean, np.sqrt(Color_F062_F087_lens_err**2 + Color_F062_F087_isochrone_uncertainty**2) ) 

        print('  Weighting with Color_F062_F087_lens = %.3f+-%.3f, considering isochrone uncertainty = %.3f mag for process %s'%(Color_F062_F087_lens_mean, Color_F062_F087_lens_err, Color_F062_F087_isochrone_uncertainty, ni))

        ### max weight = 1/np.sqrt(2*np.pi)/sigma
        max_weight *= 1./np.sqrt(2*np.pi)/np.sqrt(Color_F062_F087_lens_err**2 + Color_F062_F087_isochrone_uncertainty**2)

    else:
        Color_F062_F087_lens_weight = 1.
        max_weight *= 1.

    

    if config.use_Color_F087_F213_lens == True :

        Color_F087_F213_lens_mean, Color_F087_F213_lens_err = config.Color_F087_F213_lens_mean, config.Color_F087_F213_lens_err
        Color_F087_F213_isochrone_uncertainty = config.Color_F087_F213_isochrone_uncertainty

        Color_F087_F213_lens_weight = Gaussian1D(Color_F087_F213_lens, Color_F087_F213_lens_mean, np.sqrt(Color_F087_F213_lens_err**2 + Color_F087_F213_isochrone_uncertainty**2) ) 
        print('  Weighting with Color_F087_F213_lens = %.3f+-%.3f, considering isochrone uncertainty = %.3f mag for process %s'%(Color_F087_F213_lens_mean, Color_F087_F213_lens_err, Color_F087_F213_isochrone_uncertainty, ni))

        ### max weight = 1/np.sqrt(2*np.pi)/sigma
        max_weight *= 1./np.sqrt(2*np.pi)/np.sqrt(Color_F087_F213_lens_err**2 + Color_F087_F213_isochrone_uncertainty**2)

    else:
        Color_F087_F213_lens_weight = 1.
        max_weight *= 1.
    


    if config.use_F213_lens == True :

        F213_lens_mean, F213_lens_err = config.F213_lens_mean, config.F213_lens_err
        
        F213_lens_weight = Gaussian1D(F213_lens, F213_lens_mean, F213_lens_err) 
        print('  Weighting with F213_lens = %.3f+-%.3f for process %s'%(F213_lens_mean, F213_lens_err, ni))
        
        ### max weight = 1/np.sqrt(2*np.pi)/sigma
        max_weight *= 1./np.sqrt(2*np.pi)/F213_lens_err

    else:
        F213_lens_weight = 1.
        max_weight *= 1.

    # if config.use_thetaE==True:
    #     if config.thetaEweight_file==True:
    #         #logthetaE_weight_data = np.loadtxt(config.thetaEweight_filepath)
    #         #logthetaE_model, logthetaE_weight_model = logthetaE_weight_data[:,0],logthetaE_weight_data[:,1]
    #         #thetaEweight = griddata(logthetaE_model,logthetaE_weight_model,xi=thetaE[good0],method='cubic',fill_value=0)
    #         ##thetaEweight = griddata(logthetaE_model,logthetaE_weight_model,xi=np.log10(thetaE[good0]),method='cubic',fill_value=0) /np.log(10)/thetaE[good0]


    #         # # planet Close flat
    #         # def logrho_chi2_surface(x):  
    #         #     return  -4.68429741e+00*np.exp(3.31497129e+02*x) +  (1.91741972e+02)*x + 3.04689354e-01 + (4.45459245e+00)*np.exp(3.34848507e+02*x)
    #         # #.  return  c*np.exp(d*x) +  e*x + f + g*np.exp(h*x) #三次加指数函数拟合  #a * np.exp(-b * x) + c
    #         # #     return 3.92592775e-122*np.exp(5.34602926e+004*x) +  3.92587016e+002*x + (1.02443839e-002) + (-6.42767071e+004)*x**2 + 3.22679270e+007*x**3
    #         # #[-4.68429741e+00  3.31497129e+02  1.91741972e+02  3.04689354e-01 4.45459245e+00  3.34848507e+02]

    #         # # planet wide flat
    #         # def logrho_chi2_surface(x):  
    #         #     return  1.10877615e-01*np.exp(4.61928395e+02*x) +  (4.14237861e+01)*x + 1.72637560e-01 + (-1.28524447e-01)*np.exp(4.50924933e+02*x)
    #         # #.  return  c*np.exp(d*x) +  e*x + f + g*np.exp(h*x) #三次加指数函数拟合  #a * np.exp(-b * x) + c
    #         # #     return 3.92592775e-122*np.exp(5.34602926e+004*x) +  3.92587016e+002*x + (1.02443839e-002) + (-6.42767071e+004)*x**2 + 3.22679270e+007*x**3
    #         # #[ 1.10877615e-01  4.61928395e+02  4.14237861e+01  1.72637560e-01 -1.28524447e-01  4.50924933e+02]

    #         # # Binary Close
    #         # def logrho_chi2_surface(x):  
    #         #     return  1.57840146e+00*np.exp(3.83291247e+02*x) +  (1.52607813e+02)*x + 3.80835618e-01 + (-1.73950140e+00)*np.exp(3.74849894e+02*x)
    #         # #.  return  c*np.exp(d*x) +  e*x + f + g*np.exp(h*x) #三次加指数函数拟合  #a * np.exp(-b * x) + c
    #         # #     return 3.92592775e-122*np.exp(5.34602926e+004*x) +  3.92587016e+002*x + (1.02443839e-002) + (-6.42767071e+004)*x**2 + 3.22679270e+007*x**3
    #         # #[ 1.57840146e+00  3.83291247e+02  1.52607813e+02  3.80835618e-01 -1.73950140e+00  3.74849894e+02]

    #         # Binary Wide
    #         # def logrho_chi2_surface(x):  
    #         # #    return  4.38133353e-002*np.exp(4.87881999e+003*x) +  4.12175871e+002*x + -4.99595269e-002 + 3.49173542e-116*np.exp(2.15364953e+005*x)
    #         #     return -3.36239708e-01*np.exp(1.98910572e+01*x) +  1.49740672e+02*x + (6.67334070e-01) + (-1.77776810e+05)*x**2 + 4.59085746e+07*x**3
    #         # #[-3.36239708e-01  1.98910572e+01  1.49740672e+02  6.67334070e-01 -1.77776810e+05  4.59085746e+07]

    #         # planet_close_5e7_as_total_mass
    #         def logrho_chi2_surface(x):  
    #             x = x*0.0165985 / 2.51e-3
    #             return  1.10877615e-01*np.exp(4.61928395e+02*x) +  (4.14237861e+01)*x + 1.72637560e-01 + (-1.28524447e-01)*np.exp(4.50924933e+02*x)



    #         def pdf_thetaE_analytic(theta_Einstein):#unit:mas
    #             ### planet Close
    #             # convert_factor = 1.0029398967802388
    #             ### planet Wide
    #             # convert_factor = 1.0030082730769192
    #             ### Binary Close
    #             # convert_factor = 1.2659224249828098
    #             ### Binary Wide
    #             # convert_factor = 2.171486194055059
    #             ### Binary Wide secondary_lens_as_single
    #             # convert_factor = 1.1265671220094662

    #             ###############################################
    #             ############ theta*=0.68e-3 for planet, 0.74e-3 for binary ################
    #             ###############################################
    #             # return  np.exp(-0.5*logrho_chi2_surface(  0.74e-3/(theta_Einstein*convert_factor)  ))

    #             theta_star = 0.71e-3   # mas
    #             return  np.exp(-0.5*logrho_chi2_surface(  theta_star/(theta_Einstein)  ))
            


    #         thetaEweight = pdf_thetaE_analytic( thetaE[good0] )
    #         print('Use thetaE weight: surface funtion')

    #     else:
    #         if config.thetaE_upperlimit==True:
    #             thetaEweight = Gaussian1D(thetaE[good0],thetaEmean,thetaEerr,flat='above',truncation=[0,np.inf])
    #         else:
    #             #thetaEweight = Gaussian1D(thetaE[good0],thetaEmean,thetaEerr)
    #             thetaEweight = RatioOf2GaussianDistribution(thetaE[good0],thetasmean,rhomean,thetaserr,rhoerr)

    #     # #tEweight = Gaussian1D(tE[good0],tEmean,tEerr)
    #     # thetaEmean,thetaEerr = config.thetaEmean,config.thetaEerr
    #     # thetaEweight = Gaussian1D(thetaE[good0],thetaEmean,thetaEerr)
    #     # print('Use thetaE weight: tE = %.3f+-%.3f'%(thetaEmean,thetaEerr))
        
    # else:
    #     thetaEweight = 1.
    # #musweight = Gaussian1D(musl[good0]*Year2Day,-5.81,3.14) * Gaussian1D(musb[good0]*Year2Day,-0.20,2.69)
    # #print(musweight)



    # if config.use_murel_hel == True :
    #     if config.use_murel_hel_direction == True : 

    #         # murelE_2sun_good0, murelN_2sun_good0 = pm_lb_to_EN_from_radec(config.ra, config.dec, murell_2sun[good0], murelb_2sun[good0])    
    #         murelE_2sun_good0, murelN_2sun_good0 = gm.YZ2EN(murell_2sun[good0], murelb_2sun[good0])
    #         murel_hel_E_weight = Gaussian1D(murelE_2sun_good0, murel_hel_E_mean,murel_hel_E_err)
    #         murel_hel_N_weight = Gaussian1D(murelN_2sun_good0, murel_hel_N_mean,murel_hel_N_err)
    #         murel_hel_weight = murel_hel_E_weight * murel_hel_N_weight

    #         print('Use murel_hel weight with direction')

    #     else: 
    #         murel_hel_mean = np.sqrt(murel_hel_E_mean**2 + murel_hel_N_mean**2)
    #         murel_hel_err  = np.sqrt( (murel_hel_E_mean/murel_hel_mean*murel_hel_E_err)**2 + (murel_hel_N_mean/murel_hel_mean*murel_hel_N_err)**2)

    #         murel_hel_weight = Gaussian1D(murel_2sun[good0], murel_hel_mean,murel_hel_err)
    #         print('Use murel_hel weight: murel_hel = %.3f+-%.3f mas/yr'%(murel_hel_mean,murel_hel_err))
    # else:
    #     murel_hel_weight = 1.



    total_weight = tEweight * piEweight * Color_F062_F087_lens_weight * Color_F087_F213_lens_weight * F213_lens_weight 

    total_weight[np.isnan(total_weight)] = 0.0

    ### cut off < 1e-12 is very conservative ###
    ### consider a total of 1e10 samples, 
    ### even nearly all cut off, at most contribute 1e10 * 1e-12 = 1e-2 weights;
    ### while, there must be samples with weight ~ 1, thus 1e-2 can be neglected
    good1 = ( total_weight > (max_weight * 1e-12) )





    savename = f'output/{config.name}/{config.model}/weight_{config.name}_{config.model}_batch_{ni}.h5'
    print('  Saving data to %s ...'%(savename), flush=True)

    with h5py.File(savename, "w") as f:
        g = f.create_group("weights")

        # Use "compression=None" to disable compression. "lzf" = fast, low-CPU.
        comp = "lzf"   # or None, or "gzip"

        g.create_dataset("Ds",           data=np.asarray(Ds[good1],           np.float32), compression=comp)
        g.create_dataset("Dl",           data=np.asarray(Dl[good1],           np.float32), compression=comp)
        
        g.create_dataset("Age_lens",     data=np.asarray(Age_lens[good1],     np.float32), compression=comp)
        g.create_dataset("MH_lens",      data=np.asarray(MH_lens[good1],      np.float32), compression=comp)
        g.create_dataset("Age_lens_snapped",     data=np.asarray(Age_lens_snapped[good1],     np.float32), compression=comp)
        g.create_dataset("MH_lens_snapped",      data=np.asarray(MH_lens_snapped[good1],      np.float32), compression=comp)

        if config.binary_lens == True : 
            ## in unit of solar mass, au
            g.create_dataset("Ml_1",           data=np.asarray(Mprimary[good1],           np.float32), compression=comp) 
            g.create_dataset("Ml_2",           data=np.asarray(Msecondary[good1],           np.float32), compression=comp) 
            g.create_dataset("a_prep",           data=np.asarray(a_prep[good1],           np.float32), compression=comp) 
        else : 
            g.create_dataset("Ml",           data=np.asarray(Ml[good1],           np.float32), compression=comp) 
        
        g.create_dataset("AV_lens",      data=np.asarray(AV_lens[good1],      np.float32), compression=comp)
        g.create_dataset("RV_lens",      data=np.asarray(RV_lens[good1],      np.float32), compression=comp)

        g.create_dataset("pirel",  data=np.asarray(pirel[good1],  np.float32), compression=comp)
        g.create_dataset("thetaE",  data=np.asarray(thetaE[good1],  np.float32), compression=comp)

        g.create_dataset("murel_geo_E",  data=np.asarray(murel_geo_E[good1],  np.float32), compression=comp)
        g.create_dataset("murel_geo_N",  data=np.asarray(murel_geo_N[good1],  np.float32), compression=comp)
        g.create_dataset("murel_geo",  data=np.asarray(murel_geo[good1],  np.float32), compression=comp)

        g.create_dataset("Intrinsic_Color_F062_F087_lens", data=np.asarray(Intrinsic_Color_F062_F087_lens[good1], np.float32), compression=comp)
        g.create_dataset("Intrinsic_Color_F087_F213_lens", data=np.asarray(Intrinsic_Color_F087_F213_lens[good1], np.float32), compression=comp)
        g.create_dataset("M_F213_lens",                    data=np.asarray(M_F213_lens[good1],                    np.float32), compression=comp)

        g.create_dataset("E_F062_F087",                    data=np.asarray(E_F062_F087[good1],                    np.float32), compression=comp)
        g.create_dataset("E_F087_F213",                    data=np.asarray(E_F087_F213[good1],                    np.float32), compression=comp)
        g.create_dataset("A_F213",                         data=np.asarray(A_F213[good1],                         np.float32), compression=comp)

        ### below are observables (migt be) used to weight, thus (might be) no new information
        g.create_dataset("murel_hel_E",  data=np.asarray(murel_hel_E[good1],  np.float32), compression=comp)
        g.create_dataset("murel_hel_N",  data=np.asarray(murel_hel_N[good1],  np.float32), compression=comp)
        
        g.create_dataset("piE_E",        data=np.asarray(piE_E[good1],        np.float32), compression=comp)
        g.create_dataset("piE_N",        data=np.asarray(piE_N[good1],        np.float32), compression=comp)

        g.create_dataset("tE",           data=np.asarray(tE[good1],           np.float32), compression=comp)

        g.create_dataset("Color_F062_F087_lens", data=np.asarray(Color_F062_F087_lens[good1], np.float32), compression=comp)
        g.create_dataset("Color_F087_F213_lens", data=np.asarray(Color_F087_F213_lens[good1], np.float32), compression=comp)
        g.create_dataset("F213_lens",            data=np.asarray(F213_lens[good1],            np.float32), compression=comp)

        ### below are all weights
        g.create_dataset("total_weight",  data=np.asarray(total_weight[good1],  np.float64), compression=comp)

        if config.use_tE == True :
            g.create_dataset("tEweight",  data=np.asarray(tEweight[good1],  np.float64), compression=comp)
        
        if config.use_piE == True :
            g.create_dataset("piEweight", data=np.asarray(piEweight[good1], np.float64), compression=comp)
        
        if config.use_Color_F062_F087_lens == True :
            g.create_dataset("Color_F062_F087_lens_weight", data=np.asarray(Color_F062_F087_lens_weight[good1], np.float64), compression=comp)
        
        if config.use_Color_F087_F213_lens == True :
            g.create_dataset("Color_F087_F213_lens_weight", data=np.asarray(Color_F087_F213_lens_weight[good1], np.float64), compression=comp)
        
        if config.use_F213_lens == True :
            g.create_dataset("F213_lens_weight",            data=np.asarray(F213_lens_weight[good1],            np.float64), compression=comp)

    del Ds, Dl, Age_lens, MH_lens, Age_lens_snapped, MH_lens_snapped, Ml
    if config.binary_lens == True : 
        del Mprimary, Msecondary, a_prep
    del AV_lens, RV_lens, pirel, thetaE, murel_geo_E, murel_geo_N, murel_geo
    del Intrinsic_Color_F062_F087_lens, Intrinsic_Color_F087_F213_lens, M_F213_lens
    del E_F062_F087, E_F087_F213, A_F213

    del murel_hel_E, murel_hel_N, piE_E, piE_N, tE, Color_F062_F087_lens, Color_F087_F213_lens, F213_lens

    del total_weight, tEweight, piEweight, Color_F062_F087_lens_weight, Color_F087_F213_lens_weight, F213_lens_weight
    del good1


    
    print(f'process {ni} done in {time.time()-time_begin:.2f}s', flush=True)

    return savename



if __name__ == '__main__':
    time_begin = time.time()

    ########## change here #########
    ## eventname should be consistent with the name of .cfg file want to use
    eventname = 'mock1'
    ########## change here #########
    
    config = getEventInfo(eventname)

    modelname = config.model
    mkdir('output/%s/%s'%(eventname,modelname))
    os.system('rm output/%(eventname)s/%(modelname)s/weight*h5'%vars())
    


    nstart,nrun = config.nstart,config.nrun

    # -------- parallel map over ni --------
    ni_list = list(range(nstart, nstart+nrun))
    args_iter = [(ni, f"output/{eventname}/MockEvent_{eventname}_batch_{ni}.h5", config) for ni in ni_list]

    # Use spawn for safety (works on Linux/Mac/Windows)
    with get_context("spawn").Pool(processes=config.n_process,
        initializer=_init_parsec_db,
        initargs=(NPZ_PATH, fields)) as pool:
        # imap_unordered gives results as they finish
        for _ in pool.imap_unordered(CalculateWeights, args_iter, chunksize=1):
            pass


    


    ### Combine small weight HDF5s into one big HDF5
    # combine code will not start before all subprocesses finish as long as it is placed after the with ... Pool(...) block
    import glob, re

    combine_name = f'output/{eventname}/{modelname}/combine_weight_{eventname}_{modelname}.h5'
    


    pattern = f"output/{eventname}/{modelname}/weight_{eventname}_{modelname}_batch_*.h5"
    weight_files = glob.glob(pattern)

    def _batch_id(path):
        m = re.search(r"_batch_(\d+)\.h5$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    weight_files.sort(key=_batch_id)



    # ---- datasets that are ALWAYS present in small files ----
    keys = [
        "Ds", "Dl",
        "Age_lens", "MH_lens", "Age_lens_snapped", "MH_lens_snapped",

        # mass fields depend on binary_lens; appended below

        "AV_lens", "RV_lens",
        "pirel", "thetaE",
        "murel_geo_E", "murel_geo_N", "murel_geo",

        "Intrinsic_Color_F062_F087_lens", "Intrinsic_Color_F087_F213_lens", "M_F213_lens",
        "E_F062_F087", "E_F087_F213", "A_F213",

        "murel_hel_E", "murel_hel_N",
        "piE_E", "piE_N",
        "tE",
        "Color_F062_F087_lens", "Color_F087_F213_lens", "F213_lens",

        "total_weight",
    ]

    # ---- mass fields ----
    if config.binary_lens:
        keys += ["Ml_1", "Ml_2", "a_prep"]
    else:
        keys += ["Ml"]

    # ---- optional component-weight datasets ----
    if config.use_tE:
        keys += ["tEweight"]
    if config.use_piE:
        keys += ["piEweight"]
    if config.use_Color_F062_F087_lens:
        keys += ["Color_F062_F087_lens_weight"]
    if config.use_Color_F087_F213_lens:
        keys += ["Color_F087_F213_lens_weight"]
    if config.use_F213_lens:
        keys += ["F213_lens_weight"]

    # store weights as float64; everything else float32 
    float64_keys = set([
        "total_weight",
        "tEweight", "piEweight",
        "Color_F062_F087_lens_weight", "Color_F087_F213_lens_weight", "F213_lens_weight",
    ])



    # remove existing combined file if rerunning
    if os.path.exists(combine_name):
        os.remove(combine_name)

    print(f'Combining {len(weight_files)} weight files to {combine_name} ...', flush=True)

    comp = "lzf"  # or None / "gzip"

    with h5py.File(combine_name, "w") as fout:
        fout.attrs["eventname"] = eventname
        fout.attrs["modelname"] = modelname

        g_out = fout.create_group("weights")

        # create extendable datasets
        d_out = {}
        for k in keys:
            dt = np.float64 if k in float64_keys else np.float32
            d_out[k] = g_out.create_dataset(
                k, shape=(0,), maxshape=(None,),
                dtype=dt, chunks=True, compression=comp
            )

        offset = 0
        for wf in weight_files:
            with h5py.File(wf, "r") as fin:
                g_in = fin["weights"]

                # number of rows in this batch
                n = g_in["Ds"].shape[0]
                if n == 0:
                    continue

                new = offset + n

                # append each dataset
                for k in keys:
                    # if k not in g_in:
                    #     raise KeyError(f"{wf} missing dataset '{k}' (check binary_lens/use_* consistency)")
                    ds = d_out[k]
                    ds.resize((new,))
                    ds[offset:new] = g_in[k][:]

                offset = new



    

    ### after combining, remove all small files
    # os.system('rm output/%(eventname)s/%(modelname)s/weight*h5'%vars())

    time_end = time.time()
    print('\nTotally cost: %.2f min'%((time_end-time_begin)/60) )
        
    
    
    
    
    
    
    
    
    
    
    
    
    

    



    

    



    
