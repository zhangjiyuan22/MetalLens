from cal_weights import *

### mock event information ###
# single lens, M=0.5 solar mass, Ds=9kpc(Roman field mean), Dl=7.5kpc(Roman field mean, see Roman Paper 4)
# [M/H] = 0.2 , Age = 6 Gyr
# 
# from PARSEC, using actual mass grid = 0.501 solar mass, 
# M_F062 = 9.529 mag, M_F087 = 7.700 mag, M_F213 = 5.845 mag
# 
# mulens = 5 * (np.log10(7.5) + 2) = 14.3753 mag,
# thus F062 = 23.9043 mag, F087 = 22.0753 mag, F213 = 20.2203 mag
# 
# thus pirel = 0.022222 mas, thetaE = 0.3008 mas
# 
### change later
# murel_hel_E = -1.8 0.18, following OB110950, and 10% uncertainty
# murel_hel_N =  3.8 0.38
# 
# t0_for_velocity = 12745.5 (2030.09.01), toward this ra dec, 
# ve=  2.052 AU/yr
# vn= -0.618 AU/yr
#
# thus murel_geo_E = -1.8 -   2.052  * 0.022222 = -1.8456 mas/yr
#      murel_geo_N =  3.8 - (-0.618) * 0.022222 =  3.8137 mas/yr
#      murel_geo = 4.2368 mas/yr
#
# give a 5% uncertainty to tE
# tE = 25.9316 +- 1.2966 day
#
# AV = 4 mag (Roman field average), RV=2.5, 
# resulting Alambda/AV = [0.8082, 0.4676, 0.1084], 
# thus A_F062 = 3.2328 mag, A_F087 = 1.8704 mag, A_F213 = 0.4336 mag
#
# thus reddened F062 = 27.1371 mag, F087 = 23.9457 mag, F213 = 20.6539 mag
# thus (F062-F087) = 3.1914 mag, (F087-F213) = 3.2918 mag
# 
### mock event information end ###



############################################################
############### only need to change below ##################
############################################################
# Ml = np.array([0.5]) # solar mass
Ml = np.array([1.0]) # solar mass
Dl = np.array([7.5]) # kpc
Ds = np.array([9])   # kpc
# MH_lens = np.array([0.2]) # dex
# Age_lens = np.array([6]) # Gyr
MH_lens = np.array([0.0]) # dex
Age_lens = np.array([10]) # Gyr

AV_lens = np.array([4]) # mag
RV_lens = np.array([2.5]) 
murel_hel_E = np.array([-1.8]) # mas/yr
murel_hel_N = np.array([3.8]) # mas/yr

Color_F062_F087_lens_uncertainty = np.array([0.1]) # mag
Color_F087_F213_lens_uncertainty = np.array([0.05]) # mag
F213_lens_uncertainty = np.array([0.05]) # mag

tE_fractional_uncertainty = np.array([0.05])
murel_hel_fractional_uncertainty = np.array([0.10])

t0_for_velocity = 12745.5 # HJD-2450000 (2030.09.01)
ra = 15 * (17 + 53/60 + 17.48/3600) # deg
dec = -(29 + 13/60 + 43.17/3600) # deg
############################################################
############### only need to change above ##################
############################################################




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
###########################
# F062, F087, F213
fields = (FIELDS[11], FIELDS[12], FIELDS[18])
    
    
    
out, meta = interpolate_from_triplets(NPZ_PATH, MH_lens, Age_lens, Ml, fields=fields)


### absolute magnitude and intrinsic color
M_F062_lens, M_F087_lens, M_F213_lens = out[ fields[0] ], out[ fields[1] ], out[ fields[2] ]

Intrinsic_Color_F062_F087_lens = M_F062_lens - M_F087_lens
Intrinsic_Color_F087_F213_lens = M_F087_lens - M_F213_lens

MH_lens_snapped = meta["MH_grid"]
Age_lens_snapped = meta["Age_grid"]

# print('M_F062_lens, M_F087_lens, M_F213_lens', M_F062_lens, M_F087_lens, M_F213_lens)



# lookup : 2D array, shape (N_RV, 4)
#     Column 0: R_V grid (monotonic increasing)
#     Column 1: A_F062 / A_V
#     Column 2: A_F087 / A_V
#     Column 3: A_F213 / A_V

LOOKUP_CCM89_MDWARF_F062_F087_F213 = np.array([
    [2.0, 0.7815, 0.4083, 0.0927],
    [2.1, 0.7879, 0.4224, 0.0965],
    [2.2, 0.7937, 0.4353, 0.0999],
    [2.3, 0.7989, 0.4470, 0.1030],
    [2.4, 0.8037, 0.4577, 0.1058],
    [2.5, 0.8082, 0.4676, 0.1084],
    [2.6, 0.8122, 0.4767, 0.1109],
    [2.7, 0.8160, 0.4852, 0.1131],
    [2.8, 0.8195, 0.4930, 0.1152],
    [2.9, 0.8227, 0.5003, 0.1171],
    [3.0, 0.8258, 0.5071, 0.1189], 
    [3.5, 0.8382, 0.5353, 0.1264], 
    [4.0, 0.8475, 0.5565, 0.1320], 
    [4.5, 0.8548, 0.5730, 0.1364],
    [5.0, 0.8605, 0.5861, 0.1399], 
    [5.5, 0.8652, 0.5969, 0.1428], 
    [6.0, 0.8691, 0.6059, 0.1451]
])

E_F062_F087, E_F087_F213, A_F213 = interp_A_F062_F087_F213(AV_lens, RV_lens, lookup = LOOKUP_CCM89_MDWARF_F062_F087_F213)

# print('E_F062_F087, E_F087_F213, A_F213 ', E_F062_F087, E_F087_F213, A_F213)



Color_F062_F087_lens = Intrinsic_Color_F062_F087_lens + E_F062_F087
Color_F087_F213_lens = Intrinsic_Color_F087_F213_lens + E_F087_F213
## distance modulus, Dl in kpc
mu_lens = 5.0 * (np.log10(Dl) + 2.0)
F213_lens = M_F213_lens + mu_lens + A_F213

print('Color_F062_F087_lens = ', Color_F062_F087_lens, '+-', Color_F062_F087_lens_uncertainty, 'mag')
print('Color_F087_F213_lens = ', Color_F087_F213_lens, '+-', Color_F087_F213_lens_uncertainty, 'mag')
print('F213_lens            = ', F213_lens, '+-', F213_lens_uncertainty, 'mag')




### currently use murel_geo and murel_hel, 
### later should be changed to murel_Roman and murel_hel. 
### in AU/yr
v_earth_N, v_earth_E, v_earth_r = getEarthVelocity(t0_for_velocity, ra, dec)
# print('v_earth_N, v_earth_E', v_earth_N, v_earth_E)





pirel = 1./Dl-1./Ds       # au*(1/Dl-1/Ds), unit: mas
thetaE = np.sqrt(8.144*Ml*pirel) # unit:mas
# print('pirel', pirel)
# print('thetaE', thetaE)

print('murel_hel_E = ', murel_hel_E, '+-', abs(murel_hel_E * murel_hel_fractional_uncertainty), 'mas/yr')
print('murel_hel_N = ', murel_hel_N, '+-', abs(murel_hel_N * murel_hel_fractional_uncertainty), 'mas/yr')

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

Year2Day = 365.25

tE = thetaE / murel_geo * Year2Day  # unit: day

print('tE = ', tE, '+-', tE * tE_fractional_uncertainty, 'day')

