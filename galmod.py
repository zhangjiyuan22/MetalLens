import numpy as np
#import cupy as cp

def GetGalactocentricCoordinates(alpha,delta,d):
    ''' see astropy.coordinates for description '''
    ''' return the target position in Galactocentric coordinates, kpc '''
    alpha *= np.pi/180.
    delta *= np.pi/180.
    xicrs = d*np.cos(alpha)*np.cos(delta)
    yicrs = d*np.sin(alpha)*np.cos(delta)
    zicrs = d*np.sin(delta)
    ricrs = np.array([xicrs,yicrs,zicrs])
    ## a few default quantities ##
    alpha_gc = (17.+45./60.+37.224/3600.)*15*np.pi/180.
    delta_gc = (28.+56./60.+10.23/3600.)*np.pi/180.     # the following algebra has accounted for negative sign of delta_gc
    d_gc = 8.3      # distance from Sun to GC, kpc
    z_sun= 0.027    # height of Sun above the Galactic midplane, kpc
    eta = 58.5986320*np.pi/180. # extra "roll" angle, Blaauw et al., 1960
    xvec_gc = np.array([1.,0.,0.])
    ## rotation matrices ##
    rot = np.zeros((3,3))
    rot[0,0] = np.cos(alpha_gc)*np.cos(delta_gc)
    rot[0,1] = np.cos(delta_gc)*np.sin(alpha_gc)
    rot[0,2] =-np.sin(delta_gc)
    rot[1,0] = np.cos(alpha_gc)*np.sin(delta_gc)*np.sin(eta) - np.sin(alpha_gc)*np.cos(eta)
    rot[1,1] = np.sin(alpha_gc)*np.sin(delta_gc)*np.sin(eta) + np.cos(alpha_gc)*np.cos(eta)
    rot[1,2] = np.cos(delta_gc)*np.sin(eta)
    rot[2,0] = np.cos(alpha_gc)*np.sin(delta_gc)*np.cos(eta) + np.sin(alpha_gc)*np.sin(eta)
    rot[2,1] = np.sin(alpha_gc)*np.sin(delta_gc)*np.cos(eta) - np.cos(alpha_gc)*np.sin(eta)
    rot[2,2] = np.cos(delta_gc)*np.cos(eta)
    sintheta = z_sun/d_gc
    costheta = np.sqrt(1-sintheta**2)
    H = np.array([[costheta,0.,sintheta],[0.,1.,0.],[-sintheta,0.,costheta]])
    rgc = np.dot(H,np.dot(rot,ricrs)-d_gc*xvec_gc) 
    return rgc

def EN2YZ(ve,vn):
    ''' convert the velocity components in (East,North) directions to the Galactocentric (y,z) directions '''
    GalaPlaneTilt = 62.9*np.pi/180.
    vy = ve*np.cos(GalaPlaneTilt) + vn*np.sin(GalaPlaneTilt)
    vz =-ve*np.sin(GalaPlaneTilt) + vn*np.cos(GalaPlaneTilt)
    return vy,vz

def YZ2EN(vy,vz):
    ''' convert the velocity components in (y,z) directions in the Galactocentric frame to (E,N) directions '''
    GalaPlaneTilt = -62.9*np.pi/180.    # angle between celestrial plane and galactic plane
    ve = vy*np.cos(GalaPlaneTilt) + vz*np.sin(GalaPlaneTilt)
    vn =-vy*np.sin(GalaPlaneTilt) + vz*np.cos(GalaPlaneTilt)
    return ve,vn

def GetGalacticCoordinates(alpha,delta):
    ''' find Galactic coordinates (l,b) (deg) from equatorial coordinates (alpha,delta) (deg)'''
    alpha *= np.pi/180.
    delta *= np.pi/180.
    alpha_gp = 192.859508*np.pi/180.
    delta_gp = 27.128336*np.pi/180.
    lcp = 122.932*np.pi/180.
    sinb = np.sin(delta_gp)*np.sin(delta) + np.cos(delta_gp)*np.cos(delta)*np.cos(alpha-alpha_gp)
    b = np.arcsin(sinb)
    cosb = np.sqrt(1-sinb**2)
    sinlcp_l = np.cos(delta)*np.sin(alpha-alpha_gp)/cosb
    coslcp_l = (np.cos(delta_gp)*np.sin(delta) - np.sin(delta_gp)*np.cos(delta)*np.cos(alpha-alpha_gp))/cosb
    sinl = np.sin(lcp)*coslcp_l - np.cos(lcp)*sinlcp_l
    cosl = np.cos(lcp)*coslcp_l + np.sin(lcp)*sinlcp_l
    if sinl > 0:
        l = np.arccos(cosl)
    else:
        l =-np.arccos(cosl) + 2*np.pi
    return l*180./np.pi,b*180./np.pi

def BulgeStellarDensity(rgc):
    ''' return the stellar density contributed by the Bulge at given Galactocentric coordinates rgc; parameters from Robin et al. (2003) '''
    BarAngle = 30.*np.pi/180.   # G2 bar orientation w.r.t. x axis
    xp = rgc[0]*np.cos(BarAngle) - rgc[1]*np.sin(BarAngle)
    yp = rgc[0]*np.sin(BarAngle) + rgc[1]*np.cos(BarAngle)
    zp = rgc[2]
    x0,y0,z0 = 1.59,0.424,0.424 # scale length along each axis
    Rc = 2.54   # truncation distance of Bulge, in kpc
    N0 = 13.70  # fiducial number density ; in stars/pc^3
    rs = (((xp/x0)**2+(yp/y0)**2)**2+(zp/z0)**4)**0.25
    NumberDensity = N0*np.exp(-0.5*rs**2)
#    ptrunc = np.sqrt(xp**2+yp**2)-Rc
#    if ptrunc > 0:
#        NumberDensity *= np.exp(-2*ptrunc**2)
    return NumberDensity

def DiskStellarDensity(rgc, Rd=2.5):
    ''' return the stellar density contributed by the disk at given Galactocentric coordinates rgc'''
    r0 = Rd     # scale length of disk, kpc
    z0 = 0.325  # scale height of disk, kpc
#     NumberDensity0 = 0.14*np.exp(8.3/r0)
    NumberDensity0 = 0.14*np.exp(8.3/r0)
#    rho0 = 0.06*np.exp(8.3/r0)  # adopted such to be consistent with local density (see Henderson+2014)
    r = np.sqrt(rgc[0]**2+rgc[1]**2)
    z = np.abs(rgc[2])
#    rho = rho0*np.exp(-r/r0-z/z0)
    NumberDensity = NumberDensity0*np.exp(-r/r0-z/z0)
#    q,R1 = 1.5,0.75 # parameters to truncate the disk at ~1 kpc from GC
#    rho *= 0.5*(1+np.tanh(q*(r-R1)))
    return NumberDensity

def VelocityDistribution(rgc):
    ''' return the velocity mean and dispersion at given Galactocentric coordinates rgc '''
    vdisk_mean = np.array([240.,0.])
    vdisk_disp = np.array([33.,18.])
    vbulg_mean = np.array([0.,0.])
    ## Bulge has rotation speed ##
    Omega0 = 88.    # km/s/kpc
    BarAngle = 30.*np.pi/180.   # G2 bar orientation w.r.t. x axis
    vrot = Omega0*np.sqrt(rgc[0]**2+rgc[1]**2)*np.cos(BarAngle)
#    vbulg_mean += np.array([vrot,0.])
    vbulg_disp = np.array([120.,120.])
    ## for given position, find velocity mean & dispersion ##
    rhob = BulgeStellarDensity(rgc)
    rhod = DiskStellarDensity(rgc)
    vmean = (rhob*vbulg_mean+rhod*vdisk_mean)/(rhob+rhod)
    vdisp = np.sqrt((rhob**2*vbulg_disp**2+rhod**2*vdisk_disp**2)/(rhob+rhod)**2)
    return vmean,vdisp

def TransverseVelocityPriorProbability(rgc_lens,rgc_sour,dlens,dsour):
    ''' return the prior distribution of vhel, based on the positions of lens and source '''
    vdisk_mean = np.array([240.,0.])
    vsun_lsr = np.array([12.,7.]) # Solar velocity w.r.t. the LSR
    vsun_gc = vsun_lsr + vdisk_mean
    f1 = dsour*1./(dsour-dlens)
    f2 =-dlens*1./(dsour-dlens)
    vsmean,vsdisp = VelocityDistribution(rgc_sour)
    vlmean,vldisp = VelocityDistribution(rgc_lens)
    vmean = f1*vlmean + f2*vsmean - vsun_gc
    vdisp = np.sqrt(f1**2*vldisp**2 + f2**2*vsdisp**2)
    ## now we need to rotate from (y,z) to (E,N) ##
    vmeane,vmeann = YZ2EN(vmean[0],vmean[1])
    ## the covariance matrix is also changed: from decoupled (y,z) to coupled (E,N) ##
    GalTilt = 62.9*np.pi/180.   # angle between the celestrial plane and the Galactic plane
    sigx2 = vdisp[0]**2*(np.cos(GalTilt))**2 + vdisp[1]**2*(np.sin(GalTilt))**2
    sigy2 = vdisp[0]**2*(np.sin(GalTilt))**2 + vdisp[1]**2*(np.cos(GalTilt))**2
    sigxy = (vdisp[0]**2-vdisp[1]**2)*np.sin(GalTilt)*np.cos(GalTilt)
    vhel_cov = np.array([[sigx2,sigxy],[sigxy,sigy2]])
    vhel_mean = np.array([vmeane,vmeann])
    return vhel_mean,vhel_cov

def RelativeProperMotionPriorProbability(rgc_lens,rgc_sour,dlens,dsour):
    ''' return the prior distribution of mu_rel, based on the positions of lens and source '''
    vdisk_mean = np.array([240.,0.])
    vsun_lsr = np.array([12.,7.])   # Solar velocity wrt the LSR
    vsun_gc = vsun_lsr + vdisk_mean
    vsmean,vsdisp = VelocityDistribution(rgc_sour)
    vlmean,vldisp = VelocityDistribution(rgc_lens)
    mumean = vlmean/dlens - vsmean/dsour - (1./dlens-1./dsour)*vsun_gc
    mudisp = np.sqrt(vsdisp**2/dsour**2 + vldisp**2/dlens**2)
    UnitConvert = np.pi/15. # km/s/kpc = 2*pi/30 mas/yr
    mumean *= UnitConvert
    mudisp *= UnitConvert
    ## now we need to rotate from (y,z) to (E,N) ##
    mumeane,mumeann = YZ2EN(mumean[0],mumean[1])
    ## the covariance matrix is also changed: from decoupled (y,z) to coupled (E,N) ##
    GalTilt = 62.9*np.pi/180.   # angle between the celestrial plane and the Galactic plane
    sigx2 = mudisp[0]**2*(np.cos(GalTilt))**2 + mudisp[1]**2*(np.sin(GalTilt))**2
    sigy2 = mudisp[0]**2*(np.sin(GalTilt))**2 + mudisp[1]**2*(np.cos(GalTilt))**2
    sigxy = (mudisp[0]**2-mudisp[1]**2)*np.sin(GalTilt)*np.cos(GalTilt)
    murel_cov = np.array([[sigx2,sigxy],[sigxy,sigy2]])
    murel_mean = np.array([mumeane,mumeann])
    return murel_mean,murel_cov

def GaussianOverlapIntegral(a1,a2,c1,c2):
    ''' compute the overlap integral of two Gaussians; means are given by (a1,a2), and covariance matrices are given by (c1,c2)'''
    b1 = np.linalg.inv(c1)
    d1 = np.dot(b1,a1)
    b2 = np.linalg.inv(c2)
    d2 = np.dot(b2,a2)
    ## combine ##
    bnew = b1 + b2
    cnew = np.linalg.inv(bnew)
    dnew = d1 + d2
    anew = np.dot(cnew,dnew)
    prob = np.sqrt(np.linalg.det(b1)*np.linalg.det(b2)/np.linalg.det(bnew))/2./np.pi
    prob*= np.exp(-0.5*(np.dot(d1,a1)+np.dot(d2,a2)-np.dot(dnew,anew)))
    return prob

def LogMassFunction(mass,kroupa=False):
    ''' the mass function in log(m); provide flat MF & Kroupa MF forms; the input mass is in unit of Solar mass '''
    if kroupa == False:
        # if mass>0.013 and mass<1.3:
        if mass>0.013 : ## no trunction
            return 1.
        else:
            return 0.
    if mass<0.013:  # do not consider planetary lenses
        return 0.
    elif mass<0.08:
        return mass**(0.7)
    elif mass<0.5:
        return 0.08*mass**(-0.3)
    else: ## no trunction
        return 0.04*mass**(-1.3)
    # elif mass<1.3:
    #     return 0.04*mass**(-1.3)
    # else:   # MF truncated at 1.3 Solar mass, because bright lens are rare
    #     return 0.

##### Mass Function including WD,NS and BH
def MFMS(mass):
    '''Mass Function of Main Sequence Stars and Brown Dwarf
    Only for numpy arrays
    '''
    mass = list(mass)
    return np.fromiter(map(lambda x:LogMassFunction(x,kroupa=True)/x,mass),dtype='float')*0.4753865875688558
    # return np.fromiter(map(lambda x:LogMassFunction(x,kroupa=False)/x,mass),dtype='float')*0.4753865875688558
    
def MFWD(mass):
    '''Mass Function of White Dwarf'''
    sigma = 0.16
    m0 = 0.65
    pdf = 1./(2*np.pi)**0.5/sigma*np.exp(-((mass-m0)/sigma)**2/2.)*(np.sign((mass-0.3))-np.sign((mass-1.4)))*0.5
    return pdf*0.004464079212421635
    #return pdf*0.007126687790892069

def MFNS(mass):
    '''Mass Function of Neutron Star'''
    sigma = 0.2
    m0 = 1.5
    pdf = 1./(2*np.pi)**0.5/sigma*np.exp(-((mass-m0)/sigma)**2/2.)*(np.sign((mass-1.1))-np.sign((mass-2.5)))*0.5
    return pdf*0.00030698401622668026
    #return pdf*0.0004900852175837338
    
def MFBH(mass):
    '''Mass Function of Black Hole'''
    pdf = 10.**(mass/(-17.))*(np.sign(mass-2.5)-np.sign(mass-80.))*0.5
    return pdf*2.6613860126445695e-05
    #return pdf*4.2478574332843974e-05



def MassFunctionZhu(M_range,dM):
    mass = np.arange(*M_range,dM)
    pdf_ms = MFMS(mass)
    pdf_wd = MFWD(mass)
    pdf_ns = MFNS(mass)
    pdf_bh = MFBH(mass)
    pdf_All = pdf_ms+pdf_wd+pdf_ns+pdf_bh
    
    return np.array(pdf_All)
