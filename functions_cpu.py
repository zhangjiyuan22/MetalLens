#import cupy as cp
import numpy as np
import os
import galmod as gm
# np.seterr(divide='ignore')
import time
import os
# from getconfig import *
from multiprocessing import get_context
import h5py

def mkdir(path):
    Exist = os.path.exists(path)
    if Exist:
        print(path, '  already exists!')
    else:
        os.makedirs(path)
        print(path,'  created!')

# def save_path(savedir,strs,comp,mdname,tEmean,n,npz=False):
#     path = savedir + '%s-%s-tE%d%s-%d.npy'%(strs,mdname,tEmean,comp,n)
#     if npz:
#         path = path[:-1]+'z'
#     return path

# def vRvT2vl(vR,vT,R_vec,dir_vec,sign_theta):
#     '''
#     convert 2D velocity in plane (vR,vT) to vl，i.e. the velocity component in plane that is prependicular to line-of-sight
#     vR: radial velocity; 
#     vT: Tangential velocity (direction of rotation); 
#     R_vec: The vector of the galactic center pointing to the position of the star; 
#     dir_vec: Unit vector of line of sight direction;
#     sign_theta: Specifies the sign of sin (when l>0/l<0, sin>0/sin<0)
#     '''
#     cos_theta = -np.dot(R_vec,dir_vec)/np.linalg.norm(R_vec,axis=1)
#     sin_theta = sign_theta * np.sqrt(1-cos_theta**2)
#     vl = vT * cos_theta + vR * sin_theta
#     return vl

def RandomSampleFromCDF(x,CDF,n_sample,smooth=True,return_index=False):
    # '''
    # smooth: if True, smooth out discreate random values (by adding a random number in [-dx/2,dx/2])
    # '''
    x_gen = np.random.uniform(0.,CDF[-1],n_sample)
    index = np.searchsorted(CDF,x_gen)
    if smooth:
        dx = x[1]-x[0]
        x_result = x[index] + (np.random.rand(n_sample)-0.5)*dx
    else:
        x_result = x[index]
    
    if return_index:
        return x_result,index
    else:
        return x_result
    
def RandomSampleFromCDF2(xs,CDFs,ratios,n_sample,smooth=True):
    # '''
    # support multi-CDF
    # smooth: if True, smooth out discreate random values (by adding a random number in [-dx/2,dx/2])
    # '''
    n_component = int(len(ratios))
    ratios = np.array(ratios)/np.sum(ratios)
    cum_ratios = np.cumsum(ratios)
    classify = np.random.rand(n_sample)
    
    classification = np.searchsorted(cum_ratios,classify)
    x_result = np.zeros(n_sample)
    
    for c in range(n_component):
        classfilter = (classification==c)
        n_class = int(np.sum(classfilter))
        x_gen = np.random.uniform(0.,CDFs[c][-1],n_class)
        index = np.searchsorted(CDFs[c],x_gen)
        
        if smooth:
            dx = xs[c][1]-xs[c][0]
            x_result[classfilter] = xs[c][index] + (np.random.rand(n_class)-0.5)*dx
        else:
            x_result[classfilter] = xs[c][index]
    
    return x_result,classification
    
    
# def RandomSampleFromPDF(x,PDF,n_sample,smooth=True,return_index=False):
#     CDF = np.cumsum(PDF)
#     return RandomSampleFromCDF(x,CDF,n_sample,smooth,return_index)
# """
# def SampleFromTruncatedGaussian1D(mean,std,nsample,truncation=[-np.inf,np.inf],precision=100,cupy=False):
#     '''
#     mean: peak location (not necessary to be the mean value) of Gaussian distribution
#     std: the standard deviation
#     nsample: number of output sampled points
#     truncation: list-like, lower and upper truncation value
#     precision: sample rate of PDF in 1-sigma
#     '''
#     dx = std/precision
#     x_min = max(mean-10*std,truncation[0]+dx/2)
#     x_max = min(mean+10*std,truncation[1]+dx/2)
#     if cupy==False:
#         x = np.arange(x_min,x_max, dx)
#         PDF = np.exp(-0.5* ((x-mean)/std)**2 )
#         CDF = np.cumsum(PDF)
#         sample_cdf = np.random.rand(nsample)*CDF[-1]
#         sample_arg = np.searchsorted(CDF,sample_cdf)
#         sample_x = x[sample_arg]
#         sample_x = sample_x + (np.random.rand(nsample)-0.5)*dx
#     else:
#         x = cp.arange(x_min,x_max, dx)
#         PDF = cp.exp(-0.5* ((x-mean)/std)**2 )
#         CDF = cp.cumsum(PDF)
#         sample_cdf = cp.random.rand(nsample)*CDF[-1]
#         sample_arg = cp.searchsorted(CDF,sample_cdf)
#         sample_x = x[sample_arg]
#         sample_x = sample_x + (cp.random.rand(nsample)-0.5)*dx
#     return sample_x

# def SampleFromOneSideGaussian1D(mean,std,nsample,flat='below',truncation=[-np.inf,np.inf],precision=100,cupy=False):
#     '''
#     mean: peak location (not necessary to be the mean value) of Gaussian distribution
#     std: the standard deviation
#     nsample: number of output sampled points
#     flat: 'below' or 'above'
#           'below': PDF below mean is flat
#           'above': PDF above mean is flat
#     truncation: list-like, lower and upper truncation value
#     precision: sample rate of PDF in 1-sigma
#     '''
#     dx = std/precision
#     x_min = max(mean-10*std,truncation[0]+dx/2)
#     x_max = min(mean+10*std,truncation[1]+dx/2)
#     if cupy==False:
#         x = np.arange(x_min,x_max, dx)
#         PDF = np.exp(-0.5* ((x-mean)/std)**2 )
#         if flat=='below':
#             PDF[x<=mean] = 1.
#         elif flat=='above':
#             PDF[x>=mean] = 1.
#         CDF = np.cumsum(PDF)
#         sample_cdf = np.random.rand(nsample)*CDF[-1]
#         sample_arg = np.searchsorted(CDF,sample_cdf)
#         sample_x = x[sample_arg]
#         sample_x = sample_x + (np.random.rand(nsample)-0.5)*dx
#     else:
#         x = cp.arange(x_min,x_max, dx)
#         PDF = cp.exp(-0.5* ((x-mean)/std)**2 )
#         if flat=='below':
#             PDF[x<=mean] = 1.
#         elif flat=='above':
#             PDF[x>=mean] = 1.
#         CDF = cp.cumsum(PDF)
#         sample_cdf = cp.random.rand(nsample)*CDF[-1]
#         sample_arg = cp.searchsorted(CDF,sample_cdf)
#         sample_x = x[sample_arg]
#         sample_x = sample_x + (cp.random.rand(nsample)-0.5)*dx
#     return sample_x

# from math import erf
# def erf_array(x):
#     if type(x) in [float,np.float64]:
#         return erf(x)
#     result = np.fromiter(map(erf,x),dtype='float')
#     return result

# def RatioOf2GaussianDistribution(x,mean1,mean2,std1,std2):
#     '''
#     Z = X1/X2
#     X1 ~ N(mean1,std1)
#     X2 ~ N(mean2,std2)
#     '''
#     w1 = 1/(std1)**2
#     w2 = 1/(std2)**2
    
#     at = np.sqrt(x**2*w1 + w2)
#     bt = x*mean1*w1 + mean2*w2
#     rt2 = (bt/at)**2
#     ct = mean1**2*w1 + mean2**2*w2
#     dt = np.exp(0.5*(rt2 - ct))
    
#     A1 = bt*dt/(at**3)/(np.sqrt(2*np.pi)*std1*std2)*( erf_array(bt/at) )
#     A2 = 1/(at**2*np.pi*std1*std2)*np.exp(-0.5*ct)
    
#     return A1 + A2
# """

# def genVelocityDistribution_Shu(sigma_r0=38., Rd=2.5, vc=220.
#                                 ,dV=1.0/200.0, Rmin=0.05,Rmax=8.35,Rstep=0.1
#                                 ,savepath = 'velocity_data'):
#     Shuinfo = np.round(np.array([sigma_r0,Rd,vc,dV,Rmin,Rmax,Rstep]),5)
#     if os.path.exists(savepath+'/Shuconfig.npy'):
#         Shuinfo_old = np.load(savepath+'/Shuconfig.npy')
#         if (Shuinfo==Shuinfo_old).all():
#             print('Shu velocity distribution files already exist.')
#             return
        
#     from galpy.df import shudf
#     print('Generating Shu velocity distribution...',flush=True)
#     for rs in np.arange(Rmin,Rmax,Rstep): # (Rmin,Rmax,Rstep), in kpc
#         ShuDistFilenamevT = savepath+'/Shu-vT-CDF-R%.2f-dV5e-3.npy'%(rs)
#         ShuDistFilenamevR = savepath+'/Shu-vR-CDF-R%.2f-dV5e-3.npy'%(rs)

#         Vt_gen0 = np.arange(0, 3, dV)
#         Vr_gen0 = np.arange(-2, 2, dV)
#         ### 计算 Shu DF 的(vT,vR)分布CDF ###
#         print('  Generating Shu DF... (R=%.2f)'%rs,flush=True)
#         df = shudf(profileParams=(Rd/8.3,(4*Rd)/8.3,sigma_r0/vc),beta=0.)

#         vts = np.linspace(0.0,3.0,151)
#         vrs = np.linspace(-2.0,2.0,201)
# #                 Vt_gen0 = np.arange(0, 3, dV)
#         pdf_Vt_Disk = ( lambda x: np.sum([df(np.array([rs/8.3,vr,x])) for vr in vrs]) )
#         Vt_pdf = np.array([pdf_Vt_Disk(xi) for xi in Vt_gen0])
#         Vt_cdf = np.cumsum(Vt_pdf)
#         np.save(ShuDistFilenamevT,np.array(Vt_cdf))
# #                 Vr_gen0 = np.arange(-2, 2, dV)
#         pdf_Vr_Disk = ( lambda x: np.sum([df(np.array([rs/8.3,x,vt])) for vt in vts]) )
#         Vr_pdf = np.array([pdf_Vr_Disk(xi) for xi in Vr_gen0])
#         Vr_cdf = np.cumsum(Vr_pdf)
#         np.save(ShuDistFilenamevR,np.array(Vr_cdf))
#     np.save(savepath+'/Shuconfig.npy',Shuinfo)
#     print('Done',flush=True)
#     return

# def Gaussian1D(x,mean,sigma):
#     A = 1/np.sqrt(2*np.pi)/sigma
#     cx = (x-mean)/sigma
#     #chi = cx**2
#     return A*np.exp(-0.5*cx**2)

# def Gaussian2D(x1,x2,mean1,mean2,sigma1,sigma2,rho12):
#     A = 1/(2*np.pi*sigma1*sigma2*np.sqrt(1-rho12**2))
#     cx1 = (x1-mean1)/sigma1
#     cx2 = (x2-mean2)/sigma2
#     chi = 1/(1-rho12**2)*( cx1**2 -2*rho12*cx1*cx2 + cx2**2 )
#     return A*np.exp(-0.5*chi)

####################################
## Look up for lens mass range #####
####################################
def _nearest_index_uniform(x, x0, dx, n):
    #"""Nearest integer index on a uniform axis; robust to tiny float jitter."""
    idx = np.rint((x - x0) / dx).astype(np.int64)
    return np.clip(idx, 0, n - 1)

def mass_bounds_for_samples(npz_path, Age_lens, MH_lens):
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
    DB = np.load(npz_path, allow_pickle=False)

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





def simulate_one_run(args):
    # """
    # Worker: perform one 'ni' run and save one output file.
    # We pass a single tuple (ni, ctx) because Pool.map only maps one arg.
    # """
    
    ni, ctx = args

    # unpack context (everything computed once in SimulateEvents)
    (eventname, ndots, 
     Dss, cdfDs, ratioDs, Dls, cdfDl, ratioDl,
     Disk_min,
     Age_lens_min, Age_lens_max, MH_lens_min, MH_lens_max, 
     murel_hel_E_mean, murel_hel_E_err, 
     murel_hel_N_mean, murel_hel_N_err, 
     murel_hel_rhoEN, 
     AV_lens_min, AV_lens_max, RV_lens_mean, RV_lens_err) = ctx


    # Per-process seed for reproducibility and independence
    np.random.seed( 1234567 + int(ni) )


    time_begin = time.time()
    print('%dth run...'%ni, flush=True)
    # savename = 'output/%s/MCevent-%s_%d.dat'%(eventname,eventname,ni)
    # savename = 'output/%s/MCevent-%s_%d.npy'%(eventname,eventname,ni)

    print('  Generating Ds for process %s'%ni, flush=True)
    # generate source position. sloc=(0,1): (bulge,disk);
    Ds_gen,sloc = RandomSampleFromCDF2(xs=Dss,CDFs=cdfDs,ratios=ratioDs,n_sample=ndots)
    # Rs_gen_vec = (Ds_gen*direction[:2,np.newaxis]+Earth[:2,np.newaxis]).T # 与银心的距离(向量),不考虑z
    # Rs_gen = np.linalg.norm(Rs_gen_vec,axis=1) # 取模

    # generate lens position. lloc=(0,1): (bulge,disk)
    Dl_gen,lloc = RandomSampleFromCDF2(xs=Dls,CDFs=cdfDl,ratios=ratioDl,n_sample=ndots)
    # Rl_gen_vec = (Dl_gen*direction[:2,np.newaxis]+Earth[:2,np.newaxis]).T # 与银心的距离(向量),不考虑z
    # Rl_gen = np.linalg.norm(Rl_gen_vec,axis=1) # 取模



    # Drop Dl>Ds pairs
    # print('  Drop unphysical source-lens pairs...',end='    ',flush=True)
    FineQ  = (Dl_gen < Ds_gen)
    Ds, Dl = Ds_gen[FineQ], Dl_gen[FineQ]
    # Source_loc, Lens_loc =  sloc[FineQ], lloc[FineQ]
    # Rs, Rl = Rs_gen[FineQ], Rl_gen[FineQ]
    # Rs_vec, Rl_vec = Rs_gen_vec[FineQ], Rl_gen_vec[FineQ]
    nFine = int(np.sum(FineQ))
    del FineQ

    # sBulgQ = (Source_loc==0)
    # sDiskQ = (Source_loc==1)
    # lBulgQ = (  Lens_loc==0)
    # lDiskQ = (  Lens_loc==1)

    # ns_Bulg = int(np.sum(sBulgQ))
    # ns_Disk = int(np.sum(sDiskQ))
    # nl_Bulg = int(np.sum(lBulgQ))
    # nl_Disk = int(np.sum(lDiskQ))

    del Ds_gen, Dl_gen, sloc, lloc #, Rs_gen, Rl_gen
    # print('done',flush=True)

    print('  Generating Dl for process %s'%ni, flush=True)
    ### resample Dl
    Dl = Ds * np.random.uniform(0.0, 1.0, size=nFine)
    mask = (Dl < Disk_min)
    while np.any(mask):
        Dl[mask] = Ds[mask] * np.random.uniform(0.0, 1.0, size=mask.sum())
        mask = (Dl < Disk_min)

    del mask
    


    print('  Generating lens Age for process %s'%ni, flush=True)
    # Uniform age in e.g. [1, 10] Gyr
    Age_lens = np.random.uniform(Age_lens_min, Age_lens_max, size=nFine)

    print('  Generating lens [M/H] for process %s'%ni, flush=True)
    # Uniform [M/H] in e.g. [-1.0, 0.5]
    MH_lens  = np.random.uniform(MH_lens_min, MH_lens_max, size=nFine)

    print('  Look up for lens mass range for each sampled lens (Age, [M/H]) pair', flush=True)
    mass_min_arr, mass_max_arr = mass_bounds_for_samples("PARSEC_isochrone/final_isochrone_label012_vista_roman_euclid_ogle2_csst_merged_no_repeating_mass.npz", Age_lens, MH_lens)

    print('  Generating (primary) lens masses for process %s'%ni, flush=True)
    # Uniform (primary) lens mass in the mass range allowed by isochrone of (Age, [M/H]) pair
    Ml = np.random.uniform(low=mass_min_arr, high=mass_max_arr, size=nFine)

    del mass_min_arr, mass_max_arr



    print('  Generating murel_hel_E, murel_hel_N for process %s'%ni, flush=True)
    
    murel_hel_mean = np.array([murel_hel_E_mean, murel_hel_N_mean])

    murel_hel_cov = np.array([
        [murel_hel_E_err**2,                                   murel_hel_rhoEN * murel_hel_E_err * murel_hel_N_err],
        [murel_hel_rhoEN * murel_hel_E_err * murel_hel_N_err,  murel_hel_N_err**2]
    ])

    # Shape (N, 2)
    murel_hel_2D = np.random.multivariate_normal(murel_hel_mean, murel_hel_cov, size=nFine)
    murel_hel_E = murel_hel_2D[:, 0]
    murel_hel_N = murel_hel_2D[:, 1]

    del murel_hel_2D



    print('  Generating lens AV, RV for process %s'%ni, flush=True)
    # Uniform AV in e.g. [0, 10] mag (mean AV in Roman GBTDS field is around 4 mag)
    AV_lens = np.random.uniform(AV_lens_min, AV_lens_max, size=nFine)
    
    # 1. CCM89 RV from Normal(2.5, 0.2), adopting from Nataf et al. 2013; 
    # 2. for CCM89, changing RV only changes the direction of the extinction vector; 
    #    while for F99, changing RV only changes the length of the extinction vector, not the direction;
    #    thus by using CCM89, we already considered the worse case, as only direction has influence to the metallicity constraint; 
    # 3. here use such a large RV range to show the method still works even with very uncertain extinction curve;
    #    because RV ~ Normal(2.5, 0.2) is the RV dispersion for the whole OGLE3 bulge field, 
    #    thus for a given line-of-sight, the RV can be constrained better
    # 4. in future, we must use local(line-of-sight dependent) extinction curve determined from Roman data itself; 
    #    no RV parametrization or law form assumption(like CCM89 or F99) needed then, 
    #    Alambda/AV will be directly constrained in Roman filters from variable stars for each line-of-sight. 
    RV_lens = np.random.normal(RV_lens_mean, RV_lens_err, size=nFine)
    # truncated to 2.0<RV<6.0, as CCM89 is only defined in this RV range
    # corresponds to truncate at 2.5 sigma, thus OK
    bad = (RV_lens < 2.0) | (RV_lens > 6.0)
    while np.any(bad):
        RV_lens[bad] = np.random.normal(RV_lens_mean, RV_lens_err, size=bad.sum())
        bad = (RV_lens < 2.0) | (RV_lens > 6.0)
    
    del bad
    


    savename = f'output/{eventname}/MockEvent_{eventname}_batch_{ni}.h5'
    print('  Saving data to %s ...'%(savename), flush=True)

    with h5py.File(savename, "w") as f:
        g = f.create_group("events")

        # Use "compression=None" to disable compression. "lzf" = fast, low-CPU.
        comp = "lzf"   # or None, or "gzip"

        g.create_dataset("Ds",           data=np.asarray(Ds,           np.float32), compression=comp)
        g.create_dataset("Dl",           data=np.asarray(Dl,           np.float32), compression=comp)
        
        g.create_dataset("Age_lens",     data=np.asarray(Age_lens,     np.float32), compression=comp)
        g.create_dataset("MH_lens",      data=np.asarray(MH_lens,      np.float32), compression=comp)
        g.create_dataset("Ml",           data=np.asarray(Ml,           np.float32), compression=comp) 
        
        g.create_dataset("murel_hel_E",  data=np.asarray(murel_hel_E,  np.float32), compression=comp)
        g.create_dataset("murel_hel_N",  data=np.asarray(murel_hel_N,  np.float32), compression=comp)
        
        g.create_dataset("AV_lens",      data=np.asarray(AV_lens,      np.float32), compression=comp)
        g.create_dataset("RV_lens",      data=np.asarray(RV_lens,      np.float32), compression=comp)

        
    print(f'process {ni} done in {time.time()-time_begin:.2f}s', flush=True)

    del Ds, Dl, Age_lens, MH_lens, Ml, murel_hel_E, murel_hel_N, AV_lens, RV_lens

    return savename





def SimulateEvents(config):
    # k_trans = 0.0005775483273639937   # km/s/kpc -> mas/day: 0.0005775483273639937
    # Year2Day = 365.25

    ###################################
    ### read parameters from config ###
    eventname = config.name
    ndots = config.ndot
    nstart,nrun = config.nstart,config.nrun
    n_process = config.n_process
    # Dl_profile = config.Dl_profile
    # Ds_profile = config.Ds_profile
    mkdir('output/%s'%eventname)
    
    # alphas,deltas = [(17.+45./60.+37.224/3600.)*15],[-(28.+56./60.+10.23/3600.)]
    # alphaGC,deltaGC = (17.+45./60.+37.224/3600.)*15 , -(28.+56./60.+10.23/3600.)
    alphas,deltas = config.ra, config.dec
    Galactic_ls,Galactic_bs = gm.GetGalacticCoordinates(alphas,deltas)
    print('Galactic_ls,Galactic_bs',Galactic_ls,Galactic_bs)


    # Galactic model parameters
    gamma = config.gamma
    Rd = config.Rd
    # vc = config.vc
    # sigma_t0 = config.sigma_Vt
    # sigma_b0 = config.sigma_Vz
    # sigma_r0 = config.sigma_VR
    # DiskModel = config.DiskModel
    # print('### Jiyuan ###')
    # print(DiskModel)
    # print('###  end   ###')
    

    ### Galactic components
    # bulge_source = config.bulge_source
    # disk_source  = config.disk_source
    # bulge_lens   = config.bulge_lens
    # disk_lens    = config.disk_lens
    Disk_min = config.Disk_min
    Disk_max = config.Disk_max
    # lDisk_min = config.lDisk_min
    # lDisk_max = config.lDisk_max
    Bulg_min = config.Bulg_min
    Bulg_max = config.Bulg_max

    # SourceFromPM = config.SourceFromPM
    # if SourceFromPM == True:
    #     muslmean,muslerr = config.muslmean,config.muslerr
    #     musbmean,musberr = config.musbmean,config.musberr
    #     muscor = config.muscor

    #     mus_mean = [muslmean,musbmean]
    #     mus_cov = [[muslerr**2,muscor*muslerr*musberr],[muscor*muslerr*musberr,musberr**2]]

    # # Lens type
    # main_sequence = config.main_sequence
    # white_dwarf   = config.white_dwarf
    # neutron_star  = config.neutron_star
    # black_hole    = config.black_hole

    # output option
    # SolarMotion = config.SolarMotion
    # EarthMotion = config.EarthMotion
    # Earth_Vl = config.Earth_Vl
    # Earth_Vb = config.Earth_Vb
    # Solar_Vl = config.Solar_Vl
    # Solar_Vb = config.Solar_Vb

    murel_hel_E_mean, murel_hel_E_err = config.murel_hel_E_mean,config.murel_hel_E_err
    murel_hel_N_mean, murel_hel_N_err = config.murel_hel_N_mean,config.murel_hel_N_err
    murel_hel_rhoEN = config.murel_hel_rhoEN

    AV_lens_min = config.AV_lens_min
    AV_lens_max = config.AV_lens_max

    RV_lens_mean, RV_lens_err = config.RV_lens_mean, config.RV_lens_err

    MH_lens_min = config.MH_lens_min
    MH_lens_max = config.MH_lens_max

    Age_lens_min = config.Age_lens_min
    Age_lens_max = config.Age_lens_max

    ### read parameters from config ###
    ###################################


    ### Get event direction ###
    ### set to Galactic center
    rgcl = gm.GetGalactocentricCoordinates(alphas, deltas, 8.3)
    Earth = gm.GetGalactocentricCoordinates(0.,0.,0.)
    direction = rgcl - Earth
    direction = direction/8.3  #normalized vector
    direction,Earth = np.array(direction), np.array(Earth)
    print('direction,Earth',direction,Earth)

    # if DiskModel=='Shu':
    #     dV=1.0/200.0
    #     Rmin,Rmax,Rstep=0.05,8.35,0.1
    #     genVelocityDistribution_Shu(sigma_r0=sigma_r0, Rd=Rd, vc=vc
    #                                 ,dV=dV, Rmin=Rmin,Rmax=Rmax,Rstep=Rstep
    #                                 ,savepath = 'velocity_data')


    ### Calculate cumulatives ###
    print('Calculating Ds probability distribution (PDF/CDF)...',end='    ',flush=True)
    dDs = 1.0/2000.0
    dDl = 1.0/2000.0

    Ds_Bulg = np.arange(Bulg_min,Bulg_max,dDs)   # source
    Ds_Disk = np.arange(Disk_min,Disk_max,dDs)   # source

    Dl_Bulg = np.arange(Bulg_min,Bulg_max,dDl) # lens in bulge
    Dl_Disk = np.arange(Disk_min,Disk_max,dDl) # lens in disk 
    # Dl_all = cp.array(Dl_Disk)

    # len_Ds_Bulg = len(Ds_Bulg)
    # len_Dl_Bulg = len(Dl_Bulg)

    # calculate position in galactic-centric coordinate (x,y,z) 
    posits_Bulg = Ds_Bulg * direction[:,np.newaxis] + Earth[:,np.newaxis]
    posits_Disk = Ds_Disk * direction[:,np.newaxis] + Earth[:,np.newaxis]

    positl_Bulg = Dl_Bulg * direction[:,np.newaxis] + Earth[:,np.newaxis]
    positl_Disk = Dl_Disk * direction[:,np.newaxis] + Earth[:,np.newaxis]

    # pdfs_Bulg = gm.BulgeStellarDensity(posits_Bulg)*(Ds_Bulg**(2-gamma+Ds_profile))
    # pdfs_Disk = gm.DiskStellarDensity(posits_Disk,Rd=Rd)*(Ds_Disk**(2-gamma+Ds_profile))
    pdfs_Bulg = gm.BulgeStellarDensity(posits_Bulg)       *(Ds_Bulg**(2-gamma))
    pdfs_Disk = gm.DiskStellarDensity(posits_Disk, Rd=Rd) *(Ds_Disk**(2-gamma))
    cdfs_Bulg = np.cumsum(pdfs_Bulg)
    cdfs_Disk = np.cumsum(pdfs_Disk)
    # if bulge_source ==False:
    #     cdfs_Bulg = np.zeros_like(Ds_Bulg)
    # if disk_source  ==False:
    #     cdfs_Disk = np.zeros_like(Ds_Disk)
    Dss  = [  Ds_Bulg,  Ds_Disk]
    cdfDs = [cdfs_Bulg,cdfs_Disk]
    ratioDs = [float(cdfs_Bulg[-1]),float(cdfs_Disk[-1])]

    # sBulgRatio = np.sum(pdfs_Bulg)/(np.sum(pdfs_Bulg)+np.sum(pdfs_Disk))

    # pdfl_Bulg = gm.BulgeStellarDensity(positl_Bulg)*(Dl_Bulg**Dl_profile)
    # pdfl_Disk = gm.DiskStellarDensity(positl_Disk,Rd=Rd)*(Dl_Disk**Dl_profile)
    pdfl_Bulg = gm.BulgeStellarDensity(positl_Bulg)       *(Dl_Bulg**2)
    pdfl_Disk = gm.DiskStellarDensity(positl_Disk, Rd=Rd) *(Dl_Disk**2)
    cdfl_Bulg = np.cumsum(pdfl_Bulg)
    cdfl_Disk = np.cumsum(pdfl_Disk)
    # if bulge_lens ==False:
    #     cdfl_Bulg = np.zeros_like(Dl_Bulg)
    # if disk_lens  ==False:
    #     cdfl_Disk = np.zeros_like(Dl_Disk)
    Dls  = [  Dl_Bulg,  Dl_Disk]
    cdfDl = [cdfl_Bulg,cdfl_Disk]
    ratioDl = [float(cdfl_Bulg[-1]),float(cdfl_Disk[-1])]

    print('done',flush=True)




    # print('Calculating Ml probability distribution (PDF/CDF)...',end='    ',flush=True)
    # dM = 1.0/2000.0
    # # Ml_MS = np.arange(0.01,1.3,dM)
    # Ml_MS = np.arange(0.01, 3.0, dM)

    # Ml_WD = np.arange(0.15,1.5,dM)
    # #Ml_WD = cp.arange(0.60,0.72,dM) # for kb193289
    # Ml_NS = np.arange(0.60,3.2,dM)
    # Ml_BH = np.arange(1.50,100,dM)
    # cdfM_MS = np.cumsum(np.array( gm.MFMS(Ml_MS) ))
    # cdfM_WD = np.cumsum(np.array( gm.MFWD(Ml_WD) ))
    # #cdfM_WD = cp.cumsum(cp.array( np.ones_like(Ml_WD) )) # for kb193289
    # cdfM_NS = np.cumsum(np.array( gm.MFNS(Ml_NS) ))
    # cdfM_BH = np.cumsum(np.array( gm.MFBH(Ml_BH) ))
    # if main_sequence == False:
    #     cdfM_MS = np.zeros_like(Ml_MS)
    # if white_dwarf == False:
    #     cdfM_WD = np.zeros_like(Ml_WD)
    # if neutron_star == False:
    #     cdfM_NS = np.zeros_like(Ml_NS)
    # if black_hole  == False:
    #     cdfM_BH = np.zeros_like(Ml_BH)
    # Mls   = [  Ml_MS,  Ml_WD,  Ml_NS,  Ml_BH]
    # cdfMs = [cdfM_MS,cdfM_WD,cdfM_NS,cdfM_BH]
    # ratioMs = [float(cdfM_MS[-1]),float(cdfM_WD[-1]),float(cdfM_NS[-1]),float(cdfM_BH[-1])]


    # ### jiyuan ###
    # # Ml_MS_Bulg = np.arange(0.01,1.1,dM)#here is the difference from above
    # Ml_MS_Bulg = np.arange(0.01, 3.0, dM)

    # Ml_WD_Bulg = np.arange(0.15,1.5,dM)
    # Ml_NS_Bulg = np.arange(0.60,3.2,dM)
    # Ml_BH_Bulg = np.arange(1.50,100,dM)
    # cdfM_MS_Bulg = np.cumsum(np.array( gm.MFMS(Ml_MS_Bulg) ))
    # cdfM_WD_Bulg = np.cumsum(np.array( gm.MFWD(Ml_WD_Bulg) ))
    # cdfM_NS_Bulg = np.cumsum(np.array( gm.MFNS(Ml_NS_Bulg) ))
    # cdfM_BH_Bulg = np.cumsum(np.array( gm.MFBH(Ml_BH_Bulg) ))
    # if main_sequence == False:
    #     cdfM_MS_Bulg = np.zeros_like(Ml_MS_Bulg)
    # if white_dwarf == False:
    #     cdfM_WD_Bulg = np.zeros_like(Ml_WD_Bulg)
    # if neutron_star == False:
    #     cdfM_NS_Bulg = np.zeros_like(Ml_NS_Bulg)
    # if black_hole  == False:
    #     cdfM_BH_Bulg = np.zeros_like(Ml_BH_Bulg)
    # Mls_Bulg   = [  Ml_MS_Bulg,  Ml_WD_Bulg,  Ml_NS_Bulg,  Ml_BH_Bulg]
    # cdfMs_Bulg = [cdfM_MS_Bulg,cdfM_WD_Bulg,cdfM_NS_Bulg,cdfM_BH_Bulg]
    # ratioMs_Bulg = [float(cdfM_MS_Bulg[-1]),float(cdfM_WD_Bulg[-1]),float(cdfM_NS_Bulg[-1]),float(cdfM_BH_Bulg[-1])]
    # ###   end  ###

    # print('done',flush=True)

    # Build the read-only context we broadcast to workers:
    ctx =  (eventname, ndots, 
            Dss, cdfDs, ratioDs, Dls, cdfDl, ratioDl,
            Disk_min,
            Age_lens_min, Age_lens_max, MH_lens_min, MH_lens_max, 
            murel_hel_E_mean, murel_hel_E_err, 
            murel_hel_N_mean, murel_hel_N_err, 
            murel_hel_rhoEN, 
            AV_lens_min, AV_lens_max, RV_lens_mean, RV_lens_err)
    
    # -------- parallel map over ni --------
    ni_list = list(range(nstart, nstart+nrun))
    args_iter = [(ni, ctx) for ni in ni_list]

    # Use spawn for safety (works on Linux/Mac/Windows)
    with get_context("spawn").Pool(processes=n_process) as pool:
        # imap_unordered gives results as they finish
        for _ in pool.imap_unordered(simulate_one_run, args_iter, chunksize=1):
            pass
        
        
        
