import configparser
import numpy as np

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class EventInfo:
    eventname = 'mock_event'
    ra, dec = 270,-30
    
    # VelocityDistributionModel = ''
    # vc = 220.
    Rd = 2.5
    
    # sigma_Vt = 33.
    # sigma_Vz = 18.
    # sigma_VR = 38.
    
    # EarthMotion = False
    
def getEventInfo(eventname,cfgfile=False):
    if cfgfile==False:
        filename = 'config/%s.cfg'%((eventname).lower())
    else:
        filename = cfgfile
    p = configparser.RawConfigParser()
    eventinfo = EventInfo()
    p.read(filename)
    # print(filename)
    
    eventinfo.fullname = p.get('Event Info','name')
    eventinfo.name = eventname
    eventinfo.model = p.get('Event Info','model')

    # Get event position
    ra_str = p.get('Event Info','ra')
    dec_str = p.get('Event Info','dec')
    ra = np.array(ra_str.split(':')).astype('float')
    eventinfo.ra = 15*(ra[0] + ra[1]/60 + ra[2]/3600)
    dec = np.array(dec_str.split(':')).astype('float')
    eventinfo.dec = np.sign(dec[0])*(abs(dec[0]) + dec[1]/60 + dec[2]/3600)
    
    # Get runnning options
    eventinfo.nstart = int( p.get('Running Options','nstart') )
    eventinfo.nrun   = int( p.get('Running Options','nrun') )
    eventinfo.ndot   = int( p.get('Running Options','ndot') )
    eventinfo.n_process  = int( p.get('Running Options','n_process') )
    # eventinfo.Dl_profile = float( p.get('Running Options','Dl_profile') )
    # eventinfo.Ds_profile = float( p.get('Running Options','Ds_profile') )
    
    # Get weight options
    eventinfo.use_tE = str2bool( p.get('Weight Options','use_tE') )
    # try:
    #     eventinfo.tEweight_file = str2bool( p.get('Weight Options','tEweight_file') )
    # except configparser.NoOptionError:
    #     eventinfo.tEweight_file = False
    if eventinfo.use_tE==True: # and eventinfo.tEweight_file==False:
        eventinfo.tEmean,eventinfo.tEerr = [float(pi) for pi in p.get('Weight Options','tEvalue').split()]
    # if eventinfo.tEweight_file==True:
    #     eventinfo.tEweight_filepath = p.get('Weight Options','tEweight_filepath')
    
    eventinfo.use_piE = str2bool( p.get('Weight Options','use_piE') )
    if eventinfo.use_piE==True:
        eventinfo.piE_E_mean, eventinfo.piE_E_err = [float(pi) for pi in p.get('Weight Options','piE_E_value').split()]
        eventinfo.piE_N_mean, eventinfo.piE_N_err = [float(pi) for pi in p.get('Weight Options','piE_N_value').split()]
        eventinfo.piE_rhoEN = float( p.get('Weight Options','piE_rhoEN') )

    # eventinfo.use_thetaE = str2bool( p.get('Weight Options','use_thetaE') )
    # if eventinfo.use_thetaE==True:
    #     eventinfo.thetaE_upperlimit = str2bool( p.get('Weight Options','thetaE_upperlimit') )
    #     eventinfo.rhomean,eventinfo.rhoerr = [float(pi) for pi in p.get('Weight Options','rhovalue').split()]
    #     eventinfo.thetasmean,eventinfo.thetaserr = [float(pi) for pi in p.get('Weight Options','thetasvalue').split()]
    #     eventinfo.thetaEweight_file = str2bool( p.get('Weight Options','thetaEweight_file') )
    #     eventinfo.thetaEweight_filepath = str(p.get('Weight Options','thetaEweight_filepath'))
    #     #eventinfo.Rstarmean,eventinfo.Rstarerr = [float(pi) for pi in p.get('Weight Options','Rstarvalue').split()]
    # if eventinfo.use_thetaE==True and eventinfo.thetaEweight_file==False:
    #     eventinfo.thetaEmean,eventinfo.thetaEerr = [float(pi) for pi in p.get('Weight Options','thetaEvalue').split()]

    eventinfo.use_murel_hel = str2bool( p.get('Weight Options','use_murel_hel') )
    if eventinfo.use_murel_hel == True :
        eventinfo.murel_hel_E_mean, eventinfo.murel_hel_E_err = [float(pi) for pi in p.get('Weight Options','murel_hel_E_value').split()]
        eventinfo.murel_hel_N_mean, eventinfo.murel_hel_N_err = [float(pi) for pi in p.get('Weight Options','murel_hel_N_value').split()]
        eventinfo.murel_hel_rhoEN = float( p.get('Weight Options','murel_hel_rhoEN') )
        # eventinfo.use_murel_hel_direction = str2bool( p.get('Weight Options','use_murel_hel_direction') )

    eventinfo.use_Color_F062_F087_lens = str2bool( p.get('Weight Options','use_Color_F062_F087_lens') )
    if eventinfo.use_Color_F062_F087_lens == True : 
        eventinfo.Color_F062_F087_lens_mean, eventinfo.Color_F062_F087_lens_err = [float(pi) for pi in p.get('Weight Options','Color_F062_F087_lens_value').split()]

    eventinfo.use_Color_F087_F213_lens = str2bool( p.get('Weight Options','use_Color_F087_F213_lens') )
    if eventinfo.use_Color_F087_F213_lens == True : 
        eventinfo.Color_F087_F213_lens_mean, eventinfo.Color_F087_F213_lens_err = [float(pi) for pi in p.get('Weight Options','Color_F087_F213_lens_value').split()]

    eventinfo.use_F213_lens = str2bool( p.get('Weight Options','use_F213_lens') )
    if eventinfo.use_F213_lens == True : 
        eventinfo.F213_lens_mean, eventinfo.F213_lens_err = [float(pi) for pi in p.get('Weight Options','F213_lens_value').split()]



    eventinfo.Color_F062_F087_isochrone_uncertainty = float( p.get('Isochrone Uncertainty','Color_F062_F087_isochrone_uncertainty') )
    eventinfo.Color_F087_F213_isochrone_uncertainty = float( p.get('Isochrone Uncertainty','Color_F087_F213_isochrone_uncertainty') )



    eventinfo.binary_lens = str2bool( p.get('Binary Lens','binary_lens') )
    if eventinfo.binary_lens == True :
        eventinfo.q_mean, eventinfo.q_err = [float(pi) for pi in p.get('Binary Lens','q_value').split()]
        eventinfo.s_mean, eventinfo.s_err = [float(pi) for pi in p.get('Binary Lens','s_value').split()]

    # Get Galactic Components
    # eventinfo.bulge_source = str2bool( p.get('Galactic Component','bulge_source') )
    # eventinfo.disk_source  = str2bool( p.get('Galactic Component','disk_source')  )
    # eventinfo.bulge_lens   = str2bool( p.get('Galactic Component','bulge_lens')   )
    # eventinfo.disk_lens    = str2bool( p.get('Galactic Component','disk_lens')    )
    eventinfo.Disk_min = float(p.get('Galactic Component','Disk_min'))
    eventinfo.Disk_max = float(p.get('Galactic Component','Disk_max'))
    # eventinfo.lDisk_min = float(p.get('Galactic Component','lDisk_min'))
    # eventinfo.lDisk_max = float(p.get('Galactic Component','lDisk_max'))
    eventinfo.Bulg_min = float(p.get('Galactic Component','Bulg_min'))
    eventinfo.Bulg_max = float(p.get('Galactic Component','Bulg_max'))
    # eventinfo.SourceFromPM = str2bool( p.get('Galactic Component','SourceFromPM') )
    # if eventinfo.SourceFromPM == True:
    #     eventinfo.muslmean,eventinfo.muslerr = [float(pi) for pi in p.get('Galactic Component','musl').split()]
    #     eventinfo.musbmean,eventinfo.musberr = [float(pi) for pi in p.get('Galactic Component','musb').split()]
    #     eventinfo.muscor = float( p.get('Galactic Component','muscor') )

    
    # Get Lens type (Main-sequence, WD, NS, BH)
    # eventinfo.main_sequence = str2bool( p.get('Mass Function','main_sequence') )
    # eventinfo.white_dwarf   = str2bool( p.get('Mass Function','white_dwarf')   )
    # eventinfo.neutron_star  = str2bool( p.get('Mass Function','neutron_star')  )
    # eventinfo.black_hole    = str2bool( p.get('Mass Function','black_hole')    )
    # eventinfo.max_main_sequence_mass_bulge = float(p.get('Mass Function','max_main_sequence_mass_bulge'))
    # eventinfo.max_main_sequence_mass_disk  = float(p.get('Mass Function','max_main_sequence_mass_disk'))

    # Get lens age and metallicity prior type (informative, uniform)
    # eventinfo.Lens_Age_Metallicity_Prior = p.get('Lens Age Metallicity Prior', 'Lens_Age_Metallicity_Prior')
    eventinfo.Age_lens_min = float(p.get('Lens Age Metallicity','Age_lens_min'))
    eventinfo.Age_lens_max = float(p.get('Lens Age Metallicity','Age_lens_max'))
    eventinfo.MH_lens_min  = float(p.get('Lens Age Metallicity','MH_lens_min'))
    eventinfo.MH_lens_max  = float(p.get('Lens Age Metallicity','MH_lens_max'))

    eventinfo.AV_lens_min  = float(p.get('Extinction','AV_lens_min'))
    eventinfo.AV_lens_max  = float(p.get('Extinction','AV_lens_max'))
    eventinfo.RV_lens_mean, eventinfo.RV_lens_err = [float(pi) for pi in p.get('Extinction','RV_lens_value').split()]


    # Get output options (only effect the output format)
    # eventinfo.SolarMotion = str2bool(p.get('Output Options','SolarMotion'))
    # eventinfo.EarthMotion = str2bool(p.get('Output Options','EarthMotion'))
    # eventinfo.Earth_Vl = float(p.get('Output Options','Earth_Vl'))
    # eventinfo.Earth_Vb = float(p.get('Output Options','Earth_Vb'))
    # eventinfo.Solar_Vl = float(p.get('Output Options','Solar_Vl'))
    # eventinfo.Solar_Vb = float(p.get('Output Options','Solar_Vb'))
    # eventinfo.v_earth_E = float(p.get('Output Options','v_earth_E'))
    # eventinfo.v_earth_N = float(p.get('Output Options','v_earth_N'))
    eventinfo.t0_for_velocity = float(p.get('Frame Transform','t0_for_velocity'))
    
    # Get Disk Model parameters
    eventinfo.gamma = float(p.get('Source Profile','gamma'))
    
    # eventinfo.DiskModel = p.get('Disk Model','VelocityDistribution')
    # eventinfo.vc = float(p.get('Disk Model','vc'))
    eventinfo.Rd = float(p.get('Disk Model','Rd'))
    # eventinfo.sigma_Vt = float(p.get('Disk Model','sigma_Vt'))
    # eventinfo.sigma_Vz = float(p.get('Disk Model','sigma_Vz'))
    # eventinfo.sigma_VR = float(p.get('Disk Model','sigma_VR'))
    
    
    return eventinfo
    
# if __name__=='__main__':
#     config = EventInfo()
#     config = getEventInfo('mock_event')
