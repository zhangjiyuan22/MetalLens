# MetalLens
Enabling Metallicity Constraints on Microlensing Lenses from Multi-band Lens Flux Measurements

We present a method that can, for the first time, routinely constrain the metallicities of the dominant population of microlensing lenses: low-mass stars located in the bulge and inner disk. 
Gravitational microlensing is uniquely sensitive to cold planets beyond the snow line around such hosts, but to date the metallicities of these distant low-mass lenses have remained essentially unconstrained. 
Our method exploits the fact that the angular Einstein radius, $\theta_{\rm E}$, plays an analogous role to the parallax in classical stellar work: when combined with multi-band photometry, it breaks the degeneracy between stellar mass, distance, metallicity, and extinction, allowing these quantities to be jointly constrained. 
Lens fluxes measured in three bands from high-resolution imaging are required, including a $V/R$-like band, an $I/z$-like band, and a $K_{\rm s}$-like band (e.g., Roman F062 + F087 + F213, with F062 replaceable by CSST $r$, Euclid VIS, or HST $V/R$-like filters). 
Mock-recovery experiments with our new forward-modeling code, \textsc{MetalLens}, show that, with $\theta_{\rm E}$ determined with $\sim 10 percent$ precision and lens colors measured with uncertainties of $\sigma_{(F062-F087)} \simeq 0.10$~mag and $\sigma_{(F087-F213)} \simeq 0.05$~mag, the metallicity can be recovered with a precision of $\sim 0.3$~dex for a $0.5\,M_\odot$ bulge lens. 
High-resolution imaging campaigns with Roman, CSST, and other facilities can therefore assemble a large sample of microlensing cold-planet systems with well-measured host-star metallicities, enabling the first robust tests of how the occurrence of cold planets across a wide range of masses around low-mass stars depends on host metallicity, and providing a new probe of planet-formation theories such as core accretion. 

## how to run a mock-recovery experiment?
1. run mock_an_event.py to derive the observables for a mock event (change mock event's physical parameters around line 40 to 60)
2. copy the derived observables and their uncertainties into config/mock1.cfg (line 93 for tE, line 122-124 for murel_hel, line 136 for Color_F062_F087_lens_value, line 141 for Color_F087_F213_lens_value, line 146 for F213_lens_value)
3. change n_process in config/mock1.cfg to the number of cores on your device
4. run simulate_events.py, which will simulate 5e9 simulated microlensing events, typically taking 2 min on a 96-core node
5. run cal_weights.py, which will weight the 5e9 simulated events by the observables of the mock event, typically taking 2 min on a 96-core node
6. run plot_1D_PDF.py to generate a plot, which contains all parameters' probability distribution and 16/50/84 percentile values; the plot is stored as output/mock1/single_lens/1D_PDF.pdf
