# MetalLens
Enabling Metallicity Constraints on Microlensing Lenses from Multi-band Lens Flux Measurements

## how to run a mock-recovery experiment?
1. run mock_an_event.py to derive the observables for a mock event (change mock event's physical parameters around line 40 to 60)
2. copy the derived observables and their uncertainties into config/mock1.cfg
3. change n_process in config/mock1.cfg to the number of cores on your device
4. run simulate_events.py, which will simulate 5e9 simulated microlensing events, typically taking 2 min on a 96-core node
5. run cal_weights.py, which will weight the 5e9 simulated events by the observables of the mock event, typically taking 2 min on a 96-core node
6. run plot_1D_PDF.py to generate a plot, which contains all parameters' probability distribution and 16/50/84 percentile values
