import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import scipy as sp
import optuna as op
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

'''
Class for calibratopn process via Optuna 
(read tutorial in Covasim documentation https://docs.idmod.org/projects/covasim/en/latest/tutorials/t7.html and Optuna documentation https://optuna.readthedocs.io/en/stable/https://optuna.readthedocs.io/en/stable/)
'''

class Calibration:


    def __init__(self, storage, pdict, location, pop_location, start_day, end_day, 
                 datafile, cal, cal_keys, cal_values, n_trials=100, n_workers=1, n_runs=7,
                 school_days=None, school_changes=None, to_plot = ['new_diagnoses',  'new_deaths', 'n_critical']):
        '''
        Args:
            storage - storage for Optuna
            pdict - dictionary with parameters bounds
            location - city/region/country (string)
            pop_location - population in location
            start_day - start day of modellibg
            end_day - end day of modelling
            datafile - smoothed statistics
            cal - list with previous calibrated parameters
            cal_keys - which statistics to include in functional
            cal_values - weights for cal_keys
            school_days, school_changes - for school intervention (if provided)
        '''
        # Settings
        self.pop_size = 100e3 # Number of agents
        self.location=location
        self.pop_location=pop_location
        self.start_day = start_day
        self.end_day = end_day
        self.datafile = datafile


        # Saving and running
        self.n_trials  = n_trials # Number of sequential Optuna trials
        self.n_workers = n_workers # Number of parallel Optuna threads -- incompatible with n_runs > 1
        self.n_runs    = n_runs # Number of sims being averaged together in a single trial -- incompatible with n_workers > 1
        self.storage   = storage # Database location
        self.name      = 'covasim' # Optuna study name -- not important but required
        
        # For school interventions
        self.school_days=school_days
        self.school_changes=school_changes 
        
        # For calibration
        self.cal_keys=cal_keys        # keys of calibrated statistics (dict)
        self.cal_values=cal_values    # values of calibrated statistics (dict)
        self.pdict = pdict            # bounds for parameters
        self.cal = cal                # list of lists of calibrated parameters
        
        assert self.n_workers == 1 or self.n_runs == 1, f'Since daemons cannot spawn, you cannot parallelize both workers ({self.n_workers}) and sims per worker ({self.n_runs})'

        # Control plotting
        self.to_plot = to_plot



    def create_sim(self, x, verbose=0):
        ''' Create the simulation from the parameters '''

        
        if isinstance(x, dict):
            pars, pkeys = self.get_bounds() # Get parameter guesses
            x = [x[k] for k in pkeys]
        

        # Define and load the data
        self.calibration_parameters = x
        
        # Parameters
        assert len(x) == len(self.pdict), 'shape of x and pdict does not match'
        
        # First calibration consists of 5 parameters
        if len(x) == 5:
            pop_infected = x[0]
            beta         = x[1]
            beta_day     = x[2]
            beta_change  = x[3]
            symp_test    = x[4]

            pars = dict(
                pop_size     = self.pop_size,
                pop_scale    = self.pop_location/self.pop_size,
                pop_infected = pop_infected,
                pop_type = 'hybrid',
                beta = beta,
                start_day = self.start_day,  
                end_day   = self.end_day, 
                location=self.location,
                verbose=0,
                rescale = True)
            
            # Create the sim
            sim = cv.Sim(pars, datafile=self.datafile)
            interventions = [cv.change_beta(days=beta_day, changes=beta_change), # different beta_changes
                        cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=symp_test) # testing routine
                        ]
            if self.school_days is not None and self.school_changes is not None:
                interventions+=[cv.clip_edges(days=self.school_days, changes=self.school_changes, layers='s')] # schools closed 
            
        # Other calibrations consist of 2 parameters    
        if len(x) == 2:
            beta_day    = x[0]
            beta_change = x[1]
            symp_test=self.cal[0][4]
            
            
            pars = dict(
                pop_size    = self.pop_size,
                pop_scale    = self.pop_location/self.pop_size,
                pop_infected = self.cal[0][0],
                pop_type = 'hybrid',
                beta = self.cal[0][1],
                start_day = self.start_day,  
                end_day   = self.end_day, 
                location=self.location,
                verbose=0,
                rescale = True)
            # Create the sim
            sim = cv.Sim(pars, datafile=self.datafile)
    
            # Add interventions
            interventions = [cv.change_beta(days=self.cal[i][2], changes=self.cal[i][3]) 
                              for i in range(len(self.cal))]
            interventions += [cv.change_beta(days=beta_day, changes=beta_change)]
            interventions += [cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=symp_test)]
            
            if self.school_days is not None and self.school_changes is not None:
                interventions+=[cv.clip_edges(days=self.school_days, changes=self.school_changes, layers='s')] # schools closed
        # Update
        sim.update_pars(interventions=interventions)

        self.sim = sim

        return sim


    def get_bounds(self):
        ''' Set parameter starting points and bounds -- NB, only lower and upper bounds used for fitting '''

        # Convert from dicts to arrays
        pars = sc.objdict()
        for key in ['best', 'lb', 'ub']:
            pars[key] = np.array([v[key] for v in self.pdict.values()])

        return pars, self.pdict.keys()


    def smooth(self, y, sigma=3):
        ''' Optional smoothing if using daily death data '''
        return sp.ndimage.gaussian_filter1d(y, sigma=sigma)


    
    def run_msim(self):
        if self.n_runs == 1:
            sim = self.sim
            sim.run()
        else:
            msim = cv.MultiSim(base_sim=self.sim)
            msim.run(n_runs=self.n_runs)
            sim = msim.reduce(output=True)
            
        weights={self.cal_keys[i] : self.cal_values[i] for i in range(len(self.cal_keys))}
        sim.compute_fit(keys=self.cal_keys, weights=weights, output=False)
        self.sim = sim
        self.mismatch = sim.results.fit.mismatch
         
        return sim

    # Functions for Optuna
    def objective(self, x):
        ''' Define the objective function we are trying to minimize '''
        self.create_sim(x=x)
        self.run_msim()
        return self.mismatch


    def op_objective(self, trial):
        ''' Define the objective for Optuna '''
        pars, pkeys = self.get_bounds() # Get parameter guesses
        x = np.zeros(len(pkeys))
        for k,key in enumerate(pkeys):
            x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

        return self.objective(x)

    def worker(self):
        ''' Run a single Optuna worker '''
        study = op.load_study(storage=self.storage, study_name=self.name)
        return study.optimize(self.op_objective, n_trials=self.n_trials)


    def run_workers(self):
        ''' Run allworkers -- parallelized if each sim is not parallelized '''
        if self.n_workers == 1:
            self.worker()
        else:
            sc.parallelize(self.worker, self.n_workers)
        return


    def make_study(self):
        try: op.delete_study(storage=self.storage, study_name=self.name)
        except: pass
        return op.create_study(storage=self.storage, study_name=self.name)


    def load_study(self):
        return op.load_study(storage=self.storage, study_name=self.name)


    def get_best_pars(self, print_mismatch=True):
        ''' Get the outcomes of a calibration '''
        study = self.load_study()
        output = study.best_params
        if print_mismatch:
            print(f'Mismatch: {study.best_value}')
        return output


    def calibrate(self):
        ''' Perform the calibration '''
        self.make_study()
        self.run_workers()
        output = self.get_best_pars()
        return output


    def save(self):
        pars_calib = self.get_best_pars()
        sc.savejson(f'calibrated_parameters_{self.until}_{self.state}.json', pars_calib)


# Modelling after calibration
        
def model(cal):
    
    recalibrate = True # Whether to run the calibration
    do_plot     = True # Whether to plot results
    verbose     = 0.1 # How much detail to print

   

    # Plot initial
    if do_plot:
        print('Running initial uncalibrated simulation...')
        pars, pkeys = cal.get_bounds() # Get parameter guesses
        sim = cal.create_sim(pars.best, verbose=verbose)
        sim.run()
        sim.plot(to_plot=cal.to_plot)
        pl.gcf().suptitle('Initial parameter values')
        cal.objective(pars.best)
        pl.pause(1.0) # Ensure it has time to render

    # Calibrate
    if recalibrate:
        print(f'Starting calibration for Novosibirsk')
        T = sc.tic()
        pars_calib = cal.calibrate()
        sc.toc(T)
    else:
        pars_calib = cal.get_best_pars()

    # Plot result
    if do_plot:
        print('Plotting result...')
        x = [pars_calib[k] for k in pkeys]
        sim = cal.create_sim(x, verbose=verbose)
        sim.run()
        sim.plot(to_plot=cal.to_plot)
        pl.gcf().suptitle('Calibrated parameter values')
    return sim, pars_calib   



