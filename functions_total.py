# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:26:38 2021

@author: sosma
"""


'''
File with functions for calibration, running and plotting of the sim

@author: sosma
'''
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pylab as pl
import sciris as sc
import scipy as sp
import covasim as cv
import optuna as op
import pandas as pd
from datetime import datetime, date, timedelta

from tqdm import tqdm
from func import *
from SARIMAX import *

import warnings
warnings.filterwarnings("ignore")

import calibration_total as st
from calibration_total import model

# Supporting functions 

def smooth(y):
    '''
    Function for smoothing pd.Seria
    '''
    return sp.ndimage.gaussian_filter1d(y, sigma=3)
def smooth_pd(df):
    '''
    Function for smoothing pd.DataFrame
    '''
    for col in df.columns:
        df[col]=smooth(df[col])
    return df

def past_extr(df,series,n,C=1.1):
    '''
    Function for filling unknown tests at the beggining of modeling
    Args:
        df - dataframe with statistics
        series - statistic to extrapolate (for example df['new_tests'])
        n - how many points to extrapolate (better to fill Nans: n=df['new_tests'].isna().sum())
        C - parameter for rolling (default = 1.1)
    '''
    back = series[::-1]
    back.index = np.arange(back.shape[0])
    for i in range(back.isna().sum()):
        back = back.fillna(back.rolling(n).apply(lambda x: x.mean()*C**-i).shift())
    back.index = df.index[::-1]
    back = round(back)[::-1]
    return back


def future_extr(filename, end_day, n_future, n_past=2, dday=None):
    '''
    Function for filling unknown tests in future
    Args:
        filename - unsmoothed dataframe with statistics
        end_day - date until we know number of new tests
        n_future - how many points to extrapolate (better to fill Nans: n=df['new_tests'].isna().sum())'
        
    '''
   
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    if dday is not None:
        df=df[:dday]
    series = df['new_tests'].dropna()
    date = cv.daydiff(pd.to_datetime(series.index[0]).date(), end_day)+1
    n = n_past+n_future
    
    
    lmb = None
    if series.min() <= 0:
        new_tests_box, lmb = scs.boxcox(series + abs(series.min()) + 1, lmbda=lmb) # прибавляем единицу, так как в исходном ряде есть нули
    else:
        new_tests_box,lmb = scs.boxcox(series, lmbda=lmb)
    new_tests_box = pd.Series(new_tests_box,index=series.index)
    
    d = 1
    D = 1
    parameters_list = [1,2,0,1]
    model_SARIMAX, params_SARIMAX, aic_SARIMAX = best_SARIMAX(new_tests_box,d,D,n_past,parameters_list)
    
    forecast_SARIMAX = get_predict(series,model_SARIMAX, n_past, n_future, lmb)
    forecast=forecast_SARIMAX[date:]
    
    return forecast
    

def bounds_of_per(start_day, end_day, window=30):
    days_in_df=cv.daydiff(start_day, end_day)
    nmb_of_periods=int(days_in_df/window)
    bounds_of_periods=[]
    for i in range(1,nmb_of_periods+1):
        bounds_of_periods.append(cv.date(window*i,start_date=start_day,as_date=False))
    return bounds_of_periods


# Calibration 
def if_new_calib(previous_cal_day, timedelt):
    
    '''
    Function for defining if we need to run new calibration
    Args:
        previous_cal_day - previous calibration day
        timedelt - period of recalibration 
    '''
    
    print("Previous calibration day: ", previous_cal_day)
    
    # day today
    dt=datetime.today()
    day_today=cv.date(date(dt.year, dt.month, dt.day),as_date=False)
    print("Date today : ", day_today)
    
    
    # define how many days from the previous calibration passed
    days_no_cal=cv.daydiff(previous_cal_day, day_today)
    print("Days from the previous calibration passed: ",days_no_cal)
    if days_no_cal > timedelt:
        run_calibration = True
        print("More than",timedelt, "days between calibrations - let's update the parameters!")
        previous_cal_day = day_today
    else:
        run_calibration = False
        print("Less than",timedelt, "days between calibrations - run the model!")
    return previous_cal_day, day_today, run_calibration

def calibration_process(param, location, pop_location, datafile, start_day, end_day, pop_inf, cal_keys, cal_values,
                        beta_ch=[0.5,0.3,1.2], school_days=None, school_changes=None, print_cal_param=False):
    '''
     Function for calibration parameters from new period
     Args:
         param - previous calibrated parameters
         datafile - smoothed dataset with statistics
         start_day - day to start modelling
         end day - day until to calibrate
         pop_inf - list of bounds (best, upper bound, lower bound) for parameter 'pop_infected'. Example: pop_inf=[20, 1, 100]
         beta_ch - list of bounds (best, upper bound, lower bound) for parameter 'beta_change'. Can be modified.
         school_days, school_changes - for school intervention (if provided)
         n_trials, n_runs - for Optuna, read documentation 
         print_cal_param - to print calibrated parameters or not
    '''
    
    sm_data=datafile.copy() # smoothed datafile
    start_day=start_day
    end_day=end_day
    
    print("Modeling ", start_day , " - ", end_day)
    
    b_day_ubound=cv.daydiff(start_day,end_day) # end day is an upper bound for calibration
    
    # cal_param - special list of lists for auto calibration
    if not param: # if list of params is empty -- this is the first calibration 
        cal_param=[]
    else:
        cal_param=[[param[0]['pop_infected'],param[0]['beta'],param[0]['beta_day_1'],param[0]['beta_change_1'],param[0]['symp_test']]]
        for i in range(len(param)-1):
            cal_param.append([None,None,param[i+1][f'beta_day_{i+2}'],param[i+1][f'beta_change_{i+2}'], None])
        b_day_lbound=list(param[-1].values())[-1] # previous beta_day is a lower bound for calibration
        b_day_best=(b_day_lbound+b_day_ubound)/2    
    
    if print_cal_param:
        print(cal_param)
    storage = f'sqlite:///calibration.db' # Optuna database location

    if b_day_ubound < datafile.shape[0]:  # if datafile has data after end day - cut it
        sm_data=sm_data[:b_day_ubound+1]
    
    # new calibrated parameters
    if not param:
        pdict = sc.objdict(
            pop_infected = dict(best=pop_inf[0], lb=pop_inf[1],  ub=pop_inf[2]),
            beta         = dict(best=0.016, lb=0.01, ub=0.025),
            beta_day     = dict(best=25,    lb=1,    ub=b_day_ubound),
            beta_change  = dict(best=beta_ch[0],   lb=beta_ch[1],   ub=beta_ch[2]),
            symp_test    = dict(best=30,    lb=1,     ub=200)
        )
    else:
        pdict = sc.objdict(
                beta_day = dict(best=b_day_best,    lb=b_day_lbound,    ub=b_day_ubound),
                beta_change = dict(best=beta_ch[0],   lb=beta_ch[1],   ub=beta_ch[2]))
        
    # Create calibration
    calibr = st.Calibration(storage=storage, pdict=pdict, location=location, pop_location=pop_location, cal_keys=cal_keys,cal_values=cal_values,
			     end_day=end_day, start_day=start_day,
                            school_days=school_days, school_changes=school_changes, datafile=sm_data, 
                            cal=cal_param,)
    sim, pars_calib=model(calibr)
    param.append(pars_calib)
    
    param[-1][f'beta_change_{len(param)}']=param[-1].pop('beta_change')
    param[-1][f'beta_day_{len(param)}']=param[-1].pop('beta_day')
    par=param.copy()
    
    return sim, par



# Running and plotting 
    
def run_model(p, location, pop_location, start_day, end_day, b_days, b_changes, data, run, 
              school_days=None, school_changes=None,
              to_plot=['new_diagnoses','new_deaths', 'new_infections']):
    
    '''
     Function for running Covasim model
     Args:
         p - calibrated parameters of the model
         location - city/region/country (string)
         pop_location - population in location
         start_day - day to start modelling
         end day - day or date to run until
         b_days - list of days of beta_change (got from list 'p')
         b_changes - list of beta_change (got from list 'p')
         data - smoothed dataset with statistics
         run - wheter to run sim or not 
         school_days, school_changes - for school intervention (if provided)
         to_plot - list to plot
    '''
    
    
    datafile = data
    pop_size = 100e3
    pars = dict(
        pop_size = pop_size,
        pop_scale = pop_location/pop_size,
        pop_infected = p[0]['pop_infected'],
        pop_type = 'hybrid',
        beta = p[0]['beta'],
        start_day = start_day,  
        end_day   = end_day,  
        location=location,
        verbose=0.1,
        rescale = True,
        )
    sim = cv.Sim(pars, datafile=datafile)

  
    interventions = [ cv.change_beta(days=b_days, changes=b_changes), # different beta_changes
                        cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=p[0]['symp_test']) # testing routine
                        ]
    if school_days is not None and school_changes is not None:
        interventions+=[cv.clip_edges(days=school_days, changes=school_changes, layers='s')] # schools closed 
           
    sim.update_pars(interventions=interventions)
    if run:
        sim.run()
        sim.plot(to_plot=to_plot)
    return sim


# Plotting 
    
def plot_reff(sim, figsize=(18,5), fontsize=18, linewidth=2, ticks=50, rotation=0, dday=-1):
    '''
     Function for plotting effective reproduction number R(t)
     Args:
         sim - already run sim from Covasim
         others - parameters for mathplotlib
    '''
    r_eff=sim.results['r_eff'].values[:dday]
    r_eff=pd.Series(r_eff,index=sim.datafile.index[:dday])
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Effective reproduction number',{'fontsize': fontsize})
    plt.hlines(1,sim['start_day'],sim['end_day'], colors='red', linestyles='dashed')
    ax.plot(r_eff.index,r_eff,color='black', linewidth=linewidth)
    plt.xticks(r_eff.index,rotation=rotation,fontsize=fontsize);
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks))
    plt.yticks(fontsize=fontsize)
    plt.show()
    plt.savefig('r_eff.eps', format='eps')
    return fig
    

def plot_prop_diagnosed(sim, df1, figsize=(18,5), fontsize=18, fontsize_legend=15, linewidth=2, ticks=50, rotation=0, dday=-1):
    '''
     Function for plotting proportion of diagnosed people out of tested
     Args:
         sim - already run sim from Covasim
         df1 - smoothed statistics (for comparison)
         others - parameters for mathplotlib
    '''
    
    new_prop=sim.results['new_diagnoses'].values[:dday]/sim.results['new_tests'].values[:dday]
    new_prop=pd.Series(new_prop,index=df1.index)
    
    df1['new_prop']=df1['new_diagnoses'][:dday]/df1['new_tests'][:dday]
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Proportion of diagnosed',{'fontsize': fontsize})
    plt.xlabel('Date',{'fontsize': fontsize})
    plt.ylabel('Proportion',{'fontsize':fontsize})
    ax.plot(new_prop.index,new_prop,color='red',linewidth=linewidth,markersize=3, label='model')
    ax.plot(new_prop.index,df1['new_prop'],color='navy',linewidth=linewidth,markersize=3, label='data')
    plt.xticks(new_prop.index,rotation=rotation,fontsize=fontsize);
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks))
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.savefig('prop_diagnosed.eps', format='eps')

    plt.show()
    
    
def plot_interventions(df, b_days, dates,save,
                       figsize=(18,5), fontsize=18, fontsize_legend=15, linewidth=2, ticks=50, rotation=0, dday=-1):
    '''
    Function for plotting real interventions (blue) and model interventions (red).
    Args:
        df - statistical data (not smoothed)
        b_change_model - list of beta_change parameters (model)
        dates - dates with interventions in your location (real)
   
    '''
    if dday !=-1:
        for i in range(len(b_days)):
            if dday<b_days[i]:
                for j in range(len(b_days)-i):
                    del b_days[-1]
                break
    
    b_change_model=cv.date(b_days,start_date=start_day, as_date=False) # dates of beta_days
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Number of new diagnoses',{'fontsize':fontsize})
    
    ax.plot(df.index[:dday], df.new_diagnoses[:dday],color='black', linewidth=linewidth)
    for date in dates:
        if date==dates[0]:
            plt.vlines(date,0,df['new_diagnoses'].max(),linestyles='dashdot',color='blue',label='Real inteventions')
        else:
            plt.vlines(date,0,df['new_diagnoses'].max(),linestyles='dashdot',color='blue')
    for date in b_change_model:
        if date==b_change_model[0]:
            plt.vlines(date,0,df['new_diagnoses'].max(),linestyles='--',color='red', label='Model inteventions')
        else:
            plt.vlines(date,0,df['new_diagnoses'].max(),linestyles='--',color='red')
    plt.legend(fontsize=fontsize_legend)
    plt.xticks(df.index[:dday],rotation=rotation,fontsize=fontsize);
    plt.yticks(fontsize=fontsize)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks))
    if save:
        plt.savefig('interv_plot.eps', format='eps')

    plt.show()
    
    
    
def run_msim_conf(sim, to_plot, n_runs=10, save=False, namemsim=None, plot=True):
    '''
    Function for calculation of confidential interval
    Args:
        sim - already initialized sim from Covasim
        n_runs - how many sims to average
        to_plot - list of statistics to plot
   
    '''
    msim = cv.MultiSim(sim, n_runs = n_runs,verbose=0) 
    msim.run()
    msim.reduce()
    if save:
        msim.save(namemsim)
    print('msim successfully run')
    if plot:
        msim.plot(to_plot=to_plot)
    
def prognose_check(df1, data_csv, location, pop_location, start_day, date1, date2, n_params, p, b_days, b_changes,
                    msimname, school_days=None, school_changes=None, run_sim=False):
    '''
   Function to check the-goodness-of prognose on the historical data
   Args:
       df1 - the whole statistical data until today (smoothed)
       data_csv - name of dataset file
       location - city/region/country (string)
       pop_location - population in location
       start_day - start day of modellibg
       date1 - date of the end of modelling period (example: '2020-09-01')
       date2 - date  of the end of testing period (example: '2020-10-01')
       n_params - number of parameters from p to use for this period of historical data
       p - calibrated parameters
       statistics- which statistic to prognose
       school_days - for this period of historical data (optional)
       school_changes - for this period of historical data (optional)
       run_sim - to run the sim or not (False if need to run msim)
    '''
    date_1 = cv.daydiff(start_day, date1)+1  # convert dates into days (returns number of days from start_day to date1)
    date_2 = cv.daydiff(start_day, date2)+1 
    if date_2<date_1:
        print('Error:', date1, 'must be before', date2)
        
    model_data=df1.copy()
    model_data=model_data[:date_1]
    
    n_future=date_2-date_1 
    forecast=future_extr(filename=data_csv, df1=df1, end_day=date1, n_future=n_future, dday=date_1)

    
    forecast=pd.Series(smooth(forecast),index=[df1.index[-1] + timedelta(days=i) for i in range(1, n_future+2)])
    forecast.name ='new_tests'
    forecast = forecast.to_frame()
    
    
    forecast_data=pd.concat([df1,forecast])
    forecast_data['date']=forecast_data.index
    
    p=p[:n_params]
    b_days=b_days[:n_params]
    b_changes=b_changes[:n_params]

    sim=run_model(p=p, location=location, pop_location=pop_location, start_day=start_day, end_day=date2, 
                  b_days=b_days, b_changes=b_changes, data=forecast_data, run=run_sim,
                  school_days=school_days, school_changes=school_changes)
    
    msim = cv.MultiSim(sim, n_runs = 10) # confidential interval
    msim.run()
    msim.save(msimname)


def plot_prognose(df1, start_day, date1, date2, msimname, statistics, savename, n_params, b_days,
                   color_stat, name_stat,
                   school_days=None, figsize=(18,5), fontsize=18, fontsize_legend=15, linewidth=2, 
                   ticks=50, rotation=0):
    '''
    Function for plotting prognose of new diagnoses on the historical data
    Args:
        model_data, mod_data, tests_data, df_test - all from  def prognose(..)
        low - array with 90% quantile of modelling (from def prognose(..), need for confidential interval)
        high - array with 10% quantile of modelling (from def prognose(..), need for confidential interval)
        school_days - close or open school (dates)
        b_dates - changes of beta_parameters
    '''
    b_days=b_days[:n_params]
    date_1 = cv.daydiff(start_day, date1)+1
    date_2 = cv.daydiff(start_day, date2)+1 
    test=df1[:date_2]
    msim=cv.load(msimname)
    low, high=msim.reduce(low_high=True)    
    mod=pd.Series(smooth(msim.results[statistics]),index=test['date'])
    mod_data=mod[:date_1+1]
    tests_data=mod[date_1:date_2]
    df_test=test[statistics]
    b_dates=cv.date(b_days,start_date=start_day, as_date=False)
    if school_days is not None:
        sch_dates=school_days
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(f'Number of {name_stat}',{'fontsize': fontsize})
    #lt.xlabel('Date',{'fontsize': fontsize})
    ax.fill_between(test.index, smooth(low), smooth(high), alpha=0.2, color=color_stat)
    ax.plot(mod_data.index,mod_data,color='black', linewidth=linewidth, label='Model data')
    ax.plot(tests_data.index,tests_data,color='red', linewidth=linewidth, label='Test data',linestyle='--')
    ax.scatter(test.index,df_test,color=color_stat, s=10, label='Real data')
    # for date in sch_dates:
    #      plt.vlines(date,0,high.max(),linestyles='--',color='green')
    # for date in b_dates:
    #      plt.vlines(date,0,high.max(),linestyles='--',color='black')
    plt.xticks(test.index,rotation=rotation,fontsize=fontsize);
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks))
    plt.legend(fontsize=fontsize_legend)
    plt.yticks(fontsize=fontsize)
    plt.show()
    plt.savefig(savename, format='eps')

    
def prognose(forecast_data, start_day, location, pop_location, p, to_plot, b_days, b_changes, 
             school_days=None, school_changes=None, run_sim=False, n_runs=10, save=False, namemsim=None, plot=True):
    '''
   Function to do prognose.
   Args:
       forecast_data - the whole statistical data with prognose for tests
       start_day - start day of modellibg

       location - city/region/country (string)
       pop_location - population in location
       p - calibrated parameters
       to_plot - statistics to plot
       namemsim - name of the file to save
       school_days - for this period of historical data
       school_changes - for this period of historical data
       run_sim - to run the sim or not (False if need to run msim)
    '''
    #end_date=cv.date(forecast_data.index[-1].to_pydatetime().date())
    end_date=cv.date(forecast_data.index[-1])
    sim=run_model(p=p, location=location, pop_location=pop_location, start_day=start_day, end_day=end_date, 
                  b_days=b_days, b_changes=b_changes, data=forecast_data, run=run_sim,
                  school_days=school_days, school_changes=school_changes)
    
    run_msim_conf(sim=sim, to_plot=to_plot, n_runs=n_runs, save=save, namemsim=namemsim, plot=plot)
  
    
# Functions for scenarios


def scenarios(forecast_data, p, location, pop_location, b_days,b_changes, school_days, 
               school_changes, namescen,  start_day, n_runs=10):
    # Run options
    end_date=cv.date(forecast_data.index[-1].to_pydatetime().date())
    
    print('Modelling ', start_day, ' - ', end_date)
    
    sim=run_model(p=p, location=location, pop_location=pop_location, start_day=start_day, end_day=end_date, 
                  b_days=b_days, b_changes=b_changes, data=forecast_data, run=False,
                  school_days=school_days, school_changes=school_changes)
    
    # Scenario metaparameters
    metapars = dict(
        n_runs    = n_runs, # Number of parallel runs; change to 3 for quick, 11 for real
        rand_seed = 1,
        quantiles = {'low':0.1, 'high':0.9},
    )
    
    # changes of beta (for intervention change_beta)
    b_days=[]
    b_changes=[]
    for i in range(len(p)):
        b_days.append(p[i][f'beta_day_{i+1}'])
        b_changes.append(p[i][f'beta_change_{i+1}'])
    b_days=list(map(int, b_days))


    # dates of beta_change
    b_change_model=cv.date(b_days,start_date=start_day, as_date=False)


    # all base interventions in the model
    b_change_model=b_change_model+school_days
    
    # interventions
    baseline_interv = [
                cv.clip_edges(days=school_days, changes=school_changes, layers='s'), # schools closed
                cv.change_beta(days=b_days, changes=b_changes),
                cv.test_num(daily_tests=sim.data['new_tests'], symp_test=p[0]['symp_test']),
                ]
    #more = [cv.change_beta(days='2021-02-01', changes=0.49)]
    more = [cv.change_beta(days='2021-02-10', changes=1.5, layers='c')]
    more_interv = baseline_interv + more
    
    #less = [cv.change_beta(days='2021-02-01',changes=0.4)]
    less = [cv.change_beta(days='2021-02-10',changes=0.5, layers='c')]

    less_interv = baseline_interv + less
    
    
    #Define the actual scenarios
    scenarios = {
        'baseline': {
                  'name':'Baseline',
                  'pars': {
                      'interventions': baseline_interv}},
                 
          
                'more': {
                  'name':'Increase mobility',
                  'pars': {
                      'interventions' : more_interv}},
                 
                 'less': {
                  'name':'Decrease mobility',
                  'pars': {
                      'interventions' : less_interv}}}
                 
    scens = cv.Scenarios(sim=sim,metapars=metapars, scenarios=scenarios)
    scens.run(verbose=0)
    scens.save(namescen)
    
    
    
# ------------------------------------------------------------------------------------------    

def vaccinate_by_age(sim, prob_of_vaccine, ages=[20,60]):
    '''
    Define the vaccine subtargeting (function from Covasim tutorial https://docs.idmod.org/projects/covasim/en/latest/tutorials/t5.html)
    Args:
        sim  - Covasim sim
        prob_of_vaccine - list of probabilities to be tested by age
        ages - slises for age groups: young, middle, old
    Return:
        dictionary for vaccine intervention
    '''    
    young  = cv.true(sim.people.age < ages[0]) # cv.true() returns indices of people matching this condition, i.e. people under 50
    middle = cv.true((sim.people.age >= ages[0]) * (sim.people.age < ages[1])) # Multiplication means "and" here
    old    = cv.true(sim.people.age > ages[1])
    inds = sim.people.uid # Everyone in the population -- equivalent to np.arange(len(sim.people))
    vals = np.ones(len(sim.people)) # Create the array
    vals[young] = prob_of_vaccine[0] #  probability for young people 
    vals[middle] = prob_of_vaccine[1] #  probability for people middle aged
    vals[old] = prob_of_vaccine[2] #  probaility for old people 
    output = dict(inds=inds, vals=vals)
    return output

    
def run_vaccine_uniform(p, end_day,date_vaccine, school_days, school_changes,b_days,b_changes, data, run,
                    to_plot=['new_diagnoses','new_deaths', 'new_infections']):

	'''
	 Function for 
	 Args:
	    
	'''


	datafile = data
	pop_size = 100e3
	pop_Novosib=2798170
	pars = dict(
	    pop_size = pop_size,
	    pop_scale = pop_Novosib/pop_size,
	    pop_infected = p[0]['pop_infected'],
	    pop_type = 'hybrid',
	    beta = p[0]['beta'],
	    start_day = '2020-03-12',  
	    end_day   = end_day,  
	    location='Novosibirsk',
	    verbose=0.1,
	    rescale = True,
	    )
	sim = cv.Sim(pars, datafile=datafile)
	sim.initialize()
	interventions = [cv.clip_edges(days=school_days, changes=school_changes, layers='s'), # schools closed
		            cv.change_beta(days=b_days, changes=b_changes), # different beta_changes
		            cv.vaccine(days=date_vaccine, rel_sus=0.8, rel_symp=0.06,
		                      subtarget=vaccinate_by_age(sim=sim,prob_of_vaccine=[0,0.5,0.5])),
		            cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=p[0]['symp_test']) # testing routine
		            ]
		            
	sim.update_pars(interventions=interventions)
	if run:
	    sim.run()
	    sim.plot(to_plot=to_plot)
	return sim


