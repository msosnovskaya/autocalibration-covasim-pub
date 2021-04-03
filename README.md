Files for autocalibration process in COVASIM. 
 - functions_total.py - all auxiliary functions;
 - calibration_total.py - Class for autocalibration with OPTUNA;
 - func.py, SARIMA.py - additional files for data analysis and extropolation;
 - autocalibration_total_Novosibirsk.ipynb - Notebook with example of code for Novosibirsk.

The main idea of autocalibration is Cliff's - to devide modeling period into parts (here duration of one period is 1 month) and calibrate at the first period parameters
beta, pop_inf, symp_test and beta_day_1, beta_change_1. In the following periods calibrate only beta_day_i, beta_change_i in every month (i=1,..,number of months). Also added some 
functions for checking prognose on history data, tests extrapolation (to predict number of diagnoses) and some plotting. This code was validated also for New York State and UK.
