import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize

from itertools import product
from random import shuffle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')
from func import *
from tqdm import tqdm_notebook as tqdm     

def invboxcox(series,lmb):
    """
    Обратное преобразование Бокса-Кокса.
    
    Args:
        series (pd.Series): временной ряд
        lmb (float): параметр преобразования Бокса-Кокса
    
    Returns:
        1. result: результат обратного преобразования
    """
    if lmb == 0:
        return(np.exp(series))
    else:
        return(np.exp(np.log(lmb*series+1)/lmb))
    
def tsplot(series, lags=None):
    """
    Функция отрисовки ряда, авткороллеяции и частичной автокорреляции, а также проведения расширенного \
теста Дики-Фуллера о нестационарности.
    
    Args:
        series (pd.Series): временной ряд
        lags (int): максимальный лаг на графике автокорреляции
    """
    fig = plt.figure(figsize=(12,8))
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    series.plot(ax=ts_ax)
    ts_ax.set_title('Time Series Analysis Plots')
    smt.graphics.plot_acf(series, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(series, lags=lags, ax=pacf_ax, alpha=0.5)
    print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(series)[1])
    plt.tight_layout()
    
def best_SARIMAX(series, d,D, n_past, parameters_list=None,args={}):
    """
    Находит модель SARIMAX с оптимальными параметрами на основании минимизации AIC.
    
    Args:
        series (pd.Series): временной ряд
        d (int): порядок разности временного ряда, после которого ряд становится стационарным
        D (int): порядок сезонной разности временного ряда, после которого ряд становится стационарным
        n_past (int): количество дней временного ряда, откладываемые на тест
        parameters_list (list): список из определенныз параметров [p,q,P,Q] модели SARIMA, с которыми должна\
строиться модель (подбор оптимального набора параметров не производится).
        args (dict): словарь некоторых аргументов со значениями функции sm.tsa.statespace.SARIMAX()
    
    Returns:
        1. best_model: модель SARIMAX
        2. best_params: набор оптимальных параметров
        3. best_aic: минимальное значение AIC метрики с оптимальными параметрами
    """
    if parameters_list is None:
        ps = range(0, 7)
        qs = range(0, 7)
        Ps = [0,1]
        Qs = [0,1]
        parameters_list = product(ps, qs, Ps, Qs)
        parameters_list = list(parameters_list)
    else:
        parameters_list = [parameters_list]
        
    best_aic = float("inf")
    
    for param in tqdm(parameters_list):
        model=sm.tsa.statespace.SARIMAX(series[:-n_past], 
                                        order = (param[0], d, param[1]), 
                                        seasonal_order = (param[2], D, param[3], 7),
                                        **args).fit(disp=-1)
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_params = param
    return best_model, best_params, best_aic

def get_predict(series,model,n_past,n_future,lmb):
    result = invboxcox(model.fittedvalues, lmb)
    
    forecast = invboxcox(model.predict(start = series.index.size - n_past, 
                                       end = series.index.size + n_future), lmb)
    forecast.index = [series.index[-n_past] + timedelta(days=i) for i in range(n_past+n_future+1)]
    forecast = result.append(forecast)
    #forecast = nan_interpolation(forecast,interpolation_method='linear',q=0.997)
    
    return forecast

def plot_SARIMAX(series,model,n_past, n_future, lmb):
    """
    Функция отрисовки временного ряда и модели SARIMAX, предсказывающей значения спрогнозированного \
показателя на обучающей и на тестовой выборке.

    Args:
        series (pd.Series): временной ряд
        model (sm.tsa.statespace.SARIMAX): предварительно обученная модель типа SARIMAX
        n_past (int): количество дней временного ряда, предварительно отложенные на тест
        n_future (int): количество дней прогноза от конца временного ряда в будущее
        lmb (float): параметр преобразования Бокса-Кокса
    """
    forecast = get_predict(series,model,n_past,n_future,lmb)
    n = n_past+n_future
    plt.figure(figsize=(25, 10))
    plt.plot(forecast, label="SARIMAX")
    plt.plot(series, label=series.name)
    plt.axvspan(forecast.index[-n], forecast.index[-n_future], alpha=0.5, color='lightgrey')
    plt.legend(loc='best',fontsize=20)
    plt.grid(True)
    plt.axis('tight')
    plt.show()
