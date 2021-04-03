import pandas as pd
import numpy as np
from datetime import datetime
import os

from scipy import sparse
from scipy.sparse.linalg import spsolve
from statsmodels.tools.validation import array_like, PandasWrapper

import warnings
warnings.filterwarnings("ignore")


def parse_data(state='all-states'):
    """
    Парсинг и преобразование в таблицу данных с сайта https://covidtracking.com/ по определенному штату.\\
По умолчанию скачивается информация по всем штатам.
    """
    
    state_url = f'https://covidtracking.com/data/download/{state}-history.csv'

    df = pd.read_csv(state_url, index_col="date", parse_dates=True, na_values=['nan'])
    df = df.sort_values(by=['date'])
    df.to_csv(f'data/{state}.csv')
    
    return df


def ewma(df,series,n,alpha,w):
    """
    Применяет двойное сглаживание скользящего среднего ко временному ряду.
    """
    df_c = df.loc[series.index,:]
    series_w = pd.concat([series,df_c.apply(lambda x: w[x.day],axis=1)],axis=1)
    
    series_w['w'] = series_w.iloc[:,0]*series_w.iloc[:,1]
    cum_sum = series_w.iloc[:,1].rolling(n).sum()
    
    wma = (series_w.iloc[:,2].rolling(n).sum()/cum_sum)
    
    ewma = wma.rolling(n).apply(lambda x: sum([x[i]*(1-alpha)**(abs(i-n+1)) 
                                                       for i in range(n)])/
                               sum([(1-alpha)**(abs(i-n+1)) for i in range(n)]))
    
    return ewma


def hpfilter(x, lamb=1600):
    pw = PandasWrapper(x)
    x = array_like(x, 'x', ndim=1)
    nobs = len(x)
    I = sparse.eye(nobs, nobs)  # noqa:E741
    offsets = np.array([0, 1, 2])
    data = np.repeat([[1.], [-2.], [1.]], nobs, axis=1)
    K = sparse.dia_matrix((data, offsets), shape=(nobs - 2, nobs))

    use_umfpack = True
    trend = spsolve(I+lamb*K.T.dot(K), x, use_umfpack=use_umfpack)

    cycle = x - trend
    return pw.wrap(cycle, append='cycle'), pw.wrap(trend, append='trend')

def smooth(series):
    """
    Применяет ...
    """
    return hpfilter(series,100)[1]
    

def past_extrapolation(series,n,C=1.03):
    """
    Добавляет гладкую экстраполяцию прошлого с нормальным шумом к передаваемому временному ряду.
    """
    back = series[::-1]
    back.index = np.arange(back.shape[0])
    for i in range(back.isna().sum()):
        back = back.fillna(back.rolling(n).apply(lambda x: x.mean()*C**-i).shift())
    back.index = series.index[::-1]
    back = round(back)[::-1]
    return back


def nan_interpolation(series,interpolation_method='akima',q=0.997):
    """
    Интерполяция пропусков и выбросных значений с пмощью одного из методов \
функции pd.DataFrame.interpolate. По умолчанию используются Akima-сплайны.
    """
    if series.name.find('cum') == -1:
        result = series.map(lambda x: np.nan if x <= 0 or x > series.quantile(q) 
                                         else x).interpolate(method=interpolation_method)
    else:
        result = series.map(lambda x: np.nan if x <= 0 else x).interpolate(method=interpolation_method)
        
    return result

def Rt(n):
    return df.new_diagnoses.rolling(n).sum()/df.new_diagnoses.shift(n).rolling(n).sum()

def full_preprocessing(df,interpolation=True):
    """
    Объединение кода файлов preprocessing.ipynb и EDA.ipynb в функцию для быстрой обработки данных с сайта и добавления новых признаков.
    """

    a = df.recovered[1:].copy()
    a = pd.Series((df.recovered[1:].values - df.recovered[:-1].values),index=a.index)
    a.loc[df.index[0]] = np.nan
    a = a.sort_index()
    df['recoveredIncrease'] = a.values.copy()
    
    df = df.groupby(by=df.index.map(lambda x: datetime(x.year,x.month,x.day))).mean()
    
    curr = ['hospitalizedCurrently','inIcuCurrently','onVentilatorCurrently']
    new = ['positiveIncrease', 'negativeIncrease','totalTestResultsIncrease', 'hospitalizedIncrease','recoveredIncrease','deathIncrease']
    cum = ['positive','negative','totalTestResults','hospitalizedCumulative','recovered','death']
    df = df[curr+new+cum]
    
    dict_columns  = {
                 'hospitalizedCurrently'     : 'curr_hospitalized',
                 'inIcuCurrently'            : 'curr_icu',
                 'onVentilatorCurrently'     : 'curr_vent',
                 'positiveIncrease'          : 'new_diagnoses',
                 'negativeIncrease'          : 'new_negatives',
                 'totalTestResultsIncrease'  : 'new_tests',
                 'hospitalizedIncrease'      : 'new_hospitalized',
                 'recoveredIncrease'         : 'new_recovered',
                 'deathIncrease'             : 'new_deaths',
                 'positive'                  : 'cum_diagnoses',
                 'negative'                  : 'cum_negatives',
                 'totalTestResults'          : 'cum_tests',
                 'hospitalizedCumulative'    : 'cum_hospitalized',
                 'recovered'                 : 'cum_recovered',
                 'death'                     : 'cum_deaths'
                }
    df.columns = [dict_columns[col] for col in df.columns]
    
    df = df.loc[:,[a for a in df.columns if df[a].isna().sum() < int(df.index.size*0.75)]]
    df = df.loc[:,[a for a in df.columns if df[a].unique().size > 3]]
    
             
    df = df.fillna(0)
    for a in df.columns[:15]:
        df[a] = df[a].astype('int64')
    
    df['day'] = df.index.map(lambda x: x.day_name().lower()[:2])
    df.day = df.day.map(lambda x: 1 if x == 'mo' else
                                  2 if x == 'tu' else
                                  3 if x == 'we' else
                                  4 if x == 'th' else
                                  5 if x == 'fr' else
                                  6 if x == 'sa' else
                                  7)
    
    w = (df.new_tests.groupby(df['day']).sum()/df.new_tests.sum())[range(1,8)]
    
    for col in [c for c in df.columns if c.find('new') != -1]:
        if interpolation and df[col].sum() != 0:
            df[col] = nan_interpolation(df[col])
        df[col] = past_extrapolation(df[col],7)
        df[f'{col}_w'] = smooth(df[col])
        df[f'{col}_pc'] = df[f'{col}_w'].pct_change(7)
    
    if 'new_diagnoses_w' in df.columns.tolist():
        df['new_prop'] = df.new_diagnoses_w/df.new_tests_w
        df['new_prop_w'] = ewma(df,df['new_prop'],7,0.5,w)
        df['new_prop_w'] = df['new_prop_w'].map(lambda x: 1 if x > 1. else x)
    else:
        df['new_diagnoses_w'] = np.zeros_like(df.iloc[:,0].values)
        df['new_prop_w'] = np.zeros_like(df.iloc[:,0].values)
        df['new_diagnoses_pc'] = np.zeros_like(df.iloc[:,0].values)
    
    df = df.fillna(0)
    
    for a in [c for c in df.columns if c.find('prop') == -1 and c.find('day') == -1 and c.find('pc') == -1]:
    	df[a] = df[a].astype('int64')
    
    return df

def full_preprocessing_RUS(df,interpolation=True):
    """
    Объединение кода файлов preprocessing.ipynb и EDA.ipynb в функцию для быстрой обработки данных с сайта и добавления новых признаков.
    """

    
    curr = ['ventilation','reanimation','cur_children']
    new = ['new_diagnoses', 'new_tests','new_children', 'adults','new_recoveries','new_deaths']
    cum = ['cum_diagnoses','cum_recoveries','cum_deaths','cum_tests','cum_children']
    df = df[curr+new+cum]
    
    dict_columns  = {
                 'ventilation'     : 'curr_vent',
                 'reanimation'     : 'curr_icu',
                 'new_diagnoses'   : 'new_diagnoses',
                 'new_tests'       : 'new_tests',
                 'cur_children'    : 'curr_children',
                 'new_children'    : 'new_children',
                 'adults'          : 'new_adults',
                 'new_recoveries'  : 'new_recovered',
                 'new_deaths'      : 'new_deaths',
                 'cum_diagnoses'   : 'cum_diagnoses',
                 'cum_recoveries'  : 'cum_recovered',
                 'cum_deaths'      : 'cum_deaths',
                 'cum_tests'       : 'cum_tests',
                 'cum_children'    : 'cum_children'
                }
    df.columns = [dict_columns[col] for col in df.columns]
    
    df = df.loc[:,[a for a in df.columns if df[a].isna().sum() < int(df.index.size*0.75)]]
    df = df.loc[:,[a for a in df.columns if df[a].unique().size > 3]]
    
             
    
    df['day'] = df.index.map(lambda x: x.day_name().lower()[:2])
    df.day = df.day.map(lambda x: 1 if x == 'mo' else
                                  2 if x == 'tu' else
                                  3 if x == 'we' else
                                  4 if x == 'th' else
                                  5 if x == 'fr' else
                                  6 if x == 'sa' else
                                  7)
    
    w = (df.new_tests.groupby(df['day']).sum()/df.new_tests.sum())[range(1,8)]
    
    df = df.fillna(0)
    
    for col in [c for c in df.columns if c.find('new') != -1 or c.find('curr') != -1 or c.find('cum') != -1]:
        if interpolation and df[col].sum() != 0:
            df[col] = nan_interpolation(df[col],'from_derivatives')
        df[col] = past_extrapolation(df[col],7)
        df[f'{col}_w'] = smooth(df[col])
        df[f'{col}_pc'] = df[f'{col}_w'].pct_change(7)
    
    if 'new_diagnoses_w' in df.columns.tolist():
        df['new_prop'] = df.new_diagnoses_w/df.new_tests_w
        df['new_prop_w'] = smooth(df['new_prop'])
        df['new_prop_w'] = df['new_prop_w'].map(lambda x: 1 if x > 1. else x)
    else:
        df['new_diagnoses_w'] = np.zeros_like(df.iloc[:,0].values)
        df['new_prop_w'] = np.zeros_like(df.iloc[:,0].values)
        df['new_diagnoses_pc'] = np.zeros_like(df.iloc[:,0].values)
    
    df = df.fillna(0)
    
    for a in [c for c in df.columns if c.find('prop') == -1 and c.find('day') == -1 and c.find('pc') == -1]:
        df[a] = df[a].astype('int64')   
        
    for a in [c for c in df.columns if c.find('day') == -1 and c.find('pc') == -1]:
        df[a] = df[a].map(lambda x: 0 if x < 0 else x)
    
    return df

