# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:46:53 2022

@author: zacha
"""

import os
import re
import numpy as np
import pandas as pd
from SALib.sample import latin
from joblib import Parallel, delayed
import re
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import scipy.stats as stats
import seaborn as sns
from datetime import datetime
from matplotlib.lines import Line2D
import os


# set random seed for reproducibility
seed_value = 123

# directory where the data is stored
data_dir = 'C:/Users/zacha/Documents/UNC/SP2016_StateMod'

# template file as a source for modification
template_file = os.path.join(data_dir, "SP2016_H_original.ddm")

# directory to write modified files to
output_dir = "C:/Users/zacha/Documents/UNC/SP2016_StateMod"

# scenario name
scenario = "test"

# character indicating row is a comment
comment = "#"

# dictionary to hold values for each field
d = {"yr": [], 
     "id": [], 
     "jan": [], 
     "feb": [], 
     "mar": [], 
     "apr": [], 
     "may": [], 
     "jun": [], 
     "jul": [], 
     "aug": [],
     "sep": [],
     "oct": [],
     "nov": [],
     "dec": [],
     "total": []}

# define the column widths for the output file
column_widths = {"yr": 4, 
                 "id": 9, 
                 "jan": 10, 
                 "feb": 7, 
                 "mar": 7, 
                 "apr": 7, 
                 "may": 7, 
                 "jun": 7, 
                 "jul": 7, 
                 "aug": 7,
                 "sep": 7,
                 "oct": 7,
                 "nov": 7,
                 "dec": 7,
                 "total": 9}

# list of columns to process
column_list = ["yr", "id", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "total"]

# list of value columns that may be modified
value_columns = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "total"]

#%%time

# empty string to hold header data
header = ""

capture = False
with open(template_file) as template:
    
    for idx, line in enumerate(template):
        
        if capture:
            value_list = line.strip().split()
            

            # comprehension only used to build dict
            x = [d[column_list[idx]].append(i) for idx, i in enumerate(value_list)]

        else:
            
            # store any header and preliminary lines to use in restoration
            header += line
            
            # passes uncommented date range line before data start and all commented lines in header
            if line[0] != comment:
                capture = True
                
# convert dictionary to a pandas data frame  
df = pd.DataFrame(d)


# convert value column types to float
df[value_columns] = df[value_columns].astype(np.float64)

df

df.describe()

loveland_indoor_demand_df = pd.DataFrame()
loveland_indoor_demand_stats = pd.DataFrame()
loveland_indoor_demand_norm = pd.DataFrame()
loveland_outdoor_demand_df = pd.DataFrame()
longmont_indoor_demand_df = pd.DataFrame()
longmont_outdoor_demand_df = pd.DataFrame()
boulder_indoor_demand_df = pd.DataFrame()
boulder_outdoor_demand_df = pd.DataFrame()
denver_indoor_demand_df = pd.DataFrame()
denver_outdoor_demand_df = pd.DataFrame()
aurora_indoor_demand_df = pd.DataFrame()
aurora_outdoor_demand_df = pd.DataFrame()
englewood_indoor_demand_df = pd.DataFrame()
englewood_outdoor_demand_df = pd.DataFrame()
northglenn_indoor_demand_df = pd.DataFrame()
northglenn_outdoor_demand_df = pd.DataFrame()
westminster_indoor_demand_df = pd.DataFrame()
westminster_outdoor_demand_df = pd.DataFrame()
thornton_indoor_demand_df = pd.DataFrame()
thornton_outdoor_demand_df = pd.DataFrame()
lafayette_indoor_demand_df = pd.DataFrame()
lafayette_outdoor_demand_df = pd.DataFrame()
louisville_indoor_demand_df = pd.DataFrame()
louisville_outdoor_demand_df = pd.DataFrame()
arvada_indoor_demand_df = pd.DataFrame()
arvada_outdoor_demand_df = pd.DataFrame()
conmutual_indoor_demand_df = pd.DataFrame()
conmutual_outdoor_demand_df = pd.DataFrame()
golden_indoor_demand_df = pd.DataFrame()
golden_outdoor_demand_df = pd.DataFrame()

synthetic_series_length = 756

#LOVELAND INDOOR

loveland_indoor_demand_df = df.loc[df['id'] == '04_Lovelnd_I']
loveland_indoor_demand_df['yr'] = loveland_indoor_demand_df['yr'].astype(int)
print(loveland_indoor_demand_df.dtypes)
loveland_indoor_demand_df_selection = loveland_indoor_demand_df.iloc[46:63, 2:14]
loveland_indoor_demand_df_selection_update = pd.DataFrame(loveland_indoor_demand_df_selection.values.ravel(),columns = ['column'])
loveland_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(loveland_indoor_demand_df_selection_update), freq='M')
loveland_indoor_demand_df_selection_update.index = loveland_indoor_demand_df_selection_update['date']
loveland_indoor_demand_df_selection_update['month'] = loveland_indoor_demand_df_selection_update.index.month
loveland_indoor_demand_df_selection_update['year'] = loveland_indoor_demand_df_selection_update.index.year
loveland_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = loveland_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == loveland_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    loveland_indoor_demand_df_selection_update['norm'][i] = loveland_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = loveland_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = loveland_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

loveland_indoor_demand_df_selection_update['deseas'] = loveland_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    loveland_indoor_demand_df_selection_update['deseas'].loc[loveland_indoor_demand_df_selection_update['month'] == i] = (loveland_indoor_demand_df_selection_update['deseas'].loc[loveland_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(loveland_indoor_demand_df_selection_update['deseas'])

loveland_indoor_demand_df_selection_update['deseas_l1'] = np.nan
loveland_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(loveland_indoor_demand_df_selection_update.head())
print()

loveland_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = loveland_indoor_demand_df_selection_update['deseas'].values[:-1]
loveland_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = loveland_indoor_demand_df_selection_update['deseas'].values[:-12]

loveland_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=loveland_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

loveland_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(loveland_indoor_demand_df_selection_update)

plt.figure()
plt.plot(loveland_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(loveland_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(loveland_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

loveland_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
loveland_indoor_demands['synthetic2_noise'] = norm.rvs(loveland_indoor_demand_df_selection_update['ar_resid'].mean(), loveland_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(loveland_indoor_demand_df_selection_update['ar_resid'])
plt.plot(loveland_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = loveland_indoor_demands.shape[0]

synth_ar = list(loveland_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = loveland_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
loveland_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
loveland_indoor_demands

loveland_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(loveland_indoor_demands), freq='M')
loveland_indoor_demands.index = loveland_indoor_demands['date']
loveland_indoor_demands['month'] = loveland_indoor_demands.index.month
loveland_indoor_demands['year'] = loveland_indoor_demands.index.year


loveland_indoor_demands['synthetic_demand'] = loveland_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    loveland_indoor_demands['synthetic_demand'].loc[loveland_indoor_demands['month'] == i] = loveland_indoor_demands['synthetic_demand'].loc[loveland_indoor_demands['month'] == i] * sigma + mu
    
loveland_indoor_demands['6000'] = loveland_indoor_demands['synthetic_demand']*7200

plt.figure()
plt.plot(loveland_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('Loveland Indoor Demands')
plt.plot(loveland_indoor_demands['6000'])
ax = plt.gca()
ax.set_ylim([350, 850])

plt.figure()
plt.title('Loveland Historical')
ax = plt.gca()
ax.set_ylim([350, 850])
plt.plot(loveland_indoor_demand_df_selection_update['column'])
plt.axhline(y = 600, color = 'r', linestyle = '-')




#LOVELAND OUTDOOR

loveland_outdoor_demand_df = df.loc[df['id'] == '04_Lovelnd_O']
loveland_outdoor_demand_df['yr'] = loveland_outdoor_demand_df['yr'].astype(int)
print(loveland_outdoor_demand_df.dtypes)
loveland_outdoor_demand_df_historical = loveland_outdoor_demand_df.iloc[0:63, 2:14]
loveland_outdoor_demand_df_historical_ravel = pd.DataFrame(loveland_outdoor_demand_df_historical.values.ravel(),columns = ['column'])
loveland_outdoor_demand_df_historical_ravel['date'] = pd.date_range(start='1/1/1950', periods=len(loveland_outdoor_demand_df_historical_ravel), freq='M')
loveland_outdoor_demand_df_historical_ravel.index = loveland_outdoor_demand_df_historical_ravel['date']
loveland_outdoor_demand_df_historical_ravel['month'] = loveland_outdoor_demand_df_historical_ravel.index.month
loveland_outdoor_demand_df_historical_ravel['year'] = loveland_outdoor_demand_df_historical_ravel.index.year


plt.figure()
plt.title('Loveland Outdoor Demands')
plt.plot(loveland_outdoor_demand_df_historical_ravel['column'])



loveland_outdoor_demand_df_selection = loveland_outdoor_demand_df.iloc[46:63, 2:14]
loveland_outdoor_demand_df_selection_update = pd.DataFrame(loveland_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
loveland_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(loveland_outdoor_demand_df_selection_update), freq='M')
loveland_outdoor_demand_df_selection_update.index = loveland_outdoor_demand_df_selection_update['date']
loveland_outdoor_demand_df_selection_update['month'] = loveland_outdoor_demand_df_selection_update.index.month
loveland_outdoor_demand_df_selection_update['year'] = loveland_outdoor_demand_df_selection_update.index.year
loveland_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = loveland_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == loveland_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    loveland_outdoor_demand_df_selection_update['norm'][i] = loveland_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = loveland_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = loveland_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

loveland_outdoor_demand_df_selection_update['deseas'] = loveland_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    loveland_outdoor_demand_df_selection_update['deseas'].loc[loveland_outdoor_demand_df_selection_update['month'] == i] = (loveland_outdoor_demand_df_selection_update['deseas'].loc[loveland_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma

loveland_outdoor_demand_df_selection_update.fillna(0)
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(loveland_outdoor_demand_df_selection_update['deseas'])

loveland_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
loveland_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(loveland_outdoor_demand_df_selection_update.head())
print()

loveland_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = loveland_outdoor_demand_df_selection_update['deseas'].values[:-1]
loveland_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = loveland_outdoor_demand_df_selection_update['deseas'].values[:-12]

loveland_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=loveland_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

loveland_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(loveland_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(loveland_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(loveland_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(loveland_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

loveland_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
loveland_outdoor_demands['synthetic2_noise'] = norm.rvs(loveland_outdoor_demand_df_selection_update['ar_resid'].mean(), loveland_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(loveland_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(loveland_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = loveland_outdoor_demands.shape[0]

synth_ar = list(loveland_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = loveland_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
loveland_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
loveland_outdoor_demands

loveland_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(loveland_outdoor_demands), freq='M')
loveland_outdoor_demands.index = loveland_outdoor_demands['date']
loveland_outdoor_demands['month'] = loveland_outdoor_demands.index.month
loveland_outdoor_demands['year'] = loveland_outdoor_demands.index.year


loveland_outdoor_demands['synthetic_demand'] = loveland_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    loveland_outdoor_demands['synthetic_demand'].loc[loveland_outdoor_demands['month'] == i] = loveland_outdoor_demands['synthetic_demand'].loc[loveland_outdoor_demands['month'] == i] * sigma + mu

loveland_outdoor_demands['synthetic_demand'][loveland_outdoor_demands['synthetic_demand'] < 0] = 0
    
loveland_outdoor_demands['8000'] = loveland_outdoor_demands['synthetic_demand']*8000

plt.figure()
plt.plot(loveland_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('Loveland Outdoor Demands')
plt.plot(loveland_outdoor_demands['8000'])

plt.figure()
plt.title('Loveland Historical')
plt.plot(loveland_outdoor_demand_df_selection_update['column'])

#LONGMONT INDOOR

longmont_indoor_demand_df = df.loc[df['id'] == '05LONG_IN']
longmont_indoor_demand_df['yr'] = longmont_indoor_demand_df['yr'].astype(int)
print(longmont_indoor_demand_df.dtypes)
longmont_indoor_demand_df_selection = longmont_indoor_demand_df.iloc[46:63, 2:14]
longmont_indoor_demand_df_selection_update = pd.DataFrame(longmont_indoor_demand_df_selection.values.ravel(),columns = ['column'])
longmont_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(longmont_indoor_demand_df_selection_update), freq='M')
longmont_indoor_demand_df_selection_update.index = longmont_indoor_demand_df_selection_update['date']
longmont_indoor_demand_df_selection_update['month'] = longmont_indoor_demand_df_selection_update.index.month
longmont_indoor_demand_df_selection_update['year'] = longmont_indoor_demand_df_selection_update.index.year
longmont_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = longmont_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == longmont_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    longmont_indoor_demand_df_selection_update['norm'][i] = longmont_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = longmont_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = longmont_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

longmont_indoor_demand_df_selection_update['deseas'] = longmont_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    longmont_indoor_demand_df_selection_update['deseas'].loc[longmont_indoor_demand_df_selection_update['month'] == i] = (longmont_indoor_demand_df_selection_update['deseas'].loc[longmont_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(longmont_indoor_demand_df_selection_update['deseas'])

longmont_indoor_demand_df_selection_update['deseas_l1'] = np.nan
longmont_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(longmont_indoor_demand_df_selection_update.head())
print()

longmont_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = longmont_indoor_demand_df_selection_update['deseas'].values[:-1]
longmont_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = longmont_indoor_demand_df_selection_update['deseas'].values[:-12]

longmont_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=longmont_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

longmont_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(longmont_indoor_demand_df_selection_update)

plt.figure()
plt.plot(longmont_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(longmont_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(longmont_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

longmont_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
longmont_indoor_demands['synthetic2_noise'] = norm.rvs(longmont_indoor_demand_df_selection_update['ar_resid'].mean(), longmont_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(longmont_indoor_demand_df_selection_update['ar_resid'])
plt.plot(longmont_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = longmont_indoor_demands.shape[0]

synth_ar = list(longmont_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = longmont_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
longmont_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
longmont_indoor_demands

longmont_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(longmont_indoor_demands), freq='M')
longmont_indoor_demands.index = longmont_indoor_demands['date']
longmont_indoor_demands['month'] = longmont_indoor_demands.index.month
longmont_indoor_demands['year'] = longmont_indoor_demands.index.year


longmont_indoor_demands['synthetic_demand'] = longmont_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    longmont_indoor_demands['synthetic_demand'].loc[longmont_indoor_demands['month'] == i] = longmont_indoor_demands['synthetic_demand'].loc[longmont_indoor_demands['month'] == i] * sigma + mu
    
longmont_indoor_demands['6000'] = longmont_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(longmont_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('Longmont indoor Demands')
plt.plot(longmont_indoor_demands['6000'])

plt.figure()
plt.title('Longmont Historical')
plt.plot(longmont_indoor_demand_df_selection_update['column'])

#LONGMONT OUTDOOR
longmont_outdoor_demand_df = df.loc[df['id'] == '05LONG_OUT']
longmont_outdoor_demand_df['yr'] = longmont_outdoor_demand_df['yr'].astype(int)
print(longmont_outdoor_demand_df.dtypes)
longmont_outdoor_demand_df_selection = longmont_outdoor_demand_df.iloc[46:63, 2:14]
longmont_outdoor_demand_df_selection_update = pd.DataFrame(longmont_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
longmont_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(longmont_outdoor_demand_df_selection_update), freq='M')
longmont_outdoor_demand_df_selection_update.index = longmont_outdoor_demand_df_selection_update['date']
longmont_outdoor_demand_df_selection_update['month'] = longmont_outdoor_demand_df_selection_update.index.month
longmont_outdoor_demand_df_selection_update['year'] = longmont_outdoor_demand_df_selection_update.index.year
longmont_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = longmont_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == longmont_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    longmont_outdoor_demand_df_selection_update['norm'][i] = longmont_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = longmont_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = longmont_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

longmont_outdoor_demand_df_selection_update['deseas'] = longmont_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    longmont_outdoor_demand_df_selection_update['deseas'].loc[longmont_outdoor_demand_df_selection_update['month'] == i] = (longmont_outdoor_demand_df_selection_update['deseas'].loc[longmont_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(longmont_outdoor_demand_df_selection_update['deseas'])

longmont_outdoor_demand_df_selection_update = longmont_outdoor_demand_df_selection_update.fillna(0)

longmont_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
longmont_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(longmont_outdoor_demand_df_selection_update.head())
print()

longmont_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = longmont_outdoor_demand_df_selection_update['deseas'].values[:-1]
longmont_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = longmont_outdoor_demand_df_selection_update['deseas'].values[:-12]

longmont_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=longmont_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

longmont_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(longmont_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(longmont_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(longmont_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(longmont_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

longmont_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
longmont_outdoor_demands['synthetic2_noise'] = norm.rvs(longmont_outdoor_demand_df_selection_update['ar_resid'].mean(), longmont_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(longmont_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(longmont_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = longmont_outdoor_demands.shape[0]

synth_ar = list(longmont_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = longmont_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
longmont_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
longmont_outdoor_demands

longmont_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(longmont_outdoor_demands), freq='M')
longmont_outdoor_demands.index = longmont_outdoor_demands['date']
longmont_outdoor_demands['month'] = longmont_outdoor_demands.index.month
longmont_outdoor_demands['year'] = longmont_outdoor_demands.index.year


longmont_outdoor_demands['synthetic_demand'] = longmont_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    longmont_outdoor_demands['synthetic_demand'].loc[longmont_outdoor_demands['month'] == i] = longmont_outdoor_demands['synthetic_demand'].loc[longmont_outdoor_demands['month'] == i] * sigma + mu

longmont_outdoor_demands['synthetic_demand'][longmont_outdoor_demands['synthetic_demand'] < 0] = 0
    
longmont_outdoor_demands['6000'] = longmont_outdoor_demands['synthetic_demand']*6000



plt.figure()
plt.plot(longmont_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('Longmont Outdoor Demands')
plt.plot(longmont_outdoor_demands['6000'])

plt.figure()
plt.title('Longmont Historical')
plt.plot(longmont_outdoor_demand_df_selection_update['column'])

#BOULDER INDOOR

boulder_indoor_demand_df = df.loc[df['id'] == '06BOULDER_I']
boulder_indoor_demand_df['yr'] = boulder_indoor_demand_df['yr'].astype(int)
print(boulder_indoor_demand_df.dtypes)
boulder_indoor_demand_df_selection = boulder_indoor_demand_df.iloc[46:63, 2:14]
boulder_indoor_demand_df_selection_update = pd.DataFrame(boulder_indoor_demand_df_selection.values.ravel(),columns = ['column'])
boulder_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(boulder_indoor_demand_df_selection_update), freq='M')
boulder_indoor_demand_df_selection_update.index = boulder_indoor_demand_df_selection_update['date']
boulder_indoor_demand_df_selection_update['month'] = boulder_indoor_demand_df_selection_update.index.month
boulder_indoor_demand_df_selection_update['year'] = boulder_indoor_demand_df_selection_update.index.year
boulder_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = boulder_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == boulder_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    boulder_indoor_demand_df_selection_update['norm'][i] = boulder_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = boulder_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = boulder_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

boulder_indoor_demand_df_selection_update['deseas'] = boulder_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    boulder_indoor_demand_df_selection_update['deseas'].loc[boulder_indoor_demand_df_selection_update['month'] == i] = (boulder_indoor_demand_df_selection_update['deseas'].loc[boulder_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(boulder_indoor_demand_df_selection_update['deseas'])

boulder_indoor_demand_df_selection_update = boulder_indoor_demand_df_selection_update.fillna(0)

boulder_indoor_demand_df_selection_update['deseas_l1'] = np.nan
boulder_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(boulder_indoor_demand_df_selection_update.head())
print()

boulder_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = boulder_indoor_demand_df_selection_update['deseas'].values[:-1]
boulder_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = boulder_indoor_demand_df_selection_update['deseas'].values[:-12]

boulder_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=boulder_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

boulder_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(boulder_indoor_demand_df_selection_update)

plt.figure()
plt.plot(boulder_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(boulder_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(boulder_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

boulder_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
boulder_indoor_demands['synthetic2_noise'] = norm.rvs(boulder_indoor_demand_df_selection_update['ar_resid'].mean(), boulder_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(boulder_indoor_demand_df_selection_update['ar_resid'])
plt.plot(boulder_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = boulder_indoor_demands.shape[0]

synth_ar = list(boulder_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = boulder_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
boulder_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
boulder_indoor_demands

boulder_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(boulder_indoor_demands), freq='M')
boulder_indoor_demands.index = boulder_indoor_demands['date']
boulder_indoor_demands['month'] = boulder_indoor_demands.index.month
boulder_indoor_demands['year'] = boulder_indoor_demands.index.year


boulder_indoor_demands['synthetic_demand'] = boulder_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    boulder_indoor_demands['synthetic_demand'].loc[boulder_indoor_demands['month'] == i] = boulder_indoor_demands['synthetic_demand'].loc[boulder_indoor_demands['month'] == i] * sigma + mu
    
boulder_indoor_demands['6000'] = boulder_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(boulder_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('boulder Indoor Demands')
plt.plot(boulder_indoor_demands['6000'])

plt.figure()
plt.title('boulder Historical')
plt.plot(boulder_indoor_demand_df_selection_update['column'])

#BOULDER OUTDOOR
boulder_outdoor_demand_df = df.loc[df['id'] == '06BOULDER_O']
boulder_outdoor_demand_df['yr'] = boulder_outdoor_demand_df['yr'].astype(int)
print(boulder_outdoor_demand_df.dtypes)
boulder_outdoor_demand_df_selection = boulder_outdoor_demand_df.iloc[46:63, 2:14]
boulder_outdoor_demand_df_selection_update = pd.DataFrame(boulder_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
boulder_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(boulder_outdoor_demand_df_selection_update), freq='M')
boulder_outdoor_demand_df_selection_update.index = boulder_outdoor_demand_df_selection_update['date']
boulder_outdoor_demand_df_selection_update['month'] = boulder_outdoor_demand_df_selection_update.index.month
boulder_outdoor_demand_df_selection_update['year'] = boulder_outdoor_demand_df_selection_update.index.year
boulder_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = boulder_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == boulder_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    boulder_outdoor_demand_df_selection_update['norm'][i] = boulder_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = boulder_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = boulder_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

boulder_outdoor_demand_df_selection_update['deseas'] = boulder_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    boulder_outdoor_demand_df_selection_update['deseas'].loc[boulder_outdoor_demand_df_selection_update['month'] == i] = (boulder_outdoor_demand_df_selection_update['deseas'].loc[boulder_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(boulder_outdoor_demand_df_selection_update['deseas'])

boulder_outdoor_demand_df_selection_update = boulder_outdoor_demand_df_selection_update.fillna(0)

boulder_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
boulder_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(boulder_outdoor_demand_df_selection_update.head())
print()

boulder_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = boulder_outdoor_demand_df_selection_update['deseas'].values[:-1]
boulder_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = boulder_outdoor_demand_df_selection_update['deseas'].values[:-12]

boulder_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=boulder_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

boulder_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(boulder_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(boulder_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(boulder_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(boulder_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

boulder_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
boulder_outdoor_demands['synthetic2_noise'] = norm.rvs(boulder_outdoor_demand_df_selection_update['ar_resid'].mean(), boulder_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(boulder_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(boulder_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = boulder_outdoor_demands.shape[0]

synth_ar = list(boulder_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = boulder_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
boulder_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
boulder_outdoor_demands

boulder_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(boulder_outdoor_demands), freq='M')
boulder_outdoor_demands.index = boulder_outdoor_demands['date']
boulder_outdoor_demands['month'] = boulder_outdoor_demands.index.month
boulder_outdoor_demands['year'] = boulder_outdoor_demands.index.year


boulder_outdoor_demands['synthetic_demand'] = boulder_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    boulder_outdoor_demands['synthetic_demand'].loc[boulder_outdoor_demands['month'] == i] = boulder_outdoor_demands['synthetic_demand'].loc[boulder_outdoor_demands['month'] == i] * sigma + mu

boulder_outdoor_demands['synthetic_demand'][boulder_outdoor_demands['synthetic_demand'] < 0] = 0
    
boulder_outdoor_demands['6000'] = boulder_outdoor_demands['synthetic_demand']*6000



plt.figure()
plt.plot(boulder_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('Boulder Outdoor Demands')
plt.plot(boulder_outdoor_demands['6000'])

plt.figure()
plt.title('Boulder Historical')
plt.plot(boulder_outdoor_demand_df_selection_update['column'])

#DENVER INDOOR

denver_indoor_demand_df = df.loc[df['id'] == '08_Denver_I']
denver_indoor_demand_df['yr'] = denver_indoor_demand_df['yr'].astype(int)
print(denver_indoor_demand_df.dtypes)
denver_indoor_demand_df_selection = denver_indoor_demand_df.iloc[46:63, 2:14]
denver_indoor_demand_df_selection_update = pd.DataFrame(denver_indoor_demand_df_selection.values.ravel(),columns = ['column'])
denver_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(denver_indoor_demand_df_selection_update), freq='M')
denver_indoor_demand_df_selection_update.index = denver_indoor_demand_df_selection_update['date']
denver_indoor_demand_df_selection_update['month'] = denver_indoor_demand_df_selection_update.index.month
denver_indoor_demand_df_selection_update['year'] = denver_indoor_demand_df_selection_update.index.year
denver_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = denver_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == denver_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    denver_indoor_demand_df_selection_update['norm'][i] = denver_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = denver_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = denver_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

denver_indoor_demand_df_selection_update['deseas'] = denver_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    denver_indoor_demand_df_selection_update['deseas'].loc[denver_indoor_demand_df_selection_update['month'] == i] = (denver_indoor_demand_df_selection_update['deseas'].loc[denver_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(denver_indoor_demand_df_selection_update['deseas'])

denver_indoor_demand_df_selection_update['deseas_l1'] = np.nan
denver_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(denver_indoor_demand_df_selection_update.head())
print()

denver_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = denver_indoor_demand_df_selection_update['deseas'].values[:-1]
denver_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = denver_indoor_demand_df_selection_update['deseas'].values[:-12]

denver_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=denver_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

denver_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(denver_indoor_demand_df_selection_update)

plt.figure()
plt.plot(denver_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(denver_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(denver_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

denver_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
denver_indoor_demands['synthetic2_noise'] = norm.rvs(denver_indoor_demand_df_selection_update['ar_resid'].mean(), denver_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(denver_indoor_demand_df_selection_update['ar_resid'])
plt.plot(denver_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = denver_indoor_demands.shape[0]

synth_ar = list(denver_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = denver_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
denver_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
denver_indoor_demands

denver_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(denver_indoor_demands), freq='M')
denver_indoor_demands.index = denver_indoor_demands['date']
denver_indoor_demands['month'] = denver_indoor_demands.index.month
denver_indoor_demands['year'] = denver_indoor_demands.index.year


denver_indoor_demands['synthetic_demand'] = denver_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    denver_indoor_demands['synthetic_demand'].loc[denver_indoor_demands['month'] == i] = denver_indoor_demands['synthetic_demand'].loc[denver_indoor_demands['month'] == i] * sigma + mu
    
denver_indoor_demands['130000'] = denver_indoor_demands['synthetic_demand']*130000
plt.figure()
plt.plot(denver_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('Denver Indoor Demands')
plt.plot(denver_indoor_demands['130000'])

plt.figure()
plt.title('Denver Historical')
plt.plot(denver_indoor_demand_df_selection_update['column'])


#DENVER OUTDOOR

denver_outdoor_demand_df = df.loc[df['id'] == '08_Denver_O']
denver_outdoor_demand_df['yr'] = denver_outdoor_demand_df['yr'].astype(int)
print(denver_outdoor_demand_df.dtypes)
denver_outdoor_demand_df_selection = denver_outdoor_demand_df.iloc[46:63, 2:14]
denver_outdoor_demand_df_selection_update = pd.DataFrame(denver_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
denver_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(denver_outdoor_demand_df_selection_update), freq='M')
denver_outdoor_demand_df_selection_update.index = denver_outdoor_demand_df_selection_update['date']
denver_outdoor_demand_df_selection_update['month'] = denver_outdoor_demand_df_selection_update.index.month
denver_outdoor_demand_df_selection_update['year'] = denver_outdoor_demand_df_selection_update.index.year
denver_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = denver_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == denver_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    denver_outdoor_demand_df_selection_update['norm'][i] = denver_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = denver_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = denver_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

denver_outdoor_demand_df_selection_update['deseas'] = denver_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    denver_outdoor_demand_df_selection_update['deseas'].loc[denver_outdoor_demand_df_selection_update['month'] == i] = (denver_outdoor_demand_df_selection_update['deseas'].loc[denver_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(denver_outdoor_demand_df_selection_update['deseas'])

denver_outdoor_demand_df_selection_update = denver_outdoor_demand_df_selection_update.fillna(0)

denver_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
denver_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(denver_outdoor_demand_df_selection_update.head())
print()

denver_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = denver_outdoor_demand_df_selection_update['deseas'].values[:-1]
denver_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = denver_outdoor_demand_df_selection_update['deseas'].values[:-12]

denver_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=denver_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

denver_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(denver_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(denver_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(denver_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(denver_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

denver_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
denver_outdoor_demands['synthetic2_noise'] = norm.rvs(denver_outdoor_demand_df_selection_update['ar_resid'].mean(), denver_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(denver_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(denver_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = denver_outdoor_demands.shape[0]

synth_ar = list(denver_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = denver_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
denver_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
denver_outdoor_demands

denver_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(denver_outdoor_demands), freq='M')
denver_outdoor_demands.index = denver_outdoor_demands['date']
denver_outdoor_demands['month'] = denver_outdoor_demands.index.month
denver_outdoor_demands['year'] = denver_outdoor_demands.index.year


denver_outdoor_demands['synthetic_demand'] = denver_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    denver_outdoor_demands['synthetic_demand'].loc[denver_outdoor_demands['month'] == i] = denver_outdoor_demands['synthetic_demand'].loc[denver_outdoor_demands['month'] == i] * sigma + mu

denver_outdoor_demands['synthetic_demand'][denver_outdoor_demands['synthetic_demand'] < 0] = 0
    
denver_outdoor_demands['80000'] = denver_outdoor_demands['synthetic_demand']*80000



plt.figure()
plt.plot(denver_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('Denver Outdoor Demands')
plt.plot(denver_outdoor_demands['80000'])

plt.figure()
plt.title('Denver Historical')
plt.plot(denver_outdoor_demand_df_selection_update['column'])

#AURORA INDOOR

aurora_indoor_demand_df = df.loc[df['id'] == '08_Aurora_I']
aurora_indoor_demand_df['yr'] = aurora_indoor_demand_df['yr'].astype(int)
print(aurora_indoor_demand_df.dtypes)
aurora_indoor_demand_df_selection = aurora_indoor_demand_df.iloc[46:63, 2:14]
aurora_indoor_demand_df_selection_update = pd.DataFrame(aurora_indoor_demand_df_selection.values.ravel(),columns = ['column'])
aurora_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(aurora_indoor_demand_df_selection_update), freq='M')
aurora_indoor_demand_df_selection_update.index = aurora_indoor_demand_df_selection_update['date']
aurora_indoor_demand_df_selection_update['month'] = aurora_indoor_demand_df_selection_update.index.month
aurora_indoor_demand_df_selection_update['year'] = aurora_indoor_demand_df_selection_update.index.year
aurora_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = aurora_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == aurora_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    aurora_indoor_demand_df_selection_update['norm'][i] = aurora_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = aurora_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = aurora_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

aurora_indoor_demand_df_selection_update['deseas'] = aurora_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    aurora_indoor_demand_df_selection_update['deseas'].loc[aurora_indoor_demand_df_selection_update['month'] == i] = (aurora_indoor_demand_df_selection_update['deseas'].loc[aurora_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(aurora_indoor_demand_df_selection_update['deseas'])

aurora_indoor_demand_df_selection_update['deseas_l1'] = np.nan
aurora_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(aurora_indoor_demand_df_selection_update.head())
print()

aurora_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = aurora_indoor_demand_df_selection_update['deseas'].values[:-1]
aurora_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = aurora_indoor_demand_df_selection_update['deseas'].values[:-12]

aurora_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=aurora_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

aurora_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(aurora_indoor_demand_df_selection_update)

plt.figure()
plt.plot(aurora_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(aurora_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(aurora_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

aurora_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
aurora_indoor_demands['synthetic2_noise'] = norm.rvs(aurora_indoor_demand_df_selection_update['ar_resid'].mean(), aurora_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(aurora_indoor_demand_df_selection_update['ar_resid'])
plt.plot(aurora_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = aurora_indoor_demands.shape[0]

synth_ar = list(aurora_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = aurora_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
aurora_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
aurora_indoor_demands

aurora_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(aurora_indoor_demands), freq='M')
aurora_indoor_demands.index = aurora_indoor_demands['date']
aurora_indoor_demands['month'] = aurora_indoor_demands.index.month
aurora_indoor_demands['year'] = aurora_indoor_demands.index.year


aurora_indoor_demands['synthetic_demand'] = aurora_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    aurora_indoor_demands['synthetic_demand'].loc[aurora_indoor_demands['month'] == i] = aurora_indoor_demands['synthetic_demand'].loc[aurora_indoor_demands['month'] == i] * sigma + mu
    
aurora_indoor_demands['6000'] = aurora_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(aurora_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('Aurora Indoor Demands')
plt.plot(aurora_indoor_demands['6000'])

plt.figure()
plt.title('Aurora Historical')
plt.plot(aurora_indoor_demand_df_selection_update['column'])

#AURORA OUTDOOR

aurora_outdoor_demand_df = df.loc[df['id'] == '08_Aurora_I']
aurora_outdoor_demand_df['yr'] = aurora_outdoor_demand_df['yr'].astype(int)
print(aurora_outdoor_demand_df.dtypes)
aurora_outdoor_demand_df_selection = aurora_outdoor_demand_df.iloc[46:63, 2:14]
aurora_outdoor_demand_df_selection_update = pd.DataFrame(aurora_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
aurora_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(aurora_outdoor_demand_df_selection_update), freq='M')
aurora_outdoor_demand_df_selection_update.index = aurora_outdoor_demand_df_selection_update['date']
aurora_outdoor_demand_df_selection_update['month'] = aurora_outdoor_demand_df_selection_update.index.month
aurora_outdoor_demand_df_selection_update['year'] = aurora_outdoor_demand_df_selection_update.index.year
aurora_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = aurora_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == aurora_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    aurora_outdoor_demand_df_selection_update['norm'][i] = aurora_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = aurora_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = aurora_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

aurora_outdoor_demand_df_selection_update['deseas'] = aurora_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    aurora_outdoor_demand_df_selection_update['deseas'].loc[aurora_outdoor_demand_df_selection_update['month'] == i] = (aurora_outdoor_demand_df_selection_update['deseas'].loc[aurora_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(aurora_outdoor_demand_df_selection_update['deseas'])

aurora_outdoor_demand_df_selection_update = aurora_outdoor_demand_df_selection_update.fillna(0)

aurora_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
aurora_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(aurora_outdoor_demand_df_selection_update.head())
print()

aurora_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = aurora_outdoor_demand_df_selection_update['deseas'].values[:-1]
aurora_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = aurora_outdoor_demand_df_selection_update['deseas'].values[:-12]

aurora_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=aurora_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

aurora_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(aurora_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(aurora_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(aurora_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(aurora_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

aurora_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
aurora_outdoor_demands['synthetic2_noise'] = norm.rvs(aurora_outdoor_demand_df_selection_update['ar_resid'].mean(), aurora_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(aurora_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(aurora_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = aurora_outdoor_demands.shape[0]

synth_ar = list(aurora_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = aurora_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
aurora_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
aurora_outdoor_demands

aurora_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(aurora_outdoor_demands), freq='M')
aurora_outdoor_demands.index = aurora_outdoor_demands['date']
aurora_outdoor_demands['month'] = aurora_outdoor_demands.index.month
aurora_outdoor_demands['year'] = aurora_outdoor_demands.index.year


aurora_outdoor_demands['synthetic_demand'] = aurora_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    aurora_outdoor_demands['synthetic_demand'].loc[aurora_outdoor_demands['month'] == i] = aurora_outdoor_demands['synthetic_demand'].loc[aurora_outdoor_demands['month'] == i] * sigma + mu

aurora_outdoor_demands['synthetic_demand'][aurora_outdoor_demands['synthetic_demand'] < 0] = 0
    
aurora_outdoor_demands['80000'] = aurora_outdoor_demands['synthetic_demand']*80000



plt.figure()
plt.plot(aurora_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('Aurora Outdoor Demands')
plt.plot(aurora_outdoor_demands['80000'])

plt.figure()
plt.title('Aurora Historical')
plt.plot(aurora_outdoor_demand_df_selection_update['column'])

#ENGLEWOOD INDOOR
englewood_indoor_demand_df = df.loc[df['id'] == '08_Englwd_O']
englewood_indoor_demand_df['yr'] = englewood_indoor_demand_df['yr'].astype(int)
print(englewood_indoor_demand_df.dtypes)
englewood_indoor_demand_df_selection = englewood_indoor_demand_df.iloc[46:63, 2:14]
englewood_indoor_demand_df_selection_update = pd.DataFrame(englewood_indoor_demand_df_selection.values.ravel(),columns = ['column'])
englewood_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(englewood_indoor_demand_df_selection_update), freq='M')
englewood_indoor_demand_df_selection_update.index = englewood_indoor_demand_df_selection_update['date']
englewood_indoor_demand_df_selection_update['month'] = englewood_indoor_demand_df_selection_update.index.month
englewood_indoor_demand_df_selection_update['year'] = englewood_indoor_demand_df_selection_update.index.year
englewood_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = englewood_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == englewood_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    englewood_indoor_demand_df_selection_update['norm'][i] = englewood_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = englewood_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = englewood_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

englewood_indoor_demand_df_selection_update['deseas'] = englewood_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    englewood_indoor_demand_df_selection_update['deseas'].loc[englewood_indoor_demand_df_selection_update['month'] == i] = (englewood_indoor_demand_df_selection_update['deseas'].loc[englewood_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(englewood_indoor_demand_df_selection_update['deseas'])

englewood_indoor_demand_df_selection_update = englewood_indoor_demand_df_selection_update.fillna(0)

englewood_indoor_demand_df_selection_update['deseas_l1'] = np.nan
englewood_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(englewood_indoor_demand_df_selection_update.head())
print()

englewood_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = englewood_indoor_demand_df_selection_update['deseas'].values[:-1]
englewood_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = englewood_indoor_demand_df_selection_update['deseas'].values[:-12]

englewood_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=englewood_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

englewood_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(englewood_indoor_demand_df_selection_update)

plt.figure()
plt.plot(englewood_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(englewood_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(englewood_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

englewood_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
englewood_indoor_demands['synthetic2_noise'] = norm.rvs(englewood_indoor_demand_df_selection_update['ar_resid'].mean(), englewood_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(englewood_indoor_demand_df_selection_update['ar_resid'])
plt.plot(englewood_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = englewood_indoor_demands.shape[0]

synth_ar = list(englewood_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = englewood_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
englewood_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
englewood_indoor_demands

englewood_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(englewood_indoor_demands), freq='M')
englewood_indoor_demands.index = englewood_indoor_demands['date']
englewood_indoor_demands['month'] = englewood_indoor_demands.index.month
englewood_indoor_demands['year'] = englewood_indoor_demands.index.year


englewood_indoor_demands['synthetic_demand'] = englewood_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    englewood_indoor_demands['synthetic_demand'].loc[englewood_indoor_demands['month'] == i] = englewood_indoor_demands['synthetic_demand'].loc[englewood_indoor_demands['month'] == i] * sigma + mu
    
englewood_indoor_demands['6000'] = englewood_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(englewood_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('englewood Indoor Demands')
plt.plot(englewood_indoor_demands['6000'])

plt.figure()
plt.title('englewood Historical')
plt.plot(englewood_indoor_demand_df_selection_update['column'])

#ENGLEWOOD OUTDOOR

englewood_outdoor_demand_df = df.loc[df['id'] == '08_Englwd_O']
englewood_outdoor_demand_df['yr'] = englewood_outdoor_demand_df['yr'].astype(int)
print(englewood_outdoor_demand_df.dtypes)
englewood_outdoor_demand_df_selection = englewood_outdoor_demand_df.iloc[46:63, 2:14]
englewood_outdoor_demand_df_selection_update = pd.DataFrame(englewood_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
englewood_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(englewood_outdoor_demand_df_selection_update), freq='M')
englewood_outdoor_demand_df_selection_update.index = englewood_outdoor_demand_df_selection_update['date']
englewood_outdoor_demand_df_selection_update['month'] = englewood_outdoor_demand_df_selection_update.index.month
englewood_outdoor_demand_df_selection_update['year'] = englewood_outdoor_demand_df_selection_update.index.year
englewood_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = englewood_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == englewood_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    englewood_outdoor_demand_df_selection_update['norm'][i] = englewood_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = englewood_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = englewood_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

englewood_outdoor_demand_df_selection_update['deseas'] = englewood_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    englewood_outdoor_demand_df_selection_update['deseas'].loc[englewood_outdoor_demand_df_selection_update['month'] == i] = (englewood_outdoor_demand_df_selection_update['deseas'].loc[englewood_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(englewood_outdoor_demand_df_selection_update['deseas'])

englewood_outdoor_demand_df_selection_update = englewood_outdoor_demand_df_selection_update.fillna(0)

englewood_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
englewood_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(englewood_outdoor_demand_df_selection_update.head())
print()

englewood_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = englewood_outdoor_demand_df_selection_update['deseas'].values[:-1]
englewood_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = englewood_outdoor_demand_df_selection_update['deseas'].values[:-12]

englewood_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=englewood_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

englewood_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(englewood_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(englewood_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(englewood_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(englewood_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

englewood_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
englewood_outdoor_demands['synthetic2_noise'] = norm.rvs(englewood_outdoor_demand_df_selection_update['ar_resid'].mean(), englewood_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(englewood_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(englewood_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = englewood_outdoor_demands.shape[0]

synth_ar = list(englewood_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = englewood_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
englewood_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
englewood_outdoor_demands

englewood_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(englewood_outdoor_demands), freq='M')
englewood_outdoor_demands.index = englewood_outdoor_demands['date']
englewood_outdoor_demands['month'] = englewood_outdoor_demands.index.month
englewood_outdoor_demands['year'] = englewood_outdoor_demands.index.year


englewood_outdoor_demands['synthetic_demand'] = englewood_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    englewood_outdoor_demands['synthetic_demand'].loc[englewood_outdoor_demands['month'] == i] = englewood_outdoor_demands['synthetic_demand'].loc[englewood_outdoor_demands['month'] == i] * sigma + mu

englewood_outdoor_demands['synthetic_demand'][englewood_outdoor_demands['synthetic_demand'] < 0] = 0
    
englewood_outdoor_demands['2500'] = englewood_outdoor_demands['synthetic_demand']*2500



plt.figure()
plt.plot(englewood_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('Englewood Outdoor Demands')
plt.plot(englewood_outdoor_demands['2500'])

plt.figure()
plt.title('Englewood Historical')
plt.plot(englewood_outdoor_demand_df_selection_update['column'])

#NORTHGLENN INDOOR

northglenn_indoor_demand_df = df.loc[df['id'] == '02_Nglenn_I']
northglenn_indoor_demand_df['yr'] = northglenn_indoor_demand_df['yr'].astype(int)
print(northglenn_indoor_demand_df.dtypes)
northglenn_indoor_demand_df_selection = northglenn_indoor_demand_df.iloc[46:63, 2:14]
northglenn_indoor_demand_df_selection_update = pd.DataFrame(northglenn_indoor_demand_df_selection.values.ravel(),columns = ['column'])
northglenn_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(northglenn_indoor_demand_df_selection_update), freq='M')
northglenn_indoor_demand_df_selection_update.index = northglenn_indoor_demand_df_selection_update['date']
northglenn_indoor_demand_df_selection_update['month'] = northglenn_indoor_demand_df_selection_update.index.month
northglenn_indoor_demand_df_selection_update['year'] = northglenn_indoor_demand_df_selection_update.index.year
northglenn_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = northglenn_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == northglenn_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    northglenn_indoor_demand_df_selection_update['norm'][i] = northglenn_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = northglenn_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = northglenn_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

northglenn_indoor_demand_df_selection_update['deseas'] = northglenn_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    northglenn_indoor_demand_df_selection_update['deseas'].loc[northglenn_indoor_demand_df_selection_update['month'] == i] = (northglenn_indoor_demand_df_selection_update['deseas'].loc[northglenn_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(northglenn_indoor_demand_df_selection_update['deseas'])

northglenn_indoor_demand_df_selection_update = northglenn_indoor_demand_df_selection_update.fillna(0)

northglenn_indoor_demand_df_selection_update['deseas_l1'] = np.nan
northglenn_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(northglenn_indoor_demand_df_selection_update.head())
print()

northglenn_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = northglenn_indoor_demand_df_selection_update['deseas'].values[:-1]
northglenn_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = northglenn_indoor_demand_df_selection_update['deseas'].values[:-12]

northglenn_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=northglenn_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

northglenn_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(northglenn_indoor_demand_df_selection_update)

plt.figure()
plt.plot(northglenn_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(northglenn_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(northglenn_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

northglenn_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
northglenn_indoor_demands['synthetic2_noise'] = norm.rvs(northglenn_indoor_demand_df_selection_update['ar_resid'].mean(), northglenn_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(northglenn_indoor_demand_df_selection_update['ar_resid'])
plt.plot(northglenn_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = northglenn_indoor_demands.shape[0]

synth_ar = list(northglenn_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = northglenn_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
northglenn_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
northglenn_indoor_demands

northglenn_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(northglenn_indoor_demands), freq='M')
northglenn_indoor_demands.index = northglenn_indoor_demands['date']
northglenn_indoor_demands['month'] = northglenn_indoor_demands.index.month
northglenn_indoor_demands['year'] = northglenn_indoor_demands.index.year


northglenn_indoor_demands['synthetic_demand'] = northglenn_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    northglenn_indoor_demands['synthetic_demand'].loc[northglenn_indoor_demands['month'] == i] = northglenn_indoor_demands['synthetic_demand'].loc[northglenn_indoor_demands['month'] == i] * sigma + mu
    
northglenn_indoor_demands['6000'] = northglenn_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(northglenn_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('Northglenn Indoor Demands')
plt.plot(northglenn_indoor_demands['6000'])

plt.figure()
plt.title('Northglenn Historical')
plt.plot(northglenn_indoor_demand_df_selection_update['column'])

#NORTHGLENN OUTDOOR

northglenn_outdoor_demand_df = df.loc[df['id'] == '02_Nglenn_O']
northglenn_outdoor_demand_df['yr'] = northglenn_outdoor_demand_df['yr'].astype(int)
print(northglenn_outdoor_demand_df.dtypes)
northglenn_outdoor_demand_df_selection = northglenn_outdoor_demand_df.iloc[46:63, 2:14]
northglenn_outdoor_demand_df_selection_update = pd.DataFrame(northglenn_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
northglenn_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(northglenn_outdoor_demand_df_selection_update), freq='M')
northglenn_outdoor_demand_df_selection_update.index = northglenn_outdoor_demand_df_selection_update['date']
northglenn_outdoor_demand_df_selection_update['month'] = northglenn_outdoor_demand_df_selection_update.index.month
northglenn_outdoor_demand_df_selection_update['year'] = northglenn_outdoor_demand_df_selection_update.index.year
northglenn_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = northglenn_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == northglenn_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    northglenn_outdoor_demand_df_selection_update['norm'][i] = northglenn_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = northglenn_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = northglenn_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

northglenn_outdoor_demand_df_selection_update['deseas'] = northglenn_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    northglenn_outdoor_demand_df_selection_update['deseas'].loc[northglenn_outdoor_demand_df_selection_update['month'] == i] = (northglenn_outdoor_demand_df_selection_update['deseas'].loc[northglenn_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(northglenn_outdoor_demand_df_selection_update['deseas'])

northglenn_outdoor_demand_df_selection_update = northglenn_outdoor_demand_df_selection_update.fillna(0)

northglenn_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
northglenn_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(northglenn_outdoor_demand_df_selection_update.head())
print()

northglenn_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = northglenn_outdoor_demand_df_selection_update['deseas'].values[:-1]
northglenn_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = northglenn_outdoor_demand_df_selection_update['deseas'].values[:-12]

northglenn_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=northglenn_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

northglenn_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(northglenn_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(northglenn_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(northglenn_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(northglenn_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

northglenn_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
northglenn_outdoor_demands['synthetic2_noise'] = norm.rvs(northglenn_outdoor_demand_df_selection_update['ar_resid'].mean(), northglenn_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(northglenn_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(northglenn_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = northglenn_outdoor_demands.shape[0]

synth_ar = list(northglenn_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = northglenn_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
northglenn_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
northglenn_outdoor_demands

northglenn_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(northglenn_outdoor_demands), freq='M')
northglenn_outdoor_demands.index = northglenn_outdoor_demands['date']
northglenn_outdoor_demands['month'] = northglenn_outdoor_demands.index.month
northglenn_outdoor_demands['year'] = northglenn_outdoor_demands.index.year


northglenn_outdoor_demands['synthetic_demand'] = northglenn_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    northglenn_outdoor_demands['synthetic_demand'].loc[northglenn_outdoor_demands['month'] == i] = northglenn_outdoor_demands['synthetic_demand'].loc[northglenn_outdoor_demands['month'] == i] * sigma + mu
    
northglenn_outdoor_demands['6000'] = northglenn_outdoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(northglenn_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('Northglenn Outdoor Demands')
plt.plot(northglenn_outdoor_demands['6000'])

plt.figure()
plt.title('Northglenn Historical')
plt.plot(northglenn_outdoor_demand_df_selection_update['column'])



# #WESTMINSTER INDOOR

westminster_indoor_demand_df = df.loc[df['id'] == '02_Westy_I']
westminster_indoor_demand_df['yr'] = westminster_indoor_demand_df['yr'].astype(int)
print(westminster_indoor_demand_df.dtypes)
westminster_indoor_demand_df_selection = westminster_indoor_demand_df.iloc[46:63, 2:14]
westminster_indoor_demand_df_selection_update = pd.DataFrame(westminster_indoor_demand_df_selection.values.ravel(),columns = ['column'])
westminster_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(westminster_indoor_demand_df_selection_update), freq='M')
westminster_indoor_demand_df_selection_update.index = westminster_indoor_demand_df_selection_update['date']
westminster_indoor_demand_df_selection_update['month'] = westminster_indoor_demand_df_selection_update.index.month
westminster_indoor_demand_df_selection_update['year'] = westminster_indoor_demand_df_selection_update.index.year
westminster_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = westminster_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == westminster_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    westminster_indoor_demand_df_selection_update['norm'][i] = westminster_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = westminster_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = westminster_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

westminster_indoor_demand_df_selection_update['deseas'] = westminster_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    westminster_indoor_demand_df_selection_update['deseas'].loc[westminster_indoor_demand_df_selection_update['month'] == i] = (westminster_indoor_demand_df_selection_update['deseas'].loc[westminster_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(westminster_indoor_demand_df_selection_update['deseas'])

westminster_indoor_demand_df_selection_update = westminster_indoor_demand_df_selection_update.fillna(0)

westminster_indoor_demand_df_selection_update['deseas_l1'] = np.nan
westminster_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(westminster_indoor_demand_df_selection_update.head())
print()

westminster_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = westminster_indoor_demand_df_selection_update['deseas'].values[:-1]
westminster_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = westminster_indoor_demand_df_selection_update['deseas'].values[:-12]

westminster_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=westminster_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

westminster_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(westminster_indoor_demand_df_selection_update)

plt.figure()
plt.plot(westminster_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(westminster_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(westminster_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

westminster_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
westminster_indoor_demands['synthetic2_noise'] = norm.rvs(westminster_indoor_demand_df_selection_update['ar_resid'].mean(), westminster_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(westminster_indoor_demand_df_selection_update['ar_resid'])
plt.plot(westminster_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = westminster_indoor_demands.shape[0]

synth_ar = list(westminster_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = westminster_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
westminster_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
westminster_indoor_demands

westminster_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(westminster_indoor_demands), freq='M')
westminster_indoor_demands.index = westminster_indoor_demands['date']
westminster_indoor_demands['month'] = westminster_indoor_demands.index.month
westminster_indoor_demands['year'] = westminster_indoor_demands.index.year


westminster_indoor_demands['synthetic_demand'] = westminster_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    westminster_indoor_demands['synthetic_demand'].loc[westminster_indoor_demands['month'] == i] = westminster_indoor_demands['synthetic_demand'].loc[westminster_indoor_demands['month'] == i] * sigma + mu
    
westminster_indoor_demands['6000'] = westminster_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(westminster_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('Westminster Indoor Demands')
plt.plot(westminster_indoor_demands['6000'])

plt.figure()
plt.title('Westminster Historical')
plt.plot(westminster_indoor_demand_df_selection_update['column'])

# #WESTMINSTER OUTDOOR

westminster_outdoor_demand_df = df.loc[df['id'] == '02_Westy_O']
westminster_outdoor_demand_df['yr'] = westminster_outdoor_demand_df['yr'].astype(int)
print(westminster_outdoor_demand_df.dtypes)
westminster_outdoor_demand_df_selection = westminster_outdoor_demand_df.iloc[46:63, 2:14]
westminster_outdoor_demand_df_selection_update = pd.DataFrame(westminster_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
westminster_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(westminster_outdoor_demand_df_selection_update), freq='M')
westminster_outdoor_demand_df_selection_update.index = westminster_outdoor_demand_df_selection_update['date']
westminster_outdoor_demand_df_selection_update['month'] = westminster_outdoor_demand_df_selection_update.index.month
westminster_outdoor_demand_df_selection_update['year'] = westminster_outdoor_demand_df_selection_update.index.year
westminster_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = westminster_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == westminster_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    westminster_outdoor_demand_df_selection_update['norm'][i] = westminster_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = westminster_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = westminster_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

westminster_outdoor_demand_df_selection_update['deseas'] = westminster_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    westminster_outdoor_demand_df_selection_update['deseas'].loc[westminster_outdoor_demand_df_selection_update['month'] == i] = (westminster_outdoor_demand_df_selection_update['deseas'].loc[westminster_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(westminster_outdoor_demand_df_selection_update['deseas'])

westminster_outdoor_demand_df_selection_update = westminster_outdoor_demand_df_selection_update.fillna(0)

westminster_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
westminster_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(westminster_outdoor_demand_df_selection_update.head())
print()

westminster_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = westminster_outdoor_demand_df_selection_update['deseas'].values[:-1]
westminster_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = westminster_outdoor_demand_df_selection_update['deseas'].values[:-12]

westminster_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=westminster_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

westminster_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(westminster_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(westminster_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(westminster_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(westminster_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

westminster_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
westminster_outdoor_demands['synthetic2_noise'] = norm.rvs(westminster_outdoor_demand_df_selection_update['ar_resid'].mean(), westminster_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(westminster_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(westminster_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = westminster_outdoor_demands.shape[0]

synth_ar = list(westminster_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = westminster_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
westminster_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
westminster_outdoor_demands

westminster_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(westminster_outdoor_demands), freq='M')
westminster_outdoor_demands.index = westminster_outdoor_demands['date']
westminster_outdoor_demands['month'] = westminster_outdoor_demands.index.month
westminster_outdoor_demands['year'] = westminster_outdoor_demands.index.year


westminster_outdoor_demands['synthetic_demand'] = westminster_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    westminster_outdoor_demands['synthetic_demand'].loc[westminster_outdoor_demands['month'] == i] = westminster_outdoor_demands['synthetic_demand'].loc[westminster_outdoor_demands['month'] == i] * sigma + mu

westminster_outdoor_demands['synthetic_demand'][westminster_outdoor_demands['synthetic_demand'] < 0] = 0
    
westminster_outdoor_demands['6000'] = westminster_outdoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(westminster_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('westminster outdoor Demands')
plt.plot(westminster_outdoor_demands['6000'])

plt.figure()
plt.title('westminster Historical')
plt.plot(westminster_outdoor_demand_df_selection_update['column'])


#THORNTON INDOOR

thornton_indoor_demand_df = df.loc[df['id'] == '02_Thorn_I']
thornton_indoor_demand_df['yr'] = thornton_indoor_demand_df['yr'].astype(int)
print(thornton_indoor_demand_df.dtypes)
thornton_indoor_demand_df_selection = thornton_indoor_demand_df.iloc[46:63, 2:14]
thornton_indoor_demand_df_selection_update = pd.DataFrame(thornton_indoor_demand_df_selection.values.ravel(),columns = ['column'])
thornton_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(thornton_indoor_demand_df_selection_update), freq='M')
thornton_indoor_demand_df_selection_update.index = thornton_indoor_demand_df_selection_update['date']
thornton_indoor_demand_df_selection_update['month'] = thornton_indoor_demand_df_selection_update.index.month
thornton_indoor_demand_df_selection_update['year'] = thornton_indoor_demand_df_selection_update.index.year
thornton_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = thornton_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == thornton_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    thornton_indoor_demand_df_selection_update['norm'][i] = thornton_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = thornton_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = thornton_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

thornton_indoor_demand_df_selection_update['deseas'] = thornton_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    thornton_indoor_demand_df_selection_update['deseas'].loc[thornton_indoor_demand_df_selection_update['month'] == i] = (thornton_indoor_demand_df_selection_update['deseas'].loc[thornton_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(thornton_indoor_demand_df_selection_update['deseas'])

thornton_indoor_demand_df_selection_update = thornton_indoor_demand_df_selection_update.fillna(0)

thornton_indoor_demand_df_selection_update['deseas_l1'] = np.nan
thornton_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(thornton_indoor_demand_df_selection_update.head())
print()

thornton_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = thornton_indoor_demand_df_selection_update['deseas'].values[:-1]
thornton_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = thornton_indoor_demand_df_selection_update['deseas'].values[:-12]

thornton_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=thornton_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

thornton_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(thornton_indoor_demand_df_selection_update)

plt.figure()
plt.plot(thornton_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(thornton_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(thornton_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

thornton_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
thornton_indoor_demands['synthetic2_noise'] = norm.rvs(thornton_indoor_demand_df_selection_update['ar_resid'].mean(), thornton_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(thornton_indoor_demand_df_selection_update['ar_resid'])
plt.plot(thornton_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = thornton_indoor_demands.shape[0]

synth_ar = list(thornton_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = thornton_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
thornton_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
thornton_indoor_demands

thornton_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(thornton_indoor_demands), freq='M')
thornton_indoor_demands.index = thornton_indoor_demands['date']
thornton_indoor_demands['month'] = thornton_indoor_demands.index.month
thornton_indoor_demands['year'] = thornton_indoor_demands.index.year


thornton_indoor_demands['synthetic_demand'] = thornton_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    thornton_indoor_demands['synthetic_demand'].loc[thornton_indoor_demands['month'] == i] = thornton_indoor_demands['synthetic_demand'].loc[thornton_indoor_demands['month'] == i] * sigma + mu
    
thornton_indoor_demands['6000'] = thornton_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(thornton_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('Thornton Indoor Demands')
plt.plot(thornton_indoor_demands['6000'])

plt.figure()
plt.title('Thornton Historical')
plt.plot(thornton_indoor_demand_df_selection_update['column'])

#THORNTON OUTDOOR

thornton_outdoor_demand_df = df.loc[df['id'] == '02_Thorn_O']
thornton_outdoor_demand_df['yr'] = thornton_outdoor_demand_df['yr'].astype(int)
print(thornton_outdoor_demand_df.dtypes)
thornton_outdoor_demand_df_selection = thornton_outdoor_demand_df.iloc[46:63, 2:14]
thornton_outdoor_demand_df_selection_update = pd.DataFrame(thornton_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
thornton_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(thornton_outdoor_demand_df_selection_update), freq='M')
thornton_outdoor_demand_df_selection_update.index = thornton_outdoor_demand_df_selection_update['date']
thornton_outdoor_demand_df_selection_update['month'] = thornton_outdoor_demand_df_selection_update.index.month
thornton_outdoor_demand_df_selection_update['year'] = thornton_outdoor_demand_df_selection_update.index.year
thornton_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = thornton_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == thornton_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    thornton_outdoor_demand_df_selection_update['norm'][i] = thornton_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = thornton_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = thornton_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

thornton_outdoor_demand_df_selection_update['deseas'] = thornton_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    thornton_outdoor_demand_df_selection_update['deseas'].loc[thornton_outdoor_demand_df_selection_update['month'] == i] = (thornton_outdoor_demand_df_selection_update['deseas'].loc[thornton_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(thornton_outdoor_demand_df_selection_update['deseas'])

thornton_outdoor_demand_df_selection_update = thornton_outdoor_demand_df_selection_update.fillna(0)

thornton_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
thornton_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(thornton_outdoor_demand_df_selection_update.head())
print()

thornton_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = thornton_outdoor_demand_df_selection_update['deseas'].values[:-1]
thornton_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = thornton_outdoor_demand_df_selection_update['deseas'].values[:-12]

thornton_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=thornton_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

thornton_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(thornton_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(thornton_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(thornton_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(thornton_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

thornton_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
thornton_outdoor_demands['synthetic2_noise'] = norm.rvs(thornton_outdoor_demand_df_selection_update['ar_resid'].mean(), thornton_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(thornton_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(thornton_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = thornton_outdoor_demands.shape[0]

synth_ar = list(thornton_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = thornton_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
thornton_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
thornton_outdoor_demands

thornton_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(thornton_outdoor_demands), freq='M')
thornton_outdoor_demands.index = thornton_outdoor_demands['date']
thornton_outdoor_demands['month'] = thornton_outdoor_demands.index.month
thornton_outdoor_demands['year'] = thornton_outdoor_demands.index.year


thornton_outdoor_demands['synthetic_demand'] = thornton_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    thornton_outdoor_demands['synthetic_demand'].loc[thornton_outdoor_demands['month'] == i] = thornton_outdoor_demands['synthetic_demand'].loc[thornton_outdoor_demands['month'] == i] * sigma + mu
 
thornton_outdoor_demands['synthetic_demand'][thornton_outdoor_demands['synthetic_demand'] < 0] = 0    
 
thornton_outdoor_demands['6000'] = thornton_outdoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(thornton_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('Thornton Outdoor Demands')
plt.plot(thornton_outdoor_demands['6000'])

plt.figure()
plt.title('Thornton Historical')
plt.plot(thornton_outdoor_demand_df_selection_update['column'])


#LAFAYETTE INDOOR

lafayette_indoor_demand_df = df.loc[df['id'] == '06LAFFYT_I']
lafayette_indoor_demand_df['yr'] = lafayette_indoor_demand_df['yr'].astype(int)
print(lafayette_indoor_demand_df.dtypes)
lafayette_indoor_demand_df_selection = lafayette_indoor_demand_df.iloc[46:63, 2:14]
lafayette_indoor_demand_df_selection_update = pd.DataFrame(lafayette_indoor_demand_df_selection.values.ravel(),columns = ['column'])
lafayette_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(lafayette_indoor_demand_df_selection_update), freq='M')
lafayette_indoor_demand_df_selection_update.index = lafayette_indoor_demand_df_selection_update['date']
lafayette_indoor_demand_df_selection_update['month'] = lafayette_indoor_demand_df_selection_update.index.month
lafayette_indoor_demand_df_selection_update['year'] = lafayette_indoor_demand_df_selection_update.index.year
lafayette_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = lafayette_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == lafayette_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    lafayette_indoor_demand_df_selection_update['norm'][i] = lafayette_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = lafayette_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = lafayette_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

lafayette_indoor_demand_df_selection_update['deseas'] = lafayette_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    lafayette_indoor_demand_df_selection_update['deseas'].loc[lafayette_indoor_demand_df_selection_update['month'] == i] = (lafayette_indoor_demand_df_selection_update['deseas'].loc[lafayette_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(lafayette_indoor_demand_df_selection_update['deseas'])

lafayette_indoor_demand_df_selection_update = lafayette_indoor_demand_df_selection_update.fillna(0)

lafayette_indoor_demand_df_selection_update['deseas_l1'] = np.nan
lafayette_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(lafayette_indoor_demand_df_selection_update.head())
print()

lafayette_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = lafayette_indoor_demand_df_selection_update['deseas'].values[:-1]
lafayette_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = lafayette_indoor_demand_df_selection_update['deseas'].values[:-12]

lafayette_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=lafayette_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

lafayette_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(lafayette_indoor_demand_df_selection_update)

plt.figure()
plt.plot(lafayette_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(lafayette_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(lafayette_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

lafayette_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
lafayette_indoor_demands['synthetic2_noise'] = norm.rvs(lafayette_indoor_demand_df_selection_update['ar_resid'].mean(), lafayette_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(lafayette_indoor_demand_df_selection_update['ar_resid'])
plt.plot(lafayette_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = lafayette_indoor_demands.shape[0]

synth_ar = list(lafayette_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = lafayette_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
lafayette_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
lafayette_indoor_demands

lafayette_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(lafayette_indoor_demands), freq='M')
lafayette_indoor_demands.index = lafayette_indoor_demands['date']
lafayette_indoor_demands['month'] = lafayette_indoor_demands.index.month
lafayette_indoor_demands['year'] = lafayette_indoor_demands.index.year


lafayette_indoor_demands['synthetic_demand'] = lafayette_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    lafayette_indoor_demands['synthetic_demand'].loc[lafayette_indoor_demands['month'] == i] = lafayette_indoor_demands['synthetic_demand'].loc[lafayette_indoor_demands['month'] == i] * sigma + mu
    
lafayette_indoor_demands['6000'] = lafayette_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(lafayette_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('Lafayette Indoor Demands')
plt.plot(lafayette_indoor_demands['6000'])

plt.figure()
plt.title('Lafayette Historical')
plt.plot(lafayette_indoor_demand_df_selection_update['column'])


#LAFAYETTE OUTDOOR

lafayette_outdoor_demand_df = df.loc[df['id'] == '06LAFFYT_O']
lafayette_outdoor_demand_df['yr'] = lafayette_outdoor_demand_df['yr'].astype(int)
print(lafayette_outdoor_demand_df.dtypes)
lafayette_outdoor_demand_df_selection = lafayette_outdoor_demand_df.iloc[46:63, 2:14]
lafayette_outdoor_demand_df_selection_update = pd.DataFrame(lafayette_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
lafayette_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(lafayette_outdoor_demand_df_selection_update), freq='M')
lafayette_outdoor_demand_df_selection_update.index = lafayette_outdoor_demand_df_selection_update['date']
lafayette_outdoor_demand_df_selection_update['month'] = lafayette_outdoor_demand_df_selection_update.index.month
lafayette_outdoor_demand_df_selection_update['year'] = lafayette_outdoor_demand_df_selection_update.index.year
lafayette_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = lafayette_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == lafayette_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    lafayette_outdoor_demand_df_selection_update['norm'][i] = lafayette_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = lafayette_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = lafayette_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

lafayette_outdoor_demand_df_selection_update['deseas'] = lafayette_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    lafayette_outdoor_demand_df_selection_update['deseas'].loc[lafayette_outdoor_demand_df_selection_update['month'] == i] = (lafayette_outdoor_demand_df_selection_update['deseas'].loc[lafayette_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(lafayette_outdoor_demand_df_selection_update['deseas'])

lafayette_outdoor_demand_df_selection_update = lafayette_outdoor_demand_df_selection_update.fillna(0)

lafayette_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
lafayette_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(lafayette_outdoor_demand_df_selection_update.head())
print()

lafayette_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = lafayette_outdoor_demand_df_selection_update['deseas'].values[:-1]
lafayette_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = lafayette_outdoor_demand_df_selection_update['deseas'].values[:-12]

lafayette_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=lafayette_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

lafayette_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(lafayette_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(lafayette_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(lafayette_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(lafayette_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

lafayette_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
lafayette_outdoor_demands['synthetic2_noise'] = norm.rvs(lafayette_outdoor_demand_df_selection_update['ar_resid'].mean(), lafayette_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(lafayette_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(lafayette_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = lafayette_outdoor_demands.shape[0]

synth_ar = list(lafayette_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = lafayette_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
lafayette_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
lafayette_outdoor_demands

lafayette_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(lafayette_outdoor_demands), freq='M')
lafayette_outdoor_demands.index = lafayette_outdoor_demands['date']
lafayette_outdoor_demands['month'] = lafayette_outdoor_demands.index.month
lafayette_outdoor_demands['year'] = lafayette_outdoor_demands.index.year


lafayette_outdoor_demands['synthetic_demand'] = lafayette_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    lafayette_outdoor_demands['synthetic_demand'].loc[lafayette_outdoor_demands['month'] == i] = lafayette_outdoor_demands['synthetic_demand'].loc[lafayette_outdoor_demands['month'] == i] * sigma + mu
 
lafayette_outdoor_demands['synthetic_demand'][lafayette_outdoor_demands['synthetic_demand'] < 0] = 0    
 
lafayette_outdoor_demands['6000'] = lafayette_outdoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(lafayette_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('Lafayette Outdoor Demands')
plt.plot(lafayette_outdoor_demands['6000'])

plt.figure()
plt.title('Lafayette Historical')
plt.plot(lafayette_outdoor_demand_df_selection_update['column'])

#LOUISVILLE INDOOR

louisville_indoor_demand_df = df.loc[df['id'] == '06LOUIS_I']
louisville_indoor_demand_df['yr'] = louisville_indoor_demand_df['yr'].astype(int)
print(louisville_indoor_demand_df.dtypes)
louisville_indoor_demand_df_selection = louisville_indoor_demand_df.iloc[46:63, 2:14]
louisville_indoor_demand_df_selection_update = pd.DataFrame(louisville_indoor_demand_df_selection.values.ravel(),columns = ['column'])
louisville_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(louisville_indoor_demand_df_selection_update), freq='M')
louisville_indoor_demand_df_selection_update.index = louisville_indoor_demand_df_selection_update['date']
louisville_indoor_demand_df_selection_update['month'] = louisville_indoor_demand_df_selection_update.index.month
louisville_indoor_demand_df_selection_update['year'] = louisville_indoor_demand_df_selection_update.index.year
louisville_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = louisville_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == louisville_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    louisville_indoor_demand_df_selection_update['norm'][i] = louisville_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = louisville_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = louisville_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

louisville_indoor_demand_df_selection_update['deseas'] = louisville_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    louisville_indoor_demand_df_selection_update['deseas'].loc[louisville_indoor_demand_df_selection_update['month'] == i] = (louisville_indoor_demand_df_selection_update['deseas'].loc[louisville_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(louisville_indoor_demand_df_selection_update['deseas'])

louisville_indoor_demand_df_selection_update = louisville_indoor_demand_df_selection_update.fillna(0)

louisville_indoor_demand_df_selection_update['deseas_l1'] = np.nan
louisville_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(louisville_indoor_demand_df_selection_update.head())
print()

louisville_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = louisville_indoor_demand_df_selection_update['deseas'].values[:-1]
louisville_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = louisville_indoor_demand_df_selection_update['deseas'].values[:-12]

louisville_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=louisville_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

louisville_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(louisville_indoor_demand_df_selection_update)

plt.figure()
plt.plot(louisville_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(louisville_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(louisville_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

louisville_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
louisville_indoor_demands['synthetic2_noise'] = norm.rvs(louisville_indoor_demand_df_selection_update['ar_resid'].mean(), louisville_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(louisville_indoor_demand_df_selection_update['ar_resid'])
plt.plot(louisville_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = louisville_indoor_demands.shape[0]

synth_ar = list(louisville_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = louisville_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
louisville_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
louisville_indoor_demands

louisville_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(louisville_indoor_demands), freq='M')
louisville_indoor_demands.index = louisville_indoor_demands['date']
louisville_indoor_demands['month'] = louisville_indoor_demands.index.month
louisville_indoor_demands['year'] = louisville_indoor_demands.index.year


louisville_indoor_demands['synthetic_demand'] = louisville_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    louisville_indoor_demands['synthetic_demand'].loc[louisville_indoor_demands['month'] == i] = louisville_indoor_demands['synthetic_demand'].loc[louisville_indoor_demands['month'] == i] * sigma + mu
    
louisville_indoor_demands['6000'] = louisville_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(louisville_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('louisville Indoor Demands')
plt.plot(louisville_indoor_demands['6000'])

plt.figure()
plt.title('louisville Historical')
plt.plot(louisville_indoor_demand_df_selection_update['column'])


#LOUISVILLE OUTDOOR

louisville_outdoor_demand_df = df.loc[df['id'] == '06LOUIS_O']
louisville_outdoor_demand_df['yr'] = louisville_outdoor_demand_df['yr'].astype(int)
print(louisville_outdoor_demand_df.dtypes)
louisville_outdoor_demand_df_selection = louisville_outdoor_demand_df.iloc[46:63, 2:14]
louisville_outdoor_demand_df_selection_update = pd.DataFrame(louisville_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
louisville_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(louisville_outdoor_demand_df_selection_update), freq='M')
louisville_outdoor_demand_df_selection_update.index = louisville_outdoor_demand_df_selection_update['date']
louisville_outdoor_demand_df_selection_update['month'] = louisville_outdoor_demand_df_selection_update.index.month
louisville_outdoor_demand_df_selection_update['year'] = louisville_outdoor_demand_df_selection_update.index.year
louisville_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = louisville_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == louisville_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    louisville_outdoor_demand_df_selection_update['norm'][i] = louisville_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = louisville_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = louisville_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

louisville_outdoor_demand_df_selection_update['deseas'] = louisville_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    louisville_outdoor_demand_df_selection_update['deseas'].loc[louisville_outdoor_demand_df_selection_update['month'] == i] = (louisville_outdoor_demand_df_selection_update['deseas'].loc[louisville_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(louisville_outdoor_demand_df_selection_update['deseas'])

louisville_outdoor_demand_df_selection_update = louisville_outdoor_demand_df_selection_update.fillna(0)

louisville_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
louisville_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(louisville_outdoor_demand_df_selection_update.head())
print()

louisville_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = louisville_outdoor_demand_df_selection_update['deseas'].values[:-1]
louisville_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = louisville_outdoor_demand_df_selection_update['deseas'].values[:-12]

louisville_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=louisville_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

louisville_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(louisville_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(louisville_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(louisville_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(louisville_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

louisville_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
louisville_outdoor_demands['synthetic2_noise'] = norm.rvs(louisville_outdoor_demand_df_selection_update['ar_resid'].mean(), louisville_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(louisville_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(louisville_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = louisville_outdoor_demands.shape[0]

synth_ar = list(louisville_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = louisville_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
louisville_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
louisville_outdoor_demands

louisville_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(louisville_outdoor_demands), freq='M')
louisville_outdoor_demands.index = louisville_outdoor_demands['date']
louisville_outdoor_demands['month'] = louisville_outdoor_demands.index.month
louisville_outdoor_demands['year'] = louisville_outdoor_demands.index.year


louisville_outdoor_demands['synthetic_demand'] = louisville_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    louisville_outdoor_demands['synthetic_demand'].loc[louisville_outdoor_demands['month'] == i] = louisville_outdoor_demands['synthetic_demand'].loc[louisville_outdoor_demands['month'] == i] * sigma + mu
    
louisville_outdoor_demands['6000'] = louisville_outdoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(louisville_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('louisville outdoor Demands')
plt.plot(louisville_outdoor_demands['6000'])

plt.figure()
plt.title('louisville Historical')
plt.plot(louisville_outdoor_demand_df_selection_update['column'])

#ARVADA INDOOR

arvada_indoor_demand_df = df.loc[df['id'] == '07_Arvada_I']
arvada_indoor_demand_df['yr'] = arvada_indoor_demand_df['yr'].astype(int)
print(arvada_indoor_demand_df.dtypes)
arvada_indoor_demand_df_selection = arvada_indoor_demand_df.iloc[46:63, 2:14]
arvada_indoor_demand_df_selection_update = pd.DataFrame(arvada_indoor_demand_df_selection.values.ravel(),columns = ['column'])
arvada_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(arvada_indoor_demand_df_selection_update), freq='M')
arvada_indoor_demand_df_selection_update.index = arvada_indoor_demand_df_selection_update['date']
arvada_indoor_demand_df_selection_update['month'] = arvada_indoor_demand_df_selection_update.index.month
arvada_indoor_demand_df_selection_update['year'] = arvada_indoor_demand_df_selection_update.index.year
arvada_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = arvada_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == arvada_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    arvada_indoor_demand_df_selection_update['norm'][i] = arvada_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = arvada_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = arvada_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

arvada_indoor_demand_df_selection_update['deseas'] = arvada_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    arvada_indoor_demand_df_selection_update['deseas'].loc[arvada_indoor_demand_df_selection_update['month'] == i] = (arvada_indoor_demand_df_selection_update['deseas'].loc[arvada_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(arvada_indoor_demand_df_selection_update['deseas'])

arvada_indoor_demand_df_selection_update = arvada_indoor_demand_df_selection_update.fillna(0)

arvada_indoor_demand_df_selection_update['deseas_l1'] = np.nan
arvada_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(arvada_indoor_demand_df_selection_update.head())
print()

arvada_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = arvada_indoor_demand_df_selection_update['deseas'].values[:-1]
arvada_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = arvada_indoor_demand_df_selection_update['deseas'].values[:-12]

arvada_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=arvada_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

arvada_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(arvada_indoor_demand_df_selection_update)

plt.figure()
plt.plot(arvada_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(arvada_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(arvada_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

arvada_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
arvada_indoor_demands['synthetic2_noise'] = norm.rvs(arvada_indoor_demand_df_selection_update['ar_resid'].mean(), arvada_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(arvada_indoor_demand_df_selection_update['ar_resid'])
plt.plot(arvada_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = arvada_indoor_demands.shape[0]

synth_ar = list(arvada_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = arvada_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
arvada_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
arvada_indoor_demands

arvada_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(arvada_indoor_demands), freq='M')
arvada_indoor_demands.index = arvada_indoor_demands['date']
arvada_indoor_demands['month'] = arvada_indoor_demands.index.month
arvada_indoor_demands['year'] = arvada_indoor_demands.index.year


arvada_indoor_demands['synthetic_demand'] = arvada_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    arvada_indoor_demands['synthetic_demand'].loc[arvada_indoor_demands['month'] == i] = arvada_indoor_demands['synthetic_demand'].loc[arvada_indoor_demands['month'] == i] * sigma + mu
    
arvada_indoor_demands['6000'] = arvada_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(arvada_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('arvada Indoor Demands')
plt.plot(arvada_indoor_demands['6000'])

plt.figure()
plt.title('arvada Historical')
plt.plot(arvada_indoor_demand_df_selection_update['column'])

#ARVADA OUTDOOR

arvada_outdoor_demand_df = df.loc[df['id'] == '07_Arvada_O']
arvada_outdoor_demand_df['yr'] = arvada_outdoor_demand_df['yr'].astype(int)
print(arvada_outdoor_demand_df.dtypes)
arvada_outdoor_demand_df_selection = arvada_outdoor_demand_df.iloc[46:63, 2:14]
arvada_outdoor_demand_df_selection_update = pd.DataFrame(arvada_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
arvada_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(arvada_outdoor_demand_df_selection_update), freq='M')
arvada_outdoor_demand_df_selection_update.index = arvada_outdoor_demand_df_selection_update['date']
arvada_outdoor_demand_df_selection_update['month'] = arvada_outdoor_demand_df_selection_update.index.month
arvada_outdoor_demand_df_selection_update['year'] = arvada_outdoor_demand_df_selection_update.index.year
arvada_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = arvada_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == arvada_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    arvada_outdoor_demand_df_selection_update['norm'][i] = arvada_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = arvada_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = arvada_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

arvada_outdoor_demand_df_selection_update['deseas'] = arvada_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    arvada_outdoor_demand_df_selection_update['deseas'].loc[arvada_outdoor_demand_df_selection_update['month'] == i] = (arvada_outdoor_demand_df_selection_update['deseas'].loc[arvada_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(arvada_outdoor_demand_df_selection_update['deseas'])

arvada_outdoor_demand_df_selection_update = arvada_outdoor_demand_df_selection_update.fillna(0)

arvada_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
arvada_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(arvada_outdoor_demand_df_selection_update.head())
print()

arvada_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = arvada_outdoor_demand_df_selection_update['deseas'].values[:-1]
arvada_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = arvada_outdoor_demand_df_selection_update['deseas'].values[:-12]

arvada_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=arvada_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

arvada_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(arvada_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(arvada_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(arvada_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(arvada_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

arvada_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
arvada_outdoor_demands['synthetic2_noise'] = norm.rvs(arvada_outdoor_demand_df_selection_update['ar_resid'].mean(), arvada_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(arvada_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(arvada_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = arvada_outdoor_demands.shape[0]

synth_ar = list(arvada_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = arvada_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
arvada_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
arvada_outdoor_demands

arvada_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(arvada_outdoor_demands), freq='M')
arvada_outdoor_demands.index = arvada_outdoor_demands['date']
arvada_outdoor_demands['month'] = arvada_outdoor_demands.index.month
arvada_outdoor_demands['year'] = arvada_outdoor_demands.index.year


arvada_outdoor_demands['synthetic_demand'] = arvada_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    arvada_outdoor_demands['synthetic_demand'].loc[arvada_outdoor_demands['month'] == i] = arvada_outdoor_demands['synthetic_demand'].loc[arvada_outdoor_demands['month'] == i] * sigma + mu
    
arvada_outdoor_demands['6000'] = arvada_outdoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(arvada_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('arvada outdoor Demands')
plt.plot(arvada_outdoor_demands['6000'])

plt.figure()
plt.title('arvada Historical')
plt.plot(arvada_outdoor_demand_df_selection_update['column'])



#CONMUTUAL INDOOR

conmutual_indoor_demand_df = df.loc[df['id'] == '07_ConMut_I']
conmutual_indoor_demand_df['yr'] = conmutual_indoor_demand_df['yr'].astype(int)
print(conmutual_indoor_demand_df.dtypes)
conmutual_indoor_demand_df_selection = conmutual_indoor_demand_df.iloc[46:63, 2:14]
conmutual_indoor_demand_df_selection_update = pd.DataFrame(conmutual_indoor_demand_df_selection.values.ravel(),columns = ['column'])
conmutual_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(conmutual_indoor_demand_df_selection_update), freq='M')
conmutual_indoor_demand_df_selection_update.index = conmutual_indoor_demand_df_selection_update['date']
conmutual_indoor_demand_df_selection_update['month'] = conmutual_indoor_demand_df_selection_update.index.month
conmutual_indoor_demand_df_selection_update['year'] = conmutual_indoor_demand_df_selection_update.index.year
conmutual_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = conmutual_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == conmutual_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    conmutual_indoor_demand_df_selection_update['norm'][i] = conmutual_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = conmutual_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = conmutual_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

conmutual_indoor_demand_df_selection_update['deseas'] = conmutual_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    conmutual_indoor_demand_df_selection_update['deseas'].loc[conmutual_indoor_demand_df_selection_update['month'] == i] = (conmutual_indoor_demand_df_selection_update['deseas'].loc[conmutual_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(conmutual_indoor_demand_df_selection_update['deseas'])

conmutual_indoor_demand_df_selection_update = conmutual_indoor_demand_df_selection_update.fillna(0)

conmutual_indoor_demand_df_selection_update['deseas_l1'] = np.nan
conmutual_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(conmutual_indoor_demand_df_selection_update.head())
print()

conmutual_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = conmutual_indoor_demand_df_selection_update['deseas'].values[:-1]
conmutual_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = conmutual_indoor_demand_df_selection_update['deseas'].values[:-12]

conmutual_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=conmutual_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

conmutual_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(conmutual_indoor_demand_df_selection_update)

plt.figure()
plt.plot(conmutual_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(conmutual_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(conmutual_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

conmutual_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
conmutual_indoor_demands['synthetic2_noise'] = norm.rvs(conmutual_indoor_demand_df_selection_update['ar_resid'].mean(), conmutual_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(conmutual_indoor_demand_df_selection_update['ar_resid'])
plt.plot(conmutual_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = conmutual_indoor_demands.shape[0]

synth_ar = list(conmutual_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = conmutual_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
conmutual_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
conmutual_indoor_demands

conmutual_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(conmutual_indoor_demands), freq='M')
conmutual_indoor_demands.index = conmutual_indoor_demands['date']
conmutual_indoor_demands['month'] = conmutual_indoor_demands.index.month
conmutual_indoor_demands['year'] = conmutual_indoor_demands.index.year


conmutual_indoor_demands['synthetic_demand'] = conmutual_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    conmutual_indoor_demands['synthetic_demand'].loc[conmutual_indoor_demands['month'] == i] = conmutual_indoor_demands['synthetic_demand'].loc[conmutual_indoor_demands['month'] == i] * sigma + mu
    
conmutual_indoor_demands['6000'] = conmutual_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(conmutual_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('conmutual Indoor Demands')
plt.plot(conmutual_indoor_demands['6000'])

plt.figure()
plt.title('conmutual Historical')
plt.plot(conmutual_indoor_demand_df_selection_update['column'])


#CONMUTUAL OUTDOOR

conmutual_outdoor_demand_df = df.loc[df['id'] == '07_ConMut_O']
conmutual_outdoor_demand_df['yr'] = conmutual_outdoor_demand_df['yr'].astype(int)
print(conmutual_outdoor_demand_df.dtypes)
conmutual_outdoor_demand_df_selection = conmutual_outdoor_demand_df.iloc[46:63, 2:14]
conmutual_outdoor_demand_df_selection_update = pd.DataFrame(conmutual_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
conmutual_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(conmutual_outdoor_demand_df_selection_update), freq='M')
conmutual_outdoor_demand_df_selection_update.index = conmutual_outdoor_demand_df_selection_update['date']
conmutual_outdoor_demand_df_selection_update['month'] = conmutual_outdoor_demand_df_selection_update.index.month
conmutual_outdoor_demand_df_selection_update['year'] = conmutual_outdoor_demand_df_selection_update.index.year
conmutual_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = conmutual_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == conmutual_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    conmutual_outdoor_demand_df_selection_update['norm'][i] = conmutual_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = conmutual_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = conmutual_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

conmutual_outdoor_demand_df_selection_update['deseas'] = conmutual_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    conmutual_outdoor_demand_df_selection_update['deseas'].loc[conmutual_outdoor_demand_df_selection_update['month'] == i] = (conmutual_outdoor_demand_df_selection_update['deseas'].loc[conmutual_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(conmutual_outdoor_demand_df_selection_update['deseas'])

conmutual_outdoor_demand_df_selection_update = conmutual_outdoor_demand_df_selection_update.fillna(0)

conmutual_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
conmutual_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(conmutual_outdoor_demand_df_selection_update.head())
print()

conmutual_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = conmutual_outdoor_demand_df_selection_update['deseas'].values[:-1]
conmutual_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = conmutual_outdoor_demand_df_selection_update['deseas'].values[:-12]

conmutual_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=conmutual_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

conmutual_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(conmutual_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(conmutual_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(conmutual_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(conmutual_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

conmutual_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
conmutual_outdoor_demands['synthetic2_noise'] = norm.rvs(conmutual_outdoor_demand_df_selection_update['ar_resid'].mean(), conmutual_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(conmutual_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(conmutual_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = conmutual_outdoor_demands.shape[0]

synth_ar = list(conmutual_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = conmutual_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
conmutual_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
conmutual_outdoor_demands

conmutual_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(conmutual_outdoor_demands), freq='M')
conmutual_outdoor_demands.index = conmutual_outdoor_demands['date']
conmutual_outdoor_demands['month'] = conmutual_outdoor_demands.index.month
conmutual_outdoor_demands['year'] = conmutual_outdoor_demands.index.year


conmutual_outdoor_demands['synthetic_demand'] = conmutual_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    conmutual_outdoor_demands['synthetic_demand'].loc[conmutual_outdoor_demands['month'] == i] = conmutual_outdoor_demands['synthetic_demand'].loc[conmutual_outdoor_demands['month'] == i] * sigma + mu
    
conmutual_outdoor_demands['6000'] = conmutual_outdoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(conmutual_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('conmutual outdoor Demands')
plt.plot(conmutual_outdoor_demands['6000'])

plt.figure()
plt.title('conmutual Historical')
plt.plot(conmutual_outdoor_demand_df_selection_update['column'])

#GOLDEN INDOOR

golden_indoor_demand_df = df.loc[df['id'] == '07_Golden_I']
golden_indoor_demand_df['yr'] = golden_indoor_demand_df['yr'].astype(int)
print(golden_indoor_demand_df.dtypes)
golden_indoor_demand_df_selection = golden_indoor_demand_df.iloc[46:63, 2:14]
golden_indoor_demand_df_selection_update = pd.DataFrame(golden_indoor_demand_df_selection.values.ravel(),columns = ['column'])
golden_indoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(golden_indoor_demand_df_selection_update), freq='M')
golden_indoor_demand_df_selection_update.index = golden_indoor_demand_df_selection_update['date']
golden_indoor_demand_df_selection_update['month'] = golden_indoor_demand_df_selection_update.index.month
golden_indoor_demand_df_selection_update['year'] = golden_indoor_demand_df_selection_update.index.year
golden_indoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = golden_indoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == golden_indoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    golden_indoor_demand_df_selection_update['norm'][i] = golden_indoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = golden_indoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = golden_indoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

golden_indoor_demand_df_selection_update['deseas'] = golden_indoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    golden_indoor_demand_df_selection_update['deseas'].loc[golden_indoor_demand_df_selection_update['month'] == i] = (golden_indoor_demand_df_selection_update['deseas'].loc[golden_indoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(golden_indoor_demand_df_selection_update['deseas'])

golden_indoor_demand_df_selection_update = golden_indoor_demand_df_selection_update.fillna(0)

golden_indoor_demand_df_selection_update['deseas_l1'] = np.nan
golden_indoor_demand_df_selection_update['deseas_l12'] = np.nan
print(golden_indoor_demand_df_selection_update.head())
print()

golden_indoor_demand_df_selection_update['deseas_l1'].iloc[1:] = golden_indoor_demand_df_selection_update['deseas'].values[:-1]
golden_indoor_demand_df_selection_update['deseas_l12'].iloc[12:] = golden_indoor_demand_df_selection_update['deseas'].values[:-12]

golden_indoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=golden_indoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

golden_indoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(golden_indoor_demand_df_selection_update)

plt.figure()
plt.plot(golden_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(golden_indoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(golden_indoor_demand_df_selection_update['ar_resid'].iloc[12:])

golden_indoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
golden_indoor_demands['synthetic2_noise'] = norm.rvs(golden_indoor_demand_df_selection_update['ar_resid'].mean(), golden_indoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(golden_indoor_demand_df_selection_update['ar_resid'])
plt.plot(golden_indoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = golden_indoor_demands.shape[0]

synth_ar = list(golden_indoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = golden_indoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
golden_indoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
golden_indoor_demands

golden_indoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(golden_indoor_demands), freq='M')
golden_indoor_demands.index = golden_indoor_demands['date']
golden_indoor_demands['month'] = golden_indoor_demands.index.month
golden_indoor_demands['year'] = golden_indoor_demands.index.year


golden_indoor_demands['synthetic_demand'] = golden_indoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    golden_indoor_demands['synthetic_demand'].loc[golden_indoor_demands['month'] == i] = golden_indoor_demands['synthetic_demand'].loc[golden_indoor_demands['month'] == i] * sigma + mu
    
golden_indoor_demands['6000'] = golden_indoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(golden_indoor_demands['synthetic_demand'])


plt.figure()
plt.title('golden Indoor Demands')
plt.plot(golden_indoor_demands['6000'])

plt.figure()
plt.title('golden Historical')
plt.plot(golden_indoor_demand_df_selection_update['column'])


#GOLDEN OUTDOOR

golden_outdoor_demand_df = df.loc[df['id'] == '07_Golden_O']
golden_outdoor_demand_df['yr'] = golden_outdoor_demand_df['yr'].astype(int)
print(golden_outdoor_demand_df.dtypes)
golden_outdoor_demand_df_selection = golden_outdoor_demand_df.iloc[46:63, 2:14]
golden_outdoor_demand_df_selection_update = pd.DataFrame(golden_outdoor_demand_df_selection.values.ravel(),columns = ['column'])
golden_outdoor_demand_df_selection_update['date'] = pd.date_range(start='1/1/1996', periods=len(golden_outdoor_demand_df_selection_update), freq='M')
golden_outdoor_demand_df_selection_update.index = golden_outdoor_demand_df_selection_update['date']
golden_outdoor_demand_df_selection_update['month'] = golden_outdoor_demand_df_selection_update.index.month
golden_outdoor_demand_df_selection_update['year'] = golden_outdoor_demand_df_selection_update.index.year
golden_outdoor_demand_df_selection_update['norm'] = np.nan



yearly_sum = golden_outdoor_demand_df_selection_update.groupby('year').sum()['column']


for i in range(0,204):
    year_check = yearly_sum.index == golden_outdoor_demand_df_selection_update['year'][i]
    annual_sum = yearly_sum[year_check]
    golden_outdoor_demand_df_selection_update['norm'][i] = golden_outdoor_demand_df_selection_update['column'][i]/annual_sum

monthly_mean = golden_outdoor_demand_df_selection_update.groupby('month').mean()['norm']
monthly_std = golden_outdoor_demand_df_selection_update.groupby('month').std()['norm']
monthly_std

golden_outdoor_demand_df_selection_update['deseas'] = golden_outdoor_demand_df_selection_update['norm'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    golden_outdoor_demand_df_selection_update['deseas'].loc[golden_outdoor_demand_df_selection_update['month'] == i] = (golden_outdoor_demand_df_selection_update['deseas'].loc[golden_outdoor_demand_df_selection_update['month'] == i] - mu) / sigma
    
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(golden_outdoor_demand_df_selection_update['deseas'])

golden_outdoor_demand_df_selection_update = golden_outdoor_demand_df_selection_update.fillna(0)

golden_outdoor_demand_df_selection_update['deseas_l1'] = np.nan
golden_outdoor_demand_df_selection_update['deseas_l12'] = np.nan
print(golden_outdoor_demand_df_selection_update.head())
print()

golden_outdoor_demand_df_selection_update['deseas_l1'].iloc[1:] = golden_outdoor_demand_df_selection_update['deseas'].values[:-1]
golden_outdoor_demand_df_selection_update['deseas_l12'].iloc[12:] = golden_outdoor_demand_df_selection_update['deseas'].values[:-12]

golden_outdoor_demand_df_selection_update.head

lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=golden_outdoor_demand_df_selection_update)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

golden_outdoor_demand_df_selection_update['ar_resid'] = lm_log_ar_fit.resid
print(golden_outdoor_demand_df_selection_update)

plt.figure()
plt.plot(golden_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
plt.hist(golden_outdoor_demand_df_selection_update['ar_resid'])

plt.figure()
fig = plot_acf(golden_outdoor_demand_df_selection_update['ar_resid'].iloc[12:])

golden_outdoor_demands = pd.DataFrame()


from scipy.stats import skewnorm, gamma, norm
golden_outdoor_demands['synthetic2_noise'] = norm.rvs(golden_outdoor_demand_df_selection_update['ar_resid'].mean(), golden_outdoor_demand_df_selection_update['ar_resid'].std(), size=synthetic_series_length)

plt.figure()
plt.plot(golden_outdoor_demand_df_selection_update['ar_resid'])
plt.plot(golden_outdoor_demands['synthetic2_noise'])

def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

max_lag = 12
nrow = golden_outdoor_demands.shape[0]

synth_ar = list(golden_outdoor_demand_df_selection_update['deseas'].iloc[-max_lag:])
synth_ar

for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = golden_outdoor_demands['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
golden_outdoor_demands['synthetic2_deseas'] = synth_ar[-nrow:]
golden_outdoor_demands

golden_outdoor_demands['date'] = pd.date_range(start='1/1/1950', periods=len(golden_outdoor_demands), freq='M')
golden_outdoor_demands.index = golden_outdoor_demands['date']
golden_outdoor_demands['month'] = golden_outdoor_demands.index.month
golden_outdoor_demands['year'] = golden_outdoor_demands.index.year


golden_outdoor_demands['synthetic_demand'] = golden_outdoor_demands['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean[i]
    sigma = monthly_std[i]
    golden_outdoor_demands['synthetic_demand'].loc[golden_outdoor_demands['month'] == i] = golden_outdoor_demands['synthetic_demand'].loc[golden_outdoor_demands['month'] == i] * sigma + mu
    
golden_outdoor_demands['6000'] = golden_outdoor_demands['synthetic_demand']*6000

plt.figure()
plt.plot(golden_outdoor_demands['synthetic_demand'])


plt.figure()
plt.title('golden outdoor Demands')
plt.plot(golden_outdoor_demands['6000'])

plt.figure()
plt.title('golden Historical')
plt.plot(golden_outdoor_demand_df_selection_update['column'])



# loveland_outdoor_demand_df = df.loc[df['id'] == '04_Lovelnd_O']
# longmont_indoor_demand_df = df.loc[df['id'] == '05LONG_IN']
# longmont_outdoor_demand_df = df.loc[df['id'] == '05LONG_OUT']
# boulder_indoor_demand_df = df.loc[df['id'] == '06BOULDER_I']
# boulder_outdoor_demand_df = df.loc[df['id'] == '06BOULDER_O']
# denver_indoor_demand_df = df.loc[df['id'] == '08_Denver_I']
# denver_outdoor_demand_df = df.loc[df['id'] == '08_Denver_O']
# aurora_indoor_demand_df = df.loc[df['id'] == '08_Aurora_I']
# aurora_outdoor_demand_df = df.loc[df['id'] == '08_Aurora_O']
# englewood_indoor_demand_df = df.loc[df['id'] == '08_Englwd_I']
# englewood_outdoor_demand_df = df.loc[df['id'] == '08_Englwd_O']
# northglenn_indoor_demand_df = df.loc[df['id'] == '02_Nglenn_I']
# northglenn_outdoor_demand_df = df.loc[df['id'] == '02_Nglenn_O']
# westminster_indoor_demand_df = df.loc[df['id'] == '02_Westy_I']
# westminster_outdoor_demand_df = df.loc[df['id'] == '02_Westy_O']
# thornton_indoor_demand_df = df.loc[df['id'] == '02_Thorn_I']
# thornton_outdoor_demand_df = df.loc[df['id'] == '02_Thorn_O']
# lafayette_indoor_demand_df = df.loc[df['id'] == '06LAFFYT_I']
# lafayette_outdoor_demand_df = df.loc[df['id'] == '06LAFFYT_O']
# louisville_indoor_demand_df = df.loc[df['id'] == '06LOUIS_I']
# louisville_outdoor_demand_df = df.loc[df['id'] == '06LOUIS_O']
# arvada_indoor_demand_df = df.loc[df['id'] == '07_Arvada_I']
# arvada_outdoor_demand_df = df.loc[df['id'] == '07_Arvada_O']
# conmutual_indoor_demand_df = df.loc[df['id'] == '07_ConMut_I']
# conmutual_outdoor_demand_df = df.loc[df['id'] == '07_ConMut_O']
# golden_indoor_demand_df = df.loc[df['id'] == '07_Golden_I']
# golden_outdoor_demand_df = df.loc[df['id'] == '07_Golden_O']

synthetic_demands = pd.DataFrame()
synthetic_demands['04_Lovelnd_I'] = loveland_indoor_demands['synthetic_demand']
synthetic_demands['04_Lovelnd_O'] = loveland_outdoor_demands['synthetic_demand']
synthetic_demands['05LONG_IN'] = longmont_indoor_demands['synthetic_demand']
synthetic_demands['05LONG_OUT'] = longmont_outdoor_demands['synthetic_demand']
synthetic_demands['06BOULDER_I'] = boulder_indoor_demands['synthetic_demand']
synthetic_demands['06BOULDER_O'] = boulder_outdoor_demands['synthetic_demand']
synthetic_demands['08_Denver_I'] = denver_indoor_demands['synthetic_demand']
synthetic_demands['08_Denver_O'] = denver_outdoor_demands['synthetic_demand']
synthetic_demands['08_Aurora_I'] = aurora_indoor_demands['synthetic_demand']
synthetic_demands['08_Aurora_O'] = aurora_outdoor_demands['synthetic_demand']
synthetic_demands['08_Englwd_I'] = englewood_indoor_demands['synthetic_demand']
synthetic_demands['08_Englwd_O'] = englewood_outdoor_demands['synthetic_demand']
synthetic_demands['02_Nglenn_I'] = northglenn_indoor_demands['synthetic_demand']
synthetic_demands['02_Nglenn_O'] = northglenn_outdoor_demands['synthetic_demand']
synthetic_demands['02_Westy_I'] = westminster_indoor_demands['synthetic_demand']
synthetic_demands['02_Westy_O'] = westminster_outdoor_demands['synthetic_demand']
synthetic_demands['02_Thorn_I'] = thornton_indoor_demands['synthetic_demand']
synthetic_demands['02_Thorn_O'] = thornton_outdoor_demands['synthetic_demand']
synthetic_demands['06LAFFYT_I'] = lafayette_indoor_demands['synthetic_demand']
synthetic_demands['06LAFFYT_O'] = lafayette_outdoor_demands['synthetic_demand']
synthetic_demands['06LOUIS_I'] = louisville_indoor_demands['synthetic_demand']
synthetic_demands['06LOUIS_O'] = louisville_outdoor_demands['synthetic_demand']
synthetic_demands['07_Arvada_I'] = arvada_indoor_demands['synthetic_demand']
synthetic_demands['07_Arvada_O'] = arvada_outdoor_demands['synthetic_demand']
synthetic_demands['07_ConMut_I'] = conmutual_indoor_demands['synthetic_demand']
synthetic_demands['07_ConMut_O'] = conmutual_outdoor_demands['synthetic_demand']
synthetic_demands['07_Golden_I'] = golden_indoor_demands['synthetic_demand']
synthetic_demands['07_Golden_O'] = golden_outdoor_demands['synthetic_demand']
synthetic_demands['index'] = range(0,synthetic_series_length)
synthetic_demands.index = synthetic_demands['index']


historical_means = pd.DataFrame()
historical_means['index'] = range(0,1)
historical_means.index = historical_means['index']


historical_means["04_Lovelnd_I"] = loveland_indoor_demand_df_selection_update['column'].mean()*12
historical_means['04_Lovelnd_O'] = loveland_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['05LONG_IN'] = longmont_indoor_demand_df_selection_update['column'].mean()*12
historical_means['05LONG_OUT'] = longmont_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['06BOULDER_I'] = boulder_indoor_demand_df_selection_update['column'].mean()*12
historical_means['06BOULDER_O'] = boulder_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['08_Denver_I'] = denver_indoor_demand_df_selection_update['column'].mean()*12
historical_means['08_Denver_O'] = denver_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['08_Aurora_I'] = aurora_indoor_demand_df_selection_update['column'].mean()*12
historical_means['08_Aurora_O'] = aurora_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['08_Englwd_I'] = englewood_indoor_demand_df_selection_update['column'].mean()*12
historical_means['08_Englwd_O'] = englewood_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['02_Nglenn_I'] = northglenn_indoor_demand_df_selection_update['column'].mean()*12
historical_means['02_Nglenn_O'] = northglenn_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['02_Westy_I'] = westminster_indoor_demand_df_selection_update['column'].mean()*12
historical_means['02_Westy_O'] = westminster_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['02_Thorn_I'] = thornton_indoor_demand_df_selection_update['column'].mean()*12
historical_means['02_Thorn_O'] = thornton_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['06LAFFYT_I'] = lafayette_indoor_demand_df_selection_update['column'].mean()*12
historical_means['06LAFFYT_O'] = lafayette_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['06LOUIS_I'] = louisville_indoor_demand_df_selection_update['column'].mean()*12
historical_means['06LOUIS_O'] = louisville_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['07_Arvada_I'] = arvada_indoor_demand_df_selection_update['column'].mean()*12
historical_means['07_Arvada_O'] = arvada_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['07_ConMut_I'] = conmutual_indoor_demand_df_selection_update['column'].mean()*12
historical_means['07_ConMut_O'] = conmutual_outdoor_demand_df_selection_update['column'].mean()*12
historical_means['07_Golden_I'] = golden_indoor_demand_df_selection_update['column'].mean()*12
historical_means['07_Golden_O'] = golden_outdoor_demand_df_selection_update['column'].mean()*12


demand_increases = pd.DataFrame()
demand_increases['index'] = range(0,1)
demand_increases.index = demand_increases['index']

demand_multipliers = 1

demand_increases["04_Lovelnd_I"] = loveland_indoor_demand_df_selection_update['column'].mean()*12 + loveland_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['04_Lovelnd_O'] = loveland_outdoor_demand_df_selection_update['column'].mean()*12 + loveland_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['05LONG_IN'] = longmont_indoor_demand_df_selection_update['column'].mean()*12 + longmont_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['05LONG_OUT'] = longmont_outdoor_demand_df_selection_update['column'].mean()*12 + longmont_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['06BOULDER_I'] = boulder_indoor_demand_df_selection_update['column'].mean()*12 + boulder_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['06BOULDER_O'] = boulder_outdoor_demand_df_selection_update['column'].mean()*12 + boulder_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['08_Denver_I'] = denver_indoor_demand_df_selection_update['column'].mean()*12 + denver_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['08_Denver_O'] = denver_outdoor_demand_df_selection_update['column'].mean()*12 + denver_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['08_Aurora_I'] = aurora_indoor_demand_df_selection_update['column'].mean()*12 + aurora_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['08_Aurora_O'] = aurora_outdoor_demand_df_selection_update['column'].mean()*12 + aurora_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['08_Englwd_I'] = englewood_indoor_demand_df_selection_update['column'].mean()*12 + englewood_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['08_Englwd_O'] = englewood_outdoor_demand_df_selection_update['column'].mean()*12 + englewood_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['02_Nglenn_I'] = northglenn_indoor_demand_df_selection_update['column'].mean()*12 + northglenn_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['02_Nglenn_O'] = northglenn_outdoor_demand_df_selection_update['column'].mean()*12 + northglenn_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['02_Westy_I'] = westminster_indoor_demand_df_selection_update['column'].mean()*12 + westminster_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['02_Westy_O'] = westminster_outdoor_demand_df_selection_update['column'].mean()*12 + westminster_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['02_Thorn_I'] = thornton_indoor_demand_df_selection_update['column'].mean()*12 + thornton_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['02_Thorn_O'] = thornton_outdoor_demand_df_selection_update['column'].mean()*12 + thornton_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['06LAFFYT_I'] = lafayette_indoor_demand_df_selection_update['column'].mean()*12 + lafayette_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['06LAFFYT_O'] = lafayette_outdoor_demand_df_selection_update['column'].mean()*12 + lafayette_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['06LOUIS_I'] = louisville_indoor_demand_df_selection_update['column'].mean()*12 + louisville_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['06LOUIS_O'] = louisville_outdoor_demand_df_selection_update['column'].mean()*12 + louisville_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['07_Arvada_I'] = arvada_indoor_demand_df_selection_update['column'].mean()*12 + arvada_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['07_Arvada_O'] = arvada_outdoor_demand_df_selection_update['column'].mean()*12 + arvada_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['07_ConMut_I'] = conmutual_indoor_demand_df_selection_update['column'].mean()*12 + conmutual_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['07_ConMut_O'] = conmutual_outdoor_demand_df_selection_update['column'].mean()*12 + conmutual_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['07_Golden_I'] = golden_indoor_demand_df_selection_update['column'].mean()*12 + golden_indoor_demand_df_selection_update['column'].mean()*12*demand_multipliers
demand_increases['07_Golden_O'] = golden_outdoor_demand_df_selection_update['column'].mean()*12 + golden_outdoor_demand_df_selection_update['column'].mean()*12*demand_multipliers




