# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:54:20 2022

@author: zacha
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv("colorado_snow_data.csv")  
df.columns = ['Date', 'South Platte', 'Upper Colorado']
df["Date"]= pd.to_datetime(df["Date"])

df.index = df['Date']
df['month'] = df.index.month
df['year'] = df.index.year

South_Platte_Snow = pd.DataFrame()
South_Platte_Snow['South_Platte'] = df.groupby(df.year)['South Platte'].mean() 
Upper_CO_Snow = df.groupby(df.year)['Upper Colorado'].mean() 
Threshold = South_Platte_Snow['South_Platte'].mean()

plt.figure()
plt.plot(South_Platte_Snow,color='red', label='South Platte')
#plt.plot(Upper_CO_Snow,color='blue', label='Upper CO')
plt.hlines(Threshold, 1950, 2012, linestyles='dashed', label='Threshold')

plt.xlabel('Hydrologic Year')
plt.ylabel('Snowpack (% of average)')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)

South_Platte_Snow.loc[South_Platte_Snow['South_Platte'] >= Threshold, 'Year_Type'] = 'Wet'
South_Platte_Snow.loc[South_Platte_Snow['South_Platte'] <= Threshold, 'Year_Type'] = 'Dry'

Dry_Years = South_Platte_Snow.drop(South_Platte_Snow.index[South_Platte_Snow['Year_Type'] == 'Wet'])
Dry_Years_list = pd.Series(Dry_Years.index)


Wet_Years = South_Platte_Snow.drop(South_Platte_Snow.index[South_Platte_Snow['Year_Type'] == 'Dry'])
Wet_Years_list = pd.Series(Wet_Years.index)
