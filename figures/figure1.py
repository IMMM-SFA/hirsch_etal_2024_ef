# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:49:36 2023

@author: zacha
"""

import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.colors as mcolors

## import Northern Water Res Rights ####
Northern_Res_Rights = pd.read_csv('NorthernWaterResRights.csv', header=None)
Northern_Res_Rights.columns =['WDID', 'Name', 'Adjudication Date', 'Absolute', 'Conditional', 'APEX Absolute', 'APEX Conditional']
Northern_Res_Rights['Adjudication Date'] = pd.to_datetime(Northern_Res_Rights['Adjudication Date'])
Northern_Res_Rights['Sum'] = (Northern_Res_Rights['Absolute'] + Northern_Res_Rights['Conditional'] + Northern_Res_Rights['APEX Absolute'] + Northern_Res_Rights['APEX Conditional'])
Northern_Res_Rights.index = Northern_Res_Rights['Adjudication Date']
Northern_Res_Rights['Year'] = Northern_Res_Rights.index.year
Northern_Res_Rights_Yearly = (Northern_Res_Rights.groupby(['Year'])["Sum"].sum()).astype(int)

Northern_Total_Rights = Northern_Res_Rights_Yearly
Northern_Total_Rights = Northern_Total_Rights.groupby(['Year']).sum()

Northern_Cum_Sum = Northern_Total_Rights.cumsum()
Northern_Cum_Sum_extended = Northern_Cum_Sum.reindex(range(1920,2013))
Northern_Cum_Sum_extended = Northern_Cum_Sum_extended.ffill()
Northern_Cum_Sum_extended = Northern_Cum_Sum_extended[Northern_Cum_Sum_extended.index >= 1950]



#os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline/xddparquet')
#os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet_03_29')

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline/xddparquet')



ThorntonIndoor = pd.read_parquet('02_Thorn_I.parquet', engine='pyarrow')
ThorntonOutdoor = pd.read_parquet('02_Thorn_O.parquet', engine='pyarrow')
WestminsterIndoor = pd.read_parquet('02_Westy_I.parquet', engine='pyarrow')
WestminsterOutdoor = pd.read_parquet('02_Westy_O.parquet', engine='pyarrow')
LaffayetteIndoor = pd.read_parquet('06LAFFYT_I.parquet', engine='pyarrow')
LaffayetteOutdoor = pd.read_parquet('06LAFFYT_O.parquet', engine='pyarrow')
LouisvilleIndoor = pd.read_parquet('06LOUIS_I.parquet', engine='pyarrow')
LouisvilleOutdoor = pd.read_parquet('06LOUIS_O.parquet', engine='pyarrow')
BoulderOutdoor = pd.read_parquet('06BOULDER_O.parquet', engine='pyarrow')
BoulderIndoor = pd.read_parquet('06BOULDER_I.parquet', engine='pyarrow')
LongmontIndoor = pd.read_parquet('05LONG_IN.parquet', engine='pyarrow')
LongmontOutdoor = pd.read_parquet('05LONG_OUT.parquet', engine='pyarrow')
LovelandIndoor = pd.read_parquet('04_Lovelnd_I.parquet', engine='pyarrow')
LovelandOutdoor = pd.read_parquet('04_Lovelnd_O.parquet', engine='pyarrow')
ArvadaIndoor = pd.read_parquet('07_Arvada_I.parquet', engine='pyarrow')
ArvadaOutdoor = pd.read_parquet('07_Arvada_O.parquet', engine='pyarrow')
ConMutIndoor = pd.read_parquet('07_ConMut_I.parquet', engine='pyarrow')
ConMutOutdoor = pd.read_parquet('07_ConMut_O.parquet', engine='pyarrow')
GoldenIndoor = pd.read_parquet('07_Golden_I.parquet', engine='pyarrow')
GoldenOutdoor = pd.read_parquet('07_Golden_O.parquet', engine='pyarrow')
DenverIndoor = pd.read_parquet('08_Denver_I.parquet', engine='pyarrow')
DenverOutdoor = pd.read_parquet('08_Denver_O.parquet', engine='pyarrow')
EnglewoodIndoor = pd.read_parquet('08_Englwd_I.parquet', engine='pyarrow')
EnglewoodOutdoor = pd.read_parquet('08_Englwd_O.parquet', engine='pyarrow')
AuroraIndoor = pd.read_parquet('08_Aurora_I.parquet', engine='pyarrow')
AuroraOutdoor = pd.read_parquet('08_Aurora_O.parquet', engine='pyarrow')

EstesParkIndoor = pd.read_parquet('0400518_I.parquet', engine='pyarrow')
EstesParkOutdoor = pd.read_parquet('0400518_O.parquet', engine='pyarrow')

# convert municipal demands to pandas series

Thornton_Indoor_demand = ThorntonIndoor['demand']
Thornton_Outdoor_demand = ThorntonOutdoor['demand']
Westminster_Indoor_demand = WestminsterIndoor['demand']
Westminster_Outdoor_demand = WestminsterOutdoor['demand']
Laffayette_Indoor_demand = LaffayetteIndoor['demand']
Laffayette_Outdoor_demand = LaffayetteOutdoor['demand']
Louisville_Indoor_demand = LouisvilleIndoor['demand']
Louisville_Outdoor_demand = LouisvilleOutdoor['demand']
Boulder_Outdoor_demand = BoulderIndoor['demand']
Boulder_Indoor_demand = BoulderOutdoor['demand']
Longmont_Indoor_demand = LongmontIndoor['demand']
Longmont_Outdoor_demand = LongmontOutdoor['demand']
Loveland_Indoor_demand = LovelandIndoor['demand']
Loveland_Outdoor_demand = LovelandOutdoor['demand']
Arvada_Indoor_demand = ArvadaIndoor['demand']
Arvada_Outdoor_demand = ArvadaOutdoor['demand']
ConMut_Indoor_demand = ConMutIndoor['demand']
ConMut_Outdoor_demand = ConMutOutdoor['demand']
Golden_Indoor_demand = GoldenIndoor['demand']
Golden_Outdoor_demand = GoldenOutdoor['demand']
Denver_Indoor_demand = DenverIndoor['demand']
Denver_Outdoor_demand = DenverOutdoor['demand']
Englewood_Indoor_demand = EnglewoodIndoor['demand']
Englewood_Outdoor_demand = EnglewoodOutdoor['demand']
Aurora_Indoor_demand = AuroraIndoor['demand']
Aurora_Outdoor_demand = AuroraOutdoor['demand']
EstesPark_Indoor_demand = EstesParkIndoor['demand']
EstesPark_Outdoor_demand = EstesParkOutdoor['demand']


Thornton_Total_demand = Thornton_Indoor_demand + Thornton_Outdoor_demand
Westminster_Total_demand = Westminster_Indoor_demand + Westminster_Outdoor_demand
Laffayette_Total_demand = Laffayette_Indoor_demand + Laffayette_Outdoor_demand
Louisville_Total_demand = Louisville_Indoor_demand + Louisville_Outdoor_demand
Boulder_Total_demand = Boulder_Indoor_demand + Boulder_Outdoor_demand
Longmont_Total_demand = Longmont_Indoor_demand + Longmont_Outdoor_demand
Loveland_Total_demand = Loveland_Indoor_demand + Loveland_Outdoor_demand
Arvada_Total_demand = Arvada_Indoor_demand + Arvada_Outdoor_demand
ConMut_Total_demand = ConMut_Indoor_demand + ConMut_Outdoor_demand
Golden_Total_demand = Golden_Indoor_demand + Golden_Outdoor_demand
Denver_Total_demand = Denver_Indoor_demand + Denver_Outdoor_demand
Englewood_Total_demand = Englewood_Indoor_demand + Englewood_Outdoor_demand
Aurora_Total_demand = Aurora_Indoor_demand + Aurora_Outdoor_demand
EstesPark_Total_demand = EstesPark_Indoor_demand + EstesPark_Outdoor_demand

Thornton_Total_demand_Sum = pd.DataFrame().assign(Year=ThorntonIndoor['year'],demand=Thornton_Total_demand)
Thornton_Total_demand_Sum = Thornton_Total_demand_Sum.groupby('Year').sum()

Westminster_Total_demand_Sum = pd.DataFrame().assign(Year=WestminsterIndoor['year'],demand=Westminster_Total_demand)
Westminster_Total_demand_Sum = Westminster_Total_demand_Sum.groupby('Year').sum()

Laffayette_Total_demand_Sum = pd.DataFrame().assign(Year=LaffayetteIndoor['year'],demand=Laffayette_Total_demand)
Laffayette_Total_demand_Sum = Laffayette_Total_demand_Sum.groupby('Year').sum()

Louisville_Total_demand_Sum = pd.DataFrame().assign(Year=LouisvilleIndoor['year'],demand=Louisville_Total_demand)
Louisville_Total_demand_Sum = Louisville_Total_demand_Sum.groupby('Year').sum()

Boulder_Total_demand_Sum = pd.DataFrame().assign(Year=BoulderIndoor['year'],demand=Boulder_Total_demand)
Boulder_Total_demand_Sum = Boulder_Total_demand_Sum.groupby('Year').sum()

Boulder_Total_demand_Sum_H = pd.DataFrame().assign(Year=BoulderIndoor['year'],demand=Boulder_Total_demand)
Boulder_Total_demand_Sum_H = Boulder_Total_demand_Sum_H.groupby('Year').sum()

Longmont_Total_demand_Sum = pd.DataFrame().assign(Year=LongmontIndoor['year'],demand=Longmont_Total_demand)
Longmont_Total_demand_Sum = Longmont_Total_demand_Sum.groupby('Year').sum()

Loveland_Total_demand_Sum = pd.DataFrame().assign(Year=LovelandIndoor['year'],demand=Loveland_Total_demand)
Loveland_Total_demand_Sum = Loveland_Total_demand_Sum.groupby('Year').sum()

Arvada_Total_demand_Sum = pd.DataFrame().assign(Year=ArvadaIndoor['year'],demand=Arvada_Total_demand)
Arvada_Total_demand_Sum = Arvada_Total_demand_Sum.groupby('Year').sum()

ConMut_Total_demand_Sum = pd.DataFrame().assign(Year=ConMutIndoor['year'],demand=ConMut_Total_demand)
ConMut_Total_demand_Sum = ConMut_Total_demand_Sum.groupby('Year').sum()

Golden_Total_demand_Sum = pd.DataFrame().assign(Year=GoldenIndoor['year'],demand=Golden_Total_demand)
Golden_Total_demand_Sum = Golden_Total_demand_Sum.groupby('Year').sum()

Denver_Total_demand_Sum = pd.DataFrame().assign(Year=DenverIndoor['year'],demand=Denver_Total_demand)
Denver_Total_demand_Sum = Denver_Total_demand_Sum.groupby('Year').sum()

Englewood_Total_demand_Sum = pd.DataFrame().assign(Year=EnglewoodIndoor['year'],demand=Englewood_Total_demand)
Englewood_Total_demand_Sum = Englewood_Total_demand_Sum.groupby('Year').sum()

Aurora_Total_demand_Sum = pd.DataFrame().assign(Year=AuroraIndoor['year'],demand=Aurora_Total_demand)
Aurora_Total_demand_Sum = Aurora_Total_demand_Sum.groupby('Year').sum()

EstesPark_Total_demand_Sum = pd.DataFrame().assign(Year=EstesParkIndoor['year'],demand=EstesPark_Total_demand)
EstesPark_Total_demand_Sum = EstesPark_Total_demand_Sum.groupby('Year').sum()



## Sum all Northern Water municipalities shortages' ###

Northern_Water_Muni_demands = (Loveland_Total_demand_Sum + Longmont_Total_demand_Sum + Louisville_Total_demand_Sum + Laffayette_Total_demand_Sum + Boulder_Total_demand_Sum)


## FIGURE 1 #########################

af_to_m3 = 1233.4818375475
af_to_km3 = 810713.18210885

color1 = '#277DA1'
color2 = '#90BE6D'
ax1 = plt.subplot()
plt.style.use('default')
ax1.plot(Northern_Water_Muni_demands*af_to_m3/1000000, label = 'Municipal Demands', linewidth = 1.5, color = color1)
ax1.plot(Northern_Cum_Sum_extended*af_to_m3/1000000, label = 'Water Rights Holdings', linewidth = 1.5, color = color2)
ax1.set_yticklabels(ax1.get_yticklabels())
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.text(-.1, .5, 'Water Quantity (MCM)', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = 'black', fontsize = 10)
ax1.text(.5, -.1, 'Year', ha='center', va='center', transform=ax1.transAxes, color = 'black', fontsize = 10)
ax1.legend()
plt.savefig('figure_1_hi_res.png', format = 'png', dpi=600)
#legend_prop = {'weight': 'bold'}
#ax1.legend(prop = legend_prop , framealpha = 0.5)


