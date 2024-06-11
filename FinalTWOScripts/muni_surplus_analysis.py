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




Northern_Cum_Sum = Northern_Cum_Sum [Northern_Cum_Sum.index <= 2011]
Northern_Cum_Sum = Northern_Cum_Sum [Northern_Cum_Sum.index >= 2000]

## Select Boulder Reservoir Rights ###
Boulder_res = [604172, 504515, 604238, 604236, 604269, 604489, 604252, 604253, 604254]
Boulder_Res_Rights = Northern_Res_Rights.loc[Northern_Res_Rights['WDID'].isin(Boulder_res)]
Boulder_Res_Rights['Sum'] = (Boulder_Res_Rights['Absolute'] + Boulder_Res_Rights['Conditional'] + Boulder_Res_Rights['APEX Absolute'] + Boulder_Res_Rights['APEX Conditional'])
Boulder_Res_Rights.index = Boulder_Res_Rights['Adjudication Date']
Boulder_Res_Rights['Year'] = Boulder_Res_Rights.index.year
Boulder_Res_Rights_Yearly = (Boulder_Res_Rights.groupby(['Year'])["Sum"].sum()).astype(int)

Boulder_Total_Rights = Boulder_Res_Rights_Yearly
Boulder_Total_Rights = Boulder_Total_Rights.groupby(['Year']).sum()

Boulder_Cum_Sum = Boulder_Total_Rights.cumsum()
Boulder_Cum_Sum = Boulder_Cum_Sum [Boulder_Cum_Sum.index <= 2011]




plt.plot(Boulder_Cum_Sum)
## import known CBI data ####
os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')
CBI = pd.read_csv('cbi_timeseries.csv', header= None)
CBI.columns =['date', 'snowpack', 'diversion', 'storage', 'cbi']
CBI['date'] = pd.to_datetime(CBI["date"])
CBI.index = CBI['date']
CBI['year'] = CBI.index.year
CBI_Yearly = CBI.groupby('year').mean()['cbi']
CBI_Yearly = CBI_Yearly[CBI_Yearly.index <= 2011]
CBI_Yearly_Selection = pd.DataFrame(CBI_Yearly[CBI_Yearly.index >= 2001])


### import known Boulder Surplus data - from Boulder Muni Bonds #####
os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')
Boulder_Surplus = pd.read_csv('Boulder_Surplus_Leasing.csv', header= None)
Boulder_Surplus.columns =['year', 'af']
Boulder_Surplus.index = Boulder_Surplus['year']
Boulder_Surplus = Boulder_Surplus[Boulder_Surplus.index <= 2011]


X = Boulder_Surplus[['af']]
y = CBI_Yearly_Selection

reg = LinearRegression().fit(X, y)

print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)


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

## check Greeley ###

Greeley = pd.read_parquet('0400702.parquet', engine='pyarrow')
Greeley_demand = Greeley['demand']
Greeley_Total_demand_Sum = pd.DataFrame().assign(Year=Greeley['year'],demand=Greeley_demand)
Greeley_Total_demand_Sum = Greeley_Total_demand_Sum.groupby('Year').sum()

## Sum all Northern Water municipalities shortages' ###

Northern_Water_Muni_demands = (Loveland_Total_demand_Sum + Longmont_Total_demand_Sum + Louisville_Total_demand_Sum + Laffayette_Total_demand_Sum + Boulder_Total_demand_Sum)


## FIGURE 1 #########################

ax1 = plt.subplot()
ax1.plot(Northern_Water_Muni_demands/1000, label = 'Municipal Demands')
ax1.plot(Northern_Cum_Sum_extended/1000, label = 'Water Rights Holdings')
#plt.plot(Northern_Cum_Sum_extended * 0.35, label = 'Dry-Year Right Fulfillment', linestyle = 'dashed')
#ax1.xlabel('Year')
#ax1.ylabel('Water Quantity (kAF)')
# ticks =  ax1.get_yticks()
# for label in ax1.get_yticklabels():
#     label.set_color("black")
# ax1.set_yticklabels([f'{abs(tick):.1f}' for tick in ticks], fontsize=12, weight = 'bold')
# xticks =  ax1.get_xticks()
# for label in ax1.get_xticklabels():
#     label.set_color("black")
# ax1.set_xticklabels([f'{abs(xtick):.1f}' for xtick in xticks], fontsize=12, weight = 'bold')
ax1.legend()


# ax3 = plt.subplot(3, 1, 3, sharex = ax1)
# data_5 = dry_year_two_cost_sum/1000000
# data_6 = wet_year_two_cost_sum/1000000
# plt.bar(x, data_6, color = 'blue') 
# plt.bar(x, -(data_5), color = 'red')
# plt.ylim([-.7,.7])

# ticks =  ax3.get_yticks()
# for label in ax3.get_yticklabels()[0:4]:
#     label.set_color("red")
# for label in ax3.get_yticklabels()[5:9]:
#     label.set_color("blue")
# # set labels to absolute values and with integer representation
# ax3.set_yticklabels([f'{abs(tick):.1f}' for tick in ticks], fontsize=20, weight = 'bold')

# ax3.axhline(y=0, color='black')
# #plt.ylabel('TWO Cost ($)')
# ax3.text(-.11, .75, 'Ag', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = 'blue', weight = 'bold', fontsize = 20)
# ax3.text(-.11, .25, 'Urban', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = 'red', weight = 'bold', fontsize = 20)
# ax3.text(-.15, 0.45, 'Two-Way Option Cost ($M)', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = 'black', weight = 'bold', fontsize=20)
# plt.subplots_adjust(hspace=0.4)













Northern_Water_Muni_demands = Northern_Water_Muni_demands[Northern_Water_Muni_demands.index <= 2011]
Northern_Water_Muni_demands = Northern_Water_Muni_demands[Northern_Water_Muni_demands.index >= 2001]
Northern_Water_Muni_demands['rights'] = Northern_Cum_Sum.loc[Northern_Cum_Sum.index == 2009].values[0]
Northern_Water_Muni_demands['surplus'] = Northern_Water_Muni_demands['rights'] - Northern_Water_Muni_demands['demand']


plt.figure()
plt.plot(Northern_Water_Muni_demands)
plt.xlabel('Hydrologic Year')
plt.ylabel('Demands (AF)')
#plt.ylim([0,6000])
plt.title('Northern Water Municipal Demands')


os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod')

plt.figure()
#plt.plot(EstesPark_Total_demand_Sum['demand'],color='purple', label='Estes Park')
plt.plot(Loveland_Total_demand_Sum['demand'],color='green', label='Loveland')
#plt.plot(Greeley_Total_demand_Sum['demand'],color='yellow', label='Greeley')
plt.plot(Longmont_Total_demand_Sum['demand'],color='red', label='Longmont')
plt.plot(Louisville_Total_demand_Sum['demand'],color='orange', label='Louisville')
plt.plot(Laffayette_Total_demand_Sum['demand'],color='purple', label='Lafayette')
plt.plot(Boulder_Total_demand_Sum['demand'],color='blue', label='Boulder')
plt.xlabel('Hydrologic Year')
plt.ylabel('Demand (AF)')
plt.ylim([0,26000])
#plt.title('Northern Water Adapted Municipal Demands')
plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right', borderaxespad=0)

plt.figure()
plt.plot(Boulder_Cum_Sum)
plt.plot(Boulder_Total_demand_Sum['demand'])




Boulder_Total_demand_Sum = Boulder_Total_demand_Sum[Boulder_Total_demand_Sum.index <= 2011]
Boulder_Total_demand_Sum = Boulder_Total_demand_Sum[Boulder_Total_demand_Sum.index >= 2001]
Boulder_Total_demand_Sum['rights'] = Boulder_Cum_Sum.loc[Boulder_Cum_Sum.index == 2000].values[0]
Boulder_Total_demand_Sum['surplus'] = Boulder_Total_demand_Sum['rights'] - Boulder_Total_demand_Sum['demand']





Boulder_Surplus_Percent = pd.DataFrame(Boulder_Surplus['af']/Boulder_Total_demand_Sum['surplus'])

## SUPPLEMENTAL FIGURE ####
plt.figure()
plt.scatter(Boulder_Surplus_Percent*100, CBI_Yearly_Selection, color = 'black')
plt.xlabel('% of Surplus Leased')
plt.ylabel('CBI')
plt.savefig('CBI_vs_Surplus_Percent_SI_hi_res.png', format = 'png', dpi=600, bbox_inches='tight')


plt.figure()
Northern_Lease_amounts = pd.DataFrame(Northern_Water_Muni_demands['surplus']*Boulder_Surplus_Percent[0])
plt.scatter(Northern_Lease_amounts, CBI_Yearly_Selection)

X = CBI_Yearly_Selection
y = Northern_Lease_amounts

reg = LinearRegression().fit(X, y)

print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')
CBI = pd.read_csv('cbi_timeseries.csv', header= None)
CBI.columns =['date', 'snowpack', 'diversion', 'storage', 'cbi']
CBI['date'] = pd.to_datetime(CBI["date"])
CBI.index = CBI['date']
CBI['year'] = CBI.index.year
CBI_Yearly = CBI.groupby('year').mean()['cbi']
CBI_Yearly = CBI_Yearly[CBI_Yearly.index <= 2012]

Lease_Amounts_by_CBI = CBI_Yearly*(reg.coef_[0][0]) + reg.intercept_[0]

plt.figure()
plt.scatter(CBI_Yearly,Lease_Amounts_by_CBI)
plt.xlabel('CBI')
plt.ylabel('Available Lease Volumes (AF)')
plt.title('Current Water Rights Holdings')


############## FOR HISTORICAL RIGHTS ########################################

Boulder_res = [604172, 504515, 604238, 604236, 604269, 604489, 604252, 604253, 604254]
Boulder_Res_Rights = Northern_Res_Rights.loc[Northern_Res_Rights['WDID'].isin(Boulder_res)]
Boulder_Res_Rights['Sum'] = (Boulder_Res_Rights['Absolute'] + Boulder_Res_Rights['Conditional'] + Boulder_Res_Rights['APEX Absolute'] + Boulder_Res_Rights['APEX Conditional'])
Boulder_Res_Rights.index = Boulder_Res_Rights['Adjudication Date']
Boulder_Res_Rights['Year'] = Boulder_Res_Rights.index.year
Boulder_Res_Rights_Yearly = (Boulder_Res_Rights.groupby(['Year'])["Sum"].sum()).astype(int)

Boulder_Total_Rights = Boulder_Res_Rights_Yearly
Boulder_Total_Rights = Boulder_Total_Rights.groupby(['Year']).sum()

Boulder_Cum_Sum = Boulder_Total_Rights.cumsum()
Boulder_Cum_Sum = Boulder_Cum_Sum [Boulder_Cum_Sum.index <= 2011]

Boulder_Total_demand_Sum_H = Boulder_Total_demand_Sum_H[Boulder_Total_demand_Sum_H.index <= 2011]
Boulder_Total_demand_Sum_H = Boulder_Total_demand_Sum_H[Boulder_Total_demand_Sum_H.index >= 2001]
Boulder_Total_demand_Sum_H['rights'] = Boulder_Cum_Sum.loc[Boulder_Cum_Sum.index == 1974].values[0]
Boulder_Total_demand_Sum_H['surplus'] = Boulder_Total_demand_Sum_H['rights'] - Boulder_Total_demand_Sum_H['demand']

Northern_Cum_Sum_H = Northern_Total_Rights.cumsum()


Northern_Water_Muni_demands_H = (Loveland_Total_demand_Sum + Longmont_Total_demand_Sum + Louisville_Total_demand_Sum + Laffayette_Total_demand_Sum + Boulder_Total_demand_Sum)
Northern_Water_Muni_demands_H = Northern_Water_Muni_demands_H[Northern_Water_Muni_demands_H.index <= 2011]
Northern_Water_Muni_demands_H = Northern_Water_Muni_demands_H[Northern_Water_Muni_demands_H.index >= 2001]
Northern_Water_Muni_demands_H['rights'] = Northern_Cum_Sum_H.loc[Northern_Cum_Sum_H.index == 1971].values[0]
Northern_Water_Muni_demands_H['surplus'] = Northern_Water_Muni_demands_H['rights'] - Northern_Water_Muni_demands_H['demand']

plt.figure()
Northern_Lease_amounts_H = pd.DataFrame(Northern_Water_Muni_demands_H['surplus']*Boulder_Surplus_Percent[0])
plt.scatter(Northern_Lease_amounts_H, CBI_Yearly_Selection)

X = CBI_Yearly_Selection
y = Northern_Lease_amounts_H

reg = LinearRegression().fit(X, y)

print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')
CBI = pd.read_csv('cbi_timeseries.csv', header= None)
CBI.columns =['date', 'snowpack', 'diversion', 'storage', 'cbi']
CBI['date'] = pd.to_datetime(CBI["date"])
CBI.index = CBI['date']
CBI['year'] = CBI.index.year
CBI_Yearly = CBI.groupby('year').mean()['cbi']
CBI_Yearly = CBI_Yearly[CBI_Yearly.index <= 2012]

Lease_Amounts_by_CBI_H = CBI_Yearly*(reg.coef_[0][0]) + reg.intercept_[0]

plt.figure()
plt.scatter(CBI_Yearly,Lease_Amounts_by_CBI_H)
plt.xlabel('CBI')
plt.ylabel('Available Lease Volumes (AF)')
plt.title('Historical (1971) Water Rights Holdings')
