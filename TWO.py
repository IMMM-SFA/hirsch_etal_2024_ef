# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 21:31:53 2023

@author: zacha
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import co_snow_metrics as cosnow
import statemod_sp_adapted as muni
import matplotlib.colors as colors
import extract_ipy_southplatte as ipy
import sp_irrigation_final as spirr

### import lake granby data ############

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xreparquet_ucrb_02_17/')

Granby_Reservoir_Data = pd.read_parquet('5104055+.parquet', engine = 'pyarrow')
Granby = Granby_Reservoir_Data[Granby_Reservoir_Data['account'] == 0]


Granby_Spills = Granby.groupby('year').sum()['spill']

plt.figure()
plt.plot(Granby_Spills)
plt.xlabel('Hydrologic Year')
plt.ylabel('Spill (AF)')
plt.title('Lake Granby')
plt.xlim([1950, 2012])


os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet_02_22')

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

########################################################################################################################################

# convert municipal demand shortages to pandas series

Thornton_Indoor_Shortage = ThorntonIndoor['shortage']
Thornton_Outdoor_Shortage = ThorntonOutdoor['shortage']
Westminster_Indoor_Shortage = WestminsterIndoor['shortage']
Westminster_Outdoor_Shortage = WestminsterOutdoor['shortage']
Laffayette_Indoor_Shortage = LaffayetteIndoor['shortage']
Laffayette_Outdoor_Shortage = LaffayetteOutdoor['shortage']
Louisville_Indoor_Shortage = LouisvilleIndoor['shortage']
Louisville_Outdoor_Shortage = LouisvilleOutdoor['shortage']
Boulder_Outdoor_Shortage = BoulderIndoor['shortage']
Boulder_Indoor_Shortage = BoulderOutdoor['shortage']
Longmont_Indoor_Shortage = LongmontIndoor['shortage']
Longmont_Outdoor_Shortage = LongmontOutdoor['shortage']
Loveland_Indoor_Shortage = LovelandIndoor['shortage']
Loveland_Outdoor_Shortage = LovelandOutdoor['shortage']
Arvada_Indoor_Shortage = ArvadaIndoor['shortage']
Arvada_Outdoor_Shortage = ArvadaOutdoor['shortage']
ConMut_Indoor_Shortage = ConMutIndoor['shortage']
ConMut_Outdoor_Shortage = ConMutOutdoor['shortage']
Golden_Indoor_Shortage = GoldenIndoor['shortage']
Golden_Outdoor_Shortage = GoldenOutdoor['shortage']
Denver_Indoor_Shortage = DenverIndoor['shortage']
Denver_Outdoor_Shortage = DenverOutdoor['shortage']
Englewood_Indoor_Shortage = EnglewoodIndoor['shortage']
Englewood_Outdoor_Shortage = EnglewoodOutdoor['shortage']
Aurora_Indoor_Shortage = AuroraIndoor['shortage']
Aurora_Outdoor_Shortage = AuroraOutdoor['shortage']

EstesPark_Indoor_Shortage = EstesParkIndoor['shortage']
EstesPark_Outdoor_Shortage = EstesParkOutdoor['shortage']

########################################################################################################################################

# total indoor and outdoor municipal shortages and plot time series

Thornton_Total_Shortage = Thornton_Indoor_Shortage + Thornton_Outdoor_Shortage
Westminster_Total_Shortage = Westminster_Indoor_Shortage + Westminster_Outdoor_Shortage
Laffayette_Total_Shortage = Laffayette_Indoor_Shortage + Laffayette_Outdoor_Shortage
Louisville_Total_Shortage = Louisville_Indoor_Shortage + Louisville_Outdoor_Shortage
Boulder_Total_Shortage = Boulder_Indoor_Shortage + Boulder_Outdoor_Shortage
Longmont_Total_Shortage = Longmont_Indoor_Shortage + Longmont_Outdoor_Shortage
Loveland_Total_Shortage = Loveland_Indoor_Shortage + Loveland_Outdoor_Shortage
Arvada_Total_Shortage = Arvada_Indoor_Shortage + Arvada_Outdoor_Shortage
ConMut_Total_Shortage = ConMut_Indoor_Shortage + ConMut_Outdoor_Shortage
Golden_Total_Shortage = Golden_Indoor_Shortage + Golden_Outdoor_Shortage
Denver_Total_Shortage = Denver_Indoor_Shortage + Denver_Outdoor_Shortage
Englewood_Total_Shortage = Englewood_Indoor_Shortage + Englewood_Outdoor_Shortage
Aurora_Total_Shortage = Aurora_Indoor_Shortage + Aurora_Outdoor_Shortage
EstesPark_Total_Shortage = EstesPark_Indoor_Shortage + EstesPark_Outdoor_Shortage

Thornton_Total_Shortage_Sum = pd.DataFrame().assign(Year=ThorntonIndoor['year'],Shortage=Thornton_Total_Shortage)
Thornton_Total_Shortage_Sum = Thornton_Total_Shortage_Sum.groupby('Year').sum()

Westminster_Total_Shortage_Sum = pd.DataFrame().assign(Year=WestminsterIndoor['year'],Shortage=Westminster_Total_Shortage)
Westminster_Total_Shortage_Sum = Westminster_Total_Shortage_Sum.groupby('Year').sum()

Laffayette_Total_Shortage_Sum = pd.DataFrame().assign(Year=LaffayetteIndoor['year'],Shortage=Laffayette_Total_Shortage)
Laffayette_Total_Shortage_Sum = Laffayette_Total_Shortage_Sum.groupby('Year').sum()

Louisville_Total_Shortage_Sum = pd.DataFrame().assign(Year=LouisvilleIndoor['year'],Shortage=Louisville_Total_Shortage)
Louisville_Total_Shortage_Sum = Louisville_Total_Shortage_Sum.groupby('Year').sum()

Boulder_Total_Shortage_Sum = pd.DataFrame().assign(Year=BoulderIndoor['year'],Shortage=Boulder_Total_Shortage)
Boulder_Total_Shortage_Sum = Boulder_Total_Shortage_Sum.groupby('Year').sum()

Longmont_Total_Shortage_Sum = pd.DataFrame().assign(Year=LongmontIndoor['year'],Shortage=Longmont_Total_Shortage)
Longmont_Total_Shortage_Sum = Longmont_Total_Shortage_Sum.groupby('Year').sum()

Loveland_Total_Shortage_Sum = pd.DataFrame().assign(Year=LovelandIndoor['year'],Shortage=Loveland_Total_Shortage)
Loveland_Total_Shortage_Sum = Loveland_Total_Shortage_Sum.groupby('Year').sum()

Arvada_Total_Shortage_Sum = pd.DataFrame().assign(Year=ArvadaIndoor['year'],Shortage=Arvada_Total_Shortage)
Arvada_Total_Shortage_Sum = Arvada_Total_Shortage_Sum.groupby('Year').sum()

ConMut_Total_Shortage_Sum = pd.DataFrame().assign(Year=ConMutIndoor['year'],Shortage=ConMut_Total_Shortage)
ConMut_Total_Shortage_Sum = ConMut_Total_Shortage_Sum.groupby('Year').sum()

Golden_Total_Shortage_Sum = pd.DataFrame().assign(Year=GoldenIndoor['year'],Shortage=Golden_Total_Shortage)
Golden_Total_Shortage_Sum = Golden_Total_Shortage_Sum.groupby('Year').sum()

Denver_Total_Shortage_Sum = pd.DataFrame().assign(Year=DenverIndoor['year'],Shortage=Denver_Total_Shortage)
Denver_Total_Shortage_Sum = Denver_Total_Shortage_Sum.groupby('Year').sum()

Englewood_Total_Shortage_Sum = pd.DataFrame().assign(Year=EnglewoodIndoor['year'],Shortage=Englewood_Total_Shortage)
Englewood_Total_Shortage_Sum = Englewood_Total_Shortage_Sum.groupby('Year').sum()

Aurora_Total_Shortage_Sum = pd.DataFrame().assign(Year=AuroraIndoor['year'],Shortage=Aurora_Total_Shortage)
Aurora_Total_Shortage_Sum = Aurora_Total_Shortage_Sum.groupby('Year').sum()
 
EstesPark_Total_Shortage_Sum = pd.DataFrame().assign(Year=EstesParkIndoor['year'],Shortage=EstesPark_Total_Shortage)
EstesPark_Total_Shortage_Sum = EstesPark_Total_Shortage_Sum.groupby('Year').sum()

plt.figure()
plt.plot(Boulder_Total_Shortage_Sum['Shortage'],color='blue', label='Boulder')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 
plt.figure()
plt.plot(Loveland_Total_Shortage_Sum['Shortage'],color='green', label='Loveland')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 
plt.figure()
plt.plot(Longmont_Total_Shortage_Sum['Shortage'],color='red', label='Longmont')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 
plt.figure()
plt.plot(Thornton_Total_Shortage_Sum['Shortage'],color='yellow', label='Thornton')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 
plt.figure()
plt.plot(Westminster_Total_Shortage_Sum['Shortage'],color='black', label='Westminster')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 
plt.figure()
plt.plot(Laffayette_Total_Shortage_Sum['Shortage'],color='brown', label='Laffayette')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 
plt.figure()
plt.plot(Louisville_Total_Shortage_Sum['Shortage'],color='orange', label='Louisville')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 
#plt.plot(Arvada_Total_Shortage_Sum['Shortage'],color='purple', label='Arvada')
plt.figure()
plt.plot(ConMut_Total_Shortage_Sum['Shortage'],color='pink', label='ConMut')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 
plt.figure()
plt.plot(Golden_Total_Shortage_Sum['Shortage'],color='gray', label='Golden')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.ylim(-35, 700)
plt.legend() 
#plt.plot(Denver_Total_Shortage_Sum['Shortage'],color='gold', label='Denver')
plt.figure()
plt.plot(Englewood_Total_Shortage_Sum['Shortage'],color='aqua', label='Englewood')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 

plt.figure()
plt.plot(Aurora_Total_Shortage_Sum['Shortage'], color= 'gold', label = 'Aurora')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend() 

# plt.ylabel('Shortage (AF)')
# plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right', borderaxespad=0)

plt.figure()
plt.plot(Denver_Total_Shortage_Sum['Shortage'],color='gold', label='Denver')
plt.xlabel('Hydrologic Year')
# plt.plot(Aurora_Total_Shortage_Sum['Shortage'], color= 'gold', label = 'Aurora')
# plt.plot(Thornton_Total_Shortage_Sum['Shortage'],color='yellow', label='Thornton')
# plt.plot(Westminster_Total_Shortage_Sum['Shortage'],color='black', label='Westminster')
#plt.plot(Arvada_Total_Shortage_Sum['Shortage'],color='purple', label='Arvada')
plt.ylabel('Shortage (AF)')
plt.legend() 

plt.figure()
#plt.plot(Denver_Total_Shortage_Sum['Shortage'],color='gold', label='Denver')
# plt.plot(Aurora_Total_Shortage_Sum['Shortage'], color= 'gold', label = 'Aurora')
# plt.plot(Thornton_Total_Shortage_Sum['Shortage'],color='yellow', label='Thornton')
# plt.plot(Westminster_Total_Shortage_Sum['Shortage'],color='black', label='Westminster')
plt.xlabel('Hydrologic Year')
plt.plot(Arvada_Total_Shortage_Sum['Shortage'],color='purple', label='Arvada')
plt.ylabel('Shortage (AF)')
plt.legend() 

########################################################################################################################################

plt.figure()
plt.plot(Boulder_Total_Shortage_Sum['Shortage'],color='blue', label='Boulder')
plt.plot(Loveland_Total_Shortage_Sum['Shortage'],color='green', label='Loveland')
plt.plot(Longmont_Total_Shortage_Sum['Shortage'],color='red', label='Longmont')
plt.plot(Thornton_Total_Shortage_Sum['Shortage'],color='yellow', label='Thornton')
plt.plot(Westminster_Total_Shortage_Sum['Shortage'],color='black', label='Westminster')
plt.plot(Laffayette_Total_Shortage_Sum['Shortage'],color='brown', label='Laffayette')
plt.plot(Louisville_Total_Shortage_Sum['Shortage'],color='orange', label='Louisville')
plt.plot(ConMut_Total_Shortage_Sum['Shortage'],color='pink', label='ConMut')
plt.plot(Golden_Total_Shortage_Sum['Shortage'],color='gray', label='Golden')
plt.plot(Englewood_Total_Shortage_Sum['Shortage'],color='aqua', label='Englewood')
plt.plot(Aurora_Total_Shortage_Sum['Shortage'], color= 'gold', label = 'Aurora')
plt.plot(Denver_Total_Shortage_Sum['Shortage'],color='gold', label='Denver')
plt.plot(Arvada_Total_Shortage_Sum['Shortage'],color='purple', label='Arvada')
plt.plot(EstesPark_Total_Shortage_Sum['Shortage'], color='red', label='Estes Park')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right', borderaxespad=0)



##################################### NORTHERN WATER ONLY ########################################

## check Greeley ###

Greeley = pd.read_parquet('0400702.parquet', engine='pyarrow')
Greeley_Shortage = Greeley['shortage']
Greeley_Total_Shortage_Sum = pd.DataFrame().assign(Year=Greeley['year'],Shortage=Greeley_Shortage)
Greeley_Total_Shortage_Sum = Greeley_Total_Shortage_Sum.groupby('Year').sum()

## Sum all Northern Water municipalities shortages' ###

Northern_Water_Muni_Shortages = (EstesPark_Total_Shortage_Sum + Loveland_Total_Shortage_Sum + Greeley_Total_Shortage_Sum + Longmont_Total_Shortage_Sum +
                        Louisville_Total_Shortage_Sum + Laffayette_Total_Shortage_Sum + Boulder_Total_Shortage_Sum)

plt.figure()
plt.plot(Northern_Water_Muni_Shortages)
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.title('Northern Water Municipal Shortages')

### find the year with the highest amt of shortage (driest year) for use to describe two-way option market ###
Northern_Water_Muni_Shortages.max()
Driest_Year = int(Northern_Water_Muni_Shortages.idxmax(0))

### dry year two-way option ####

dry_year_two_dict = {}

for y in range(1950,2013):
    dry_year_two = pd.DataFrame()
    irrigation_set_dry_two = spirr.irrigation_uses_by_year_dry[y]
    irrigation_set_dry_two['MUNIUSAGE'] = 0
    water_avail_dry_year = muni.Northern_Water_Muni_Shortages['Shortage'].loc[y]
    for crop in range(len(irrigation_set_dry_two)):
        if water_avail_dry_year > irrigation_set_dry_two['CONSUMPTIVE_USE_TOTAL'].iloc[crop]:
            use = irrigation_set_dry_two['CONSUMPTIVE_USE_TOTAL'].iloc[crop]
            print(use)
        else:
            use = max(water_avail_dry_year, 0)
            print(use)
        irrigation_set_dry_two['MUNIUSAGE'].iloc[crop] = use   
        water_avail_dry_year -= irrigation_set_dry_two['CONSUMPTIVE_USE_TOTAL'].iloc[crop]
    dry_year_two = pd.concat([dry_year_two, irrigation_set_dry_two])

    dry_year_two['mv_h2o_sum'] = dry_year_two['MARGINAL_VALUE_OF_WATER']*dry_year_two['MUNIUSAGE']
        
    dry_year_two_dict[y] = dry_year_two

dry_year_max_payout = {}    
for y in range(1950,2013):
    dry_year_max_payout[y] = dry_year_two_dict[y]['mv_h2o_sum'].sum()
    

dry_year_max_payout_df = pd.DataFrame.from_dict(dry_year_max_payout, orient ='index')
dry_year_max_payout_df['payout'] = dry_year_max_payout_df[0]
dry_year_max_payout_series = pd.Series(dry_year_max_payout_df[0])
dry_year_max_payout_list = dry_year_max_payout_series.to_list()
plt.figure()
plt.plot(dry_year_max_payout_df)
plt.xlabel('Hydrologic Year')
plt.ylabel('Dry Year Payout($)')

dry_year_avg_payout = dry_year_max_payout_df.mean()

# ### wet-year two-way option ###

muni_surplus_water = 6406

wet_year_two_dict = {}

for y in range(1950,2013):
    wet_year_two = pd.DataFrame()
    irrigation_set_wet_two = spirr.irrigation_uses_by_year_wet[y]
    irrigation_set_wet_two['MUNISURPLUS'] = 0
    water_avail_wet_year = muni_surplus_water
    for crop in range(len(irrigation_set_wet_two)):
        if water_avail_wet_year > irrigation_set_wet_two['USAGE'].iloc[crop]:
            use = irrigation_set_wet_two['USAGE'].iloc[crop]
            print(use)
        else:
            use = max(water_avail_wet_year, 0)
            print(use)
        irrigation_set_wet_two['MUNISURPLUS'].iloc[crop] = use   
        water_avail_wet_year -= irrigation_set_wet_two['USAGE'].iloc[crop]
    wet_year_two = pd.concat([wet_year_two, irrigation_set_wet_two])

##### MUNIS ARE PROFIT MAXIMIZERS ####
    wet_year_two['mv_h2o_sum'] = wet_year_two['MARGINAL_VALUE_OF_WATER']*wet_year_two['MUNISURPLUS']

##### MUNIS ARE LEASING AT $30/AF ####
    wet_year_two['mv_h2o_sum_muni30'] = 30*wet_year_two['MUNISURPLUS']
        
    wet_year_two_dict[y] = wet_year_two

wet_year_max_payout_s1 = {}
wet_year_max_payout_s2 = {}    
for y in range(1950,2013):
    wet_year_max_payout_s1[y] = wet_year_two_dict[y]['mv_h2o_sum'].sum()
    wet_year_max_payout_s2[y] = wet_year_two_dict[y]['mv_h2o_sum_muni30'].sum()
    

wet_year_max_payout_df_s1 = pd.DataFrame.from_dict(wet_year_max_payout_s1, orient ='index')
wet_year_max_payout_df_s1['payout'] = wet_year_max_payout_df_s1[0]
wet_year_max_payout_series_s1 = pd.Series(wet_year_max_payout_df_s1[0])
wet_year_max_payout_list_s1 = wet_year_max_payout_series_s1.to_list()
plt.figure()
plt.plot(wet_year_max_payout_df_s1)
plt.title('Scenario 1')
plt.xlabel('Hydrologic Year')
plt.ylabel('Wet Year Payout($)')

wet_year_avg_payout_s1 = wet_year_max_payout_df_s1.mean()

wet_year_max_payout_df_s2 = pd.DataFrame.from_dict(wet_year_max_payout_s2, orient ='index')
wet_year_max_payout_df_s2['payout'] = wet_year_max_payout_df_s2[0]
wet_year_max_payout_series_s2 = pd.Series(wet_year_max_payout_df_s2[0])
wet_year_max_payout_list_s2 = wet_year_max_payout_series_s2.to_list()
plt.figure()
plt.plot(wet_year_max_payout_df_s2)
plt.title('Scenario 2')
plt.xlabel('Hydrologic Year')
plt.ylabel('Wet Year Payout($)')

wet_year_avg_payout_s2 = wet_year_max_payout_df_s2.mean()

##########################################################################
######### wang transform function ########################################
############## Returns dataframe with net payout #########################
##########################################################################

## lam is set to 0.25 unless otherwise specified (risk adjustment)
## df should be dataframe with payout per year 
def wang_slim(payouts, lam = 0.25, contract = 'call', from_user = False):  
  if from_user == True: 
      payouts = -payouts
  if contract == 'put': 
      lam = -abs(lam)
  unique_pays = pd.DataFrame()
  unique_pays['unique'] = payouts.payout.unique()
  unique_pays['prob'] = 0 
  for j in range(len(unique_pays)):  
      count = 0
      val = unique_pays['unique'].iloc[j]
      for i in np.arange(len(payouts)): 
          if payouts['payout'].iloc[i] == val: 
              count += 1
    #  print(count)
      unique_pays['prob'].iloc[j] = count/len(payouts)
      
  unique_pays.sort_values(inplace=True, by='unique')
  dum = unique_pays['prob'].cumsum()  # asset cdf
  dum = st.norm.cdf(st.norm.ppf(dum) + lam)  # risk transformed payout cdf
  dum = np.append(dum[0], np.diff(dum))  # risk transformed asset pdf
  prem = (dum * unique_pays['unique']).sum()
  print(prem)
  payouts.sort_index(inplace=True)

  if from_user == False: 
      whole = (payouts['payout'] - prem)
  else: 
      payouts.sort_index(inplace=True)
      whole = (prem - payouts['payout'])
  
  return prem, whole

dry_year_prems, dry_year_whole = wang_slim(dry_year_max_payout_df, lam = 0.25, contract = 'call', from_user = False)
wet_year_s1_prems, wet_year_s1_whole = wang_slim(wet_year_max_payout_df_s1, lam = 0.25, contract = 'call', from_user = False)
wet_year_s2_prems, wet_year_s2_whole = wang_slim(wet_year_max_payout_df_s2, lam = 0.25, contract = 'call', from_user = False)


plt.plot(dry_year_whole)
plt.plot(wet_year_s1_whole)
plt.plot(wet_year_s2_whole)

premiums_s1 = dry_year_prems + wet_year_s1_prems
premiums_s2 = dry_year_prems + wet_year_s2_prems

premiums_s1_peraf = (dry_year_prems + wet_year_s1_prems)/muni_surplus_water
premiums_s2_peraf = (dry_year_prems + wet_year_s2_prems)/muni_surplus_water

# def id_payouts_index(inputs, strike, cap, model): 
#     payouts = pd.DataFrame(index = np.arange(0, len(inputs)),columns=['payout']) 
#     payouts['payout'] = 0
    
#     for i in np.arange(len(inputs)):
#         if inputs['Dalls ARF'].iloc[i] < strike: 
#             ## model will predict a value that represents losses essentially
#             payouts.iloc[i,0] = model.predict(inputs.iloc[i,:])
#        #     print(payouts.iloc[i,0])
#             ## constrain so that if predicted value is > 0, BPA does not get paid
#             if payouts.iloc[i,0] > 0:
#                 payouts.iloc[i,0] = 0
#             ## cap payouts
#             if payouts.iloc[i,0] < cap: 
#                 payouts.iloc[i,0] = cap     
#             ## NOTE: these are left negative to work within the wang transform function
#     return payouts

# def id_payment_swaps(input, strike, cap = 200000000, slope = False, swap = False, strike2 = ''): 
#     payouts = pd.DataFrame(index=np.arange(0, len(input)),columns=['payout']) 
#     payouts['payout'] = 0
#     mean = input.iloc[:,0].mean() 
    
#     for i in np.arange(len(input)):
#         if swap == True: 
#             if input['Dalls ARF'].iloc[i] >= strike: 
#                 ## no modifier
#                 if slope == False: 
#                     payouts.iloc[i,0] = abs(est_above2.predict(input.iloc[i, 1:]))/payout_mod
#                 else: 
#                     slope = input.iloc[i,0]/mean
#                     payouts.iloc[i,0] = abs(est_above2.predict(input.iloc[i, 1:]) * slope)/payout_mod
#         if swap == False: 
#             if input['Dalls ARF'].iloc[i] >= strike2: 
#                 ## no modifier
#                 if slope == False: 
#                     payouts.iloc[i,0] = abs(est_above2.predict(input.iloc[i, 1:]))/payout_mod
#                 else: 
#                     slope = input.iloc[i,0]/mean
#                     payouts.iloc[i,0] = abs(est_above2.predict(input.iloc[i, 1:]) * slope)/payout_mod
#         ## constrain payout 
#         if payouts.iloc[i,0] < 0:
#             payouts.iloc[i,0] = 0
#         ## cap payouts
#         if payouts.iloc[i,0] > cap: 
#             payouts.iloc[i,0] = cap 
#     ## return negative because these are payments from BPA to the counterparty 
#     return -payouts




os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod')

plt.figure()
plt.plot(EstesPark_Total_Shortage_Sum['Shortage'],color='purple', label='Estes Park')
plt.plot(Loveland_Total_Shortage_Sum['Shortage'],color='green', label='Loveland')
plt.plot(Greeley_Total_Shortage_Sum['Shortage'],color='yellow', label='Greeley')
plt.plot(Longmont_Total_Shortage_Sum['Shortage'],color='red', label='Longmont')
plt.plot(Louisville_Total_Shortage_Sum['Shortage'],color='orange', label='Louisville')
plt.plot(Laffayette_Total_Shortage_Sum['Shortage'],color='brown', label='Laffayette')
plt.plot(Boulder_Total_Shortage_Sum['Shortage'],color='blue', label='Boulder')
plt.xlabel('Hydrologic Year')
plt.ylabel('Shortage (AF)')
plt.title('Northern Water Municipal Shortages')
plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right', borderaxespad=0)

