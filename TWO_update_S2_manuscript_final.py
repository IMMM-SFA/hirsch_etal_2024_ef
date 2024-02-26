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
#import co_snow_metrics as cosnow
import statemod_sp_adapted as muni
import matplotlib.colors as colors
import extract_ipy_southplatte as ipy
import new_irrigation_test as new
import seaborn as sns
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.colors as mcolors


manuscript_scenario_no = 2
scenario_type = 'current water rights'

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xreparquet_ucrb_02_17/')


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

Wet_Year_Triggers = Granby_Spills.loc[Granby_Spills > 0]

Wet_Years = pd.DataFrame(index=range(1950,2013))
Wet_Years['value'] = 0

for i in Wet_Year_Triggers.index:
    Wet_Years['value'].loc[i] = Wet_Year_Triggers[i]



## import CBI ###
## CREATE THRESHnewS BASED ON DIFFERENT CBI VALUES ##

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')
CBI = pd.read_csv('cbi_timeseries.csv', header= None)
CBI.columns =['date', 'snowpack', 'diversion', 'storage', 'cbi']
CBI['date'] = pd.to_datetime(CBI["date"])
CBI.index = CBI['date']
CBI['year'] = CBI.index.year
CBI['month'] = CBI.index.month

plt.figure()
plt.plot(CBI['cbi'])
#plt.title('CBI')
plt.xlabel('Hydrologic Year')
plt.ylabel('CBI (tAF)')

CBI_Yearly = CBI.groupby('year').nth(2)['cbi']
#CBI_Yearly = CBI.groupby('year').mean()['cbi']
CBI_Yearly = CBI_Yearly[CBI_Yearly.index <= 2012]

plt.figure()
plt.plot(CBI_Yearly)
# plt.axhline(y = 550, color = 'r', linestyle = 'dashed')
# plt.axhline(y = 600, color = 'r', linestyle = 'dashed')
# plt.axhline(y = 650, color = 'r', linestyle = 'dashed')
# plt.axhline(y = 700, color = 'r', linestyle = 'dashed')
plt.axhline(y = 800, color = 'r', linestyle = 'dashed', label = '')
#plt.axhline(y = 675, color = 'g', linestyle = 'dashed')
plt.xlabel('Hydrologic Year')
plt.ylabel('CBI (tAF)')

CBI_S4 = CBI_Yearly.loc[CBI_Yearly <= 575]
CBI_S3 = CBI_Yearly.loc[CBI_Yearly <= 625]
CBI_S2 = CBI_Yearly.loc[CBI_Yearly <= 700]
CBI_S1 = CBI_Yearly.loc[CBI_Yearly <= 725]
CBI_S0 = CBI_Yearly.loc[CBI_Yearly <= 775]

CBI_Surplus = CBI_Yearly.loc[CBI_Yearly >= 800]

### CBI vs. Northern Water Shortages ###

plt.figure()
plt.scatter(CBI_Yearly, Wet_Years, c = 'red')
plt.axvline(x = 800, color = 'r', linestyle = 'dashed')
plt.xlabel('CBI')
plt.ylabel('Granby Spills (AF)')


### dry year two-way option ####

exercise_fees = [5, 7.5, 10, 12.5, 15]
added_mv_dry = pd.Series(index = range(1950,2013))
max_mv_dict = {}
for y in range(1950,2013):
    new.TWO_Selection[y]['year'] = y
    #max_mv = list(new.TWO_Selection[y].loc[new.TWO_Selection[y]['TWO_USAGE'] > 0]['MARGINAL_VALUE_OF_WATER'])[-1]
    if y in CBI_S2.index:
        #max_mv = list(new.TWO_Selection[y].loc[new.TWO_Selection[y]['TWO_USAGE'] > 0]['MARGINAL_VALUE_OF_WATER'])[-1]
        #list(new.TWO_Selection[y].loc[new.TWO_Selection[y]['TWO_USAGE'] > 0]['MARGINAL_VALUE_OF_WATER'])[-1]
        ###THIS IS THE MARKET CLEARING PRICE ######
        max_mv = list(new.TWO_Selection[y].loc[new.TWO_Selection[y]['TWO_USAGE'] > 0]['MARGINAL_VALUE_OF_WATER'])[-1]
        max_mv_dict[y] = max_mv
        print(max_mv)
        ag_max_rev_price = 1000
        ## this is where you change price for scenarios ####
        new.TWO_Selection[y]['TWO_BASELINE_SCENARIO_MV'] = new.TWO_Selection[y]['TWO_USAGE']*max_mv
        for n in exercise_fees:
            new.TWO_Selection[y][f"exercise_fee_{n}"] = new.TWO_Selection[y]['TWO_USAGE']*n
            new.TWO_Selection[y][f"TWO_BASELINE_SCENARIO_E{n}"] = new.TWO_Selection[y]['TWO_BASELINE_SCENARIO_MV']-new.TWO_Selection[y][f"exercise_fee_{n}"] 
    else:
        new.TWO_Selection[y]['TWO_BASELINE_SCENARIO_MV'] = 0
    print(new.TWO_Selection[y])
    added_mv_dry[y]= new.TWO_Selection[y]['TWO_BASELINE_SCENARIO_MV'].sum()

dry_year_options = {}
individual_dry_year_contracts = pd.DataFrame()
for i in (new.TWO_Selection[y].index.unique()):
    contracts = pd.DataFrame()
    for y in range(1950,2013):
        contract = new.TWO_Selection[y].loc[new.TWO_Selection[y].index== i] #could be uses_of_water_dry in [] if things break
        individual_dry_year_contracts = pd.concat([contract, individual_dry_year_contracts])

dry_year_options = {}
for i in (new.TWO_Selection[y].index.unique()):
    contracts = individual_dry_year_contracts.loc[individual_dry_year_contracts.index== i]
    contracts = contracts.sort_values(by=['year'], ascending=True)
    contracts = contracts.set_index("year")

    for n in exercise_fees:
        contracts[f"TWO_BASELINE_EXERCISE{n}"] = 0
        for y in range(1950,2013):
            if y in CBI_S2.index:
                contracts[f"TWO_BASELINE_EXERCISE{n}"][y] = contracts[f"TWO_BASELINE_SCENARIO_E{n}"][y]
            else:
                contracts[f"TWO_BASELINE_EXERCISE{n}"][y] = contracts['TWO_BASELINE_SCENARIO_MV'][y]

                
    dry_year_options[i] = contracts
    




dry_year_baseline_exercise = {}
for n in exercise_fees:
    dry_year_baseline_exercise[n] = pd.DataFrame()
    for i in (new.uses_of_water_dry[y].index.unique()): 
        dry_year_baseline_exercise[n][i] = dry_year_options[i][f"TWO_BASELINE_EXERCISE{n}"]

dry_year_total_water_reallocated = pd.DataFrame()
for i in (new.uses_of_water_dry[y].index.unique()): 
    dry_year_total_water_reallocated[i] = dry_year_options[i]['TWO_USAGE']

dry_year_option_timeseries = {} 
dry_year_totals_timeseries = {}   
for e in exercise_fees:
    dry_year_selected_exercise_fee = e
    
    dry_year_payouts = {}
    
    for name in (new.uses_of_water_dry[y].index.unique()):
        for n in [dry_year_selected_exercise_fee]:
            df_name = str(name) + '_df'
            dry_year_payouts[df_name] = pd.DataFrame(dry_year_options[name][f"TWO_BASELINE_EXERCISE{n}"])
            dry_year_payouts[df_name] = dry_year_payouts[df_name].rename(columns={f"TWO_BASELINE_EXERCISE{n}": 'payout'})
            
    
    ##########################################################################
    ######### wang transform function ########################################
    ############## Returns dataframe with net payout #########################
    ##########################################################################
    
    ## lam is set to 0.25 unless otherwise specified (risk adjustment)
    ## df should be dataframe with payout per year 
    ## modified slightly -- from user = False means that it is from the insurer's perspective
    def wang_slim(payouts, lam = 0.25, contract = 'put', from_user = False):  
      if from_user == True: 
          ## switch the signs because you are feeding a negative value where the payout occurs
          ## all other values are zero
          payouts = -payouts
      if contract == 'put': 
          lam = -abs(lam)
      unique_pays = pd.DataFrame()
      unique_pays['unique'] = payouts.payout.unique()
      unique_pays['lam'] = lam
      #print(unique_pays['unique'])
      unique_pays['payment probability'] = 0 
      for j in range(len(unique_pays)):  
          count = 0
          val = unique_pays['unique'].iloc[j]
          for i in np.arange(len(payouts)): 
              if payouts['payout'].iloc[i] == val: 
                  count += 1
        #  print(count)
          unique_pays['payment probability'].iloc[j] = count/len(payouts)
          #print(unique_pays)
          
      unique_pays.sort_values(inplace=True, by='unique')
      #print(unique_pays)
      dum1 = unique_pays['payment probability'].cumsum()  # asset cdf
      #print(dum1)
      unique_pays['dum1'] = dum1
      unique_pays['dum1ppf'] = st.norm.ppf(dum1)
      unique_pays['dum1ppf_riskadj'] = (st.norm.ppf(dum1) + lam)

      sns.kdeplot(dum1, color = 'blue')
      unique_pays['dum2_preriskadj'] = st.norm.cdf(st.norm.ppf(dum1))
      dum2 = st.norm.cdf(st.norm.ppf(dum1) + lam)  # risk transformed payout cdf
      unique_pays['dum2'] = dum2

      sns.kdeplot(dum2, color ='red')
      dum3 = np.append(dum2[0], np.diff(dum2))  # risk transformed asset pdf
      unique_pays['dum3'] = dum3
      unique_pays['new'] = dum3 * unique_pays['unique']

      
      prem = (dum3 * unique_pays['unique']).sum()

      payouts.sort_index(inplace=True)
    
      if from_user == True: 
         ## want the insurer's perspective
          whole = (prem - payouts['payout'])
    
      else: 
    #      payouts.sort_index(inplace=True)
         whole = (payouts['payout'] - prem)
    
      
      return prem, whole, unique_pays
    
#############################################################################################################################################
#############################################################################################################################################

    dry_year_prems = {}
    dry_year_whole = {}
    dry_year_unique_pays = {}
    dry_year_prems_total = pd.DataFrame()
    dry_year_whole_total = pd.DataFrame()
    
    for i in (new.TWO_Selection[y].index.unique()):
        dry_year_prems[i], dry_year_whole[i], dry_year_unique_pays[i] = wang_slim(dry_year_payouts[f"{i}_df"], lam = 0.25, \
                                               contract = 'call', from_user = True)
    
    
    dry_year_prems_total = {}
    for i in (new.uses_of_water_dry[y].index.unique()):
        dry_year_prems_total[i] = dry_year_prems[i] * len(range(1950,2013))
    
    dry_year_option_fee = pd.DataFrame([dry_year_prems_total])
    dry_year_summed_option_fee = pd.DataFrame()
    dry_year_summed_option_fee = dry_year_option_fee.T.sum (axis=0)
    
    dry_year_summed_water_reallocated = pd.DataFrame()
    dry_year_summed_water_reallocated_by_year = dry_year_total_water_reallocated.T.sum (axis=0)
    dry_year_summed_water_reallocated = dry_year_summed_water_reallocated_by_year.sum()
    
        
    dry_year_option_fee = dry_year_summed_option_fee/len(range(1950,2013))/dry_year_summed_water_reallocated_by_year.max()
    dry_year_option_fee = abs(dry_year_option_fee[0])
    
    ## check ####
    dry_year_option_check = sum(dry_year_prems.values())/dry_year_summed_water_reallocated_by_year.max()
    
    

    
    dry_year_two_cost = pd.Series(index=range(1950,2013))
    for y in range(1950,2013):
        if y in CBI_S2.index:
            dry_year_two_cost[y] = dry_year_option_fee + dry_year_selected_exercise_fee
        else:
            dry_year_two_cost[y] = dry_year_option_fee
    
    dry_year_two_cost_sum = pd.Series(index=range(1950,2013))
    dry_year_ag_gain = pd.Series(index=range(1950,2013))
    for y in range(1950,2013):
        if y in CBI_S2.index:
            dry_year_two_cost_sum[y] = (dry_year_summed_water_reallocated_by_year.max() * dry_year_option_fee) + (dry_year_summed_water_reallocated_by_year[y] * dry_year_selected_exercise_fee)
            dry_year_ag_gain[y] = (dry_year_summed_water_reallocated_by_year.max() * dry_year_option_fee) + (dry_year_summed_water_reallocated_by_year[y] * dry_year_selected_exercise_fee)
        else:
            dry_year_two_cost_sum[y] = dry_year_summed_water_reallocated_by_year.max() * dry_year_option_fee
            dry_year_ag_gain[y] = dry_year_summed_water_reallocated_by_year.max() * dry_year_option_fee
            
            
    dry_year_option_timeseries[e] = dry_year_two_cost
    dry_year_totals_timeseries[e] = dry_year_two_cost_sum



plt.figure()
for e in exercise_fees:
    plt.plot(dry_year_option_timeseries[e], label = f"Dry Year Option with ${e} Exercise Fee")
    plt.legend(bbox_to_anchor=(1.75, 1), loc='upper right', borderaxespad=0, frameon = False)
    plt.ylabel('Cost ($/AF)')
    plt.xlabel('Hydrologic Year')






dry_year_summed_water_reallocated = pd.DataFrame()
dry_year_summed_water_reallocated = dry_year_total_water_reallocated.T.sum (axis=0)
dry_year_summed_water_reallocated = dry_year_summed_water_reallocated.sum()




### wet year two-way option ####

exercise_fees = [5, 7.5, 10, 12.5, 15]
added_mv_wet = pd.Series(index = range(1950,2013))
min_mv_dict = {}
for y in range(1950,2013):
    new.TWO_Selection_Wet[y]['year'] = y
    new.TWO_Selection_Wet[y]['TWO_MV'] = 0

    if y in CBI_Surplus.index:

        min_mv = list(new.TWO_Selection_Wet[y].loc[new.TWO_Selection_Wet[y]['TWO_USAGE'] > 0]['MARGINAL_VALUE_OF_WATER'])[-1]
        min_mv_dict[y] = min_mv
        new.TWO_Selection_Wet[y]['TWO_MV'] = new.TWO_Selection_Wet[y]['TWO_USAGE'] * new.TWO_Selection_Wet[y]['MARGINAL_VALUE_OF_WATER']
        new.TWO_Selection_Wet[y]['TWO_BASELINE_SCENARIO_MVH2O'] = new.TWO_Selection_Wet[y]['TWO_USAGE']*min_mv
        new.TWO_Selection_Wet[y]['TWO_BASELINE_SCENARIO_MV30'] = new.TWO_Selection_Wet[y]['TWO_USAGE']*30
        for n in exercise_fees:
            new.TWO_Selection_Wet[y][f"exercise_fee_{n}"] = new.TWO_Selection_Wet[y]['TWO_USAGE']*n
            new.TWO_Selection_Wet[y][f"TWO_BASELINE_SCENARIO_MVH2O_E{n}"] = new.TWO_Selection_Wet[y]['TWO_BASELINE_SCENARIO_MVH2O']-new.TWO_Selection_Wet[y][f"exercise_fee_{n}"]
            new.TWO_Selection_Wet[y][f"TWO_BASELINE_SCENARIO_MV30_E{n}"] = new.TWO_Selection_Wet[y]['TWO_BASELINE_SCENARIO_MV30']-new.TWO_Selection_Wet[y][f"exercise_fee_{n}"]
    else:
        new.TWO_Selection_Wet[y]['TWO_BASELINE_SCENARIO_MVH2O'] = 0
        new.TWO_Selection_Wet[y]['TWO_BASELINE_SCENARIO_MV30'] = 0
    print(new.TWO_Selection_Wet[y])
    added_mv_wet[y]= new.TWO_Selection_Wet[y]['TWO_MV'].sum()

wet_year_options = {}
individual_wet_year_contracts = pd.DataFrame()
for i in (new.TWO_Selection_Wet[y].index.unique()):
    contracts_wet = pd.DataFrame()
    for y in range(1950,2013):
        contracts_wet = new.TWO_Selection_Wet[y].loc[new.TWO_Selection_Wet[y].index== i] #could be uses_of_water_dry in [] if things break
        individual_wet_year_contracts = pd.concat([contracts_wet, individual_wet_year_contracts])

wet_year_options = {}
for i in (new.TWO_Selection_Wet[y].index.unique()):
    contracts_wet = individual_wet_year_contracts.loc[individual_wet_year_contracts.index== i]
    contracts_wet = contracts_wet.sort_values(by=['year'], ascending=True)
    contracts_wet = contracts_wet.set_index("year")

    for n in exercise_fees:
        contracts_wet[f"TWO_BASELINE_MVH2O_EXERCISE{n}"] = 0
        contracts_wet[f"TWO_BASELINE_MV30_EXERCISE{n}"] = 0
        for y in range(1950,2013):
            if y in CBI_Surplus.index:
                contracts_wet[f"TWO_BASELINE_MVH2O_EXERCISE{n}"][y] = contracts_wet[f"TWO_BASELINE_SCENARIO_MVH2O_E{n}"][y]
                contracts_wet[f"TWO_BASELINE_MV30_EXERCISE{n}"][y] = contracts_wet[f"TWO_BASELINE_SCENARIO_MV30_E{n}"][y]
            else:
                contracts_wet[f"TWO_BASELINE_MVH2O_EXERCISE{n}"][y] = contracts_wet['TWO_MV'][y]
                contracts_wet[f"TWO_BASELINE_MV30_EXERCISE{n}"][y] = contracts_wet['TWO_MV'][y]

                
    wet_year_options[i] = contracts_wet
    



wet_year_baseline_exercise = {}
for n in exercise_fees:
    wet_year_baseline_exercise[n] = pd.DataFrame()
    for i in (new.uses_of_water_wet[y].index.unique()): 
        wet_year_baseline_exercise[n][i] = wet_year_options[i][f"TWO_BASELINE_MV30_EXERCISE{n}"]

wet_year_total_water_reallocated = pd.DataFrame()
for i in (new.uses_of_water_wet[y].index.unique()): 
    wet_year_total_water_reallocated[i] = wet_year_options[i]['TWO_USAGE']
    
wet_year_option_timeseries_mv30 = {}    
for e in exercise_fees:
    wet_year_selected_exercise_fee = e
    
    wet_year_payouts = {}
    
    for name in (new.uses_of_water_wet[y].index.unique()):
        for n in [wet_year_selected_exercise_fee]:
            df_name = str(name) + '_df'
            
            ### this is where you change the wet year scenario ####
            wet_year_payouts[df_name] = pd.DataFrame(wet_year_options[name][f"TWO_BASELINE_SCENARIO_MV30_E{n}"])
            wet_year_payouts[df_name] = wet_year_payouts[df_name].rename(columns={f"TWO_BASELINE_SCENARIO_MV30_E{n}": 'payout'})
            
    
    
    
    
    
    
    
    
    
    wet_year_prems = {}
    wet_year_whole = {}
    wet_year_unique_pays = {}
    wet_year_prems_total = pd.DataFrame()
    wet_year_whole_total = pd.DataFrame()
    for i in (new.TWO_Selection_Wet[y].index.unique()):
        wet_year_prems[i], wet_year_whole[i], wet_year_unique_pays[i] = wang_slim(wet_year_payouts[f"{i}_df"], lam = 0.25, \
                                               contract = 'call', from_user = True)
    
    
    wet_year_prems_total = {}
    for i in (new.uses_of_water_wet[y].index.unique()):
        wet_year_prems_total[i] = wet_year_prems[i] * len(range(1950,2013))
        
        
        
    wet_year_cost_per_af = {}
    for i in (new.uses_of_water_wet[y].index.unique()):
        wet_year_cost_per_af[i] = wet_year_prems_total[i] / wet_year_total_water_reallocated[i]
        
    
    
    wet_year_option_fee = pd.DataFrame([wet_year_prems_total])
    wet_year_summed_option_fee = pd.DataFrame()
    wet_year_summed_option_fee = wet_year_option_fee.T.sum (axis=0)
    
    wet_year_summed_water_reallocated = pd.DataFrame()
    wet_year_summed_water_reallocated_by_year = wet_year_total_water_reallocated.T.sum (axis=0)
    wet_year_summed_water_reallocated = wet_year_summed_water_reallocated_by_year.sum()
    
        
    wet_year_option_fee = wet_year_summed_option_fee/len(range(1950,2013))/wet_year_summed_water_reallocated_by_year.max()
    wet_year_option_fee = abs(wet_year_option_fee[0])
    

    
    wet_year_two_cost = pd.Series(index=range(1950,2013))
    for y in range(1950,2013):
        if y in CBI_Surplus.index:
            wet_year_two_cost[y] = wet_year_option_fee + wet_year_selected_exercise_fee
        else:
            wet_year_two_cost[y] = wet_year_option_fee

    wet_year_option_timeseries_mv30[e] = wet_year_two_cost
    
    
wet_year_selected_exercise_fee = 25
    
wet_year_two_cost_sum = pd.Series(index=range(1950,2013))
for y in range(1950,2013):
     if y in CBI_Surplus.index:
         wet_year_two_cost_sum[y] = (wet_year_summed_water_reallocated_by_year.max() * wet_year_option_fee) + (wet_year_summed_water_reallocated_by_year[y] * wet_year_selected_exercise_fee)
     else:
         wet_year_two_cost_sum[y] = wet_year_summed_water_reallocated_by_year.max() * wet_year_option_fee

plt.figure()
for e in exercise_fees:
    plt.plot(wet_year_option_timeseries_mv30[e], label = f"Wet Year Option with ${e} Exercise Fee")
    plt.legend(bbox_to_anchor=(1.75, 1), loc='upper right', borderaxespad=0, frameon = False)
    plt.ylabel('Cost ($/AF)')
    plt.xlabel('Hydrologic Year')
    
    
#total net benefits to ag   

two_nb_to_ag = pd.Series(index=range(1950,2013))
for y in range(1950,2013):    
    two_nb_to_ag[y] = dry_year_two_cost_sum[y] - wet_year_two_cost_sum[y]


    


########### ANALYSIS SELECTED EXERCISE FEE ################################################

dry_year_analysis_exercise_fee = 15
wet_year_analysis_exercise_fee = 15



import matplotlib.colors as mcolors

color1 = '#335C67'
color2 = '#540B0E'
color3 = '#99A88C'
color4 = '#E09F3E'
color5 = '#66827A'
color6 = '#BDA465'
color7 = '#9A5526'

af_to_m3 = 1233.4818375475
af_to_km3 = 810713.18210885
CBI_km3_yearly = CBI_Yearly*1000/af_to_km3

from matplotlib.gridspec import GridSpec
x = dry_year_option_timeseries[dry_year_analysis_exercise_fee].index
data_1 = dry_year_option_timeseries[dry_year_analysis_exercise_fee]
data_2 = dry_year_summed_water_reallocated_by_year*af_to_m3/1000000
data_3 = wet_year_option_timeseries_mv30[wet_year_analysis_exercise_fee]
data_4 = wet_year_summed_water_reallocated_by_year*af_to_m3/1000000

# Set the size of the figure
plt.figure(figsize=(10, 15))

ax1 = plt.subplot(3, 1, 1)
plt.bar(x, data_4, color = color1) 
plt.bar(x, -(data_2), color = color2, label = 'Dry Year')
plt.ylim([-25,25])

ticks =  ax1.get_yticks()
for label in ax1.get_yticklabels()[0:3]:
    label.set_color(color2)
for label in ax1.get_yticklabels()[4:6]:
    label.set_color(color1)
# set labels to absolute values and with integer representation
ax1.set_yticklabels([int(abs(tick)) for tick in ticks], fontsize=20, fontname='Myriad Pro')

ax1.axhline(y=0, color='black')
twin1 = plt.twinx()
twin1.plot(x, CBI_km3_yearly, color = color5) 
twin1.set_ylabel('CBI ($km^{3}$)', color = color5, fontsize=20, fontname='Myriad Pro') 
plt.ylim([0.5,1.13])
twin1.tick_params(labelcolor = color5, labelsize=20)
ticks =  twin1.get_yticks()
twin1.set_yticklabels([f'{abs(tick):.1f}' for tick in ticks], fontsize=20)
#twin1.set_yticklabels([int(abs(tick)) for tick in ticks], fontsize=20)
ax1.text(-.1, 0.75, 'To Ag', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = color1, fontsize=20, fontname='Myriad Pro')
ax1.text(-.1, 0.25, 'To Urban', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = color2, fontsize=20, fontname='Myriad Pro')
ax1.text(-.15, 0.50, 'Water Re-allocated (MCM)', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = 'black', fontsize=20, fontname='Myriad Pro')
ax1.text(0.985, 0.985, 'A', fontsize=30, fontname='Myriad Pro', transform=ax1.transAxes, ha='right', va='top')

ax2 = plt.subplot(3, 1, 3, sharex = ax1)

y1 = two_nb_to_ag/1000000
y2 = added_mv_wet/1000000

ax2.bar(x, y2, bottom = y1, color = color4, label = 'Increased Ag Productivity')
ax2.bar(x, y1, color = color3, label = 'TWO Revenue/Cost')
plt.ylim([-0.9,2.3])
ax2.text(-.15, .5, 'Ag Costs/Gains ($M)', ha='center', va='center', transform=ax2.transAxes, rotation='vertical', color = 'black', fontsize=20, fontname='Myriad Pro')
ax2.text(0.985, 0.985, 'C', fontsize=30, fontname='Myriad Pro', transform=ax2.transAxes, ha='right', va='top')
for label in ax2.get_yticklabels()[0:7]:
    label.set_color("black")
    label.set_fontsize(20)
    label.set_fontname('Myriad Pro')

legend = plt.legend(fontsize=15, loc='upper left')
for text in legend.get_texts():
    text.set_fontname('Myriad Pro')
plt.xlabel('Year', fontsize=20, fontname='Myriad Pro') # Add this line


ax3 = plt.subplot(3, 1, 2, sharex = ax1)
data_5 = dry_year_two_cost_sum/1000000
data_6 = wet_year_two_cost_sum/1000000
plt.bar(x, data_6, color = color6) 
plt.bar(x, -(data_5), color = color7)
plt.ylim([-.9,.9])

ticks =  ax3.get_yticks()
for label in ax3.get_yticklabels()[0:5]:
    label.set_color(color7)
for label in ax3.get_yticklabels()[6:10]:
    label.set_color(color6)
# set labels to absolute values and with integer representation
ax3.set_yticklabels([f'{abs(tick):.1f}' for tick in ticks], fontsize=20, fontname='Myriad Pro')

ax3.axhline(y=0, color='black')
#plt.ylabel('TWO Cost ($)')
ax3.text(-.11, .75, 'Ag', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = color6, fontsize = 20, fontname='Myriad Pro')
ax3.text(-.11, .25, 'Urban', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = color7, fontsize = 20, fontname='Myriad Pro')
ax3.text(-.15, 0.45, 'Two-Way Option Cost ($M)', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = 'black', fontsize=20, fontname='Myriad Pro')
ax3.text(0.985, 0.985, 'B', fontsize=30, fontname='Myriad Pro', transform=ax3.transAxes, ha='right', va='top')
plt.subplots_adjust(hspace=0.4)


ticks =  ax1.get_xticks()
ax1.set_xticklabels([int(abs(tick)) for tick in ticks], fontsize=15, fontname='Myriad Pro')
ax2.set_xticklabels([int(abs(tick)) for tick in ticks], fontsize=15, fontname='Myriad Pro')
ax3.set_xticklabels([int(abs(tick)) for tick in ticks], fontsize=15, fontname='Myriad Pro')
#plt.show()
plt.savefig('TWO_update_S2_hi_res.png', format = 'png', dpi=600, bbox_inches='tight')
## FINAL TWO COST ############

ag_yearly_option_fee_payment = wet_year_summed_water_reallocated_by_year.max()*wet_year_option_fee*63
ag_exercise_payment_sum = wet_year_summed_water_reallocated_by_year.sum()*wet_year_selected_exercise_fee
ag_two_cost_per_af = (ag_yearly_option_fee_payment + ag_exercise_payment_sum)/wet_year_summed_water_reallocated_by_year.sum()

urban_yearly_option_fee_payment = dry_year_summed_water_reallocated_by_year.max()*dry_year_option_fee*63
urban_exercise_payment_sum = dry_year_summed_water_reallocated_by_year.sum()*dry_year_selected_exercise_fee
urban_two_cost_per_af = (urban_yearly_option_fee_payment + urban_exercise_payment_sum)/dry_year_summed_water_reallocated_by_year.sum()

### MEAN LEASE PRICES #####
min_mv_avg_df = pd.DataFrame.from_dict(min_mv_dict, orient='index')
min_mv_avg = min_mv_avg_df.mean()[0]

max_mv_avg_df = pd.DataFrame.from_dict(max_mv_dict, orient='index')
max_mv_avg = max_mv_avg_df.mean()[0]















