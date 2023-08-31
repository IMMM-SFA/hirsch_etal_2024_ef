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

scenario_no = 2
scenario_type = 'current water rights'

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xreparquet_ucrb_02_17/')

# import pandas as pd
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as st
# #import co_snow_metrics as cosnow
# #import statemod_sp_adapted as muni
# import matplotlib.colors as colors
# import extract_ipy_southplatte as ipy
# import new_irrigation_test_historicalrights as new
# import seaborn as sns
# from scipy.stats import norm
# from scipy.integrate import quad

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

plt.figure()
plt.plot(CBI['cbi'])
#plt.title('CBI')
plt.xlabel('Hydrologic Year')
plt.ylabel('CBI (tAF)')


CBI_Yearly = CBI.groupby('year').mean()['cbi']
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
CBI_S2 = CBI_Yearly.loc[CBI_Yearly <= 677]
CBI_S1 = CBI_Yearly.loc[CBI_Yearly <= 725]
CBI_S0 = CBI_Yearly.loc[CBI_Yearly <= 775]

CBI_Surplus = CBI_Yearly.loc[CBI_Yearly >= 800]

### CBI vs. Northern Water Shortages ###

plt.figure()
plt.scatter(CBI_Yearly, Wet_Years, c = 'red')
plt.axvline(x = 800, color = 'r', linestyle = 'dashed')
plt.xlabel('CBI')
plt.ylabel('Granby Spills (AF)')

# plt.figure()
# plt.scatter(CBI_Yearly, muni.Northern_Water_Muni_Shortages, c = 'red')
# plt.axvline(x = 800, color = 'r', linestyle = 'dashed')
# plt.xlabel('CBI')
# plt.ylabel('Northern Water Shortage (AF)')

# os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')
# Boulder_Surplus = pd.read_csv('Boulder_Surplus_Leasing.csv', header= None)
# Boulder_Surplus.columns =['year', 'af']
# Boulder_Surplus.index = Boulder_Surplus['year']
    

# plt.figure()
# plt.plot(CBI_Yearly, Boulder_Surplus['af'], c = 'red')



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
    #contracts['OPTION_FEE'] = max(contracts['POSSIBLEMUNIUSAGE']) * option_fee_per_af
    for n in exercise_fees:
        contracts[f"TWO_BASELINE_EXERCISE{n}"] = 0
        for y in range(1950,2013):
            if y in CBI_S2.index:
                contracts[f"TWO_BASELINE_EXERCISE{n}"][y] = contracts[f"TWO_BASELINE_SCENARIO_E{n}"][y]
            else:
                contracts[f"TWO_BASELINE_EXERCISE{n}"][y] = contracts['TWO_BASELINE_SCENARIO_MV'][y]
        # for n in range(len(contracts)):
        #     if contracts['TWO_BASELINE_SCENARIO_MV'].iloc[n] < 1:
        #         contracts['TWO_BASELINE_SCENARIO_MV'] = contracts['TWO_BASELINE_SCENARIO_MV'] - contracts['OPTION_FEE']
                
    dry_year_options[i] = contracts
    
# for i in (new.uses_of_water_dry[y].index.unique()):
#     dry_year_options[i] = dry_year_options[i].set_index("year")



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
      # plt.figure()
      # #plt.plot(unique_pays['unique'], dum1)
      # plt.xlabel('payment probability')
      # plt.ylabel('Density')
      sns.kdeplot(dum1, color = 'blue')
      unique_pays['dum2_preriskadj'] = st.norm.cdf(st.norm.ppf(dum1))
      dum2 = st.norm.cdf(st.norm.ppf(dum1) + lam)  # risk transformed payout cdf
      unique_pays['dum2'] = dum2
      #plt.plot(unique_pays['unique'], dum2)
      #print(dum2)
      sns.kdeplot(dum2, color ='red')
      dum3 = np.append(dum2[0], np.diff(dum2))  # risk transformed asset pdf
      unique_pays['dum3'] = dum3
      unique_pays['new'] = dum3 * unique_pays['unique']
      #print(dum3)
      #print(dum3)
      # plt.figure()
      # sns.kdeplot(dum3, color ='green')
      # plt.xlim(0,1)
      
      
      prem = (dum3 * unique_pays['unique']).sum()
      #print(prem)
      #sns.kdeplot(prem, color ='blue')
      #sns.kdeplot(prem, color = 'red')
      #print(prem)
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
    
    
    # S1_Exercise = pd.Series(index=range(1950,2013))
    # for i in range(1950,2013):
    #     if i in Wet_Year_Triggers:
    #        S1_Exercise[i] = (S1_Diff/(Wet_Years['value'].loc[i]))
    #     else:
    #        S1_Exercise[i] = 0
    
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

for e in [5,15]:
    ax1 = plt.subplot()
    plt.style.use('default')
    ax1.plot(dry_year_option_timeseries[e], label = f"Dry Year Option with ${e} Exercise Fee", linewidth = 2.5)
    ax1.set_ylim(ymin=0)
    ax1.set_ylim(ymax=26)
    ax1.set_yticklabels(ax1.get_yticklabels(), weight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, weight='bold')
    ax1.text(-.1, .5, 'Cost ($/AF)', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = 'black', weight = 'bold', fontsize = 10)
    ax1.text(.5, -.1, 'Hydrologic Year', ha='center', va='center', transform=ax1.transAxes, color = 'black', weight = 'bold', fontsize = 10)
    legend_prop = {'weight': 'bold'}
    ax1.legend(bbox_to_anchor=(1.75, 1), loc='upper right', borderaxespad=0, frameon = False, prop = legend_prop , framealpha = 0.5)

    


# ## make the dry year portion map ###

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# # plot the same data on both axes
# ax2.plot(dry_year_two_cost, linestyle = 'solid', marker = 'x', markersize = 6, label='Dry Year Option (Exercise Fee = $20)')
# # ax1.plot(S2, linestyle = 'solid', marker = 'd', markersize = 6, label= 'Wet Year Option (at $30/AF)')
# # ax1.plot(Annualized_Perm_Right, linestyle = 'solid', marker = '.', markersize = 6, label= 'Permanent Water Right Purchase')
# # ax1.plot(DryYear, linestyle = 'solid', marker = 'h', markersize = 6, label= 'Dry Year Option')
# # ax2.plot(S1, linestyle = 'solid', marker = 'x', markersize = 6, label='Wet Year Option (MV of Water/AF)')
# # ax2.plot(S2, linestyle = 'solid', marker = 'd', markersize = 6, label= 'Wet Year Option (at $30/AF)')
# # ax2.plot(Annualized_Perm_Right, linestyle = 'solid', marker = '.', markersize = 6, label= 'Permanent Water Right Purchase')
# # ax2.plot(DryYear, linestyle = 'solid', marker = 'h', markersize = 6, label= 'Dry Year Option')

# # zoom-in / limit the view to different portions of the data
# ax1.set_ylim(7070, 7100)  # outliers only
# ax2.set_ylim(0, 100)  # most of the data

# # hide the spines between ax and ax2
# ax1.spines.bottom.set_visible(False)
# ax2.spines.top.set_visible(False)
# #ax1.xaxis.tick_top()
# ax1.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()


# d = .5  # proportion of vertical to horizontal extent of the slanted line
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
# ax1.legend(bbox_to_anchor=(1.5, 1), loc='upper right', borderaxespad=0, frameon = False)
# plt.xlabel('Hydrologic Year')
# plt.ylabel('Cost ($/AF)')

# plt.show()

    

# fig, ax1 = plt.subplots() 

# ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# ax1.plot(dry_year_summed_water_reallocated_by_year.index, dry_year_summed_water_reallocated_by_year , color = 'blue', linewidth = 4) 
# ax1.tick_params(axis ='y', labelcolor = 'blue') 
# ax2 = ax1.twinx() 

# ax2.set_ylabel('Urban-to-Ag Funds Transferred ($)', color = 'lightgreen') 
# ax2.set_ylim([0,dry_year_totals_timeseries[600].max()])
# ax2.plot(dry_year_totals_timeseries[600].index, dry_year_totals_timeseries[600] , color = 'lightgreen', linestyle = 'dashed', linewidth = 2) 
# ax2.tick_params(axis ='y', labelcolor = 'lightgreen') 

    

# #print(dry_year_max_payout_df['payout'].mean())
# print(dry_year_prems)
# print((dry_year_prems + dry_year_max_payout_df['payout']).mean())
# print(dry_year_whole.mean())




# dry_year_max_payout_series = pd.Series(dry_year_max_payout_df[0])
# dry_year_max_payout_list = dry_year_max_payout_series.to_list()

# plt.figure()
# plt.plot(dry_year_max_payout_df)
# plt.xlabel('Hydrologic Year')
# plt.ylabel('Dry Year Payout($)')

# dry_year_summed_payouts = pd.DataFrame()
# dry_year_summed_payouts = dry_year_max_payout_df.T.sum (axis=0)


dry_year_summed_water_reallocated = pd.DataFrame()
dry_year_summed_water_reallocated = dry_year_total_water_reallocated.T.sum (axis=0)
dry_year_summed_water_reallocated = dry_year_summed_water_reallocated.sum()


# dry_year_options = {}
# individual_dry_year_contracts = pd.DataFrame()
# for i in (new.uses_of_water_dry[y].index.unique()):
#     contracts = pd.DataFrame()
#     for y in range(1950,2013):
#         contract = new.uses_of_water_dry[y].loc[new.uses_of_water_dry[y].index== i]
#     individual_dry_year_contracts = pd.concat([contract, contracts])    





# for i in new.irrigation_structure_ids_TWO:
#     dry_year_mv = pd.DataFrame()
#     for y in range(1950,2013):
#         irrigation_set_dry_two = new.uses_of_water_dry[y].loc[new.uses_of_water_dry[y]['StateMod_Structure'] == i]
#         for crop in range(len(irrigation_set_dry_two)):
#             irrigation_set_dry_two_selection = irrigation_set_dry_two.loc[crop]
            
    
# for y in range(1950,2013):
#     dry_year_mv = pd.DataFrame()
#     for i in new.irrigation_structure_ids_TWO:
#         irrigation_set_dry_two = new.uses_of_water_dry.loc[new.uses_of_water_dry['StateMod_Structure'] == i]
#         irrigation_set_dry_two['TWO_BASELINE_SCENARIO_MV'] = 0
#         if y in CBI_S2.index:
#             irrigation_set_wet['TWO_BASELINE_SCENARIO_MV'] = new.uses_of_water_dry['POSSIBLEMUNIUSAGE']*new.uses_of_water_dry['MARGINAL_VALUE_OF_WATER']
#         else:
#             irrigation_set_wet['TWO_BASELINE_SCENARIO_MV'] = 0
#         dry_year_mv = pd.concat([dry_year_mv, irrigation_set_wet])

# dry_year_two_dict = {}

### wet year two-way option ####

exercise_fees = [5, 7.5, 10, 12.5, 15]
added_mv_wet = pd.Series(index = range(1950,2013))
min_mv_dict = {}
for y in range(1950,2013):
    new.TWO_Selection_Wet[y]['year'] = y
    new.TWO_Selection_Wet[y]['TWO_MV'] = 0
    #max_mv = list(new.TWO_Selection[y].loc[new.TWO_Selection[y]['TWO_USAGE'] > 0]['MARGINAL_VALUE_OF_WATER'])[-1]
    if y in CBI_Surplus.index:
        #max_mv = list(new.TWO_Selection[y].loc[new.TWO_Selection[y]['TWO_USAGE'] > 0]['MARGINAL_VALUE_OF_WATER'])[-1]
        #list(new.TWO_Selection[y].loc[new.TWO_Selection[y]['TWO_USAGE'] > 0]['MARGINAL_VALUE_OF_WATER'])[-1]
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
    #contracts['OPTION_FEE'] = max(contracts['POSSIBLEMUNIUSAGE']) * option_fee_per_af
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
        # for n in range(len(contracts)):
        #     if contracts['TWO_BASELINE_SCENARIO_MV'].iloc[n] < 1:
        #         contracts['TWO_BASELINE_SCENARIO_MV'] = contracts['TWO_BASELINE_SCENARIO_MV'] - contracts['OPTION_FEE']
                
    wet_year_options[i] = contracts_wet
    
# for i in (new.uses_of_water_dry[y].index.unique()):
#     dry_year_options[i] = dry_year_options[i].set_index("year")



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
            wet_year_payouts[df_name] = pd.DataFrame(wet_year_options[name][f"TWO_BASELINE_SCENARIO_MVH2O_E{n}"])
            wet_year_payouts[df_name] = wet_year_payouts[df_name].rename(columns={f"TWO_BASELINE_SCENARIO_MVH2O_E{n}": 'payout'})
            
    
    
    
    
    
    
    
    
    
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
for e in [5,15]:
    plt.plot(wet_year_option_timeseries_mv30[e], label = f"Wet Year Option with ${e} Exercise Fee", linewidth = 2.5)
    plt.legend(bbox_to_anchor=(1.75, 1), loc='upper right', borderaxespad=0, frameon = False)
    plt.ylabel('Cost ($/AF)')
    plt.xlabel('Hydrologic Year')
 
for e in [5,15]:
    ax1 = plt.subplot()
    plt.style.use('default')
    ax1.plot(wet_year_option_timeseries_mv30[e], label = f"Wet Year Option with ${e} Exercise Fee", linewidth = 2.5)
    ax1.set_ylim(ymin=0)
    ax1.set_ylim(ymax=26)
    ax1.set_yticklabels(ax1.get_yticklabels(), weight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, weight='bold')
    ax1.text(-.1, .5, 'Cost ($/AF)', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = 'black', weight = 'bold', fontsize = 10)
    ax1.text(.5, -.1, 'Hydrologic Year', ha='center', va='center', transform=ax1.transAxes, color = 'black', weight = 'bold', fontsize = 10)
    legend_prop = {'weight': 'bold'}
    ax1.legend(bbox_to_anchor=(1.75, 1), loc='upper right', borderaxespad=0, frameon = False, prop = legend_prop , framealpha = 0.5)

    
    
#total net benefits to ag   

two_nb_to_ag = pd.Series(index=range(1950,2013))
for y in range(1950,2013):    
    two_nb_to_ag[y] = dry_year_two_cost_sum[y] - wet_year_two_cost_sum[y]


    
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import gridspec

# # Simple data to display in various forms
# x = dry_year_option_timeseries[e].index
# y = dry_year_option_timeseries[e]
# z = wet_year_option_timeseries_mv30[e]

# fig = plt.figure()
# # set height ratios for subplots
# gs = gridspec.GridSpec(2, 1) 

# # the first subplot
# ax0 = plt.subplot(gs[0])
# # log scale for axis Y of the first subplot
# #ax0.set_yscale("log")
# #ax0.set_ylabel('Cost ($/AF)')
# line0, = ax0.plot(x, z, color='blue')
# plt.ylim([0, 50])

# # the second subplot
# # shared axis X
# ax1 = plt.subplot(gs[1], sharex = ax0)
# line1, = ax1.plot(x, y, color='red')
# plt.setp(ax0.get_xticklabels(), visible=False)
# plt.ylim([0, 60])
# # remove last tick label for the second subplot
# yticks = ax1.yaxis.get_major_ticks()
# yticks[-1].label1.set_visible(False)
# fig.text(0.06, 0.5, 'Cost ($/AF)', ha='center', va='center', rotation='vertical')
# fig.text(0.5, 0.04, 'Hydrologic Year', ha='center', va='center')

# # put legend on first subplot
# ax0.legend((line0, line1), ('Ag-to-Urban Payments', 'Urban-to-Ag Payments'), bbox_to_anchor=(1.5, 1), loc='upper right', borderaxespad=0, frameon = True)

# # remove vertical gap between subplots
# plt.subplots_adjust(hspace=.0)
# plt.show()

###################################################################################################
# Import Library

# import numpy as np 
# import matplotlib.pyplot as plt 
  
# # Define Data

# x = dry_year_option_timeseries[25].index
# data_1 = dry_year_option_timeseries[25]
# data_2 = dry_year_summed_water_reallocated_by_year
  
# # Create Plot

# fig, ax1 = plt.subplots() 
  
# ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Cost($/AF)', color = 'green') 
# ax1.plot(x, data_1, color = 'green') 
# ax1.tick_params(axis ='y', labelcolor = 'green') 

  
# # Adding Twin Axes

# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# ax2.plot(x, data_2, color = 'blue') 
# ax2.tick_params(axis ='y', labelcolor = 'blue') 
 
# # Show plot

# plt.show()
###########################################################################################
#### WORKS ####################
# from matplotlib.gridspec import GridSpec
# x = dry_year_option_timeseries[25].index
# data_1 = dry_year_option_timeseries[25]
# data_2 = dry_year_summed_water_reallocated_by_year
# data_3 = wet_year_option_timeseries_mv30[25]
# data_4 = wet_year_summed_water_reallocated_by_year

# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=500, sharex=True)
# fig.suptitle('Two-Way Option (Baseline)', weight = 'bnew' )

# plt.subplot(2,1,1)
# #ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Urban-to-Ag($/AF)', color = 'green') 
# ax1.plot(x, data_1, color = 'green') 
# ax1.tick_params(axis ='y', labelcolor = 'green') 
# plt.ylim([0, 50])
# twin1 = ax1.twinx()
# #twin1.plot(x,data_2,label='curvey1', markersize=0.1, linewidth=0.2, color = 'blue')
# #twin1.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# twin1.plot(x, data_2, color = 'blue') 
# twin1.set_ylabel('Ag-to-Urban(AF)', color = 'blue') 
# twin1.tick_params(axis ='y', labelcolor = 'blue') 

# # p3, = twin1.plot(x,data_2, label='curvey2', markersize=0.1, linewidth=0.2, color = 'blue')
 
# # ax1.set_xlabel('Hydrologic Year')
# # twin1.set_ylabel('Cost ($/AF)')
# # twin1.set_ylabel('Water Re-allocated (AF)')
# fig.add_gridspec(hspace=0)
# plt.subplot(2,1,2)
# ax2.set_xlabel('Hydrologic Year') 
# ax2.set_ylabel('Ag-to-Urban($/AF)', color = 'green') 
# ax2.plot(x, data_3, color = 'green') 
# ax2.tick_params(axis ='y', labelcolor = 'green') 
# plt.ylim([0, 50])
# twin2 = ax2.twinx()
# #twin1.plot(x,data_4,label='curvey1', markersize=0.1, linewidth=0.2, color = 'blue')
# #twin2.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# twin2.plot(x, data_4, color = 'blue')
# twin2.set_ylabel('Urban-to-Ag(AF)', color = 'blue')  
# twin2.tick_params(axis ='y', labelcolor = 'blue') 
# fig.add_gridspec(hspace=0)
# #fig.text(1.0, .5, 'Water Re-allocated (AF)', ha='center', va='center', rotation='vertical', color = 'blue')
# plt.show()
# plt.close()
########## BENEFITS TO MUNI ANALYSIS###############################

wet_years = [1962,1970,1971,1972,1974,1980,1984,1985,1986,1988,1996,1997,2011]


os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet_03_29')




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

#os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet_03_29')


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

# munis_of_interest = ['Loveland', 'Longmont', 'Louisville', 'Laffayette', 'Boulder']
# dfs = {}
# for m in munis_of_interest:
#     dfs[m] = f"{m}_Total_demand_Sum"['demand']
# # Filter the dataframes to only include the years in wet_years
# wet_years = [1962, 1970, 1971, 1972, 1974, 1980, 1984, 1985, 1986, 1988, 1996, 1997, 2011]
# for m in munis_of_interest:
#     dfs[m] = dfs[m].loc[wet_years]

# # Calculate the average demand for each municipality
# demand_avg = {}
# for m in munis_of_interest:
#     demand_avg[m] = dfs[m]['Total_demand_sum'].mean()

# print(demand_avg)

Loveland_avg = Loveland_Total_demand_Sum.loc[wet_years].mean()[0]
Longmont_avg = Longmont_Total_demand_Sum.loc[wet_years].mean()[0]
Louisville_avg = Louisville_Total_demand_Sum.loc[wet_years].mean()[0]
Laffayette_avg = Laffayette_Total_demand_Sum.loc[wet_years].mean()[0]
Boulder_avg = Boulder_Total_demand_Sum.loc[wet_years].mean()[0]


elasticity = (-0.3)


Loveland_alpha = {}
Longmont_alpha = {}
Louisville_alpha = {}
Laffayette_alpha = {}
Boulder_alpha = {}

#boulder block 1 price in 2022
boulder_price_per_1000 = 4.22

#loveland single family home 2023
loveland_price_per_1000 = 3.95

#longmont up to 5000 gal 2023
longmont_price_per_1000 = 4

#laffayette up to 3999 gal
laffayette_price_per_1000 = 3.55

## louisville additional 1000 gal price after 5000 gal 2021
louisville_price_per_1000 = 5.58

af_conversion_1000_gal = 0.0030688832459704

boulder_price_per_af = boulder_price_per_1000 / af_conversion_1000_gal
loveland_price_per_af = loveland_price_per_1000 / af_conversion_1000_gal
longmont_price_per_af = longmont_price_per_1000 / af_conversion_1000_gal
laffayette_price_per_af = laffayette_price_per_1000 / af_conversion_1000_gal
louisville_price_per_af = louisville_price_per_1000 / af_conversion_1000_gal

Loveland_alpha = Loveland_avg/(loveland_price_per_af)**elasticity
Longmont_alpha = Longmont_avg/(longmont_price_per_af)**elasticity
Louisville_alpha = Louisville_avg/(louisville_price_per_af)**elasticity
Laffayette_alpha = Laffayette_avg/(laffayette_price_per_af)**elasticity
Boulder_alpha = Boulder_avg/(boulder_price_per_af)**elasticity

# Loveland_MNB = pd.DataFrame()
# for y in range(1950,2013):
#     Loveland_MNB[y]= (Loveland_avg-Loveland_Total_shortage_sum[y]/Loveland_alpha)**(1/elasticity) - 900


os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod')

plt.figure()
#plt.plot(EstesPark_Total_demand_Sum['demand'],color='purple', label='Estes Park')
plt.plot(Loveland_Total_demand_Sum['demand'],color='green', label='Loveland')
#plt.plot(Greeley_Total_demand_Sum['demand'],color='yellow', label='Greeley')
plt.plot(Longmont_Total_demand_Sum['demand'],color='red', label='Longmont')
plt.plot(Louisville_Total_demand_Sum['demand'],color='orange', label='Louisville')
plt.plot(Laffayette_Total_demand_Sum['demand'],color='brown', label='Laffayette')
plt.plot(Boulder_Total_demand_Sum['demand'],color='blue', label='Boulder')
plt.xlabel('Hydrologic Year')
plt.ylabel('Demand (AF)')
plt.ylim([0,25000])
plt.title('Northern Water Adapted Municipal Demands')
plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right', borderaxespad=0)

Thornton_Indoor_shortage = ThorntonIndoor['shortage']
Thornton_Outdoor_shortage = ThorntonOutdoor['shortage']
Westminster_Indoor_shortage = WestminsterIndoor['shortage']
Westminster_Outdoor_shortage = WestminsterOutdoor['shortage']
Laffayette_Indoor_shortage = LaffayetteIndoor['shortage']
Laffayette_Outdoor_shortage = LaffayetteOutdoor['shortage']
Louisville_Indoor_shortage = LouisvilleIndoor['shortage']
Louisville_Outdoor_shortage = LouisvilleOutdoor['shortage']
Boulder_Outdoor_shortage = BoulderIndoor['shortage']
Boulder_Indoor_shortage = BoulderOutdoor['shortage']
Longmont_Indoor_shortage = LongmontIndoor['shortage']
Longmont_Outdoor_shortage = LongmontOutdoor['shortage']
Loveland_Indoor_shortage = LovelandIndoor['shortage']
Loveland_Outdoor_shortage = LovelandOutdoor['shortage']
Arvada_Indoor_shortage = ArvadaIndoor['shortage']
Arvada_Outdoor_shortage = ArvadaOutdoor['shortage']
ConMut_Indoor_shortage = ConMutIndoor['shortage']
ConMut_Outdoor_shortage = ConMutOutdoor['shortage']
Golden_Indoor_shortage = GoldenIndoor['shortage']
Golden_Outdoor_shortage = GoldenOutdoor['shortage']
Denver_Indoor_shortage = DenverIndoor['shortage']
Denver_Outdoor_shortage = DenverOutdoor['shortage']
Englewood_Indoor_shortage = EnglewoodIndoor['shortage']
Englewood_Outdoor_shortage = EnglewoodOutdoor['shortage']
Aurora_Indoor_shortage = AuroraIndoor['shortage']
Aurora_Outdoor_shortage = AuroraOutdoor['shortage']
EstesPark_Indoor_shortage = EstesParkIndoor['shortage']
EstesPark_Outdoor_shortage = EstesParkOutdoor['shortage']


Thornton_Total_shortage = Thornton_Indoor_shortage + Thornton_Outdoor_shortage
Westminster_Total_shortage = Westminster_Indoor_shortage + Westminster_Outdoor_shortage
Laffayette_Total_shortage = Laffayette_Indoor_shortage + Laffayette_Outdoor_shortage
Louisville_Total_shortage = Louisville_Indoor_shortage + Louisville_Outdoor_shortage
Boulder_Total_shortage = Boulder_Indoor_shortage + Boulder_Outdoor_shortage
Longmont_Total_shortage = Longmont_Indoor_shortage + Longmont_Outdoor_shortage
Loveland_Total_shortage = Loveland_Indoor_shortage + Loveland_Outdoor_shortage
Arvada_Total_shortage = Arvada_Indoor_shortage + Arvada_Outdoor_shortage
ConMut_Total_shortage = ConMut_Indoor_shortage + ConMut_Outdoor_shortage
Golden_Total_shortage = Golden_Indoor_shortage + Golden_Outdoor_shortage
Denver_Total_shortage = Denver_Indoor_shortage + Denver_Outdoor_shortage
Englewood_Total_shortage = Englewood_Indoor_shortage + Englewood_Outdoor_shortage
Aurora_Total_shortage = Aurora_Indoor_shortage + Aurora_Outdoor_shortage
EstesPark_Total_shortage = EstesPark_Indoor_shortage + EstesPark_Outdoor_shortage

Thornton_Total_shortage_Sum = pd.DataFrame().assign(Year=ThorntonIndoor['year'],shortage=Thornton_Total_shortage)
Thornton_Total_shortage_Sum = Thornton_Total_shortage_Sum.groupby('Year').sum()

Westminster_Total_shortage_Sum = pd.DataFrame().assign(Year=WestminsterIndoor['year'],shortage=Westminster_Total_shortage)
Westminster_Total_shortage_Sum = Westminster_Total_shortage_Sum.groupby('Year').sum()

Laffayette_Total_shortage_Sum = pd.DataFrame().assign(Year=LaffayetteIndoor['year'],shortage=Laffayette_Total_shortage)
Laffayette_Total_shortage_Sum = Laffayette_Total_shortage_Sum.groupby('Year').sum()

Louisville_Total_shortage_Sum = pd.DataFrame().assign(Year=LouisvilleIndoor['year'],shortage=Louisville_Total_shortage)
Louisville_Total_shortage_Sum = Louisville_Total_shortage_Sum.groupby('Year').sum()

Boulder_Total_shortage_Sum = pd.DataFrame().assign(Year=BoulderIndoor['year'],shortage=Boulder_Total_shortage)
Boulder_Total_shortage_Sum = Boulder_Total_shortage_Sum.groupby('Year').sum()

Longmont_Total_shortage_Sum = pd.DataFrame().assign(Year=LongmontIndoor['year'],shortage=Longmont_Total_shortage)
Longmont_Total_shortage_Sum = Longmont_Total_shortage_Sum.groupby('Year').sum()

Loveland_Total_shortage_Sum = pd.DataFrame().assign(Year=LovelandIndoor['year'],shortage=Loveland_Total_shortage)
Loveland_Total_shortage_Sum = Loveland_Total_shortage_Sum.groupby('Year').sum()

Arvada_Total_shortage_Sum = pd.DataFrame().assign(Year=ArvadaIndoor['year'],shortage=Arvada_Total_shortage)
Arvada_Total_shortage_Sum = Arvada_Total_shortage_Sum.groupby('Year').sum()

ConMut_Total_shortage_Sum = pd.DataFrame().assign(Year=ConMutIndoor['year'],shortage=ConMut_Total_shortage)
ConMut_Total_shortage_Sum = ConMut_Total_shortage_Sum.groupby('Year').sum()

Golden_Total_shortage_Sum = pd.DataFrame().assign(Year=GoldenIndoor['year'],shortage=Golden_Total_shortage)
Golden_Total_shortage_Sum = Golden_Total_shortage_Sum.groupby('Year').sum()

Denver_Total_shortage_Sum = pd.DataFrame().assign(Year=DenverIndoor['year'],shortage=Denver_Total_shortage)
Denver_Total_shortage_Sum = Denver_Total_shortage_Sum.groupby('Year').sum()

Englewood_Total_shortage_Sum = pd.DataFrame().assign(Year=EnglewoodIndoor['year'],shortage=Englewood_Total_shortage)
Englewood_Total_shortage_Sum = Englewood_Total_shortage_Sum.groupby('Year').sum()

Aurora_Total_shortage_Sum = pd.DataFrame().assign(Year=AuroraIndoor['year'],shortage=Aurora_Total_shortage)
Aurora_Total_shortage_Sum = Aurora_Total_shortage_Sum.groupby('Year').sum()

EstesPark_Total_shortage_Sum = pd.DataFrame().assign(Year=EstesParkIndoor['year'],shortage=EstesPark_Total_shortage)
EstesPark_Total_shortage_Sum = EstesPark_Total_shortage_Sum.groupby('Year').sum()



def cobb_douglas_MNB(Q, alpha, elasticity, price):
    MNB = (Q / (alpha)) ** (1 / elasticity) - price
    return MNB



loveland_mnb = {}
loveland_err = {}
for y in CBI_S2.index:
    loveland_mnb[y], loveland_err[y] = quad(cobb_douglas_MNB, Loveland_avg-Loveland_Total_shortage_Sum['shortage'][y], Loveland_avg, args=(Loveland_alpha, elasticity, loveland_price_per_af))

boulder_mnb = {}
boulder_err = {}
for y in CBI_S2.index:
    boulder_mnb[y], boulder_err[y] = quad(cobb_douglas_MNB, Boulder_avg-Boulder_Total_shortage_Sum['shortage'][y], Boulder_avg, args=(Boulder_alpha, elasticity, boulder_price_per_af))

longmont_mnb = {}
longmont_err = {}
for y in CBI_S2.index:
    longmont_mnb[y], longmont_err[y] = quad(cobb_douglas_MNB, Longmont_avg-Longmont_Total_shortage_Sum['shortage'][y], Longmont_avg, args=(Longmont_alpha, elasticity, longmont_price_per_af))

louisville_mnb = {}
louisville_err = {}
for y in CBI_S2.index:
    louisville_mnb[y], louisville_err[y] = quad(cobb_douglas_MNB, Louisville_avg-Louisville_Total_shortage_Sum['shortage'][y], Louisville_avg, args=(Louisville_alpha, elasticity, louisville_price_per_af))

laffayette_mnb = {}
laffayette_err = {}
for y in CBI_S2.index:
    laffayette_mnb[y], laffayette_err[y] = quad(cobb_douglas_MNB, Laffayette_avg-Laffayette_Total_shortage_Sum['shortage'][y], Laffayette_avg, args=(Laffayette_alpha, elasticity, laffayette_price_per_af))


total_mnb = {}
for y in CBI_S2.index:
    total_mnb[y] = abs(loveland_mnb[y] + boulder_mnb[y] + longmont_mnb[y] + louisville_mnb[y] + laffayette_mnb[y]) - dry_year_two_cost_sum[y]

sum(total_mnb.values())

########### ANALYSIS SELECTED EXERCISE FEE ################################################

dry_year_analysis_exercise_fee = 15
wet_year_analysis_exercise_fee = 15

total_muni_nb = pd.Series(index = range(1950,2013))
total_muni_nb.update(total_mnb)


# ag_to_muni_exercise = pd.Series(index = range(1950,2013))
# ag_to_muni_exercise = wet_year_summed_water_reallocated_by_year*analysis_exercise_fee

# total_muni_nb = pd.concat([total_muni_nb, ag_to_muni_exercise], axis=1)
# total_muni_nb = total_muni_nb.sum(axis=1)

########### FINAL FIGURE TEST ###################################################
from matplotlib.gridspec import GridSpec
x = dry_year_option_timeseries[dry_year_analysis_exercise_fee].index
data_1 = dry_year_option_timeseries[dry_year_analysis_exercise_fee]
data_2 = dry_year_summed_water_reallocated_by_year/1000
data_3 = wet_year_option_timeseries_mv30[wet_year_analysis_exercise_fee]
data_4 = wet_year_summed_water_reallocated_by_year/1000

# Set the size of the figure
plt.figure(figsize=(10, 15))

ax1 = plt.subplot(3, 1, 1)
plt.bar(x, data_4, color = 'blue') 
plt.bar(x, -(data_2), color = 'red', label = 'Dry Year')
plt.ylim([-25,25])

ticks =  ax1.get_yticks()
for label in ax1.get_yticklabels()[0:3]:
    label.set_color("red")
for label in ax1.get_yticklabels()[4:6]:
    label.set_color("blue")
# set labels to absolute values and with integer representation
ax1.set_yticklabels([int(abs(tick)) for tick in ticks], fontsize=20, weight = 'bold')

ax1.axhline(y=0, color='black')
twin1 = plt.twinx()
twin1.plot(x, CBI_Yearly, color = 'gray') 
twin1.set_ylabel('CBI', color = 'gray', weight = 'bold', fontsize=20) 
twin1.tick_params(labelcolor = 'gray', labelsize=20)
ticks =  twin1.get_yticks()
twin1.set_yticklabels([int(abs(tick)) for tick in ticks], fontsize=20, weight = 'bold')
ax1.text(-.1, 0.75, 'To Ag', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = 'blue', weight = 'bold', fontsize=20)
ax1.text(-.1, 0.25, 'To Urban', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = 'red', weight = 'bold', fontsize=20)
ax1.text(-.15, 0.50, 'Water Re-allocated (kAF)', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = 'black', weight = 'bold', fontsize=20)


ax2 = plt.subplot(3, 1, 2, sharex = ax1)

y1 = two_nb_to_ag/1000000
y2 = added_mv_wet/1000000

ax2.bar(x, y2, bottom = y1, color = 'orange', label = 'Increased Ag Productivity')
ax2.bar(x, y1, color = 'green', label = 'TWO Revenue/Cost')
plt.ylim([-0.7,2])
ax2.text(-.15, .5, 'Ag Gains/Losses ($M)', ha='center', va='center', transform=ax2.transAxes, rotation='vertical', color = 'black', weight = 'bold', fontsize=20)
for label in ax2.get_yticklabels()[0:7]:
    label.set_color("black")
    label.set_fontsize(20)
    label.set_fontweight('bold')

legend = plt.legend(fontsize=15)
for text in legend.get_texts():
    text.set_weight('bold')

ax3 = plt.subplot(3, 1, 3, sharex = ax1)
data_5 = dry_year_two_cost_sum/1000000
data_6 = wet_year_two_cost_sum/1000000
plt.bar(x, data_6, color = 'blue') 
plt.bar(x, -(data_5), color = 'red')
plt.ylim([-.7,.7])

ticks =  ax3.get_yticks()
for label in ax3.get_yticklabels()[0:4]:
    label.set_color("red")
for label in ax3.get_yticklabels()[5:9]:
    label.set_color("blue")
# set labels to absolute values and with integer representation
ax3.set_yticklabels([f'{abs(tick):.1f}' for tick in ticks], fontsize=20, weight = 'bold')

ax3.axhline(y=0, color='black')
#plt.ylabel('TWO Cost ($)')
ax3.text(-.11, .75, 'Ag', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = 'blue', weight = 'bold', fontsize = 20)
ax3.text(-.11, .25, 'Urban', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = 'red', weight = 'bold', fontsize = 20)
ax3.text(-.15, 0.45, 'Two-Way Option Cost ($M)', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = 'black', weight = 'bold', fontsize=20)
plt.subplots_adjust(hspace=0.4)


ticks =  ax1.get_xticks()
ax1.set_xticklabels([int(abs(tick)) for tick in ticks], fontsize=15, weight = 'bold')
ax2.set_xticklabels([int(abs(tick)) for tick in ticks], fontsize=15, weight = 'bold')
ax3.set_xticklabels([int(abs(tick)) for tick in ticks], fontsize=15, weight = 'bold')
plt.show()

sum_ag_nb = two_nb_to_ag.sum() + added_mv_wet.sum()
sum_ag_nb = sum_ag_nb.sum()
## FINAL TWO COST ############

ag_yearly_option_fee_payment = wet_year_summed_water_reallocated_by_year.max()*wet_year_option_fee*63
ag_exercise_payment_sum = wet_year_summed_water_reallocated_by_year.sum()*wet_year_selected_exercise_fee
ag_two_cost_per_af = (ag_yearly_option_fee_payment + ag_exercise_payment_sum)/wet_year_summed_water_reallocated_by_year.sum()

urban_yearly_option_fee_payment = dry_year_summed_water_reallocated_by_year.max()*dry_year_option_fee*63
urban_exercise_payment_sum = dry_year_summed_water_reallocated_by_year.sum()*dry_year_selected_exercise_fee
urban_two_cost_per_af = (urban_yearly_option_fee_payment + urban_exercise_payment_sum)/dry_year_summed_water_reallocated_by_year.sum()



# ########### FINAL FIGURE ###################################################
# from matplotlib.gridspec import GridSpec
# x = dry_year_option_timeseries[dry_year_analysis_exercise_fee].index
# data_1 = dry_year_option_timeseries[dry_year_analysis_exercise_fee]
# data_2 = dry_year_summed_water_reallocated_by_year
# data_3 = wet_year_option_timeseries_mv30[wet_year_analysis_exercise_fee]
# data_4 = wet_year_summed_water_reallocated_by_year

# # Set the size of the figure
# plt.figure(figsize=(5, 10))

# ax1 = plt.subplot(3, 1, 1)
# plt.bar(x, data_4, color = 'blue') 
# plt.bar(x, -(data_2), color = 'red', label = 'Dry Year')
# plt.ylim([-25000,25000])

# ticks =  ax1.get_yticks()
# for label in ax1.get_yticklabels()[0:3]:
#     label.set_color("red")
# for label in ax1.get_yticklabels()[4:6]:
#     label.set_color("blue")
# # set labels to absolute values and with integer representation
# ax1.set_yticklabels([int(abs(tick)) for tick in ticks])

# ax1.axhline(y=0, color='black')
# twin1 = plt.twinx()
# twin1.plot(x, CBI_Yearly, color = 'gray') 
# twin1.set_ylabel('CBI', color = 'gray') 
# twin1.tick_params(axis ='y', labelcolor = 'gray') 
# ax1.text(-.2, .85, 'Urban-to-Ag (AF)', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = 'blue')
# ax1.text(-.2, .25, 'Ag-to-Urban (AF)', ha='center', va='center', transform=ax1.transAxes, rotation='vertical', color = 'red')
# # fig.text(-0.1, 9, 'Urban-to-Ag (AF)', ha='center', va='center', rotation='vertical', color = 'blue')
# # fig.text(0, 3.3, 'Ag-to-Urban (AF)', ha='center', va='center', rotation='vertical', color = 'red')

# ax2 = plt.subplot(3, 1, 2, sharex = ax1)

# ag_mv = two_nb_to_ag
# plt.bar(x, two_nb_to_ag, color = 'orange')
# plt.bar(x, -(total_muni_nb), color = 'green', )
# plt.ylim([-20000000,20000000])
# #plt.xlabel('Hydrologic Year')
# #plt.ylabel('Ag Net Benefits ($)')
# ticks =  ax2.get_yticks()
# for label in ax2.get_yticklabels()[0:4]:
#     label.set_color("green")
# for label in ax2.get_yticklabels()[5:8]:
#     label.set_color("orange")
# # set labels to absolute values and with integer representation
# ax2.set_yticklabels([int(abs(tick)) for tick in ticks])

# ax2.axhline(y=0, color='black')
# ax2.text(-.2, .80, 'Ag NB ($)', ha='center', va='center', transform=ax2.transAxes, rotation='vertical', color = 'orange')
# ax2.text(-.2, .25, 'Urban NB ($)', ha='center', va='center', transform=ax2.transAxes, rotation='vertical', color = 'green')


# ax3 = plt.subplot(3, 1, 3, sharex = ax1)
# data_5 = dry_year_two_cost_sum
# data_6 = wet_year_two_cost_sum
# plt.bar(x, data_6, color = 'blue') 
# plt.bar(x, -(data_5), color = 'red')
# plt.ylim([-4000000,4000000])

# ticks =  ax3.get_yticks()
# for label in ax3.get_yticklabels()[0:3]:
#     label.set_color("red")
# for label in ax3.get_yticklabels()[4:7]:
#     label.set_color("blue")
# # set labels to absolute values and with integer representation
# ax3.set_yticklabels([int(abs(tick)) for tick in ticks])

# ax3.axhline(y=0, color='black')
# #plt.ylabel('TWO Cost ($)')
# ax3.text(-.2, .85, 'Ag TWO Cost ($)', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = 'blue')
# ax3.text(-.2, .20, 'Urban TWO Cost ($)', ha='center', va='center', transform=ax3.transAxes, rotation='vertical', color = 'red')

# plt.subplots_adjust(hspace=0.2)

# plt.show()












# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=500, sharex=True)
# fig.suptitle('Two-Way Option (Scenario 2)', weight = 'bnew' )

# plt.subplot(2,1,1)
# #ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Urban-to-Ag (AF)', color = 'blue') 
# ax1.bar(x, data_4, color = 'blue') 
# #ax1.plot(x, data_3, color = 'green')
# ax1.tick_params(axis ='y', labelcolor = 'green') 
# #plt.ylim([0, 40])
# twin1 = ax1.twinx()
# #twin1.plot(x,data_2,label='curvey1', markersize=0.1, linewidth=0.2, color = 'blue')
# #twin1.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# twin1.plot(x, CBI_Yearly, color = 'green') 
# twin1.set_ylabel('CBI', color = 'green') 
# twin1.tick_params(axis ='y', labelcolor = 'green') 
# #plt.ylim([0, 40])

# # p3, = twin1.plot(x,data_2, label='curvey2', markersize=0.1, linewidth=0.2, color = 'blue')
 
# # ax1.set_xlabel('Hydrologic Year')
# # twin1.set_ylabel('Cost ($/AF)')
# # twin1.set_ylabel('Water Re-allocated (AF)')
# #fig.add_gridspec(hspace=0)
# plt.subplot(2,1,2)
# ax2.set_xlabel('Hydrologic Year') 
# ax2.set_ylabel('Ag-to-Urban(AF)', color = 'red') 
# ax2.bar(x, -(data_2), color = 'red', label = 'Dry Year') 
# #ax2.plot(x, data_4, color = 'blue', label = 'Wet Year') 
# ax2.tick_params(axis ='y', labelcolor = 'red') 
# #plt.ylim([0, 50])
# # #twin1.plot(x,data_4,label='curvey1', markersize=0.1, linewidth=0.2, color = 'blue')
# # #twin2.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# # twin2.plot(x, CBI_Yearly, color = 'red')
# # twin2.set_ylabel('CBI', color = 'red')  
# ticks =  ax2.get_yticks()

# # set labels to absolute values and with integer representation
# ax2.set_yticklabels([int(abs(tick)) for tick in ticks])

# ax2.tick_params(axis ='y', labelcolor = 'red') 
# #twin2.tick_params(axis ='y', labelcolor = 'green') 
# #ax2.legend(loc = 'upper left')
# fig.add_gridspec(hspace=0)
# #fig.text(1.0, .5, 'Water Re-allocated (AF)', ha='center', va='center', rotation='vertical', color = 'blue')
# plt.show()
# plt.close()


# from matplotlib.gridspec import GridSpec
# x = dry_year_option_timeseries[25].index
# data_1 = dry_year_option_timeseries[25]
# data_2 = dry_year_summed_water_reallocated_by_year
# data_3 = wet_year_option_timeseries_mv30[25]
# data_4 = wet_year_summed_water_reallocated_by_year


# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=500, sharex=True)
# fig.suptitle('Two-Way Option (Baseline)', weight = 'bnew' )

# plt.subplot(2,1,1)
# #ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Urban-to-Ag (Wet)', color = 'blue') 
# ax1.bar(x, data_4, color = 'blue') 
# #ax1.plot(x, data_3, color = 'green')
# ax1.tick_params(axis ='y', labelcolor = 'blue') 
# #plt.ylim([0, 40])
# twin1 = ax1.twinx()
# twin1.plot(x,CBI_Yearly, markersize=1, linewidth=0.8, color = 'green')
# # #twin1.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# # twin1.plot(x, data_3, color = 'green') 
# # twin1.set_ylabel('Ag-to-Urban ($/AF)', color = 'green') 
# # twin1.tick_params(axis ='y', labelcolor = 'green') 
# # plt.ylim([0, 40])

# # p3, = twin1.plot(x,data_2, label='curvey2', markersize=0.1, linewidth=0.2, color = 'blue')
 
# # ax1.set_xlabel('Hydrologic Year')
# # twin1.set_ylabel('Cost ($/AF)')
# # twin1.set_ylabel('Water Re-allocated (AF)')
# fig.add_gridspec(hspace=0)
# plt.subplot(2,1,2)
# ax2.set_xlabel('Hydrologic Year') 
# ax2.set_ylabel('Ag-to-Urban(Dry)', color = 'red') 
# ax2.bar(x, -(data_2), color = 'red', label = 'Dry Year') 
# #ax2.plot(x, data_4, color = 'blue', label = 'Wet Year') 
# ticks =  ax2.get_yticks()

# # set labels to absolute values and with integer representation
# ax2.set_yticklabels([int(abs(tick)) for tick in ticks])

# ax2.tick_params(axis ='y', labelcolor = 'red') 

# plt.close()

plt.figure()




# ag_mv = pd.concat([added_mv, added_mv_wet])
# ag_mv = ag_mv.groupby(ag_mv.index).sum()
# plt.figure()
# plt.plot(ag_mv)
# plt.xlabel('Hydrologic Year')
# plt.ylabel('Added Benefits ($)')














# wet_year_summed_payouts_30 = pd.DataFrame()
# wet_year_summed_payouts_30 = wet_year_max_payout_df_30.T.sum (axis=0)

# wet_year_summed_payouts_mvh2o = pd.DataFrame()
# wet_year_summed_payouts_mvh2o = wet_year_max_payout_df_mvh2o.T.sum (axis=0)

# wet_year_summed_water_reallocated = pd.DataFrame()
# wet_year_summed_water_reallocated = wet_year_total_water_reallocated.T.sum (axis=0)

# baseline_payouts = pd.concat([wet_year_summed_payouts_30, dry_year_summed_payouts], axis =1)
# baseline_payouts['payouts'] = baseline_payouts[0] +baseline_payouts[1]

# baseline_reallocation = pd.concat([wet_year_summed_water_reallocated, dry_year_summed_water_reallocated], axis =1)
# baseline_reallocation['af_reallocated'] = baseline_reallocation[0] +baseline_reallocation[1]


# fig, ax1 = plt.subplots() 

# plt.title('Wet Years - TWO (Scenario 1)')   
# ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Value of Water Re-allocated ($)', color = 'lightgreen') 
# ax1.plot(baseline_reallocation.index, baseline_reallocation['af_reallocated'], color = 'blue', linewidth = 4) 
# ax1.tick_params(axis ='y', labelcolor = 'lightgreen') 
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# ax2.plot(baseline_payouts.index, baseline_payouts['payouts'] , color = 'lightgreen', linestyle = 'dashed', linewidth = 2) 
# ax2.tick_params(axis ='y', labelcolor = 'blue') 


# fig, ax1 = plt.subplots() 

# #plt.title('Dry Year - TWO')   
# ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# ax1.plot(baseline_reallocation.index, baseline_reallocation['af_reallocated'], color = 'blue', linewidth = 4) 
# ax1.tick_params(axis ='y', labelcolor = 'blue') 
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Value of Re-allocated Water ($)', color = 'lightgreen') 
# ax2.plot(baseline_payouts.index, baseline_payouts['payouts'] , color = 'lightgreen', linestyle = 'dashed', linewidth = 2) 
# ax2.tick_params(axis ='y', labelcolor = 'lightgreen') 


# two_dry_total_pricing = pd.DataFrame.from_dict(dry_year_prems_total, orient='index')
# print(two_dry_total_pricing.sum (axis=0))

# two_dry_total_water_reallocated = dry_year_summed_water_reallocated.sum(axis=0)

# print(two_dry_total_pricing.sum(axis=0)/two_dry_total_water_reallocated)

# two_wet_total_pricing = pd.DataFrame.from_dict(wet_year_prems_total, orient='index')
# print(two_wet_total_pricing.sum (axis=0))

# two_wet_total_water_reallocated = wet_year_summed_water_reallocated.sum(axis=0)

# print(two_wet_total_pricing.sum(axis=0)/two_wet_total_water_reallocated)





# for y in range(1950,2013):
#     dry_year_two_dict[y] = new.uses_of_water_dry[y].sort_values(by=['MARGINAL_VALUE_OF_WATER'], ascending=True)

# for y in range(1950,2013):
#     dry_year_two = pd.DataFrame()
#     irrigation_set_dry_two = dry_year_two_dict[y]
#     irrigation_set_dry_two['TWO_USAGE'] = 0
#     if CBI_Yearly.loc[y] <= 675:
#     #if muni.Northern_Water_Muni_Shortages['Shortage'].loc[y] > 1100:
#         water_avail_dry_year = muni.Northern_Water_Muni_Shortages['Shortage'].loc[y]
#     else:
#         water_avail_dry_year = 0
#     for crop in range(len(irrigation_set_dry_two)):
#         if water_avail_dry_year > irrigation_set_dry_two['AVAILABLE_TO_MUNI'].iloc[crop]:
#             use = irrigation_set_dry_two['AVAILABLE_TO_MUNI'].iloc[crop]
#             print(use)
#         else:
#             use = max(water_avail_dry_year, 0)
#             print(use)
#         irrigation_set_dry_two['TWO_USAGE'].iloc[crop] = use   
#         water_avail_dry_year -= irrigation_set_dry_two['AVAILABLE_TO_MUNI'].iloc[crop]
#     dry_year_two = pd.concat([dry_year_two, irrigation_set_dry_two])

#     dry_year_two['mv_h2o_sum'] = dry_year_two['MARGINAL_VALUE_OF_WATER']*dry_year_two['TWO_USAGE']
        
#     dry_year_two_dict[y] = dry_year_two


# uses_of_water_wet = {}
# #structure_revenue_wet = {}
# #year_df = pd.DataFrame()
# for y in range(1950,2013):
#     year_df = pd.DataFrame()
#     for i in irrigation_structure_ids_TWO:
#         irrigation_set_wet = map_df_aggregated_crops_sorted.loc[map_df_aggregated_crops_sorted['StateMod_Structure'] == i]
#         irrigation_set_wet['AG_DEMAND'] = 0
#         #use_of_water = pd.DataFrame()
#         water_avail_wet_year = Historical_Irrigation_Shortage_Sums[i][y]
#         for crop in reversed(range(len(irrigation_set_wet))):
#             if water_avail_wet_year > irrigation_set_wet['DELIVERED_TO_STRUCTURE'].iloc[crop]:
#                 use = irrigation_set_wet['DELIVERED_TO_STRUCTURE'].iloc[crop]
#             else:
#                 use = max(water_avail_wet_year, 0)
#             irrigation_set_wet['AG_DEMAND'].iloc[crop] = use
                 
                
#             water_avail_wet_year -= irrigation_set_wet['DELIVERED_TO_STRUCTURE'].iloc[crop]
#         #added irrigator rev here is = to the amount of revenue generated by fulfilling the shortage. since MNB is calculated
#         #at completely fulfilled 2010 irrigator plot conditions, we subtract the shortage amount to get realized rev in any year
#         # irrigation_set_wet['added_irr_rev'] = irrigation_set_wet['USAGE']*irrigation_set_wet['MARGINAL_VALUE_OF_WATER']
#         # if y in Wet_Year_Triggers:
#         #     irrigation_set_wet['rev'] = irrigation_set_wet['MNB'].sum() - irrigation_set_wet['added_irr_rev'].sum()
#         #     irrigation_set_wet['revTWO'] = irrigation_set_wet['MNB'].sum() + irrigation_set_wet['added_irr_rev'].sum()
#         # else:
#         #     irrigation_set_wet['rev'] = irrigation_set_wet['MNB'].sum() - irrigation_set_wet['added_irr_rev'].sum()
#         #     irrigation_set_wet['revTWO'] = irrigation_set_wet['rev']
#         year_df = pd.concat([year_df, irrigation_set_wet])
                      
#     uses_of_water_wet[y] = year_df    



# dry_year_max_payout = {}  
# for i in new.irrigation_structure_ids_TWO: 
#     two_dry = pd.DataFrame()
#     for y in range(1950,2013):
#         irrigation_set_dry_max = dry_year_two_dict[y].loc[dry_year_two_dict[y]['StateMod_Structure'] == i]
#         # dry_year_max_payout[i].loc[y] = irrigation_set_dry_max['mv_h2o_sum'].sum()
    

# dry_year_max_payout_df = pd.DataFrame.from_dict(dry_year_max_payout, orient ='index')
# dry_year_max_payout_df['payout'] = dry_year_max_payout_df[0]
# dry_year_max_payout_series = pd.Series(dry_year_max_payout_df[0])
# dry_year_max_payout_list = dry_year_max_payout_series.to_list()

# plt.figure()
# plt.plot(dry_year_max_payout_df)
# plt.xlabel('Hydrologic Year')
# plt.ylabel('Dry Year Payout($)')


# fig, ax1 = plt.subplots() 

# #plt.title('Dry Year - TWO')   
# ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Value of Re-allocated Water ($)', color = 'lightgreen') 
# ax1.plot(dry_year_max_payout_df.index, dry_year_max_payout_df , color = 'blue', linewidth = 4) 
# ax1.tick_params(axis ='y', labelcolor = 'lightgreen') 
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# ax2.plot(Dry_Years.index, Dry_Years , color = 'lightgreen', linestyle = 'dashed', linewidth = 2) 
# ax2.tick_params(axis ='y', labelcolor = 'blue') 

# dry_year_avg_payout = dry_year_max_payout_df.mean()


# # ### wet-year two-way option ###

# muni_surplus_water = 7690

# for i in Wet_Year_Triggers.index:
#     Wet_Years['value'].loc[i] = muni_surplus_water

# wet_year_two_dict = {}

# for y in range(1950,2013):
#     wet_year_two = pd.DataFrame()
#     irrigation_set_wet_two = spirr.irrigation_uses_by_year_wet[y]
#     irrigation_set_wet_two['MUNISURPLUS'] = 0
#     if Granby_Spills.loc[y] > 0:
#         water_avail_wet_year = muni_surplus_water
#     else:
#         water_avail_wet_year = 0
#     for crop in range(len(irrigation_set_wet_two)):
#         if water_avail_wet_year > irrigation_set_wet_two['USAGE'].iloc[crop]:
#             use = irrigation_set_wet_two['USAGE'].iloc[crop]
#             print(use)
#         else:
#             use = max(water_avail_wet_year, 0)
#             print(use)
#         irrigation_set_wet_two['MUNISURPLUS'].iloc[crop] = use   
#         water_avail_wet_year -= irrigation_set_wet_two['USAGE'].iloc[crop]
#     wet_year_two = pd.concat([wet_year_two, irrigation_set_wet_two])

# ##### MUNIS ARE PROFIT MAXIMIZERS ####
#     wet_year_two['mv_h2o_sum'] = wet_year_two['MARGINAL_VALUE_OF_WATER']*wet_year_two['MUNISURPLUS']

# ##### MUNIS ARE LEASING AT $30/AF ####
#     wet_year_two['mv_h2o_sum_muni30'] = 30*wet_year_two['MUNISURPLUS']
        
#     wet_year_two_dict[y] = wet_year_two

# wet_year_max_payout_s1 = {}
# wet_year_max_payout_s2 = {}    
# for y in range(1950,2013):
#     wet_year_max_payout_s1[y] = wet_year_two_dict[y]['mv_h2o_sum'].sum()
#     wet_year_max_payout_s2[y] = wet_year_two_dict[y]['mv_h2o_sum_muni30'].sum()
    

# wet_year_max_payout_df_s1 = pd.DataFrame.from_dict(wet_year_max_payout_s1, orient ='index')
# wet_year_max_payout_df_s1['payout'] = wet_year_max_payout_df_s1[0]
# wet_year_max_payout_series_s1 = pd.Series(wet_year_max_payout_df_s1[0])
# wet_year_max_payout_list_s1 = wet_year_max_payout_series_s1.to_list()
# plt.figure()
# plt.plot(wet_year_max_payout_df_s1)
# plt.title('Scenario 1')
# plt.xlabel('Hydrologic Year')
# plt.ylabel('Wet Year Payout($)')

# fig, ax1 = plt.subplots() 

# plt.title('Wet Years - TWO (Scenario 1)')   
# ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Ag-to-Urban Payment ($)', color = 'lightgreen') 
# ax1.plot(wet_year_max_payout_df_s1.index, wet_year_max_payout_df_s1, color = 'blue', linewidth = 4) 
# ax1.tick_params(axis ='y', labelcolor = 'lightgreen') 
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# ax2.plot(Wet_Years.index, Wet_Years , color = 'lightgreen', linestyle = 'dashed', linewidth = 2) 
# ax2.tick_params(axis ='y', labelcolor = 'blue') 


# # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# # fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# # # plot the same data on both axes
# # ax1.plot(wet_year_max_payout_df_s1.index, wet_year_max_payout_df_s1, color = 'blue', linewidth = 4) 
# # ax2.plot(wet_year_max_payout_df_s1.index, wet_year_max_payout_df_s1, color = 'blue', linewidth = 4) 
# # # zoom-in / limit the view to different portions of the data
# # ax1.set_ylim(7070, 7100)  # outliers only
# # ax2.set_ylim(0, 225)  # most of the data

# # # hide the spines between ax and ax2
# # ax1.spines.bottom.set_visible(False)
# # ax2.spines.top.set_visible(False)
# # #ax1.xaxis.tick_top()
# # ax1.tick_params(labeltop=False)  # don't put tick labels at the top
# # ax2.xaxis.tick_bottom()


# # d = .5  # proportion of vertical to horizontal extent of the slanted line
# # kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
# #               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# # ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
# # ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)







# wet_year_avg_payout_s1 = wet_year_max_payout_df_s1.mean()

# wet_year_max_payout_df_s2 = pd.DataFrame.from_dict(wet_year_max_payout_s2, orient ='index')
# wet_year_max_payout_df_s2['payout'] = wet_year_max_payout_df_s2[0]
# wet_year_max_payout_series_s2 = pd.Series(wet_year_max_payout_df_s2[0])
# wet_year_max_payout_list_s2 = wet_year_max_payout_series_s2.to_list()
# plt.figure()
# plt.plot(wet_year_max_payout_df_s2)
# plt.title('Scenario 2')
# plt.xlabel('Hydrologic Year')
# plt.ylabel('Wet Year Payout($)')

# fig, ax1 = plt.subplots() 

# plt.title('Wet Years - TWO (Scenario 2)') 
# ax1.set_xlabel('Hydrologic Year') 
# ax1.set_ylabel('Ag-to-Urban Payment ($)', color = 'lightgreen') 
# ax1.plot(wet_year_max_payout_df_s2.index, wet_year_max_payout_df_s2, color = 'blue', linewidth = 4) 
# ax1.tick_params(axis ='y', labelcolor = 'lightgreen') 
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Water Re-allocated (AF)', color = 'blue') 
# ax2.plot(Wet_Years.index, Wet_Years , color = 'lightgreen', linestyle = 'dashed', linewidth = 2) 
# ax2.tick_params(axis ='y', labelcolor = 'blue') 

# wet_year_avg_payout_s2 = wet_year_max_payout_df_s2.mean()

# ##########################################################################
# ######### wang transform function ########################################
# ############## Returns dataframe with net payout #########################
# ##########################################################################

# ## lam is set to 0.25 unless otherwise specified (risk adjustment)
# ## df should be dataframe with payout per year 
# ## modified slightly -- from user = False means that it is from the insurer's perspective
# def wang_slim(payouts, lam = 0.25, contract = 'put', from_user = False):  
#   if from_user == True: 
#       ## switch the signs because you are feeding a negative value where the payout occurs
#       ## all other values are zero
#       payouts = -payouts
#   if contract == 'put': 
#       lam = -abs(lam)
#   unique_pays = pd.DataFrame()
#   unique_pays['unique'] = payouts.payout.unique()
#   #print(unique_pays['unique'])
#   unique_pays['payment probability'] = 0 
#   for j in range(len(unique_pays)):  
#       count = 0
#       val = unique_pays['unique'].iloc[j]
#       for i in np.arange(len(payouts)): 
#           if payouts['payout'].iloc[i] == val: 
#               count += 1
#     #  print(count)
#       unique_pays['payment probability'].iloc[j] = count/len(payouts)
#       #print(unique_pays)
      
#   unique_pays.sort_values(inplace=True, by='unique')
#   #print(unique_pays)
#   dum1 = unique_pays['payment probability'].cumsum()  # asset cdf
#   #print(dum1)
#   plt.figure()
#   #plt.plot(unique_pays['unique'], dum1)
#   plt.xlabel('payment probability')
#   plt.ylabel('Density')
#   #sns.kdeplot(dum, color = 'blue')
#   dum2 = st.norm.cdf(st.norm.ppf(dum1) + lam)  # risk transformed payout cdf
#   #plt.plot(unique_pays['unique'], dum2)
#   print(dum2)
#   #sns.kdeplot(dum, color ='red')
#   dum3 = np.append(dum2[0], np.diff(dum2))  # risk transformed asset pdf
#   print(dum3)
#   #print(dum3)
#   sns.kdeplot(dum3, color ='green')
#   plt.xlim(0,1)
  
#   prem = (dum3 * unique_pays['unique']).sum()
#   #print(prem)
#   #sns.kdeplot(prem, color ='blue')
#   #sns.kdeplot(prem, color = 'red')
#   #print(prem)
#   payouts.sort_index(inplace=True)

#   if from_user == True: 
#      ## want the insurer's perspective
#       whole = (prem - payouts['payout'])

#   else: 
# #      payouts.sort_index(inplace=True)
#      whole = (payouts['payout'] - prem)

  
#   return prem, whole

# dry_year_prems, dry_year_whole = wang_slim(dry_year_max_payout_df, lam = 0.25, \
#                                            contract = 'call', from_user = True)
# print(dry_year_max_payout_df['payout'].mean())
# print(dry_year_prems)
# print((dry_year_prems + dry_year_max_payout_df['payout']).mean())
# print(dry_year_whole.mean())
# wet_year_s1_prems, wet_year_s1_whole = wang_slim(wet_year_max_payout_df_s1, \
#                                                  lam = 0.25, contract = 'call',\
#                                                      from_user = True)
# wet_year_s2_prems, wet_year_s2_whole = wang_slim(wet_year_max_payout_df_s2, \
#                                                  lam = 0.25, contract = 'call', \
#                                                      from_user = True)


    
    
    
    
    
# plt.plot(dry_year_whole)

# dry_year_whole_avg = dry_year_whole.mean()

# plt.plot(wet_year_s1_whole)
# plt.plot(wet_year_s2_whole)

# premiums_s1 = dry_year_prems + wet_year_s1_prems

# premiums_s2 = dry_year_prems + wet_year_s2_prems


# Dry_Year_Total_Water_Reallocated = Dry_Years['value'].sum()
# Dry_Year_Total_Premiums = dry_year_prems*len(dry_year_max_payout_df)

# Dry_Year_Average_Premium = Dry_Year_Total_Premiums/Dry_Year_Total_Water_Reallocated



# Wet_Year_Total_Water_Reallocated = Wet_Years['value'].sum()
# Wet_Year_S1_Total_Premiums = wet_year_s1_prems*len(wet_year_max_payout_series_s1)
# Wet_Year_S2_Total_Premiums = wet_year_s2_prems*len(wet_year_max_payout_series_s2)

# Wet_Year_S1_Average_Premium = Wet_Year_S1_Total_Premiums/Wet_Year_Total_Water_Reallocated
# Wet_Year_S2_Average_Premium = Wet_Year_S2_Total_Premiums/Wet_Year_Total_Water_Reallocated


# S1_Fixed_Cost = 0
# S1_Diff = wet_year_s1_prems - S1_Fixed_Cost

# S2_Fixed_Cost = 0
# S2_Diff = wet_year_s2_prems - S2_Fixed_Cost

# S1_Exercise = pd.Series(index=range(1950,2013))
# for i in range(1950,2013):
#     if i in Wet_Year_Triggers:
#        S1_Exercise[i] = (S1_Diff/(Wet_Years['value'].loc[i]))
#     else:
#        S1_Exercise[i] = 0

# S2_Exercise = pd.Series(index=range(1950,2013))
# for i in range(1950,2013):
#     if i in Wet_Year_Triggers:
#        S2_Exercise[i] = (S2_Diff/(Wet_Years['value'].loc[i]))
#     else:
#        S2_Exercise[i] = 0


# dry_year_fixed = -dry_year_prems/2
        
# Dry_Year_Exercise = pd.Series(index=range(1950,2013))
# for i in range(1950,2013):
#     if i in Dry_Year_Triggers:
#        Dry_Year_Exercise[i] = (dry_year_fixed/(Dry_Years['value'].loc[i]))
#        print(Dry_Years['value'][i])
#     else:
#        Dry_Year_Exercise[i] = 0
#     print(Dry_Year_Exercise[i])

# Dry_Year_total_dollars = Dry_Year_Exercise*Dry_Years['value']
# plt.figure()
# plt.plot(-dry_year_prems)

   

    

# #prem = b (AF) + fixed




# S1_Premium = -(Dry_Year_Average_Premium + Wet_Year_S1_Average_Premium)
# S2_Premium = -(Dry_Year_Average_Premium + Wet_Year_S2_Average_Premium)




# S1 = pd.DataFrame(index=range(1950,2013))
# S1['premium'] = 0

# for i in S1.index:
#     S1['premium'].loc[i] = S1_Premium
# S1['premium'] = S1['premium'].fillna(0)
    
# S2 = pd.DataFrame(index=range(1950,2013))
# S2['premium'] = 0

# for i in S2.index:
#     S2['premium'].loc[i] = S2_Premium
# S2['premium'] = S2['premium'].fillna(0)

# DryYear = pd.DataFrame(index=range(1950,2013))
# DryYear['premium'] = 0   

# for i in DryYear.index:
#     DryYear['premium'].loc[i] = -(Dry_Year_Average_Premium)
# DryYear['premium'] = DryYear['premium'].fillna(0)

# #from Brian MacPherson email, C-BT units currently selling for $75,000/unit = $97,500/AF

# CBT_Water_Right = 97500

# ## annualized at 30yrs, 6% interest, $97500 for CBT unit ###
# Annualized_Perm_Right_Cost = 7083.27

# Annualized_Perm_Right = pd.DataFrame(index=range(1950,2013))
# Annualized_Perm_Right['premium'] = 0

# for i in Annualized_Perm_Right.index:
#     Annualized_Perm_Right['premium'].loc[i] = Annualized_Perm_Right_Cost
    

# # ## Dry-year option ###

# # ## using mickelson et. al ###

# # ## exercise fee = $90 in 1988 = $227.60 today
# # Dry_Year_90 = 227.60
# # ## exercise fee = $45 in 1988 = $113.80 today
# # Dry_Year_45 = 113.80
# # ## exercise fee = $135 in 1988 = $341.40 today
# # Dry_Year_135 = 341.40

# # DryYear = pd.DataFrame(index=range(1950,2013))
# # DryYear['premium'] = 0

# # for i in S1.index:
# #     S1['premium'].loc[i] = S1_Premium + S1_Exercise[i]
    
# # S2 = pd.DataFrame(index=range(1950,2013))
# # S2['premium'] = 0

# # for i in S2.index:
# #     S2['premium'].loc[i] = S2_Premium + S2_Exercise[i]
# # S2['premium'] = S2['premium'].fillna(0)    

# # Dry_Year_Option = pd.Series(index=range(1950,2013))
# # for i in range(1950,2013):
# #     if i in Dry_Year_Triggers:
# #        Dry_Year_Option[i] = 135
# #     else:
# #        Dry_Year_Option[i] = 90


# plt.figure()
# plt.xlabel('Hydrologic Year')
# plt.ylabel('Cost ($/AF)')
# plt.plot(S1, linestyle = 'solid', marker = 'x', markersize = 6, label='Profit Maximizing Utilities')
# plt.plot(S2, linestyle = 'solid', marker = 'd', markersize = 6, label= 'Utilties lease back for $30/AF')
# plt.plot(Annualized_Perm_Right, linestyle = 'solid', marker = '.', markersize = 6, label= 'Permanent Water Right Purchase')
# plt.legend(bbox_to_anchor=(1.65, 1), loc='upper right', borderaxespad=0)


# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# # plot the same data on both axes
# ax1.plot(S1, linestyle = 'solid', marker = 'x', markersize = 6, label='Wet Year Option (MV of Water/AF)')
# ax1.plot(S2, linestyle = 'solid', marker = 'd', markersize = 6, label= 'Wet Year Option (at $30/AF)')
# ax1.plot(Annualized_Perm_Right, linestyle = 'solid', marker = '.', markersize = 6, label= 'Permanent Water Right Purchase')
# ax1.plot(DryYear, linestyle = 'solid', marker = 'h', markersize = 6, label= 'Dry Year Option')
# ax2.plot(S1, linestyle = 'solid', marker = 'x', markersize = 6, label='Wet Year Option (MV of Water/AF)')
# ax2.plot(S2, linestyle = 'solid', marker = 'd', markersize = 6, label= 'Wet Year Option (at $30/AF)')
# ax2.plot(Annualized_Perm_Right, linestyle = 'solid', marker = '.', markersize = 6, label= 'Permanent Water Right Purchase')
# ax2.plot(DryYear, linestyle = 'solid', marker = 'h', markersize = 6, label= 'Dry Year Option')

# # zoom-in / limit the view to different portions of the data
# ax1.set_ylim(7070, 7100)  # outliers only
# ax2.set_ylim(90, 190)  # most of the data

# # hide the spines between ax and ax2
# ax1.spines.bottom.set_visible(False)
# ax2.spines.top.set_visible(False)
# #ax1.xaxis.tick_top()
# ax1.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()


# d = .5  # proportion of vertical to horizontal extent of the slanted line
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
# ax1.legend(bbox_to_anchor=(1.65, 1), loc='upper right', borderaxespad=0, frameon = False)
# plt.xlabel('Hydrologic Year')
# plt.ylabel('Cost ($/AF)')

# plt.show()
    



# # def id_payouts_index(inputs, strike, cap, model): 
# #     payouts = pd.DataFrame(index = np.arange(0, len(inputs)),columns=['payout']) 
# #     payouts['payout'] = 0
    
# #     for i in np.arange(len(inputs)):
# #         if inputs['Dalls ARF'].iloc[i] < strike: 
# #             ## model will predict a value that represents losses essentially
# #             payouts.iloc[i,0] = model.predict(inputs.iloc[i,:])
# #        #     print(payouts.iloc[i,0])
# #             ## constrain so that if predicted value is > 0, BPA does not get paid
# #             if payouts.iloc[i,0] > 0:
# #                 payouts.iloc[i,0] = 0
# #             ## cap payouts
# #             if payouts.iloc[i,0] < cap: 
# #                 payouts.iloc[i,0] = cap     
# #             ## NOTE: these are left negative to work within the wang transform function
# #     return payouts

# # def id_payment_swaps(input, strike, cap = 200000000, slope = False, swap = False, strike2 = ''): 
# #     payouts = pd.DataFrame(index=np.arange(0, len(input)),columns=['payout']) 
# #     payouts['payout'] = 0
# #     mean = input.iloc[:,0].mean() 
    
# #     for i in np.arange(len(input)):
# #         if swap == True: 
# #             if input['Dalls ARF'].iloc[i] >= strike: 
# #                 ## no modifier
# #                 if slope == False: 
# #                     payouts.iloc[i,0] = abs(est_above2.predict(input.iloc[i, 1:]))/payout_mod
# #                 else: 
# #                     slope = input.iloc[i,0]/mean
# #                     payouts.iloc[i,0] = abs(est_above2.predict(input.iloc[i, 1:]) * slope)/payout_mod
# #         if swap == False: 
# #             if input['Dalls ARF'].iloc[i] >= strike2: 
# #                 ## no modifier
# #                 if slope == False: 
# #                     payouts.iloc[i,0] = abs(est_above2.predict(input.iloc[i, 1:]))/payout_mod
# #                 else: 
# #                     slope = input.iloc[i,0]/mean
# #                     payouts.iloc[i,0] = abs(est_above2.predict(input.iloc[i, 1:]) * slope)/payout_mod
# #         ## constrain payout 
# #         if payouts.iloc[i,0] < 0:
# #             payouts.iloc[i,0] = 0
# #         ## cap payouts
# #         if payouts.iloc[i,0] > cap: 
# #             payouts.iloc[i,0] = cap 
# #     ## return negative because these are payments from BPA to the counterparty 
# #     return -payouts




# os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod')

# plt.figure()
# plt.plot(EstesPark_Total_Shortage_Sum['Shortage'],color='purple', label='Estes Park')
# plt.plot(Loveland_Total_Shortage_Sum['Shortage'],color='green', label='Loveland')
# plt.plot(Greeley_Total_Shortage_Sum['Shortage'],color='yellow', label='Greeley')
# plt.plot(Longmont_Total_Shortage_Sum['Shortage'],color='red', label='Longmont')
# plt.plot(Louisville_Total_Shortage_Sum['Shortage'],color='orange', label='Louisville')
# plt.plot(Laffayette_Total_Shortage_Sum['Shortage'],color='brown', label='Laffayette')
# plt.plot(Boulder_Total_Shortage_Sum['Shortage'],color='blue', label='Boulder')
# plt.xlabel('Hydrologic Year')
# plt.ylabel('Shortage (AF)')
# plt.title('Northern Water Municipal Shortages')
# plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right', borderaxespad=0)

