# # # # -*- coding: utf-8 -*-
# # # """
# # # Created on Thu Jul  7 16:10:49 2022

# # # @author: zacha
# # # """
import os
import shutil
import statemod_data_extraction
import sys
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import csv

########################################################################################################################################

#update South Platte .ddm files

# import statemodify as stm

# # a dictionary to describe what you want to modify and the bounds for the LHS
# setup_dict = {
#     "names": ["municipal"],
#     "ids": [["04_Lovelnd_I","04_Lovelnd_O","05LONG_IN","05LONG_OUT","06BOULDER_I","06BOULDER_O"]],
#     "bounds": [[0, 1.0]]
# }

# output_directory = "C:/Users/zacha/Documents/UNC/SP2016_StateMod"
# scenario = "model_run"

# # the number of samples you wish to generate
# n_samples = 30

# # seed value for reproducibility if so desired
# seed_value = None

# # my template file.  If none passed into the `modify_ddm` function, the default file will be used.
# template_file = "SP2016_H.ddm"

# # the field that you want to use to query and modify the data
# query_field = "id"

# # generate a batch of files using generated LHS
# stm.modify_ddm(modify_dict=setup_dict,
#                 output_dir=output_directory,
#                 scenario=scenario,
#                 n_samples=n_samples,
#                 seed_value=seed_value,
#                 query_field=query_field,
#                 template_file=template_file)

######################################################################################################################################
# def update_rsp_file(test_number):
#   with open('SP2016_H.RSP','r') as f:
#     split = [x for x in f.readlines()]       
#   f.close()
#   f = open('SP2016_H.RSP','w')
#   for i in range(0, len(split)):
#     if i == 28:
#       f.write('Diversion_Demand_Monthly                = SP2016_H_scenario-model_run_sample-' + str(test_number) + '.ddm\n')
#     else:
#       f.write(split[i])
#   f.close()

#   return split

# for test_number in range(0,30):
#     update_rsp_file(test_number)
#     print('update .rsp files')
#     print('run StateMod')    
#     os.system("StateMod_Model_15.exe SP2016_H -simulate")
#     print('successful StateMod run for test #'+str(test_number))
#     os.mkdir('output'+str(test_number))
#     shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H_scenario-model_run_sample-' + str(test_number) + '.ddm', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H_scenario-model_run_sample-' + str(test_number) + '.ddm')
#     shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xdd', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/sp2016_H_S0_'+str(test_number)+'.xdd')
#     shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xre', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/sp2016_H_S0_'+str(test_number)+'.xre')
#     shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xop', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xop')
#     shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xir', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xir')
#     shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xss', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xss')
#     shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xca', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xca')
#     shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xwe', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xwe')
#     shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xpl', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xpl')
    # shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/statemod_data_extraction.py', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/statemod_data_extraction.py')
    # shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/statemod_data_extraction_xre_final.py', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/statemod_data_extraction_xre_final.py')
    # shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/ids_file.txt')
    # os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number))
    # print('extracting StateMod .xdd data to Parquet')
    # subprocess.run(["python", "statemod_data_extraction.py", "--ids", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt","--output", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/output"+str(test_number)+"/xddparquet", "sp2016_H_S0_"+str(test_number)+".xdd"])
    # print('extracting StateMod .xre data to Parquet')
    # subprocess.run(["python", "statemod_data_extraction_xre_final.py", "--ids", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt","--output", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/output"+str(test_number)+"/xreparquet", "sp2016_H_S0_"+str(test_number)+".xre"])
    # print('successful extraction')
    # os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod')
    # print('successful execution')
# print('successful termination :)')

################################################################################################################

# def update_rsp_file():
#   with open('SP2016_H.RSP','r') as f:
#     split = [x for x in f.readlines()]       
#   f.close()
#   f = open('SP2016_H.RSP','w')
#   for i in range(0, len(split)):
#     if i == 28:
#       f.write('Diversion_Demand_Monthly                = SP2016_U.ddm\n')
#     if i == 77:
#       f.write('Operational_Right                       = SP2016_U.opr\n')
#     else:
#       f.write(split[i])
#   f.close()

#   return split

# for test_number in range(0,1):
#     update_rsp_file()
#     print('update .rsp files')
#     print('run StateMod')    
#     os.system("StateMod_Model_15.exe SP2016_H -simulate")
#     print('successful StateMod run for test #'+str(test_number))
#     os.mkdir('output'+str(test_number))
    # shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H_scenario-model_run_sample-' + str(test_number) + '.ddm', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H_scenario-model_run_sample-' + str(test_number) + '.ddm')
# shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/SP2016_H.xdd', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/sp2016_H_S0_1.xdd')
# shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/SP2016_H.xre', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/sp2016_H_S0_1.xre')
#     # shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xop', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xop')
#     # shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xir', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xir')
#     # shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xss', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xss')
#     # shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xca', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xca')
#     # shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xwe', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xwe')
#     # shutil.move('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_H.xpl', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number)+'/SP2016_H.xpl')
# shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/statemod_data_extraction.py', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/statemod_data_extraction.py')
# shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/statemod_data_extraction_xre_final.py', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/statemod_data_extraction_xre_final.py')
# shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/ids_file.txt')
#     #os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number))
# print('extracting StateMod .xdd data to Parquet')
# subprocess.run(["python", "statemod_data_extraction.py", "--ids", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt","--output", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet", "sp2016_H_S0_1.xdd"])
# print('extracting StateMod .xre data to Parquet')
# subprocess.run(["python", "statemod_data_extraction_xre_final.py", "--ids", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt","--output", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xreparquet", "sp2016_H_S0_1.xre"])
# print('successful extraction')
# #os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod')
# print('successful execution')
# print('successful termination :)')



#IMPORT HISTORICAL DATA

Historical_Boulder_Water_Supplies = pd.DataFrame()
Historical_Boulder_Water_Supplies_Sum = pd.DataFrame()
Historical_Boulder_Res_Intake = pd.DataFrame()
Historical_Boulder_Barker_Pipeline = pd.DataFrame()
Historical_Boulder_City_Pipeline = pd.DataFrame()

Historical_Denver_Water_Supplies = pd.DataFrame()
Historical_Denver_Water_Supplies_Sum = pd.DataFrame()
Historical_Aurora_Water_Supplies_Sum = pd.DataFrame()
Historical_Longmont_Water_Supplies_Sum = pd.DataFrame()
Historical_Aurora_Water_Supplies = pd.DataFrame()
Historical_Longmont_Water_Supplies = pd.DataFrame()




# os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline')
# print('extracting StateMod .xdd data to Parquet')
# subprocess.run(["python", "statemod_data_extraction.py", "--ids", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt","--output", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline/xddparquet", "sp2016_H_S0_0.xdd"])
# print('extracting StateMod .xre data to Parquet')
# subprocess.run(["python", "statemod_data_extraction_xre_final.py", "--ids", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt","--output", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline/xreparquet", "sp2016_H_S0_0.xre"])
# print('successful extraction')
os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline/xddparquet')
HistoricalBoulderOutdoor = pd.read_parquet('06BOULDER_O.parquet', engine='pyarrow')
HistoricalBoulderIndoor = pd.read_parquet('06BOULDER_I.parquet', engine='pyarrow')
HistoricalLongmontIndoor = pd.read_parquet('05LONG_IN.parquet', engine='pyarrow')
HistoricalLongmontOutdoor = pd.read_parquet('05LONG_OUT.parquet', engine='pyarrow')
HistoricalLongmontCBT = pd.read_parquet('05_LongCBT.parquet', engine='pyarrow')
HistoricalLovelandIndoor = pd.read_parquet('04_Lovelnd_I.parquet', engine='pyarrow')
HistoricalLovelandOutdoor = pd.read_parquet('04_Lovelnd_O.parquet', engine='pyarrow')
HistoricalLovelandPipeline = pd.read_parquet('0400511.parquet', engine='pyarrow')
HistoricalNorthPipeline = pd.read_parquet('0500511.parquet', engine='pyarrow')
HistoricalSouthPipeline = pd.read_parquet('0500522.parquet', engine='pyarrow')
HistoricalLyonsPipeline = pd.read_parquet('0500512.parquet', engine='pyarrow')
HistoricalExporttoHorsetooth = pd.read_parquet('0400691_X.parquet', engine='pyarrow')
HistoricalAdamsTunnelImports = pd.read_parquet('0404634.parquet', engine='pyarrow')

Historical_Boulder_Res_Intake = pd.read_parquet('0600800.parquet', engine='pyarrow')
Historical_Boulder_Barker_Pipeline = pd.read_parquet('0600943.parquet', engine='pyarrow')
Historical_Boulder_City_Pipeline = pd.read_parquet('0600599.parquet', engine='pyarrow')
HistoricalAuroraIntake = pd.read_parquet('0801001.parquet', engine='pyarrow')
HistoricalMoffat = pd.read_parquet('06_MOF_IMP.parquet', engine='pyarrow')
HistoricalCon15 = pd.read_parquet('Conduit15.parquet', engine='pyarrow')
HistoricalCon20 = pd.read_parquet('0801002_D.parquet', engine='pyarrow')
HistoricalCon26 = pd.read_parquet('0801017.parquet', engine='pyarrow')

HistoricalLongmontCBT = pd.read_parquet('05_LongCBT.parquet', engine='pyarrow')




os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline/xreparquet')

HistoricalBoulderRes = pd.read_parquet('0504515+.parquet', engine='pyarrow')
#BoulderRes['totalrelease'] = BoulderRes['totalrelease'].str.replace('.','')
HistoricalBoulderRes['totalrelease'] = pd.to_numeric(HistoricalBoulderRes['totalrelease'])
HistoricalBarkerRes = pd.read_parquet('0604172+.parquet', engine='pyarrow')
HistoricalBarkerRes['totalrelease'] = pd.to_numeric(HistoricalBarkerRes['totalrelease'])
HistoricalWatershedRes = pd.read_parquet('06_WSHED+.parquet', engine='pyarrow')
#WatershedRes['totalrelease'] = WatershedRes['totalrelease'].str.replace('.','')
HistoricalWatershedRes['totalrelease'] = pd.to_numeric(HistoricalWatershedRes['totalrelease'])
HistoricalBaselineRes = pd.read_parquet('0604173+.parquet', engine='pyarrow')
#BaselineRes['totalrelease'] = BaselineRes['totalrelease'].str.replace('.','')
HistoricalBaselineRes['totalrelease'] = pd.to_numeric(HistoricalBaselineRes['totalrelease'])
HistoricalGreenGlade = pd.read_parquet('0403659+.parquet', engine='pyarrow')
#GreenGlade['totalrelease'] = GreenGlade['totalrelease'].str.replace('.','')
HistoricalButtonRockRes = pd.read_parquet('0504010+.parquet', engine='pyarrow')
#ButtonRockRes['totalrelease'] = ButtonRockRes['totalrelease'].str.replace('.','')
HistoricalUnionRes = pd.read_parquet('0503905+.parquet', engine='pyarrow')
#UnionRes['totalrelease'] = UnionRes['totalrelease'].str.replace('.','')
HistoricalSpinney = pd.read_parquet('2304013+.parquet', engine='pyarrow')
HistoricalAuroraRes = pd.read_parquet('0203379+.parquet', engine='pyarrow')
HistoricalGrossRes = pd.read_parquet('0604199+.parquet', engine='pyarrow')
HistoricalRalstonRes = pd.read_parquet('0703324+.parquet', engine='pyarrow')

#HISTORICAL LONGMONT

LongmontCBTsum = HistoricalLongmontCBT.groupby('year').sum()

LongmontCBTsum.to_csv('Longmont.csv')

#HISTORICAL AURORA

historical_aurora_supplies = ['AuroraRes']
Historical_Aurora_Water_Supplies = pd.DataFrame().assign(Year=HistoricalWatershedRes['year'],Account=HistoricalWatershedRes['account'],AuroraRes=HistoricalAuroraRes['totalrelease'])
Historical_Aurora_Water_Supplies_Sum = Historical_Aurora_Water_Supplies.groupby(["Account", "Year"])[historical_aurora_supplies].sum()
Historical_Aurora_Water_Supplies_Sum = Historical_Aurora_Water_Supplies_Sum.reset_index()

Historical_Aurora_Intake_Sum = HistoricalAuroraIntake.groupby('year').sum()

Historical_Aurora_Water_Supplies_Sum.to_csv('aurora_res.csv')
Historical_Aurora_Intake_Sum.to_csv('aurora_intake.csv')

#HISTORICAL BOULDER

historical_boulder_supplies = ['WatershedRes','BarkerRes','BoulderRes']
Historical_Boulder_Water_Supplies = pd.DataFrame().assign(Year=HistoricalWatershedRes['year'],Account=HistoricalWatershedRes['account'],WatershedRes=HistoricalWatershedRes['totalrelease'], BarkerRes=HistoricalBarkerRes['totalrelease'],BoulderRes=HistoricalBoulderRes['totalrelease'])
Historical_Boulder_Water_Supplies_Sum = Historical_Boulder_Water_Supplies.groupby(["Account", "Year"])[historical_boulder_supplies].sum()
Historical_Boulder_Water_Supplies_Sum = Historical_Boulder_Water_Supplies_Sum.reset_index()

Historical_Boulder_Total_Demand = HistoricalBoulderIndoor['demand'] + HistoricalBoulderOutdoor['demand']
Historical_Boulder_Total_Demand_Sum = pd.DataFrame().assign(Year=HistoricalBoulderOutdoor['year'],Demand=Historical_Boulder_Total_Demand)
Historical_Boulder_Total_Demand_Sum = Historical_Boulder_Total_Demand_Sum.groupby('Year').sum()
Historical_Boulder_Total_Demand_Sum = Historical_Boulder_Total_Demand_Sum.reset_index()
Historical_Boulder_Total_Demand_Sum.index = Historical_Boulder_Total_Demand_Sum['Year']

Historical_Boulder_Water_Supplies_Sum['demand'] = Historical_Boulder_Total_Demand_Sum['Demand']

Historical_Boulder_Additional_Leasing = pd.DataFrame().assign(Year=HistoricalBaselineRes['year'], Account=HistoricalBaselineRes['account'], Leases=HistoricalBaselineRes['totalrelease'])
historical_boulder_leases = ['Leases']
Historical_Boulder_Additional_Leasing_Sum = Historical_Boulder_Additional_Leasing.groupby(["Account", "Year"])[historical_boulder_leases].sum()
Historical_Boulder_Additional_Leasing_Sum = Historical_Boulder_Additional_Leasing_Sum.reset_index()
Historical_Boulder_Additional_Leases = Historical_Boulder_Additional_Leasing_Sum.loc[Historical_Boulder_Additional_Leasing_Sum['Account'] == 2]
Historical_Boulder_Additional_Leases = Historical_Boulder_Additional_Leases.reset_index()

Historical_Boulder_Municipal_Water = Historical_Boulder_Water_Supplies_Sum.loc[Historical_Boulder_Water_Supplies_Sum['Account'] == 1]
Historical_Boulder_Municipal_Water = Historical_Boulder_Municipal_Water.reset_index()

Historical_Boulder_Municipal_Water['leases'] = Historical_Boulder_Additional_Leases['Leases']

Historical_Boulder_Municipal_Water['demand'] = Historical_Boulder_Total_Demand_Sum['Demand']

Historical_Boulder_Res_Intake_Sum = Historical_Boulder_Res_Intake.groupby('year').sum()
Historical_Boulder_Barker_Pipeline_Sum = Historical_Boulder_Barker_Pipeline.groupby('year').sum()
Historical_Boulder_City_Pipeline_Sum = Historical_Boulder_City_Pipeline.groupby('year').sum()


Boulder = Historical_Boulder_Res_Intake_Sum['carried'] + Historical_Boulder_Barker_Pipeline_Sum['carried'] + Historical_Boulder_City_Pipeline_Sum['carried']
Boulder.to_csv('boulder_supplies_use.csv')
plt.figure()
plt.plot(Boulder)
plt.plot(Historical_Boulder_Total_Demand_Sum['Demand'])

#HISTORICAL DENVER

historical_denver_supplies = ['GrossRes','RalstonRes']
Historical_Denver_Water_Supplies = pd.DataFrame().assign(Year=HistoricalWatershedRes['year'],Account=HistoricalWatershedRes['account'],GrossRes=HistoricalGrossRes['totalrelease'], RalstonRes=HistoricalRalstonRes['totalrelease'])
Historical_Denver_Water_Supplies_Sum = Historical_Denver_Water_Supplies.groupby(["Account", "Year"])[historical_denver_supplies].sum()
Historical_Denver_Water_Supplies_Sum = Historical_Denver_Water_Supplies_Sum.reset_index()

Historical_Denver_Water_Supplies_Sum.to_csv('denver.csv')
Historical_Conduit15 = HistoricalCon15.groupby('year').sum()
Historical_Conduit20 = HistoricalCon20.groupby('year').sum()
Historical_Conduit26 = HistoricalCon26.groupby('year').sum()

Historical_Conduit15.to_csv('Con15.csv')
Historical_Conduit20.to_csv('Con20.csv')
Historical_Conduit26.to_csv('Con26.csv')




historical_boulder_supplies = ['WatershedRes','BarkerRes','BoulderRes']
Historical_Boulder_Water_Supplies = pd.DataFrame().assign(Year=HistoricalWatershedRes['year'],Account=HistoricalWatershedRes['account'],WatershedRes=HistoricalWatershedRes['totalrelease'], BarkerRes=HistoricalBarkerRes['totalrelease'],BoulderRes=HistoricalBoulderRes['totalrelease'])
Historical_Boulder_Water_Supplies_Sum = Historical_Boulder_Water_Supplies.groupby(["Account", "Year"])[historical_boulder_supplies].sum()
Historical_Boulder_Water_Supplies_Sum = Historical_Boulder_Water_Supplies_Sum.reset_index()

Historical_Boulder_Total_Demand = HistoricalBoulderIndoor['demand'] + HistoricalBoulderOutdoor['demand']
Historical_Boulder_Total_Demand_Sum = pd.DataFrame().assign(Year=HistoricalBoulderOutdoor['year'],Demand=Historical_Boulder_Total_Demand)
Historical_Boulder_Total_Demand_Sum = Historical_Boulder_Total_Demand_Sum.groupby('Year').sum()
Historical_Boulder_Total_Demand_Sum = Historical_Boulder_Total_Demand_Sum.reset_index()
Historical_Boulder_Total_Demand_Sum.index = Historical_Boulder_Total_Demand_Sum['Year']

Historical_Boulder_Water_Supplies_Sum['demand'] = Historical_Boulder_Total_Demand_Sum['Demand']

Historical_Boulder_Additional_Leasing = pd.DataFrame().assign(Year=HistoricalBaselineRes['year'], Account=HistoricalBaselineRes['account'], Leases=HistoricalBaselineRes['totalrelease'])
historical_boulder_leases = ['Leases']
Historical_Boulder_Additional_Leasing_Sum = Historical_Boulder_Additional_Leasing.groupby(["Account", "Year"])[historical_boulder_leases].sum()
Historical_Boulder_Additional_Leasing_Sum = Historical_Boulder_Additional_Leasing_Sum.reset_index()
Historical_Boulder_Additional_Leases = Historical_Boulder_Additional_Leasing_Sum.loc[Historical_Boulder_Additional_Leasing_Sum['Account'] == 2]
Historical_Boulder_Additional_Leases = Historical_Boulder_Additional_Leases.reset_index()

Historical_Boulder_Municipal_Water = Historical_Boulder_Water_Supplies_Sum.loc[Historical_Boulder_Water_Supplies_Sum['Account'] == 1]
Historical_Boulder_Municipal_Water = Historical_Boulder_Municipal_Water.reset_index()

Historical_Boulder_Municipal_Water['leases'] = Historical_Boulder_Additional_Leases['Leases']

Historical_Boulder_Municipal_Water['demand'] = Historical_Boulder_Total_Demand_Sum['Demand']

Historical_Boulder_Res_Intake_Sum = Historical_Boulder_Res_Intake.groupby('year').sum()
Historical_Boulder_Barker_Pipeline_Sum = Historical_Boulder_Barker_Pipeline.groupby('year').sum()
Historical_Boulder_City_Pipeline_Sum = Historical_Boulder_City_Pipeline.groupby('year').sum()


Boulder = Historical_Boulder_Res_Intake_Sum['carried'] + Historical_Boulder_Barker_Pipeline_Sum['carried'] + Historical_Boulder_City_Pipeline_Sum['carried']



# os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline')
# Historical_Boulder_Municipal_Water.to_csv('boulderhistoricalsupplies.csv')

# Historical_Boulder_Res_Intake_Sum.to_csv('boulderresintakehistoricalsupplies.csv')
# Historical_Boulder_Barker_Pipeline_Sum.to_csv('boulderbarkerpipelinehistoricalsupplies.csv')
# Historical_Boulder_City_Pipeline_Sum.to_csv('bouldercitypipelinehistoricalsupplies.csv')

#### MODEL RUNS ####


# #STORE STRUCTURES OF INTEREST DATA IN DICTIONARIES
# Boulder_Outdoor={}
# Boulder_Indoor={}
# Longmont_Indoor={}
# Longmont_Outdoor={}
# Longmont_CBT={}
# Loveland_Indoor={}
# Loveland_Outdoor={}
# Loveland_Pipeline={}
# Green_Glade={}
# North_Pipeline={}
# South_Pipeline={}
# Lyons_Pipeline={}
# Button_Rock_Res={}
# Union_Res={}
# Boulder_Res = {}
# Barker_Res= {}
# Watershed_Res={}
# Baseline_Res={}
# Boulder_Res_Intake = {}
# Boulder_Barker_Pipeline = {}
# Boulder_City_Pipeline = {}
# Export_to_Horsetooth={}
# Adams_Tunnel_Imports={}

# for test_number in range(0,30):
    
#     #MUNICIPAL AND STRUCTURE DEMANDS OF INTEREST        
#     os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet/')
#     # if test_number == 3:
#     #     continue
#     if test_number == 10:
#         continue
#     BoulderOutdoor = pd.read_parquet('06BOULDER_O.parquet', engine='pyarrow')
#     BoulderIndoor = pd.read_parquet('06BOULDER_I.parquet', engine='pyarrow')
#     LongmontIndoor = pd.read_parquet('05LONG_IN.parquet', engine='pyarrow')
#     LongmontOutdoor = pd.read_parquet('05LONG_OUT.parquet', engine='pyarrow')
#     LongmontCBT = pd.read_parquet('05_LongCBT.parquet', engine='pyarrow')
#     LovelandIndoor = pd.read_parquet('04_Lovelnd_I.parquet', engine='pyarrow')
#     LovelandOutdoor = pd.read_parquet('04_Lovelnd_O.parquet', engine='pyarrow')
#     LovelandPipeline = pd.read_parquet('0400511.parquet', engine='pyarrow')
#     NorthPipeline = pd.read_parquet('0500511.parquet', engine='pyarrow')
#     SouthPipeline = pd.read_parquet('0500522.parquet', engine='pyarrow')
#     LyonsPipeline = pd.read_parquet('0500512.parquet', engine='pyarrow')
#     BoulderResIntake = pd.read_parquet('0600800.parquet', engine='pyarrow')
#     BoulderBarkerPipeline = pd.read_parquet('0600943.parquet', engine='pyarrow')
#     BoulderCityPipeline = pd.read_parquet('0600599.parquet', engine='pyarrow')
#     ExporttoHorsetooth = pd.read_parquet('0400691_X.parquet', engine='pyarrow')
#     AdamsTunnelImports = pd.read_parquet('0404634.parquet', engine='pyarrow')
    
#     #RESERVOIRS OF INTEREST 
#     os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xreparquet/')
    
#     BoulderRes = pd.read_parquet('0504515+.parquet', engine='pyarrow')
#     #BoulderRes['totalrelease'] = BoulderRes['totalrelease'].str.replace('.','')
#     BoulderRes['totalrelease'] = pd.to_numeric(BoulderRes['totalrelease'])
#     BarkerRes = pd.read_parquet('0604172+.parquet', engine='pyarrow')
#     BarkerRes['totalrelease'] = pd.to_numeric(BarkerRes['totalrelease'])
#     WatershedRes = pd.read_parquet('06_WSHED+.parquet', engine='pyarrow')
#     #WatershedRes['totalrelease'] = WatershedRes['totalrelease'].str.replace('.','')
#     WatershedRes['totalrelease'] = pd.to_numeric(WatershedRes['totalrelease'])
#     BaselineRes = pd.read_parquet('0604173+.parquet', engine='pyarrow')
#     #BaselineRes['totalrelease'] = BaselineRes['totalrelease'].str.replace('.','')
#     BaselineRes['totalrelease'] = pd.to_numeric(BaselineRes['totalrelease'])
#     GreenGlade = pd.read_parquet('0403659+.parquet', engine='pyarrow')
#     #GreenGlade['totalrelease'] = GreenGlade['totalrelease'].str.replace('.','')
#     ButtonRockRes = pd.read_parquet('0504010+.parquet', engine='pyarrow')
#     #ButtonRockRes['totalrelease'] = ButtonRockRes['totalrelease'].str.replace('.','')
#     UnionRes = pd.read_parquet('0503905+.parquet', engine='pyarrow')
#     #UnionRes['totalrelease'] = UnionRes['totalrelease'].str.replace('.','')
    
#     Boulder_Outdoor[test_number] = BoulderOutdoor
#     Boulder_Indoor[test_number] = BoulderIndoor
#     Boulder_Res[test_number] = BoulderRes
#     Barker_Res[test_number] = BarkerRes
#     Watershed_Res[test_number] = WatershedRes
#     Baseline_Res[test_number] = BaselineRes
#     Boulder_Res_Intake[test_number] = BoulderResIntake
#     Boulder_Barker_Pipeline[test_number] = BoulderBarkerPipeline
#     Boulder_City_Pipeline[test_number] = BoulderCityPipeline
#     Longmont_Indoor[test_number] = LongmontIndoor
#     Longmont_Outdoor[test_number] = LongmontOutdoor
#     Longmont_CBT[test_number] = LongmontCBT
#     Loveland_Indoor[test_number] = LovelandIndoor
#     Loveland_Outdoor[test_number] = LovelandOutdoor
#     Loveland_Pipeline[test_number] = LovelandPipeline
#     Green_Glade[test_number] = GreenGlade
#     North_Pipeline[test_number] = NorthPipeline
#     South_Pipeline[test_number] = SouthPipeline
#     Lyons_Pipeline[test_number] = LyonsPipeline
#     Button_Rock_Res[test_number] = ButtonRockRes
#     Union_Res[test_number] = UnionRes
#     Export_to_Horsetooth[test_number] = ExporttoHorsetooth
#     Adams_Tunnel_Imports[test_number] = AdamsTunnelImports
    
    
# #ADAMS TUNNEL HISTORICAL IMPORTS

# Adams_Tunnel_Yearly_Imports = AdamsTunnelImports.groupby('year').sum()['carried']

# plt.figure()
# plt.plot(Adams_Tunnel_Yearly_Imports)
# plt.title('Adams Tunnel Imports')
# plt.ylabel('Diversions (AF)')


# #BOULDER
# Boulder_Total_Demand = {}
# Boulder_Total_Demand_Sum = {}
# Boulder_Indoor_Shortage = {}
# Boulder_Outdoor_Shortage = {}
# Boulder_Total_Shortage = {}
# Boulder_Total_Shortage_Sum ={}
# Boulder_Water_Supplies = {}
# Boulder_Water_Supplies_Sum = {}
# Boulder_Municipal_Water = {}
# Boulder_Additional_Leasing = {}
# Boulder_Additional_Leasing_Sum ={}
# Boulder_Additional_Leases = {}
# Boulder_Source_Comparison = {}
# Boulder_Res_Intake_Sum = {}
# Boulder_Barker_Pipeline_Sum = {}
# Boulder_City_Pipeline_Sum = {}


# for test_number in range(0,1):
#     #if test_number == 3:
#         #continue
#     if test_number == 10:
#         continue
#     #os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number))
#     Boulder_Total_Demand[test_number] = Boulder_Indoor[test_number]['demand'] + Boulder_Outdoor[test_number]['demand']
#     Boulder_Total_Demand_Sum[test_number] = pd.DataFrame().assign(Year=Watershed_Res[test_number]['year'],Demand=Boulder_Total_Demand[test_number])
#     Boulder_Total_Demand_Sum[test_number] = Boulder_Total_Demand_Sum[test_number].groupby('Year').sum()
#     Boulder_Total_Demand_Sum[test_number] = Boulder_Total_Demand_Sum[test_number].reset_index()
#     Boulder_Indoor_Shortage[test_number] = Boulder_Indoor[test_number]['shortage']
#     Boulder_Outdoor_Shortage[test_number] = Boulder_Outdoor[test_number]['shortage']
#     Boulder_Total_Shortage[test_number] = Boulder_Indoor_Shortage[test_number] + Boulder_Outdoor_Shortage[test_number]
#     Boulder_Total_Shortage_Sum[test_number] = pd.DataFrame().assign(Year=Watershed_Res[test_number]['year'],Shortage=Boulder_Total_Shortage[test_number])
#     Boulder_Total_Shortage_Sum[test_number] = Boulder_Total_Shortage_Sum[test_number].groupby('Year').sum()
#     boulder_supplies = ['WatershedRes','BarkerRes','BoulderRes']
#     Boulder_Water_Supplies[test_number] = pd.DataFrame().assign(Year=Watershed_Res[test_number]['year'],Account=Watershed_Res[test_number]['account'],WatershedRes=Watershed_Res[test_number]['totalrelease'], BarkerRes=Barker_Res[test_number]['totalrelease'],BoulderRes=Boulder_Res[test_number]['totalrelease'])
#     Boulder_Water_Supplies_Sum[test_number] = Boulder_Water_Supplies[test_number].groupby(["Account", "Year"])[boulder_supplies].sum()
#     Boulder_Water_Supplies_Sum[test_number] = Boulder_Water_Supplies_Sum[test_number].reset_index()
#     Boulder_Municipal_Water[test_number] = Boulder_Water_Supplies_Sum[test_number].loc[Boulder_Water_Supplies_Sum[test_number]['Account'] == 1]
    
#     Boulder_Res_Intake_Sum[test_number] = Boulder_Res_Intake[test_number].groupby('year').sum()
#     Boulder_Barker_Pipeline_Sum[test_number] = Boulder_Barker_Pipeline[test_number].groupby('year').sum()
#     Boulder_City_Pipeline_Sum[test_number] = Boulder_City_Pipeline[test_number].groupby('year').sum()
    
#     Boulder_Additional_Leasing[test_number] = pd.DataFrame().assign(Year=Baseline_Res[test_number]['year'], Account=Baseline_Res[test_number]['account'], Leases=Baseline_Res[test_number]['totalrelease'])
#     boulder_leases = ['Leases']
#     Boulder_Additional_Leasing_Sum[test_number] = Boulder_Additional_Leasing[test_number].groupby(["Account", "Year"])[boulder_leases].sum()
#     Boulder_Additional_Leasing_Sum[test_number] = Boulder_Additional_Leasing_Sum[test_number].reset_index()
#     Boulder_Additional_Leases[test_number] = Boulder_Additional_Leasing_Sum[test_number].loc[Boulder_Additional_Leasing_Sum[test_number]['Account'] == '2']
    
 
#     Boulder_Municipal_Water[test_number]['Leases'] = Boulder_Additional_Leasing_Sum[test_number]['Leases']
    
#     Boulder_Municipal_Water[test_number] = Boulder_Municipal_Water[test_number].reset_index()
#     Boulder_Municipal_Water[test_number]['totaldemand'] = Boulder_Total_Demand_Sum[test_number]['Demand']
    
#     #compare source uses based on demand changes
    
#     #Boulder_Source_Comparison[test_number] = Boulder_Municipal_Water[test_number] - Historical_Boulder_Municipal_Water
    
    
#     # #output
#     # os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number))
    
# Boulder_Res_Intake_Sum[0].to_csv('boulder_res_intake.csv')
# Boulder_Barker_Pipeline_Sum[0].to_csv('boulder_barker_pipeline.csv')
# Boulder_City_Pipeline_Sum[0].to_csv('boulder_city_pipeline.csv')
# Boulder_Additional_Leasing_Sum[0].to_csv('boulder_leases_new.csv')
    
    
#     # #Boulder_Municipal_Water[test_number].to_csv('boulder_supplies'+str(test_number)+'.csv')
#     # Boulder_Total_Demand_Sum[test_number].to_csv('boulder_demands'+str(test_number)+'.csv')
#     # Boulder_Source_Comparison[test_number].to_csv('boulder_source_comparison'+str(test_number)+'.csv')
    
# #LONGMONT
# Longmont_Indoor_Shortage = {}
# Longmont_Outdoor_Shortage = {}
# Longmont_Total_Shortage = {}
# Longmont_Total_Shortage_Sum = {}
# for test_number in range(0,1):
#     #if test_number == 3:
#         #continue
#     if test_number == 10:
#         continue
#     #os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number))
#     Longmont_Indoor_Shortage[test_number] = Longmont_Indoor[test_number]['shortage']
#     Longmont_Outdoor_Shortage[test_number] = Longmont_Outdoor[test_number]['shortage']
#     Longmont_Total_Shortage[test_number] = Longmont_Indoor_Shortage[test_number] + Longmont_Outdoor_Shortage[test_number]
#     Longmont_Total_Shortage_Sum[test_number] = pd.DataFrame().assign(Year=Watershed_Res[test_number]['year'],Shortage=Longmont_Total_Shortage[test_number])
#     Longmont_Total_Shortage_Sum[test_number] = Longmont_Total_Shortage_Sum[test_number].groupby('Year').sum()
    
# #LOVELAND
# Loveland_Indoor_Shortage = {}
# Loveland_Outdoor_Shortage = {}
# Loveland_Total_Shortage = {}
# Loveland_Total_Shortage_Sum = {}
# for test_number in range(0,1):
#     #if test_number == 3:
#         #continue
#     if test_number == 10:
#         continue
#     #os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number))
#     Loveland_Indoor_Shortage[test_number] = Loveland_Indoor[test_number]['shortage']
#     Loveland_Outdoor_Shortage[test_number] = Loveland_Outdoor[test_number]['shortage']
#     Loveland_Total_Shortage[test_number] = Loveland_Indoor_Shortage[test_number] + Loveland_Outdoor_Shortage[test_number]
#     Loveland_Total_Shortage_Sum[test_number] = pd.DataFrame().assign(Year=Watershed_Res[test_number]['year'],Shortage=Loveland_Total_Shortage[test_number])
#     Loveland_Total_Shortage_Sum[test_number] = Loveland_Total_Shortage_Sum[test_number].groupby('Year').sum()

# # def shortage(test_number):
# #     fig = plt.figure(figsize = (12,8))
# #     plt.hist(Boulder_Total_Shortage_Sum[test_number]['Shortage'], color='blue', label='Boulder')
# #     plt.hist(Longmont_Total_Shortage_Sum[test_number]['Shortage'], color='green', label='Longmont')
# #     plt.hist(Loveland_Total_Shortage_Sum[test_number]['Shortage'], color='orange', label='Loveland')
# #     # sns.kdeplot(Boulder_Total_Shortage_Sum[test_number]['Shortage'], color='blue', shade=True, label='Boulder')
# #     # sns.kdeplot(Longmont_Total_Shortage_Sum[test_number]['Shortage'], color='green', shade=True, label='Longmont')
# #     # sns.kdeplot(Loveland_Total_Shortage_Sum[test_number]['Shortage'], color='orange', shade=True, label='Loveland')
# #     plt.xlabel('Shortage (AF)')
# #     plt.ylabel('Count')
# #     plt.legend()

# plt.figure()
# plt.plot(Boulder_Total_Shortage_Sum[0]['Shortage'],color='blue', label='Boulder')
# plt.plot(Loveland_Total_Shortage_Sum[0]['Shortage'],color='green', label='Loveland')
# plt.plot(Longmont_Total_Shortage_Sum[0]['Shortage'],color='red', label='Longmont')
# plt.ylabel('Shortage (AF)')
# plt.legend()



# AverageB = {}
# AverageLong = {}
# AverageLove = {}

    
# # for test_number in range(0,30):
# #     os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/output'+str(test_number))
# #     #if test_number == 3:
# #         #continue
# #     if test_number == 10:
# #         continue
# #     shortage(test_number)
# #     plt.savefig('Shortage_Density_Plot'+str(test_number)+'.png')

# #     AverageB[test_number] = Boulder_Total_Shortage_Sum[test_number]['Shortage'].mean()
# #     AverageLong[test_number] = Longmont_Total_Shortage_Sum[test_number]['Shortage'].mean()
# #     AverageLove[test_number] = Loveland_Total_Shortage_Sum[test_number]['Shortage'].mean()
# os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet')
# demands_of_interest = ["04_Lovelnd_I","04_Lovelnd_O","05LONG_IN","05LONG_OUT","06BOULDER_I","06BOULDER_O","08_Denver_I","08_Denver_O","08_Aurora_I","08_Aurora_O","08_Englwd_I","08_Englwd_O","02_NGlenn_I","02_NGlenn_O","02_Westy_I","02_Westy_O","02_Thorn_I","02_Thorn_O","06LAFFYT_I","06LAFFYT_O","06LOUIS_I","06LOUIS_O","07_Arvada_I","07_Arvada_O","07_ConMut_I","07_ConMut_O","07_Golden_I","07_Golden_O"]
# municipalities_demands = {}
# for i in demands_of_interest:
#         municipalities_demands[i]= pd.read_parquet([i]+'.parquet', engine='pyarrow')
        
        
    
    
