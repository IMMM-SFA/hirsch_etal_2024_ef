# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 07:51:36 2022

@author: zacha
"""

# import neccessary python packages

import os
import shutil
#import statemod_data_extraction
import sys
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import csv
import ddm_extraction_timeseries as ddm
import crss_ddm_reader_cm2015b_test as crssddmt
import sprb_transbasin_distribution_test as tb
import uppercotransbasinexports as ucrb

########################################################################################################################################

# update statemod operations file to only maintain the most recent operating rules

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test')

import crss_operations_reader as crsso

print('updating statemod operations file')

operational_rights_data = crsso.read_text_file('SP2016.opr')
crsso.change_operational_rights(operational_rights_data, 'SP2016_A', 1949)

print('statemod operations file updated')

shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/SP2016_A.opr', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP2016_A.opr')
########################################################################################################################################

# update statemod demands file to include synthetic demands based on 1993-2012 average indoor/outdoor water use

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test')

print('creating synthetic demands for front range municipalities')


crssddmt.writenewDDM(crssddmt.demand_data, crssddmt.demands_of_interest, ddm.synthetic_demands, ddm.historical_means, ucrb.adams_tunnel_sp_imports, tb.transbasin_means, tb.Adapted_0400692_X, tb.Adapted_05_BRCBT, tb.infrastructure_allocations, ucrb.roberts_tunnel_sp_imports, ucrb.moffat_tunnel_sp_imports, crssddmt.start_year, 'Q')

print('successful synthetic demand creation')

########################################################################################################################################

# update statemod rsp file to include newly generated _A .ddm and .opr files

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test')

print('updating statemod rsp file')

def update_rsp_file():
  with open('SP2016_H.RSP','r') as f:
    split = [x for x in f.readlines()]       
  f.close()
  f = open('SP2016_H.RSP','w')
  for i in range(0, len(split)):
    if i == 28:
      f.write('Diversion_Demand_Monthly                = SP2016_A.ddm\n')
    elif i == 77:
      f.write('Operational_Right                       = SP2016_A.opr\n')
    else:
      f.write(split[i])
  f.close()

  return split

update_rsp_file()

print('updated statemod rsp file')

########################################################################################################################################

# run statemod (hopefully)
os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test')
print('run statemod')
os.system("StateMod_Model_15.exe SP2016_H -simulate")
print('successful StateMod run')
print('successful termination :)')

########################################################################################################################################

# modify statemod output names for use with statemodify data extractor

shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/SP2016_H.xdd', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/sp2016_H_S0_1.xdd')
shutil.copy('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/SP2016_H.xre', 'C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/sp2016_H_S0_1.xre')

# use statemodify package to extract statemod outputs to parquet files

print('extracting StateMod .xdd data to Parquet')
subprocess.run(["python", "statemod_data_extraction.py", "--ids", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt","--output", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet", "sp2016_H_S0_1.xdd"])
print('extracting StateMod .xre data to Parquet')
subprocess.run(["python", "statemod_data_extraction_xre_final.py", "--ids", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/ids_file.txt","--output", "C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xreparquet", "sp2016_H_S0_1.xre"])
print('successful extraction')

########################################################################################################################################

# convert municipal demand parquets to pandas dataframes

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet')

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
 
plt.figure()
plt.plot(Boulder_Total_Shortage_Sum['Shortage'],color='blue', label='Boulder')
plt.plot(Loveland_Total_Shortage_Sum['Shortage'],color='green', label='Loveland')
plt.plot(Longmont_Total_Shortage_Sum['Shortage'],color='red', label='Longmont')
plt.plot(Thornton_Total_Shortage_Sum['Shortage'],color='yellow', label='Thornton')
plt.plot(Westminster_Total_Shortage_Sum['Shortage'],color='black', label='Westminster')
plt.plot(Laffayette_Total_Shortage_Sum['Shortage'],color='brown', label='Laffayette')
plt.plot(Louisville_Total_Shortage_Sum['Shortage'],color='orange', label='Louisville')
plt.plot(Arvada_Total_Shortage_Sum['Shortage'],color='purple', label='Arvada')
plt.plot(ConMut_Total_Shortage_Sum['Shortage'],color='pink', label='ConMut')
plt.plot(Golden_Total_Shortage_Sum['Shortage'],color='gray', label='Golden')
#plt.plot(Denver_Total_Shortage_Sum['Shortage'],color='gold', label='Denver')
plt.plot(Englewood_Total_Shortage_Sum['Shortage'],color='aqua', label='Englewood')

plt.ylabel('Shortage (AF)')
plt.legend()   

plt.figure()
plt.plot(Denver_Total_Shortage_Sum['Shortage'],color='gold', label='Denver')
plt.ylabel('Shortage (AF)')
plt.legend() 

########################################################################################################################################

# 