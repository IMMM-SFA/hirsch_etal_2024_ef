# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 23:01:53 2022

@author: zacha
"""



import os
# import re
import numpy as np
import pandas as pd
# from SALib.sample import latin
# from joblib import Parallel, delayed
# import re
import matplotlib.pyplot as plt
# import statsmodels.formula.api as sm
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from matplotlib.colors import ListedColormap
# import scipy.stats as stats
# import seaborn as sns
# from datetime import datetime
# from matplotlib.lines import Line2D
# import os
import uppercotransbasinexports as ucrb

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline/parquet/')

#C-BT DISTRIBUTION NODES
Historical_0404634 = pd.read_parquet('0404634.parquet', engine='pyarrow')
Historical_CBT_AllPln = pd.read_parquet('CBT_AllPln.parquet', engine='pyarrow')
Historical_AdamsTunPln  = pd.read_parquet('AdamsTunPln.parquet', engine='pyarrow')
Historical_0401000 = pd.read_parquet('0401000.parquet', engine='pyarrow')
Historical_LoveCBTPln = pd.read_parquet('LoveCBTPln.parquet', engine='pyarrow')
Historical_06_SWSP_IMP = pd.read_parquet('06_SWSP_IMP.parquet', engine='pyarrow')
Historical_0400691 = pd.read_parquet('0400691.parquet', engine='pyarrow')
Historical_0400691_X = pd.read_parquet('0400691_X.parquet', engine='pyarrow')
Historical_05_LongCBT = pd.read_parquet('05_LongCBT.parquet', engine='pyarrow')
Historical_0400692 = pd.read_parquet('0400692.parquet', engine='pyarrow')
Historical_0400692_X = pd.read_parquet('0400692_X.parquet', engine='pyarrow')
Historical_0404513 = pd.read_parquet('0404513+.parquet', engine='pyarrow')
Historical_05_SVCBT = pd.read_parquet('05_SVCBT.parquet', engine='pyarrow')
Historical_05_BRCBT = pd.read_parquet('05_BRCBT.parquet', engine='pyarrow')
Historical_0600800_SV = pd.read_parquet('0600800_SV.parquet', engine='pyarrow')
Historical_BCSC = pd.read_parquet('BCSC.parquet', engine='pyarrow')
Historical_0504515 = pd.read_parquet('0504515+.parquet', engine='pyarrow')
Historical_06_WSHED = pd.read_parquet('06_WSHED+.parquet', engine='pyarrow')
Historical_0604172 = pd.read_parquet('0604172+.parquet', engine='pyarrow')
Historical_0604173 = pd.read_parquet('0604173+.parquet', engine='pyarrow')
Historical_060800_IMP = pd.read_parquet('060800_IMP.parquet', engine='pyarrow')
Historical_06_CBT_IMP = pd.read_parquet('06_CBT_IMP.parquet', engine='pyarrow')
Historical_0400691_I = pd.read_parquet('0400691_I.parquet', engine='pyarrow')
Historical_0400692_I = pd.read_parquet('0400692_I.parquet', engine='pyarrow')
Historical_0400518_I = pd.read_parquet('0400518_I.parquet', engine='pyarrow')
Historical_0400518_O = pd.read_parquet('0400518_O.parquet', engine='pyarrow')
Historical_0404128 = pd.read_parquet('0404128.parquet', engine='pyarrow')
Historical_0401001 = pd.read_parquet('0401001.parquet', engine='pyarrow')
Historical_0401002 = pd.read_parquet('0401002.parquet', engine='pyarrow')
Historical_0400702 = pd.read_parquet('0400702.parquet', engine='pyarrow')
Historical_0400503 = pd.read_parquet('0400503.parquet', engine='pyarrow')
Historical_0400520_I = pd.read_parquet('0400520_I.parquet', engine='pyarrow')
Historical_0400521_I = pd.read_parquet('0400521_I.parquet', engine='pyarrow')
Historical_0400523 = pd.read_parquet('0400523.parquet', engine='pyarrow')
Historical_0400524_I = pd.read_parquet('0400524_I.parquet', engine='pyarrow')
Historical_0400530_I = pd.read_parquet('0400530_I.parquet', engine='pyarrow')
Historical_0400532_I = pd.read_parquet('0400532_I.parquet', engine='pyarrow')
Historical_0400543_I = pd.read_parquet('0400543_I.parquet', engine='pyarrow')
Historical_05_LHCBT = pd.read_parquet('05_LHCBT.parquet', engine='pyarrow')
Historical_0404146RS = pd.read_parquet('0404146RS.parquet', engine='pyarrow')


Olympus_Tunnel = Historical_0401000.groupby('year').sum()['carried']
Adams_Tunnel = Historical_0404634.groupby('year').sum()['carried']

Historical_0400692_X_Yearly = Historical_0400692_X.groupby('year').sum()

### CHECK ###

Check_1 = Adams_Tunnel - Olympus_Tunnel

StVrain_Yearly_Exports = Historical_0400692_X.groupby('year').sum()['demand']




#DENVER DISTRIBUTION NODES
Historical_06_MOF_IMP = pd.read_parquet('06_MOF_IMP.parquet', engine='pyarrow')
Historical_MoffatWTP = pd.read_parquet('MoffatWTP.parquet', engine='pyarrow')
Historical_0604199 = pd.read_parquet('0604199+.parquet', engine='pyarrow')
Historical_0703324 = pd.read_parquet('0703324+.parquet', engine='pyarrow')
Historical_8000653 = pd.read_parquet('8000653.parquet', engine='pyarrow')
Historical_0903501 = pd.read_parquet('0903501+.parquet', engine='pyarrow')
Historical_0803983 = pd.read_parquet('0803983+.parquet', engine='pyarrow')

#AURORA DISTRIBUTION NODES
Historical_HOMSPICO = pd.read_parquet('HOMSPICO.parquet', engine='pyarrow')
#Historical_2304013 = pd.read_parquet('2304013.parquet', engine='pyarrow')

#ENGLEWOOD DISTRIBUTION NODES
Historical_2304611 = pd.read_parquet('2304611.parquet', engine='pyarrow')
Historical_0803832 = pd.read_parquet('0803832+.parquet', engine='pyarrow')

#GOLDEN DISTRIBUTION NODES
Historical_0704625 = pd.read_parquet('0704625.parquet', engine='pyarrow')
Historical_0704626 = pd.read_parquet('0704626.parquet', engine='pyarrow')
#Historical_0704630 = pd.read_parquet('0704630.parquet', engine='pyarrow')


#C-BT

Adams_Tunnel_Yearly_Imports = Historical_0404634.groupby('year').sum()['carried']



#PLAN DATA

BRCBT_Pln = pd.read_parquet('05_BRCBT_Pln+.parquet', engine='pyarrow')
LHCBT_Pln = pd.read_parquet('05_LHCBT_Pln+.parquet', engine='pyarrow')
SVCBT_Pln = pd.read_parquet('05_SVCBT_Pln+.parquet', engine='pyarrow')
CBT_SP1_Pln = pd.read_parquet('06_CBT_SP1+.parquet', engine='pyarrow')
CBT_SP2_Pln = pd.read_parquet('06_CBT_SP2+.parquet', engine='pyarrow')
CBT_ACC = pd.read_parquet('06_CBT_ACC+.parquet', engine='pyarrow')
MOF_ACC = pd.read_parquet('06_MOF_ACC+.parquet', engine='pyarrow')
Boulder_ACC = pd.read_parquet('060800_ACC+.parquet', engine='pyarrow')
Berthoud_C = pd.read_parquet('Berthoud_C+.parquet', engine='pyarrow')
Boreas_C = pd.read_parquet('Boreas_C+.parquet', engine='pyarrow')
Homestk_C = pd.read_parquet('Homestk_C+.parquet', engine='pyarrow')
LongCBT_Pln = pd.read_parquet('LongCBT_Pln+.parquet', engine='pyarrow')
RobTun_C = pd.read_parquet('RobTun_C+.parquet', engine='pyarrow')
Vidler_C = pd.read_parquet('Vidler_C+.parquet', engine='pyarrow')

CBTAllPln = pd.read_parquet('CBT_AllPln+.parquet', engine='pyarrow')
CBTAllPlnYearly= CBTAllPln.groupby('year').sum()

plt.figure()
plt.plot(CBTAllPlnYearly['totalsupply'])
plt.title('Adams Tunnel Imports')

AdamsTunPln = pd.read_parquet('AdamsTunPln+.parquet', engine='pyarrow')
AdamsTunPlnYearly= AdamsTunPln.groupby('year').sum()

LoveCBTPln = pd.read_parquet('LoveCBTPln+.parquet', engine='pyarrow')
LoveCBTPlnYearly= LoveCBTPln.groupby('year').sum()

AdamsTunPln_uses = AdamsTunPlnYearly.iloc[:,0:11]
LoveCBTPln_uses = LoveCBTPlnYearly.iloc[:,0:11]


BoulderYearly = BRCBT_Pln.groupby('year').sum()
BoulderYearlyAllocations = pd.DataFrame()
BoulderYearlyAllocations['use1'] = BoulderYearly['use1']/BoulderYearly['totalsupply']
BoulderYearlyAllocations['use2'] = BoulderYearly['use2']/BoulderYearly['totalsupply']
BoulderYearlyAllocations['use3'] = BoulderYearly['use3']/BoulderYearly['totalsupply']
BoulderYearlyAllocations['use4'] = BoulderYearly['use4']/BoulderYearly['totalsupply']


TransbasinUsesYearly = AdamsTunPln_uses + LoveCBTPln_uses

TransbasinUsesYearlyAllocations = pd.DataFrame()
TransbasinUsesYearlyAllocations['use1'] = TransbasinUsesYearly['use1']/TransbasinUsesYearly['totalsupply']
TransbasinUsesYearlyAllocations['use2'] = TransbasinUsesYearly['use2']/TransbasinUsesYearly['totalsupply']
TransbasinUsesYearlyAllocations['use3'] = TransbasinUsesYearly['use3']/TransbasinUsesYearly['totalsupply']
TransbasinUsesYearlyAllocations['use4'] = TransbasinUsesYearly['use4']/TransbasinUsesYearly['totalsupply']
TransbasinUsesYearlyAllocations['use5'] = TransbasinUsesYearly['use5']/TransbasinUsesYearly['totalsupply']
TransbasinUsesYearlyAllocations['use6'] = TransbasinUsesYearly['use6']/TransbasinUsesYearly['totalsupply']
TransbasinUsesYearlyAllocations['use7'] = TransbasinUsesYearly['use7']/TransbasinUsesYearly['totalsupply']
TransbasinUsesYearlyAllocations['use8'] = TransbasinUsesYearly['use8']/TransbasinUsesYearly['totalsupply']
TransbasinUsesYearlyAllocations['use9'] = TransbasinUsesYearly['use9']/TransbasinUsesYearly['totalsupply']
TransbasinUsesYearlyAllocations['use10'] = TransbasinUsesYearly['use10']/TransbasinUsesYearly['totalsupply']

HorsetoothExports = TransbasinUsesYearlyAllocations['use2'] 
CarterExports = TransbasinUsesYearlyAllocations['use4']

plt.figure()
plt.plot(HorsetoothExports)
plt.title('Hansen Feeder Canal C-BT Allocation')

Horsetooth_Mean = HorsetoothExports.mean()

plt.figure()
plt.plot(CarterExports)
plt.title('Carter Reservoir C-BT Allocation')

plt.figure()
#plt.plot(Carter_Yearly_Imports_1['fromcarrierother'])
plt.plot(TransbasinUsesYearly['use4'])

Transbasin_to_Horsetooth = TransbasinUsesYearlyAllocations.iloc[53:63, 1]
Transbasin_to_Horsetooth = Transbasin_to_Horsetooth.reset_index()
plt.figure()
plt.plot(Transbasin_to_Horsetooth)

Transbasin_to_Carter = TransbasinUsesYearlyAllocations.iloc[53:63, 3]
Transbasin_to_Carter = Transbasin_to_Carter.reset_index()
plt.figure()
plt.plot(Transbasin_to_Carter)

transbasin_means = pd.DataFrame()
transbasin_means['index'] = range(0,1)
transbasin_means.index = transbasin_means['index']
transbasin_means['0400691_X'] = np.mean(Transbasin_to_Horsetooth['use2'])
transbasin_means['0400692_X'] = np.mean(Transbasin_to_Carter['use4'])

Adapted_0400692_X = pd.DataFrame()
Adapted_0400692_X['carried'] = ucrb.adams_tunnel_sp_imports['column']*transbasin_means.loc[0,'0400692_X']
Adapted_0400692_X['index'] = range(0,756)

Historical_05_BRCBT_Yearly = Historical_05_BRCBT.groupby('year').sum()
BRCBT_Multipliers = Historical_05_BRCBT_Yearly['carried']/Historical_0400692_X_Yearly['demand']

Historical_05_LongCBT_Yearly = Historical_05_LongCBT.groupby('year').sum()
LongCBT_Multipliers = Historical_05_LongCBT_Yearly['carried']/Historical_0400692_X_Yearly['demand']
LongCBT_Multiplier = LongCBT_Multipliers.iloc[53:63].mean()

Historical_05LHCBT_Yearly = Historical_05_LHCBT.groupby('year').sum()
LHCBT_Multipliers = Historical_05LHCBT_Yearly['carried']/Historical_0400692_X_Yearly['demand']
LHCBT_Multiplier = LHCBT_Multipliers.iloc[53:63].mean()

Historical_05SVCBT_Yearly = Historical_05_SVCBT.groupby('year').sum()
SVCBT_Multipliers = Historical_05SVCBT_Yearly['carried']/Historical_0400692_X_Yearly['demand']
SVCBT_Multiplier = SVCBT_Multipliers.iloc[53:63].mean()

Historical_06_SWSP_IMP_Yearly = Historical_06_SWSP_IMP.groupby('year').sum()
SWSP_Multipliers = Historical_06_SWSP_IMP_Yearly['carried']/Historical_0400692_X_Yearly['demand']
SWSP_Multiplier = SWSP_Multipliers.iloc[53:63].mean()

BRCBT_Multiplier = 1 - LongCBT_Multiplier - LHCBT_Multiplier - SVCBT_Multiplier - SWSP_Multiplier
#BRCBT_Multiplier = .55
## ^ was .55
Adapted_05_BRCBT = pd.DataFrame()
Adapted_05_BRCBT['carried'] = Adapted_0400692_X['carried']*BRCBT_Multiplier
Adapted_05_BRCBT['index'] = range(0,756)

Historical_060800_IMP_Yearly = Historical_060800_IMP.groupby('year').sum()
SV060800_IMP_Multipliers = Historical_060800_IMP_Yearly['carried']/Historical_05_BRCBT_Yearly['carried']
SV060800_IMP_Multiplier = SV060800_IMP_Multipliers.iloc[53:63].mean()

Historical_06_CBT_IMP_Yearly = Historical_06_CBT_IMP.groupby('year').sum()
SV06_CBT_IMP_Multipliers = Historical_06_CBT_IMP_Yearly['carried']/Historical_05_BRCBT_Yearly['carried']
SV06_CBT_IMP_Multiplier = 1-SV060800_IMP_Multiplier

infrastructure_allocations = pd.DataFrame()
infrastructure_allocations['index'] = range(0,1)
infrastructure_allocations.index = infrastructure_allocations['index']
infrastructure_allocations['05_LongCBT'] = LongCBT_Multiplier*-1
infrastructure_allocations['05_BRCBT'] = -1
infrastructure_allocations['05_LHCBT'] = LHCBT_Multiplier*-1
infrastructure_allocations['05_SVCBT'] = SVCBT_Multiplier*-1
infrastructure_allocations['060800_IMP'] = SV060800_IMP_Multiplier*-1
infrastructure_allocations['0600800_SV'] = SV060800_IMP_Multiplier
infrastructure_allocations['06_CBT_IMP'] = SV06_CBT_IMP_Multiplier*-1
infrastructure_allocations['BCSC'] = SV06_CBT_IMP_Multiplier
infrastructure_allocations['06_SWSP_IMP'] = SWSP_Multiplier*-1
infrastructure_allocations['MoffatWTP'] = 1



# infrastructure_allocations = pd.DataFrame()
# infrastructure_allocations['index'] = range(0,1)
# infrastructure_allocations.index = infrastructure_allocations['index']
# infrastructure_allocations['05_LongCBT'] = .1*-1
# infrastructure_allocations['05_BRCBT'] = 1*-1
# infrastructure_allocations['05_LHCBT'] = .05*-1
# infrastructure_allocations['05_SVCBT'] = .2*-1
# infrastructure_allocations['060800_IMP'] = .25*-1
# infrastructure_allocations['0600800_SV'] = .25
# infrastructure_allocations['06_CBT_IMP'] = .75*-1
# infrastructure_allocations['BCSC'] = .75
# infrastructure_allocations['06_SWSP_IMP'] = .1*-1
# infrastructure_allocations['MoffatWTP'] = 1

BRCBT_Yearly = Historical_05_BRCBT.groupby('year').sum()
Historical_060800_IMP_Yearly = Historical_060800_IMP.groupby('year').sum()
Historical_06_CBT_IMP_Yearly = Historical_06_CBT_IMP.groupby('year').sum()
BWTP_Comp = Historical_060800_IMP_Yearly['carried']/BRCBT_Yearly['carried']
CBTIMP_Comp = Historical_06_CBT_IMP_Yearly['carried']/BRCBT_Yearly['carried']




AdamsTunPlnYearlyAllocations = pd.DataFrame()
AdamsTunPlnYearlyAllocations['use1'] = AdamsTunPlnYearly['use1']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use2'] = AdamsTunPlnYearly['use2']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use3'] = AdamsTunPlnYearly['use3']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use4'] = AdamsTunPlnYearly['use4']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use5'] = AdamsTunPlnYearly['use5']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use6'] = AdamsTunPlnYearly['use6']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use7'] = AdamsTunPlnYearly['use7']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use8'] = AdamsTunPlnYearly['use8']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use9'] = AdamsTunPlnYearly['use9']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use10'] = AdamsTunPlnYearly['use10']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use11'] = AdamsTunPlnYearly['use11']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use12'] = AdamsTunPlnYearly['use12']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use13'] = AdamsTunPlnYearly['use13']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use14'] = AdamsTunPlnYearly['use14']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use15'] = AdamsTunPlnYearly['use15']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use16'] = AdamsTunPlnYearly['use16']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use17'] = AdamsTunPlnYearly['use17']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use18'] = AdamsTunPlnYearly['use18']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use19'] = AdamsTunPlnYearly['use19']/AdamsTunPlnYearly['totalsupply']
AdamsTunPlnYearlyAllocations['use20'] = AdamsTunPlnYearly['use20']/AdamsTunPlnYearly['totalsupply']


LoveCBTPln = pd.read_parquet('LoveCBTPln+.parquet', engine='pyarrow')
LoveCBTPlnYearly= LoveCBTPln.groupby('year').sum()
LoveCBTPlnYearlyAllocations = pd.DataFrame()
LoveCBTPlnYearlyAllocations['use1'] = LoveCBTPlnYearly['use1']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use2'] = LoveCBTPlnYearly['use2']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use3'] = LoveCBTPlnYearly['use3']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use4'] = LoveCBTPlnYearly['use4']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use5'] = LoveCBTPlnYearly['use5']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use6'] = LoveCBTPlnYearly['use6']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use7'] = LoveCBTPlnYearly['use7']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use8'] = LoveCBTPlnYearly['use8']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use9'] = LoveCBTPlnYearly['use9']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use10'] = LoveCBTPlnYearly['use10']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use11'] = LoveCBTPlnYearly['use11']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use12'] = LoveCBTPlnYearly['use12']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use13'] = LoveCBTPlnYearly['use13']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use14'] = LoveCBTPlnYearly['use14']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use15'] = LoveCBTPlnYearly['use15']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use16'] = LoveCBTPlnYearly['use16']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use17'] = LoveCBTPlnYearly['use17']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use18'] = LoveCBTPlnYearly['use18']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use19'] = LoveCBTPlnYearly['use19']/LoveCBTPlnYearly['totalsupply']
LoveCBTPlnYearlyAllocations['use20'] = LoveCBTPlnYearly['use20']/LoveCBTPlnYearly['totalsupply']

#Olympus Tunnel

OlympusTun = Historical_0401000.groupby('year').sum()['carried']
OlympusCheck = OlympusTun - Adams_Tunnel_Yearly_Imports


#0400691 check

Hansen_Feeder_Yearly = Historical_0400691.groupby('year').sum()['carried']
Horsetooth_DDM = Historical_0400691_X.groupby('year').sum()['demand']
Horsetooth_Irr_DDM = Historical_0400691_I.groupby('year').sum()['demand']

HansenHorseCheck = Hansen_Feeder_Yearly - Horsetooth_DDM - Horsetooth_Irr_DDM

#0400692 check

St_V_Yearly = Historical_0400692.groupby('year').sum()['carried']
St_V_DDM = Historical_0400692_X.groupby('year').sum()['demand']
St_V_Irr_DDM = Historical_0400692_I.groupby('year').sum()['demand']

HansenHorseCheck = Hansen_Feeder_Yearly - Horsetooth_DDM - Horsetooth_Irr_DDM

#Carter Check

Carter_Yearly_Releases = Historical_0404513.groupby(['year','account']).sum()['totalrelease']
Carter_Yearly_Releases = Carter_Yearly_Releases.reset_index()
Carter_Yearly_Releases_1 = Carter_Yearly_Releases.loc[Carter_Yearly_Releases['account'] == 1]
Carter_Yearly_Releases_1.index = Carter_Yearly_Releases_1['year']

Carter_Release1 = Historical_0400692.groupby('year').sum()['carried']
Carter_Release1_Comp = Carter_Release1/Carter_Yearly_Releases_1['totalrelease']
Carter_Release2 = Historical_0400692_I.groupby('year').sum()['demand']
Carter_Release2_Comp = Carter_Release2/Carter_Yearly_Releases_1['totalrelease']
Carter_Release3 = Historical_0404146RS.groupby('year').sum()['other']
Carter_Release3_Comp = Carter_Release3/Carter_Yearly_Releases_1['totalrelease']

Carter_Release_Comp = Carter_Release1_Comp + Carter_Release2_Comp + Carter_Release3_Comp

SWSP_from_Carter = Historical_06_SWSP_IMP.groupby('year').sum()['carried']
STV_from_Carter = Historical_0400692_X.groupby('year').sum()['demand']


CarterReleaseCheck = Carter_Yearly_Releases_1['totalrelease'] - SWSP_from_Carter - STV_from_Carter

plt.figure()
plt.plot(CarterReleaseCheck)


Carter_Total_Imports = Historical_0400692['carried'].cumsum()
Carter_Total_Releases = Carter_Yearly_Releases_1['totalrelease'].cumsum()



LHCBT_Yearly_Exports = Historical_05_LHCBT.groupby('year').sum()['carried']
LHCBT_Yearly_Carter_Comp = LHCBT_Yearly_Exports/Adams_Tunnel_Yearly_Imports

SVCBT_Yearly_Exports = Historical_05_SVCBT.groupby('year').sum()['carried']
SVCBT_Yearly_Carter_Comp = SVCBT_Yearly_Exports/Adams_Tunnel_Yearly_Imports

BRCBT_Yearly_Exports = Historical_05_BRCBT.groupby('year').sum()['carried']
BRCBT_Yearly_Carter_Comp = BRCBT_Yearly_Exports/Adams_Tunnel_Yearly_Imports

LONG_Yearly_Exports = Historical_05_LongCBT.groupby('year').sum()['carried']
LONG_Yearly_Carter_Comp = LONG_Yearly_Exports/Adams_Tunnel_Yearly_Imports

# SWSP_Yearly_Exports = Historical_06_SWSP_IMP.groupby('year').sum()['carried']
# SWSP_Yearly_Carter_Comp = SWSP_Yearly_Exports/Adams_Tunnel_Yearly_Imports

# BOULDER_Yearly_Exports = Historical_060800_IMP.groupby('year').sum()['carried']
# BOULDER_Yearly_Carter_Comp = BOULDER_Yearly_Exports/Adams_Tunnel_Yearly_Imports

BOULDER_CBT_Yearly_Exports = Historical_06_CBT_IMP.groupby('year').sum()['carried']
BOULDER_CBT_Yearly_Carter_Comp = BOULDER_CBT_Yearly_Exports/Adams_Tunnel_Yearly_Imports

Horsetooth_Yearly_Exports = Historical_0400691.groupby('year').sum()['carried']
Horsetooth_Yearly_Carter_Comp = Horsetooth_Yearly_Exports/Adams_Tunnel_Yearly_Imports


CarterComp = LHCBT_Yearly_Carter_Comp + SVCBT_Yearly_Carter_Comp + BRCBT_Yearly_Carter_Comp + LONG_Yearly_Carter_Comp + Horsetooth_Yearly_Carter_Comp

CarterCompNew = CarterComp.loc[1954:2013,]

plt.figure()
plt.plot(StVrain_Yearly_Exports)

plt.figure()
plt.plot(BRCBT_Yearly_Carter_Comp)

plt.figure()
plt.plot(CarterCompNew)

plt.figure()
plt.plot(LHCBT_Yearly_Carter_Comp)
plt.plot(SVCBT_Yearly_Carter_Comp)
plt.plot(BRCBT_Yearly_Carter_Comp)
plt.plot(LONG_Yearly_Carter_Comp)

# MONTHLY

StVrain_Mo_Exports = Historical_0400692['carried']

LHCBT_Mo_Exports = Historical_05_LHCBT['carried']
LHCBT_Mo_Carter_Comp = LHCBT_Mo_Exports/StVrain_Mo_Exports

SVCBT_Mo_Exports = Historical_05_SVCBT['carried']
SVCBT_Mo_Carter_Comp = SVCBT_Mo_Exports/StVrain_Mo_Exports

BRCBT_Mo_Exports = Historical_05_BRCBT['carried']
BRCBT_Mo_Carter_Comp = BRCBT_Mo_Exports/StVrain_Mo_Exports

LONG_Mo_Exports = Historical_05_LongCBT['carried']
LONG_Mo_Carter_Comp = LONG_Mo_Exports/StVrain_Mo_Exports

plt.figure()
plt.plot(LHCBT_Mo_Carter_Comp)
plt.plot(SVCBT_Mo_Carter_Comp)
plt.plot(BRCBT_Mo_Carter_Comp)
plt.plot(LONG_Mo_Carter_Comp)




#First Distribution of Water from Upper CO Basin

AdamsTunPlnYearly = AdamsTunPlnYearly.reset_index()
AdamsTunPlnYearly.index = AdamsTunPlnYearly['year']
LoveCBTPlnYearly = LoveCBTPlnYearly.reset_index()
LoveCBTPlnYearly.index = LoveCBTPlnYearly['year']

HorsetoothExports = AdamsTunPlnYearly['use2'] + LoveCBTPlnYearly['use2'] 


plt.figure()
plt.plot(HorsetoothExports)
plt.title('Hansen Feeder Canal C-BT Allocation')

plt.figure()
plt.plot(CarterExports)
plt.title('Carter Reservoir C-BT Allocation')

CBT_AllPln_Yearly_Exports = Historical_CBT_AllPln.groupby('year').sum()['other']
CBT_AllPln_Yearly_Adams_Comp = CBT_AllPln_Yearly_Exports/Adams_Tunnel_Yearly_Imports
CBT_AllPln_Yearly_Adams_Comp_selection = CBT_AllPln_Yearly_Adams_Comp.iloc[46:63]
CBT_AllPln_Yearly_Adams_Comp_mean = CBT_AllPln_Yearly_Adams_Comp_selection.mean()

AdamsTunPln_Yearly_Exports = Historical_AdamsTunPln.groupby('year').sum()['carried']
AdamsTunPln_Yearly_Adams_Comp = AdamsTunPln_Yearly_Exports/Adams_Tunnel_Yearly_Imports
AdamsTunPln_Yearly_Adams_Comp_selection = AdamsTunPln_Yearly_Adams_Comp.iloc[46:63]
AdamsTunPln_Yearly_Adams_Comp_mean = AdamsTunPln_Yearly_Adams_Comp_selection.mean()

LoveCBTPln_Yearly_Exports = Historical_LoveCBTPln.groupby('year').sum()['carried']
LoveCBTPln_Yearly_Adams_Comp = LoveCBTPln_Yearly_Exports/Adams_Tunnel_Yearly_Imports
LoveCBTPln_Yearly_Adams_Comp_selection = LoveCBTPln_Yearly_Adams_Comp.iloc[46:63]
LoveCBTPln_Yearly_Adams_Comp_mean = LoveCBTPln_Yearly_Adams_Comp_selection.mean()

Return_Flow_Spills = Adams_Tunnel_Yearly_Imports - AdamsTunPln_Yearly_Exports - LoveCBTPln_Yearly_Exports
Return_Flow_Spills_Yearly_Adams_Comp = Return_Flow_Spills/AdamsTunPln_Yearly_Exports
Return_Flow_Spills_Yearly_Adams_Comp_selection = Return_Flow_Spills_Yearly_Adams_Comp.iloc[46:63]
Return_Flow_Spills_Yearly_Adams_Comp_mean = Return_Flow_Spills_Yearly_Adams_Comp_selection.mean()
plt.figure()
plt.plot(Return_Flow_Spills)


#Plan_Accounting_1 = Adams_Tunnel_Yearly_Imports - AdamsTunPln_Yearly_Exports - LoveCBTPln_Yearly_Exports - Return_Flow_Spills

#Allocate to Horsetooth first
Horsetooth_Yearly_Exports = Historical_0400691.groupby('year').sum()['carried']
Horsetooth_Yearly_Adams_Comp = Horsetooth_Yearly_Exports/AdamsTunPln_Yearly_Exports
Horsetooth_Yearly_Adams_Comp_selection = Horsetooth_Yearly_Adams_Comp.iloc[46:63]
Horsetooth_Yearly_Adams_Comp_mean = Horsetooth_Yearly_Adams_Comp_selection.mean()

plt.figure()
plt.plot(Horsetooth_Yearly_Adams_Comp)

Horsetooth_Check = HorsetoothExports - Horsetooth_Yearly_Exports


#Distribute to Carter

Carter_Yearly_Imports = Historical_0404513.groupby(['year','account']).sum()['fromcarrierother']
Carter_Yearly_Imports = Carter_Yearly_Imports.reset_index()
Carter_Yearly_Imports_1 = Carter_Yearly_Imports.loc[Carter_Yearly_Imports['account'] == 1]
Carter_Yearly_Imports_1.index = Carter_Yearly_Imports_1['year']

Carter_Yearly_Adams_Comp = Carter_Yearly_Imports_1['fromcarrierother']/AdamsTunPln_Yearly_Exports
Carter_Yearly_Adams_Comp_selection = Carter_Yearly_Adams_Comp.iloc[46:63]
Carter_Yearly_Adams_Comp_mean = Carter_Yearly_Adams_Comp_selection.mean()

#Carter Releases

Carter_Yearly_Releases = Historical_0404513.groupby(['year','account']).sum()['totalrelease']
Carter_Yearly_Releases = Carter_Yearly_Releases.reset_index()
Carter_Yearly_Releases_1 = Carter_Yearly_Releases.loc[Carter_Yearly_Releases['account'] == 1]
Carter_Yearly_Releases_1.index = Carter_Yearly_Releases_1['year']



#First node distributions

SWSP_IMP_Yearly_Exports = Historical_06_SWSP_IMP.groupby('year').sum()['carried']
SWSP_IMP_Yearly_Adams_Comp = SWSP_IMP_Yearly_Exports/AdamsTunPln_Yearly_Exports
SWSP_IMP_Yearly_Adams_Comp_selection = SWSP_IMP_Yearly_Adams_Comp.iloc[46:63]
SWSP_IMP_Yearly_Adams_Comp_mean = SWSP_IMP_Yearly_Adams_Comp_selection.mean()


LongmontCBT_Yearly_Exports = Historical_05_LongCBT.groupby('year').sum()['carried']
LongmontCBT_Yearly_Adams_Comp = LongmontCBT_Yearly_Exports/AdamsTunPln_Yearly_Exports
LongmontCBT_Yearly_Adams_Comp_selection = LongmontCBT_Yearly_Adams_Comp.iloc[46:63]
LongmontCBT_Yearly_Adams_Comp_mean = LongmontCBT_Yearly_Adams_Comp_selection.mean()

Comp_Check = Horsetooth_Yearly_Adams_Comp + Carter_Yearly_Adams_Comp + LongmontCBT_Yearly_Adams_Comp


#Second node distribution to St. Vrain via Carter

StVrain_Yearly_Exports = Historical_0400692_X.groupby('year').sum()['demand']
StVrain_Yearly_Adams_Comp = StVrain_Yearly_Exports/Carter_Yearly_Releases_1['totalrelease']
StVrain_Yearly_Adams_Comp_selection = StVrain_Yearly_Adams_Comp.iloc[46:63]
StVrain_Yearly_Adams_Comp_mean = StVrain_Yearly_Adams_Comp_selection.mean()





Carter_Yearly_Releases = Historical_0404513.groupby(['year','account']).sum()['totalrelease']
Carter_Yearly_Releases = Carter_Yearly_Releases.reset_index()
Carter_Yearly_Releases_1 = Carter_Yearly_Releases.loc[Carter_Yearly_Releases['account'] == 1]
Carter_Yearly_Releases_1.index = Carter_Yearly_Releases_1['year']


#Carter releases as a fn of StVrain demand

Carter_Excess_Supply = Carter_Yearly_Releases_1['totalrelease'] - StVrain_Yearly_Exports
Carter_StVrain_Comp = StVrain_Yearly_Exports/Carter_Yearly_Releases_1['totalrelease']

Carter_Yearly_Imports = Historical_0404513.groupby(['year','account']).sum()['fromcarrierother']
Carter_Yearly_Imports = Carter_Yearly_Imports.reset_index()
Carter_Yearly_Imports_1 = Carter_Yearly_Imports.loc[Carter_Yearly_Imports['account'] == 1]
Carter_Yearly_Imports_1.index = Carter_Yearly_Imports_1['year']

# Carter_Yearly_Storage = Carter_Yearly_Imports_1['fromcarrierother']-Carter_Yearly_Releases_1['totalrelease']
# for i in range(1950,2013):
#     if Carter_Yearly_Storage[i] <= 0:
#         Carter_Yearly_Storage[i] == 0
# Carter_Yearly_Adams_Comp = Carter_Yearly_Storage/Adams_Tunnel_Yearly_Imports

Carter_Downstream_Comp_Check = StVrain_Yearly_Exports - Carter_Yearly_Releases_1['totalrelease']

Carter_ImportExport = Carter_Yearly_Imports_1['fromcarrierother'] - Carter_Yearly_Releases_1['totalrelease']

plt.figure()
plt.plot(Carter_ImportExport)


Total_First_Dist_Exports =  Horsetooth_Yearly_Exports + StVrain_Yearly_Exports + Carter_Yearly_Imports_1['fromcarrierother']
Import_Export_Check_1 = Adams_Tunnel_Yearly_Imports - Total_First_Dist_Exports

plt.figure()
plt.plot(Carter_Yearly_Imports_1['fromcarrierother'])
plt.plot(TransbasinUsesYearly['use4'])

plt.figure()
#plt.plot(BCSC_Yearly_Exports)



os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/')