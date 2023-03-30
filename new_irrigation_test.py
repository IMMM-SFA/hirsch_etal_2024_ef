# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:24:53 2023

@author: zacha
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import contextily as cx
from shapely.geometry import Point
import matplotlib.pyplot as plt
import os 
import process_statemod_obc as obc
import statemod_irrigation_structureid_extractor as ids
import extract_ipy_southplatte as ipy

######### start analysis ################

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')

## import shapefile and transform crs ###

fp = 'Div1_Irrig_2010.shp'
map_df = gpd.read_file(fp) 
print(map_df)

#### define function to connect statemod structure id's with irrigator plots, as defined by South Platte StateMod user's manual ##

def assign_irrigation(row):
    if row in  ['0100503', '0100504', '0100710']:
        result = '0100503_I'
    elif row in ['0100506', '0100507', '0100509']:
        result = '0100507_I'
    elif row in ['0100519', '0100521', '0100522', '0100523']:
        result = '0100519_D'
    elif row in ['0100687', '6400563', '6403794']:
        result = '0100687_I'
    elif row in ['0103817']:
        result = '0103817_I'
    elif row in ['0200805', '0200902']:
        result ='0200805_I'
    elif row == '0200808':
        result ='0200808_I'
    elif row == '0200810':
        result ='0200810_I'
    elif row == '0200812':
        result ='0200812_I' 
    elif row == '0200813':
        result ='0200813_I' 
    elif row == '0200817':
        result ='0200817_I' 
    elif row == '0200824':
        result ='0200824_I'
    elif row == '0200825':
        result ='0200825_I'
    elif row in ['0200828', '0200886']:
        result ='0200828_I' 
    elif row == '0200834':
        result ='0200834_I'
    elif row == '0200837':
        result ='0200837_I' 
    elif row == '0203837':
        result ='0203837_I' 
    elif row == '0203876':
        result ='0203876_I'
    elif row == '0400520':
        result ='0400520_I'
    elif row == '0400521':
        result ='0400521_I'
    elif row == '0400524':
        result ='0400524_I'
    elif row == '0400530':
        result ='0400530_I'
    elif row in ['0400501', '0400532']:
        result ='0400532_I'
    elif row == '0400543':
        result ='0400543_I'
    elif row in ['0400588', '0404156']:
        result ='0400588_I'
    elif row == '0400691':
        result ='0400691_I'
    elif row == '0400692':
        result ='0400692_I'
    elif row == '0500526':
        result ='0500526_I'
    elif row == '0500547':
        result ='0500547_I'
    elif row == '0500564':
        result ='0500564_I'
    elif row == '0600501':
        result ='0600501_I'
    elif row in ['0600515', '0600533', '0600540']:
        result ='0600515_D'
    elif row == '0600516':
        result ='0600516_I'
    elif row in ['0600520', '0600545']:
        result ='0600520_D'
    elif row == '0600537':
        result ='0600537_I'
    elif row in ['0200552', '0600538', '0600540']:
        result ='0600515_D'
    elif row in ['0600564', '0600589']:
        result ='0600564_I'
    elif row == '0600565':
        result ='0600565_I'
    elif row in ['0600569', '0600735']:
        result ='0600569_D'
    elif row in ['0600605', '0600608', '0600609']:
        result ='0600608_D'
    elif row == '0700502':
        result ='0700502_I'
    elif row in ['0700523', '0700527', '0700528', '0700550', '0700580', '0700581', '0700595',
                  '0700599', '0700602', '0700628', '0700649', '0700650', '0700654', '0700655',
                  '0700663', '0700664', '0700677', '0700694', '0700695', '0700705', '0700706']:
        result ='0700527_D'
    elif row == '0700540':
        result ='0700540_I'
    elif row == '0700547':
        result ='0700547_I'
    elif row == '0700549':
        result ='0700549_I'
    elif row == '0700569':
        result ='0700569_I'
    elif row == '0700570':
        result ='0700570_I'
    elif row == '0700597':
        result ='0700597_I'
    elif row == '0700601':
        result ='0700601_I'
    elif row in ['0700620', '0700652']:
        result ='0700652_I'
    elif row == '0700698':
        result ='0700698_I'
    elif row in ['0801002', '0801005']:
        result ='0801002_D'
    elif row in ['0801009', '0801011', '0801462']:
        result ='0801009_D'
    elif row in ['0900522', '0900731', '0900862', '0900880']:
        result ='0900731_D'
    elif row in ['0900896', '0900962', '0900963', '0900964']:
        result ='0900963_D'
    elif row in ['2300507', '2300510', '2300511', '2300513', '2300514', '2300515', '2300519',
                  '2300520', '2300523', '2300525', '2300529', '2300530', '2300538', '2300542',
                  '2300546', '2300550', '2300551', '2300552', '2300553', '2300566', '2300574',
                  '2300575', '2300580', '2300634']:
        result ='2302900'
    elif row in ['2300622', '2300623', '2300624', '2300626', '2300627', '2300628', '2300645']:
        result ='2302901'
    elif row in ['2300686', '2300687', '2300689', '2300691', '2300694', '2300695', '2300698',
                  '2300699', '2300803', '2300807', '2301078']:
        result ='2302902'
    elif row in ['2300562', '2300700', '2300702', '2300703', '2300706', '2300707', '2300708',
                  '2300709', '2300710', '2300712', '2300714', '2300814', '2300816', '2300827', 
                  '2300829', '2300830']:
        result ='2302903'
    elif row in ['2300541', '2300601', '2300609', '2300610', '2300611', '2300612', '2300616',
                  '2300617']:
        result ='2302904'
    elif row in ['2300879', '2300882', '2300884', '2300885', '2300886', '2300887', '2300888',
                  '2300889', '2300890', '2300892', '2300894', '2300895', '2300903', '2301089']:
        result ='2302906'
    elif row in ['2300963', '2300966', '2300967', '2300976', '2300977', '2300978', '2300984',
                  '2300986']:
        result ='2302907'
    elif row in ['2301001', '2301003', '2301004', '2301006', '2301008', '2301009', '2301011',
                  '2301013']:
        result ='2302908'
    elif row in ['2300909', '2300910', '2300911', '2300920', '2300921', '2300928', '2301087']:
        result ='2302909'
    elif row in ['2302910', '2301024']:
        result ='2302910'
    elif row == '2302911':
        result ='2300797'
    elif row == '2300667':
        result ='2302912'
    elif row in ['2300620', '2300654', '2300659', '2300661', '2300663', '2300664', '2300665'
                  '2300672', '2300673', '2300675', '2300676', '2300677', '2300678', '2300680',
                  '2300683']:
        result ='2302913'
    elif row == '2300961':
        result ='2302914'
    elif row in ['2301031', '2301032', '2301036', '2301037', '2301038', '2301039', '2301040',
                  '2301041', '2301042', '2301043', '2301044', '2301045', '2301046', '2301047',
                  '2301055']:
        result ='2302915'
    elif row in ['2300802', '2300962', '2300974', '2300983']:
        result ='2302916'
    elif row in ['2301014', '2301019', '2301020', '2301029']:
        result ='2302917'
    elif row in ['2300878', '2300951', '2300952', '2300954']:
        result ='2302918'
    elif row in ['6400510', '6400511', '6400515', '6403906']:
        result ='6400511_I'
    elif row in ['6400521', '6400522']:
        result ='6400522_D'
    elif row in ['8000657', '8000739', '8000740']:
        result ='8000657_D'
    elif row in ['8000659', '8000660', '8000661', '8000662', '8000889', '8000893']:
        result ='8000662_D'
    elif row in ['8000673', '8000674']:
        result ='8000673_D'
    elif row in ['8000728', '8000729', '8000730', '8000732']:
        result ='8000732_D'
    elif row in ['8000773', '8000774', '8000776', '8000777']:
        result ='8000774_D'
    elif row in ['8000799', '8000800', '8000801', '8000845']:
        result ='8000799_D'
    elif row in ['8000825', '8000826', '8000827', '8000828', '8000829', '8000842', '8000843']:
        result ='8000829_D'
    elif row in ['8000831', '8000847', '8000848', '8000849', '8000854', '8000858']:
        result ='8000831_D'
    elif row in ['0100643', '0100644', '0100835', '0104486']:
        result ='01_ADP037'
    elif row in ['0200885', '0200887', '0200888']:
        result ='02_ADP003'
    elif row in ['0500583', '0500584', '0500587', '0500588', '0500831',
                  '0500938', '0500942', '0600732']:
        result ='05_ADP001'
    elif row in ['0500555', '0500556', '0500829']:
        result ='05_ADP002'
    elif row in ['0700526', '0700711', '0700720']:
        result ='07_ADP001'
    elif row in ['0800909', '0800910']:
        result ='08_ADP003'
    elif row in ['0801215', '0801216', '0801217', '0801230', '0801250',
                  '0801252', '0801254', '0801264', '0801266', '0801267',
                  '0801278', '0801279']:
        result ='08_ADP004'    
    elif row in ['0801360', '0801418', '0801421', '0801426', '0801427']:
        result ='08_ADP002'
    elif row == '0801483':
        result ='08_ADP003'
    elif row in ['0900739', '0900740', '0900741']:
        result ='09_ADP003'
    elif row in ['2300502', '2300503', '2300504', '2300505', '2300506',
                  '2300516', '2300564', '2300568', '2300569', '2300573',
                  '2300579', '2300583', '2300585', '2300586', '2300587', 
                  '2300631', '2300763', '2300774', '2300787', '2300788',
                  '2300789', '2300866', '2300867', '2300868', '2300869',
                  '2301138', '2301140']:
        result ='23_ADP001'
    elif row in ['2300908', '2300931', '2300932', '2300933', '2300936',
                  '2300937', '2300940', '2300948', '2300968', '2300975',
                  '2300987', '2301002', '2301005', '2301018', '2301022',
                  '2301025', '2301075', '2301083', '2301094']:
        result ='23_ADP002'
    elif row in ['8000668', '8000708', '8000709', '8000710', '8000763',
                  '8000764', '8000867', '8000895', '8000896', '8000897',
                  '8000921', '8001014']:
        result ='80_ADP001'
    else:
        result = row
    return result


## apply function to map_df #####

map_df['StateMod_Structure'] = map_df['SW_WDID1'].apply(assign_irrigation)

map_df.loc[map_df['CROP_TYPE'] == 'CORN', 'CROP_TYPE'] = 'CORN_GRAIN'
map_df.loc[map_df['CROP_TYPE'] == 'BARLEY', 'CROP_TYPE'] = 'SMALL_GRAINS'
map_df.loc[map_df['CROP_TYPE'] == 'SMALL_GRAIN', 'CROP_TYPE'] = 'SMALL_GRAINS'
map_df.loc[map_df['CROP_TYPE'] == 'WHEAT_SPRING', 'CROP_TYPE'] = 'SPRING_GRAIN'

### USED SEASONAL WATER REQUIREMENTS FROM STATECU (IN AC/IN) ###
## unknown structures IWR = average of known structures IWRs ###

agg_crop_at_structure = pd.DataFrame()
for i in ids.irrigation_structure_ids_list:
    irrigation_at_structure = map_df.loc[(map_df['StateMod_Structure'] == i)]
    for c in irrigation_at_structure['CROP_TYPE'].unique():
        crop_at_structure = irrigation_at_structure.loc[(irrigation_at_structure['CROP_TYPE'] == c)]
        z = obc.irrigation_consumptive_use_df_selection_nodups['iwr']\
            .loc[(obc.irrigation_consumptive_use_df_selection_nodups['crop'] == c) \
        & (obc.irrigation_consumptive_use_df_selection_nodups['structure_id'] == i)]
        if len(z) == 0:
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'GRASS_PASTURE', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'GRASS_PASTURE')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'ALFALFA', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'ALFALFA')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'BLUEGRASS', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'BLUEGRASS')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'CORN_GRAIN', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'CORN_GRAIN')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'SMALL_GRAINS', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'SMALL_GRAINS')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'SORGHUM_GRAIN', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'SORGHUM_GRAIN')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'SUGAR_BEETS', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'SUGAR_BEETS')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'SNAP_BEANS', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'SNAP_BEANS')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'DRY_BEANS', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'DRY_BEANS')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'POTATOES', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'POTATOES')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'SUNFLOWER', 'IWR'] = 22.0
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'ORCHARD_WO_COVER', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'ORCHARD_WO_COVER')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'VEGETABLES', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'VEGETABLES')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'SPRING_GRAIN', 'IWR'] = obc.crop_iwr_2010.loc[(obc.crop_iwr_2010['crop'] == 'SPRING_GRAIN')]['iwr'].values[0]
            crop_at_structure.loc[crop_at_structure['CROP_TYPE'] == 'WHEAT_FALL', 'IWR'] = 17.9 
        else:
            crop_at_structure['IWR'] = z.values[0]
        agg_crop_at_structure = pd.concat([crop_at_structure, agg_crop_at_structure])
        
### convert to AC/FT by dividing by 12###

agg_crop_at_structure['IWR_ACFT'] = agg_crop_at_structure['IWR']/12

### multiply by parcel-level acreage data ###

agg_crop_at_structure['CONSUMPTIVE_USE_TOTAL'] = agg_crop_at_structure['IWR_ACFT']*agg_crop_at_structure['ACRES']

## merge the structure specific conveyance efficiencies and application technology efficiencies from the .ipy ##

map_df_update = pd.merge(agg_crop_at_structure, ipy.ipydata, on="StateMod_Structure")

## divide further by sprinkler or flood irrigation efficiencies, these are structure specific values from the StateMod .ipy file from 2012 ###
map_df_update['DELIVERED_TO_FARM'] = 0
if map_df_update['IRRIG_TYPE'].loc == 'FLOOD':
    map_df_update['DELIVERED_TO_FARM'] = map_df_update['CONSUMPTIVE_USE_TOTAL']/map_df_update['floodeff']
else:
    map_df_update['DELIVERED_TO_FARM'] = map_df_update['CONSUMPTIVE_USE_TOTAL']/map_df_update['spr']

## divide further to account for conveyance efficiencies ###

map_df_update['DELIVERED_TO_STRUCTURE'] = map_df_update['DELIVERED_TO_FARM']/map_df_update['maxsurf']

#############################

map_df_aggregated_crops = map_df_update.groupby(['StateMod_Structure', 'CROP_TYPE'], as_index=False)["ACRES", "DELIVERED_TO_STRUCTURE"].sum()





### MARGINAL NET BENEFITS FROM AG PRODUCTION as defined by CSU ag extension enterprise budgets for Eastern CO
  ## and South Platte Valley where available ###

map_df_aggregated_crops['MNB'] = 0


## HB ##
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'GRASS_PASTURE', 'MNB'] = 181 * map_df_aggregated_crops['ACRES']

## CSU ag extension - alfalfa hay - northeastern colorado - - 2021, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'ALFALFA', 'MNB'] = 436.29 * map_df_aggregated_crops['ACRES']

# ## CSU ag extension - grass hay - western colorado - - 2020, net receipt before factor payments ###
# map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'BARLEY', 'MNB'] = 260.33 * map_df_aggregated_crops['ACRES']


## CSU ag extension - irrigated corn - northeastern colorado - - 2021, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'CORN_GRAIN', 'MNB'] = 352.54 * map_df_aggregated_crops['ACRES']

## HB ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SMALL_GRAINS', 'MNB'] = 75 * map_df_aggregated_crops['ACRES']


## CSU ag extension - sorghum grain - southeastern colorado - - 2017, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SORGHUM_GRAIN', 'MNB'] = 212.24 * map_df_aggregated_crops['ACRES']

## CSU ag extension - sugar beets - northeastern colorado - - 2021, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SUGAR_BEETS', 'MNB'] = 497.48 * map_df_aggregated_crops['ACRES']

## CSU ag extension - soybeans - northeastern colorado - - 2021, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'DRY_BEANS', 'MNB'] = 263.90 * map_df_aggregated_crops['ACRES']

## CSU ag extension - potatoes - san luis valley - - 2021, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'POTATOES', 'MNB'] = 4.98 * map_df_aggregated_crops['ACRES']

## CSU ag extension - sunflowers - northeastern colorado - - 2021, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SUNFLOWER', 'MNB'] = 392.93 * map_df_aggregated_crops['ACRES']

## CSU ag extension - onions - western colorado - - 2021, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'VEGETABLES', 'MNB'] = 515.35 * map_df_aggregated_crops['ACRES']

## CSU ag extension - irrigated winter wheat - south platte valley - - 2021, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SPRING_GRAIN', 'MNB'] = 153.43 * map_df_aggregated_crops['ACRES']

map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'WHEAT_FALL', 'MNB'] = 153.43 * map_df_aggregated_crops['ACRES']

## use HB's code to calculate orchard MNB ###
marginal_net_benefits = pd.DataFrame()
orchard_planting_costs = [-5183.0, -2802.0, -2802.0, 395.0, 5496.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0] 
orchard_baseline_revenue = [9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, -5183.0, -2802.0, -2802.0, 395.0, 5496.0]
total_npv_costs = 0.0
counter = 0
for cost, baseline in zip(orchard_planting_costs, orchard_baseline_revenue):
  total_npv_costs +=  (baseline - cost)/np.power(1.025, counter)
  counter += 1

map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'ORCHARD_WO_COVER', 'MNB'] = total_npv_costs * map_df_aggregated_crops['ACRES']


### CALCULATE THE MARGINAL VALUE OF WATER ###

map_df_aggregated_crops['MARGINAL_VALUE_OF_WATER'] = 0

map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'GRASS_PASTURE', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'ALFALFA', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
#map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'BARLEY', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'CORN_GRAIN', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SMALL_GRAINS', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SORGHUM_GRAIN', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SUGAR_BEETS', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'DRY_BEANS', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'POTATOES', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SUNFLOWER', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'VEGETABLES', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SPRING_GRAIN', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'WHEAT_FALL', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'ORCHARD_WO_COVER', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']


marginal_net_benefits = pd.DataFrame()
grapes_planting_costs = [-6385.0, -2599.0, -1869.0, 754.0, 2012.0, 2133.0, 2261.0] 
grapes_baseline_revenue = [2261.0, 2261.0, 2261.0, 2261.0, 2261.0, 2261.0, 2261.0]
total_npv_costs = 0.0
counter = 0
for cost, baseline in zip(grapes_planting_costs, grapes_baseline_revenue):
  total_npv_costs +=  (baseline - cost)/np.power(1.025, counter)
  counter += 1
marginal_net_benefits['GRAPES'] = total_npv_costs
orchard_planting_costs = [-5183.0, -2802.0, -2802.0, 395.0, 5496.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0] 
orchard_baseline_revenue = [9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, -5183.0, -2802.0, -2802.0, 395.0, 5496.0]
total_npv_costs = 0.0
counter = 0
for cost, baseline in zip(orchard_planting_costs, orchard_baseline_revenue):
  total_npv_costs +=  (baseline - cost)/np.power(1.025, counter)
  counter += 1

## remove 'None' rows, meaning we do not have StateMod supply/demand dynamics data ####

map_df_aggregated_crops = map_df_aggregated_crops.dropna(subset=['StateMod_Structure'])
print(map_df_aggregated_crops)


##### sort by cost ############

map_df_aggregated_crops_sorted  = map_df_aggregated_crops.sort_values(['StateMod_Structure', 'MARGINAL_VALUE_OF_WATER'], ascending=(True, False))


#### dissolve geometries ###### 

map_df_by_structure = map_df.dissolve(by='StateMod_Structure')

#### create list of irrigation structures id's represented in StateMod #####

irrigation_structure_ids = map_df_update['StateMod_Structure'].unique()
irrigation_structure_ids_list = irrigation_structure_ids.tolist()

#### read in historical irrigation parquet files from StateModify pkg ###

Historical_Irrigation = {}

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test/xddparquet_03_29/')

for i in irrigation_structure_ids_list:
    try:
        Historical_Irrigation[i]= pd.read_parquet(i + '.parquet', engine = 'pyarrow')
        pass
    except:
        continue

### sum irrigator shortages by year ####

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')


Historical_Irrigation_Shortage_Sums = {}

for i in irrigation_structure_ids_list:
    Historical_Irrigation_Shortage_Sums[i] = Historical_Irrigation[i].groupby('year').sum()['shortage']

# create connecting shortages dataframe, this is the amount of water available to irrigators in wet years ####

Historical_Irrigation_Shortages = pd.DataFrame()
for i in irrigation_structure_ids_list:
    Historical_Irrigation_Shortages[i] = Historical_Irrigation_Shortage_Sums[i]
    
Historical_Irrigation_Shortages_forattach = Historical_Irrigation_Shortages.transpose()

Historical_Irrigation_Shortages_forattach['StateMod_Structure'] = Historical_Irrigation_Shortages_forattach.index

## merge shortages and map_df ##
map_df_update = pd.merge(map_df_by_structure, Historical_Irrigation_Shortages_forattach, on="StateMod_Structure")
map_df_update.index = map_df_update['StateMod_Structure']
