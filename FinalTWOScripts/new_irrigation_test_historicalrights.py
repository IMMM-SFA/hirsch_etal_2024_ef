# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:24:53 2023

@author: zacha
"""

import os 
os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')
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
os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_historicalrights_test')
import statemod_sp_adapted_historicalrights as munihistoricalrights
os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import muni_surplus_analysis as surplus
######### start analysis ################

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')

## import shapefile and transform crs ###

fp = 'Div1_Irrig_2010.shp'
map_df = gpd.read_file(fp) 
print(map_df)
map_df.plot()
map_df.crs

map_df = map_df.to_crs("epsg:3857")
map_df.plot()


fp2 = 'Northern_Water_Boundary.shp'
northern_water_boundary = gpd.read_file(fp2) 
print(northern_water_boundary)
northern_water_boundary[['geometry']].plot(facecolor="none", edgecolor="black")
northern_water_boundary = northern_water_boundary.to_crs("epsg:3857")
northern_water_boundary[['geometry']].plot(facecolor="none", edgecolor="black")

fp3 = 'BLM_CO_ST_BNDY_20170109.shp'
colorado_state_boundary = gpd.read_file(fp3)
colorado_state_boundary = colorado_state_boundary.to_crs("epsg:3857")

fp4 = 'Upper_Colorado_River_Basin_Boundary.shp'
ucrb_boundary = gpd.read_file(fp4)
ucrb_boundary = ucrb_boundary.to_crs("epsg:3857")
ucrb_boundary[['geometry']].plot()

fp5 = 'Major_Cities_that_Use_Colorado_River_Water.shp'
co_cities = gpd.read_file(fp5)
co_cities = co_cities.to_crs("epsg:3857")
denver = gpd.GeoDataFrame(co_cities.loc[co_cities['NAME'] == 'Denver'])
denver = denver.to_crs("epsg:3857")
denver[['geometry']].plot()

## https://github.com/OpenWaterFoundation/owf-data-co-transbasin-diversions ##
fp6 = 'Colorado-Transbasin-Diversions-Primary-Gages.geojson'
transbasin_diversions = gpd.read_file(fp6)
transbasin_diversions_selection = transbasin_diversions.iloc[[9,10,14]]
transbasin_diversions_selection = transbasin_diversions_selection.to_crs("epsg:3857")
transbasin_diversions_selection[['geometry']].plot()

fp7 = 'continental-divide-co.geojson'
continental_divide = gpd.read_file(fp7)
continental_divide = continental_divide.to_crs("epsg:3857")
continental_divide[['geometry']].plot()

#granby = gpd.points_from_xy(40.15366, -105.84568, z=None, crs=3857)

# #40.1555335°N, -105.848448°W
#  (-11767000.05, 4850974.22)2.
# granby = Point((4850974.22, -11767000.05))
# granby_point = gpd.GeoSeries([granby], crs={'init': 'epsg:3857'})
# granby_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(granby))
# granby_gdf.plot()

# create a GeoDataFrame with a point
granby_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([-11782995.338802712], [4888569.808065011]))

# set the CRS to Web Mercator
granby_gdf.crs= epsg=3857

# plot the GeoDataFrame with a basemap
# ax = granby_gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
# cx.add_basemap(ax)

# # add a point to the map
# gpd.plot(ax=ax, color='red', markersize=100)

### make colorado inset map ####

fig, ax = plt.subplots(1, figsize =(24, 8))

northern_water_boundary[['geometry']].plot(facecolor="yellow", edgecolor="blue", alpha = 0.3, ax=ax, label = 'Northern Water Boundary', legend_kwds = {'label' : 'Northern Water Boundary'}, legend = True)
colorado_state_boundary[['geometry']].plot(facecolor="none", edgecolor="black",linewidth = 5.5,  ax=ax)
ucrb_boundary[['geometry']].plot(facecolor="none", edgecolor="black", ax=ax, label = 'Upper Colorado River Basin')
denver[['geometry']].plot(facecolor="green", edgecolor="black", markersize=100, ax=ax, label = 'Denver')
transbasin_diversions_selection[['geometry']].plot(facecolor="red", edgecolor="black", markersize=100, ax=ax, label = 'Transbasin Diversions')
continental_divide[['geometry']].plot(facecolor="none", edgecolor="red", linewidth = 4.5,  ax=ax, label= 'Continental Divide')
granby_gdf[['geometry']].plot(facecolor="blue", edgecolor="black", markersize = 100, ax=ax, label= 'Lake Granby')
# src_basemap = cx.providers.Stamen.Terrain
# cx.add_basemap(ax, source=src_basemap,crs = map_df.crs)
#granby_gdf[['geometry']].plot(facecolor="blue", edgecolor="black", ax=ax, label= 'Lake Granby')
ax.axis('off')
ax.grid (False)
ax.set_ylim([4429099.68, 5019999.71])
ax.set_xlim([-12149912.89, -11347000.85])
ax.legend(loc='lower right')
plt.tight_layout()

northern_water_boundary[['geometry']].plot(facecolor="none", edgecolor="blue")
colorado_state_boundary[['geometry']].plot(facecolor="none", edgecolor="black")

map_df= map_df.to_crs(epsg=3857)
northern_water_boundary = northern_water_boundary.to_crs(epsg=3857)

## drop unneccessary columns ###
#map_df = map_df.drop(map_df.loc[:, 'SW_WDID3':'COMMENTS'].columns, axis=1)

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

map_df_update.loc[map_df_update['IRRIG_TYPE'] == 'FLOOD', 'DELIVERED_TO_FARM'] = map_df_update['CONSUMPTIVE_USE_TOTAL']/map_df_update['floodeff']
map_df_update.loc[map_df_update['IRRIG_TYPE'] != 'FLOOD', 'DELIVERED_TO_FARM'] = map_df_update['CONSUMPTIVE_USE_TOTAL']/map_df_update['spr']

# if map_df_update['IRRIG_TYPE'] == 'FLOOD':
#     map_df_update['DELIVERED_TO_FARM'] = map_df_update['CONSUMPTIVE_USE_TOTAL']/map_df_update['floodeff']
# else:
#     map_df_update['DELIVERED_TO_FARM'] = map_df_update['CONSUMPTIVE_USE_TOTAL']/map_df_update['spr']

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

## CSU ag extension - onions - western colorado - - 2021, net receipt before factor payments ###
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SNAP_BEANS', 'MNB'] = 515.35 * map_df_aggregated_crops['ACRES']

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
map_df_aggregated_crops.loc[map_df_aggregated_crops['CROP_TYPE'] == 'SNAP_BEANS', 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops['MNB']/map_df_aggregated_crops['DELIVERED_TO_STRUCTURE']

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

#### read in the selected StateMod run parquet files using tools from the StateModify pkg ###

Historical_Irrigation = {}

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_historicalrights_test/xddparquet_04_12/')

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

##################################################################################################################
##################################################################################################################
### Determine which structures are recieving water in the driest year ############################################
### the following loads the StateMod .xdd output's key delivery metrics and determines who is recieving water > 0 #######################

Historical_Irrigation_SW_Deliveries = {}
for i in irrigation_structure_ids_list:
    Historical_Irrigation_SW_Deliveries[i] = Historical_Irrigation[i].groupby('year')['river-priority', 'river-storage','river-other', 'carrier-priority', 'carrier-other'].sum()

Historical_Irrigation_TOTAL_SW_DELIVERIES = {}
for i in irrigation_structure_ids_list:
    for y in range(1950,2013):
        Historical_Irrigation_TOTAL_SW_DELIVERIES[i] = Historical_Irrigation_SW_Deliveries[i]['river-priority'] + Historical_Irrigation_SW_Deliveries[i]['river-storage'] + Historical_Irrigation_SW_Deliveries[i]['river-other'] \
            + Historical_Irrigation_SW_Deliveries[i]['carrier-priority'] + Historical_Irrigation_SW_Deliveries[i]['carrier-other']

## check to ensure 2010 demand value is listed for each year at each structure ####
Historical_Irrigation_Demand_Sums = {}
for i in irrigation_structure_ids_list:
    Historical_Irrigation_Demand_Sums[i] = Historical_Irrigation[i].groupby('year').sum()['demand']
    
#### if demand - shortage = 0 in the driest year, remove them from the market pool ###
## Driest Year is determined by largest Northern Water Shortage from Adapted StateMod output ####
Driest_Year = range(1955,1956)

Driest_Year_Irrigator_Water_Right_Fulfillment = {}

for i in irrigation_structure_ids_list:
    for y in Driest_Year: 
        Driest_Year_Irrigator_Water_Right_Fulfillment[i] = int(Historical_Irrigation_Demand_Sums[i][y]) - int(Historical_Irrigation_Shortage_Sums[i][y])

### Now, we drop the irrigator structures with no water deliveries and create a new structure_id list that will be what we use to evaluate the TWO ####

dct = {k:[v] for k,v in Driest_Year_Irrigator_Water_Right_Fulfillment.items()}  # WORKAROUND
driest_year_df = pd.DataFrame(dct)
driest_year_df = driest_year_df.transpose()
driest_year_df['values'] = driest_year_df[0]

driest_year_df = driest_year_df.drop(driest_year_df[driest_year_df.values == 0].index)
driest_year_df['StateMod_Structure'] = driest_year_df.index

## pull the structure ids that are recieving water ###

irrigation_structure_ids_TWO = pd.Series(driest_year_df['StateMod_Structure'].unique())


Dry_Year_Irrigator_Water_Right_Fulfillment = {}
for y in range(1950,2013):
    dry_year_water = pd.DataFrame()
    dry_year_water_update = pd.DataFrame()
    for i in irrigation_structure_ids_TWO:
        water_avail_at_structure = int(Historical_Irrigation_Demand_Sums[i][y]) - int(Historical_Irrigation_Shortage_Sums[i][y])
        dry_year_water = pd.concat([pd.Series(i), pd.Series(water_avail_at_structure)], axis =1)
        dry_year_water_update = pd.concat([dry_year_water, dry_year_water_update])
        dry_year_water_update.index = dry_year_water_update[0]
    Dry_Year_Irrigator_Water_Right_Fulfillment[y] = dry_year_water_update


# uses_of_water_dry = {}
# #year_df = pd.DataFrame()
# for y in range(1950,2013):
#     year_df = pd.DataFrame()
#     for i in irrigation_structure_ids_TWO:
#         irrigation_set_dry = map_df_aggregated_crops_sorted.loc[map_df_aggregated_crops_sorted['StateMod_Structure'] == i]
#         irrigation_set_dry['AVAILABLE_TO_MUNI'] = 0
#         #use_of_water = pd.DataFrame()
#         water_avail_dry_year = Dry_Year_Irrigator_Water_Right_Fulfillment[y].loc[i][1]
#         for crop in range(len(irrigation_set_dry)):
#             if water_avail_dry_year > irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]:
#                 use = irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]
#             else:
#                 use = max(water_avail_dry_year, 0)
#             irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop] = use
                
                
#             water_avail_dry_year -= irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]
#         year_df = pd.concat([year_df, irrigation_set_dry])
                          
#     uses_of_water_dry[y] = year_df  
######## FOR DRY YEAR MUNICIPAL SHORTAGES - ASSIGN SHORTAGE VALUE TO CROPS BASED ON MUNICIPAL NEEDS #########

uses_of_water_dry = {}
#year_df = pd.DataFrame()
for y in range(1950,2013):
    year_df = pd.DataFrame()
    for i in irrigation_structure_ids_TWO:
        irrigation_set_dry = map_df_aggregated_crops_sorted.loc[map_df_aggregated_crops_sorted['StateMod_Structure'] == i]
        irrigation_set_dry['AVAILABLE_TO_MUNI'] = 0
        irrigation_set_dry['POSSIBLEMUNIUSAGE'] = 0
        #use_of_water = pd.DataFrame()
        water_avail_dry_year = Dry_Year_Irrigator_Water_Right_Fulfillment[y].loc[i][1]
        muni_shortage_dry_year = munihistoricalrights.Northern_Water_Muni_Shortages['Shortage'].loc[y]
        for crop in range(len(irrigation_set_dry)):
            if water_avail_dry_year > irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]:
                use = irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]
            else:
                use = max(water_avail_dry_year, 0)
            irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop] = use
                
                
            water_avail_dry_year -= irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]
        
        for crop in reversed(range(len(irrigation_set_dry))):
            if muni_shortage_dry_year > irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop]:
                use = irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop]
            else:
                use = max(muni_shortage_dry_year, 0)
            irrigation_set_dry['POSSIBLEMUNIUSAGE'].iloc[crop] = use
                
                
            water_avail_dry_year -= irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]
            muni_shortage_dry_year -= irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop]

        year_df = pd.concat([year_df, irrigation_set_dry])
                          
    uses_of_water_dry[y] = year_df  
    

irrigation_uses_by_year_dry = {}
consumptive_use_by_structure = {}
for y in range(1950,2013):
    irrigation_uses_by_year_dry[y] = uses_of_water_dry[y].groupby(['CROP_TYPE'], as_index=False)['AVAILABLE_TO_MUNI'].sum()
    consumptive_use_by_structure[y] = uses_of_water_dry[y].groupby(['StateMod_Structure'], as_index=False)['AVAILABLE_TO_MUNI'].sum()

years = pd.Series(range(1950,2013))
years = years.astype(str)








value_of_water = {}

for i in irrigation_structure_ids_TWO:
    irrigation_set_dry = uses_of_water_dry[y].loc[uses_of_water_dry[y]['StateMod_Structure'] == i]
    values_of_water = pd.Series(index=years)
    for y in range(1950,2013):
        water_avail_dry_year = munihistoricalrights.Northern_Water_Muni_Shortages['Shortage'].loc[y]
        for crop in reversed(range(len(irrigation_set_dry))):
            if water_avail_dry_year > 0:
                water_avail_dry_year -= irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop]
                if (water_avail_dry_year <= 0):
                #     # print(water_avail)
                #     # print(crop)
                #     # print(irrigation_set['TOTAL_COST'].iloc[crop])
                     values_of_water[str(y)] = irrigation_set_dry['MARGINAL_VALUE_OF_WATER'].iloc[crop]
                else:
                #     values_of_water[str(y)] = irrigation_set['MARGINAL_VALUE_OF_WATER'].iloc[crop]                    
                #     #print(y)
                #     #print(crop)
                     value_of_water[i] = values_of_water.fillna(irrigation_set_dry['MARGINAL_VALUE_OF_WATER'].iloc[-1], inplace=True)               
    value_of_water[i] = values_of_water

structures_by_yearly_water_value = pd.DataFrame()
for i in irrigation_structure_ids_TWO:
    structures_by_yearly_water_value[i] = value_of_water[i]

yearly_water_values_forattach_dry = structures_by_yearly_water_value.transpose()
max_value_dry = yearly_water_values_forattach_dry.to_numpy().max()


columnnames = {}
count = 1949
for i in yearly_water_values_forattach_dry.columns:
    count +=1
    columnnames[i] = f"value_{count}_dry"
yearly_water_values_forattach_dry.rename(columns = columnnames, inplace = True)
yearly_water_values_forattach_dry.columns

yearly_water_values_forattach_dry['StateMod_Structure'] = yearly_water_values_forattach_dry.index


map_df_update = map_df_update.drop('StateMod_Structure', axis=1)

map_df_update2 = pd.merge(map_df_update, yearly_water_values_forattach_dry, on="StateMod_Structure")


map_df_update2 = gpd.GeoDataFrame(map_df_update2)
map_df_update2 = map_df_update2.to_crs("epsg:3857")
map_df_update2.plot()




### make the dry year market map ###

for i in range(1950,2013):
    # create the colorbar
    #norm1 = colors.Normalize(vmin=map_df_update2[f"value_{i}_dry"].min(), vmax=map_df_update2[f"value_{i}_dry"].max())
    norm1 = colors.Normalize(vmin=0, vmax=max_value_dry)
    cbar = plt.cm.ScalarMappable(norm=norm1, cmap='jet')

    fig, ax = plt.subplots(1, figsize =(24, 8))
    ax.set_ylim([4820000, 5050000])
    ax.set_xlim([-11780000, -11375000])
    map_df_update2.plot(column= f"value_{i}_dry", categorical=False, cmap='jet', linewidth=.1, edgecolor='0.2',
                legend=False, ax=ax)
    northern_water_boundary[['geometry']].plot(facecolor="none", edgecolor="black",linewidth=.8, label ="Northern Water Boundary", legend = True,  ax=ax)
    ax_cbar = fig.colorbar(cbar, ax=ax)
    # add label for the colorbar
    ax_cbar.set_label(label='Marginal Value of Water ($/AF)',size = 15, weight='bold')
    #plt.legend(loc="upper right")
    ax.axis('off')
    ax.grid (False)
    #ax.set_title('South Platte Two-Way Option Market in Driest Hydrologic Year',fontsize=20, weight='bold')
    # src_basemap = cx.providers.Stamen.Terrain
    # cx.add_basemap( ax, source=src_basemap, crs = map_df_update2.crs )
    plt.tight_layout()


for y in range(1950,2013):
    for crop in irrigation_uses_by_year_dry[y]['CROP_TYPE'].unique():
        irrigation_uses_by_year_dry[y].loc[irrigation_uses_by_year_dry[y]['CROP_TYPE'] == crop, 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops_sorted.loc[map_df_aggregated_crops_sorted['CROP_TYPE'] == crop,'MARGINAL_VALUE_OF_WATER'].mean()

for y in range(1950,2013):
    irrigation_uses_by_year_dry[y] = irrigation_uses_by_year_dry[y].sort_values(by=['MARGINAL_VALUE_OF_WATER'], ascending=True)
    irrigation_uses_by_year_dry[y] = irrigation_uses_by_year_dry[y][irrigation_uses_by_year_dry[y]['AVAILABLE_TO_MUNI'] != 0]


structure_uses_by_year_dry = {}
for y in range(1950,2013):
    structure_uses_by_year_dry [y] = uses_of_water_dry[y].sort_values(by=['MARGINAL_VALUE_OF_WATER'], ascending=True)

## make structure specific supply functions
    
for i in range(1955,1956):

     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'GRASS_PASTURE', 'color'] = 'lawngreen'
     
     ## CSU ag extension - alfalfa hay - northeastern colorado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'ALFALFA', 'color'] = 'sandybrown'
     
     ## CSU ag extension - grass hay - western colorado - - 2020, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SNAP_BEANS', 'color'] = 'peru'
     
     
     ## CSU ag extension - irrigated corn - northeastern colorado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'CORN_GRAIN', 'color'] = 'yellow'
     
     ## HB ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SMALL_GRAINS', 'color'] = 'burlywood'
     
     
     ## CSU ag extension - sorghum grain - southeastern colorado - - 2017, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SORGHUM_GRAIN', 'color'] = 'wheat'
         
     ## CSU ag extension - sugar beets - northeastern colorado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SUGAR_BEETS', 'color'] = 'firebrick'
     
     ## CSU ag extension - soybeans - northeastern colorado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'DRY_BEANS', 'color'] = 'brown'
     
     ## CSU ag extension - potatoes - san luis valley - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'POTATOES', 'color'] = 'rosybrown'
     
     ## CSU ag extension - sunflowers - northeastern colorado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SUNFLOWER', 'color'] = 'gold'
     
     ## CSU ag extension - onions - western colorado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'VEGETABLES', 'color'] = 'tomato'
     
     ## CSU ag extension - irrigated winter wheat - south platte valley - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SPRING_GRAIN', 'color'] = 'goldenrod'
    
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'WHEAT_FALL', 'color'] = 'blue'
     
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'ORCHARD_WO_COVER', 'color'] = 'purple'
    
## hatch styles ###    
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'GRASS_PASTURE', 'design'] = '*'
     
     ## CSU ag extension - alfalfa hay - northeastern hado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'ALFALFA', 'design'] = 'o'
     
     ## CSU ag extension - grass hay - western hado - - 2020, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SNAP_BEANS', 'design'] = '|'
     
     
     ## CSU ag extension - irrigated corn - northeastern hado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'CORN_GRAIN', 'design'] = '+'
     
     ## HB ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SMALL_GRAINS', 'design'] = '..'
     
     
     ## CSU ag extension - sorghum grain - southeastern hado - - 2017, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SORGHUM_GRAIN', 'design'] = 'o'
         
     ## CSU ag extension - sugar beets - northeastern hado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SUGAR_BEETS', 'design'] = 'O'
     
     ## CSU ag extension - soybeans - northeastern hado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'DRY_BEANS', 'design'] = '.'
     
     ## CSU ag extension - potatoes - san luis valley - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'POTATOES', 'design'] = '*'
     
     ## CSU ag extension - sunflowers - northeastern hado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SUNFLOWER', 'design'] = '+'
     
     ## CSU ag extension - onions - western hado - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'VEGETABLES', 'design'] = 'o'
     
     ## CSU ag extension - irrigated winter wheat - south platte valley - - 2021, net receipt before factor payments ###
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'SPRING_GRAIN', 'design'] = 'O'
    
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'WHEAT_FALL', 'design'] = 'x'
     
     structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == 'ORCHARD_WO_COVER', 'design'] = '/'
    
    
     fig, ax = plt.subplots(figsize = (20,12))
     figure_name = 'land_available_for_purchase_supply_fn' + 'f{i}'+'.png'
    
     #make dictionary of total irrigated area for each crop
     #crop type is the dictionary key, crop acreage is the dictionary value
     overall_crop_areas = {}
     for index, row in structure_uses_by_year_dry[i].iterrows():
       #get total irrigated acreage (in thousand acres) for all listed crops
       overall_crop_areas[index] = row['AVAILABLE_TO_MUNI'] / 1000.0

     # #crop marginal net benefits ($/acre) come from enterpirse budget reports from the CSU ag extension
     marginal_value_of_water = {}
     for crop in range(len(structure_uses_by_year_dry[i]['CROP_TYPE'])):
         name_of_crop = structure_uses_by_year_dry[i]['CROP_TYPE'].iloc[crop]
         marginal_value_of_water[name_of_crop] = structure_uses_by_year_dry[i]['MARGINAL_VALUE_OF_WATER'].iloc[crop]

     #sort crops from highest cost to lowest cost
     water_costs = np.zeros(len(marginal_value_of_water))
     crop_list = []
     for crop_cnt, crop_name in enumerate(marginal_value_of_water):
       #MNB = $/ac; et = AF/ac; MNB/et = $/AF
       water_costs[crop_cnt] = marginal_value_of_water[crop_name] #et is ac-in, divide by 12 for ac-ft
       crop_list.append(crop_name)
     #sort crops by $/AF
     sorted_index = np.argsort(water_costs*(1.0))
     crop_list_new = np.asarray(crop_list)
     sorted_crops = crop_list_new[sorted_index]
     
     running_area = 0.0#total crop acreage across all crop types
     for crop_name in sorted_crops:
       # $/AF of fallowing for current crop type
       total_cost = marginal_value_of_water[crop_name]

       #acreage irrigated for this crop type      
       total_area = overall_crop_areas[index]
       #plot value per acre foot vs. cumulative acreage, with crops ordered from highest value crop to lowest value crop
       #plots a single 'box' for each individual crop
       color = structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == crop_name, 'color']
       #hat = structure_uses_by_year_dry[i].loc[structure_uses_by_year_dry[i]['CROP_TYPE'] == crop_name, 'design']
       # ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = color, edgecolor = 'black', linewidth = 2.0, label = crop_name)
       ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = color, edgecolor = color , linewidth = 2.0, label = crop_name)
       running_area += total_area
     
       #format plot
       #make sure y ticklabels line up with break points
       ax.set_yticks([0.0, 100.0, 200.0, 300.0, 400.0])
       ax.set_yticklabels(['$0', '$100', '$200', '$300', '$400'])
       ax.set_ylabel('Marginal Value of Water ($/AF)', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
       ax.set_xlabel('Irrigation Water Supply (tAF)', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
       ax.set_ylim([0.0, 200.0])
       ax.set_xlim([0, 1000.0])
       #ax.hlines(y = 30, xmin = 0, xmax = (muni.Northern_Water_Muni_Shortages['Shortage'].loc[i])/1000, linestyle = 'dashed', color = 'r')
       ax.legend(loc = 'upper left', prop={'size':12})
       #ax.set_xlim([0, running_area * 1.05])
     for item in (ax.get_xticklabels()):
       item.set_fontsize(32)
       item.set_fontname('Gill Sans MT')
     for item in (ax.get_yticklabels()):
       item.set_fontsize(32)
       item.set_fontname('Gill Sans MT')
     plt.show()
     plt.savefig(figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
     plt.close()   

### make the option dot plot supply function ###

for i in range(1955,1956):
    structure_uses_by_year_dry[i]['CUM_USAGE'] = 0
    water = 0
    for crop in (range(len(structure_uses_by_year_dry[i]))):  
        water += structure_uses_by_year_dry[i]['AVAILABLE_TO_MUNI'].iloc[crop]
        print(water)
        structure_uses_by_year_dry[i]['CUM_USAGE'].iloc[crop] = water 
        
        structure_uses_by_year_dry[i]['NUMERIC_CROP_TYPE'] = pd.factorize(structure_uses_by_year_dry[i]['CROP_TYPE'])[0]
    plt.scatter(structure_uses_by_year_dry[i]['CUM_USAGE'],structure_uses_by_year_dry[i]['MARGINAL_VALUE_OF_WATER'])
    
# fg = sns.FacetGrid(data=structure_uses_by_year_dry[1954], hue='CROP_TYPE', hue_order=structure_uses_by_year_dry[1954]['CROP_TYPE'].unique(), aspect=1.61)
# fg.map(plt.scatter, 'CUM_USAGE', 'MARGINAL_VALUE_OF_WATER').add_legend()


fig, ax = plt.subplots(figsize=(11, 4))
ax.scatter(data=structure_uses_by_year_dry[1955], x='CUM_USAGE', y='MARGINAL_VALUE_OF_WATER', c='NUMERIC_CROP_TYPE', cmap='Set3')
ax.set(xlabel='Available Water Supply (AF)', ylabel='Marginal Value of Water ($)')
#ax.vlines(structure_uses_by_year_dry[1954]['CUM_USAGE'], 0, structure_uses_by_year_dry[1954]['MARGINAL_VALUE_OF_WATER'], linestyle="solid", color = 'NUMERIC_CROP_TYPE')
ax.set_ylim ([0, 300])


plt.show()




## make supply functions
    
for i in range(1955,1956):

     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'GRASS_PASTURE', 'color'] = 'lawngreen'
     
     ## CSU ag extension - alfalfa hay - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'ALFALFA', 'color'] = 'sandybrown'
     
     ## CSU ag extension - grass hay - western colorado - - 2020, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SNAP_BEANS', 'color'] = 'peru'
     
     
     ## CSU ag extension - irrigated corn - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'CORN_GRAIN', 'color'] = 'yellow'
     
     ## HB ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SMALL_GRAINS', 'color'] = 'burlywood'
     
     
     ## CSU ag extension - sorghum grain - southeastern colorado - - 2017, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SORGHUM_GRAIN', 'color'] = 'wheat'
         
     ## CSU ag extension - sugar beets - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SUGAR_BEETS', 'color'] = 'firebrick'
     
     ## CSU ag extension - soybeans - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'DRY_BEANS', 'color'] = 'brown'
     
     ## CSU ag extension - potatoes - san luis valley - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'POTATOES', 'color'] = 'rosybrown'
     
     ## CSU ag extension - sunflowers - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SUNFLOWER', 'color'] = 'gold'
     
     ## CSU ag extension - onions - western colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'VEGETABLES', 'color'] = 'tomato'
     
     ## CSU ag extension - irrigated winter wheat - south platte valley - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SPRING_GRAIN', 'color'] = 'goldenrod'
    
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'WHEAT_FALL', 'color'] = 'blue'
     
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'ORCHARD_WO_COVER', 'color'] = 'purple'
    
## hatch styles ###    
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'GRASS_PASTURE', 'design'] = '+o'
     
     ## CSU ag extension - alfalfa hay - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'ALFALFA', 'design'] = 'o'
     
     ## CSU ag extension - grass hay - western hado - - 2020, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SNAP_BEANS', 'design'] = '|'
     
     
     ## CSU ag extension - irrigated corn - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'CORN_GRAIN', 'design'] = '|*'
     
     ## HB ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SMALL_GRAINS', 'design'] = '..'
     
     
     ## CSU ag extension - sorghum grain - southeastern hado - - 2017, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SORGHUM_GRAIN', 'design'] = 'o'
         
     ## CSU ag extension - sugar beets - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SUGAR_BEETS', 'design'] = 'O.'
     
     ## CSU ag extension - soybeans - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'DRY_BEANS', 'design'] = '.'
     
     ## CSU ag extension - potatoes - san luis valley - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'POTATOES', 'design'] = '**'
     
     ## CSU ag extension - sunflowers - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SUNFLOWER', 'design'] = '++'
     
     ## CSU ag extension - onions - western hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'VEGETABLES', 'design'] = 'o'
     
     ## CSU ag extension - irrigated winter wheat - south platte valley - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'SPRING_GRAIN', 'design'] = 'O'
    
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'WHEAT_FALL', 'design'] = 'x'
     
     irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == 'ORCHARD_WO_COVER', 'design'] = '/'
    
    
     fig, ax = plt.subplots(figsize = (20,12))
     figure_name = 'land_available_for_purchase_supply_fn' + 'f{i}'+'.png'
    
     #make dictionary of total irrigated area for each crop
     #crop type is the dictionary key, crop acreage is the dictionary value
     overall_crop_areas = {}
     for index, row in irrigation_uses_by_year_dry[i].iterrows():
       #get total irrigated acreage (in thousand acres) for all listed crops
       if row['CROP_TYPE'] in overall_crop_areas:
         overall_crop_areas[row['CROP_TYPE']] += row['AVAILABLE_TO_MUNI'] / 1000.0
       else:
         overall_crop_areas[row['CROP_TYPE']] = row['AVAILABLE_TO_MUNI'] / 1000.0
         
     # #crop marginal net benefits ($/acre) come from enterpirse budget reports from the CSU ag extension
     marginal_value_of_water = {}
     for crop in range(len(irrigation_uses_by_year_dry[i]['CROP_TYPE'])):
         name_of_crop = irrigation_uses_by_year_dry[i]['CROP_TYPE'].iloc[crop]
         marginal_value_of_water[name_of_crop] = irrigation_uses_by_year_dry[i]['MARGINAL_VALUE_OF_WATER'].iloc[crop]

     #sort crops from highest cost to lowest cost
     water_costs = np.zeros(len(marginal_value_of_water))
     crop_list = []
     for crop_cnt, crop_name in enumerate(marginal_value_of_water):
       #MNB = $/ac; et = AF/ac; MNB/et = $/AF
       water_costs[crop_cnt] = marginal_value_of_water[crop_name] #et is ac-in, divide by 12 for ac-ft
       crop_list.append(crop_name)
     #sort crops by $/AF
     sorted_index = np.argsort(water_costs*(1.0))
     crop_list_new = np.asarray(crop_list)
     sorted_crops = crop_list_new[sorted_index]
     
     running_area = 0.0#total crop acreage across all crop types
     for crop_name in sorted_crops:
       # $/AF of fallowing for current crop type
       total_cost = marginal_value_of_water[crop_name]

       #acreage irrigated for this crop type      
       total_area = overall_crop_areas[crop_name]
       #plot value per acre foot vs. cumulative acreage, with crops ordered from highest value crop to lowest value crop
       #plots a single 'box' for each individual crop
       #color = irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == crop_name, 'color']
       hat = irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == crop_name, 'design']
       # ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = color, edgecolor = 'black', linewidth = 2.0, label = crop_name)
       ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = 'None', hatch = hat.values[0], linewidth = 2.0, label = crop_name)
       running_area += total_area
     
       #format plot
       #make sure y ticklabels line up with break points
       ax.set_yticks([0.0, 100.0, 200.0, 300.0, 400.0])
       ax.set_yticklabels(['$0', '$100', '$200', '$300', '$400'])
       ax.set_ylabel('Marginal Value of Water ($/AF)', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
       ax.set_xlabel('Irrigation Water Supply (tAF)', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
       ax.set_ylim([0.0, 200.0])
       ax.set_xlim([0, 1000.0])
       ax.legend(loc = 'upper left', prop={'size':12})
       #ax.set_xlim([0, running_area * 1.05])
     for item in (ax.get_xticklabels()):
       item.set_fontsize(32)
       item.set_fontname('Gill Sans MT')
     for item in (ax.get_yticklabels()):
       item.set_fontsize(32)
       item.set_fontname('Gill Sans MT')
     plt.show()
     plt.savefig(figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
     plt.close()   
    
### now, we need to create a new column that selects TWO users based on water availability and mv of water ####

# TWO_Selection = {}
# for y in range(1950,2013):
#     year_df = pd.DataFrame()
#     for i in irrigation_structure_ids_TWO:
#         irrigation_set_year = uses_of_water_dry[y].sort_values(by=['MARGINAL_VALUE_OF_WATER'], ascending=True)
#         irrigation_set_dry['AVAILABLE_TO_MUNI'] = 0
#         irrigation_set_dry['POSSIBLEMUNIUSAGE'] = 0
#         #use_of_water = pd.DataFrame()
#         water_avail_dry_year = Dry_Year_Irrigator_Water_Right_Fulfillment[y].loc[i][1]
#         muni_shortage_dry_year = muni.Northern_Water_Muni_Shortages['Shortage'].loc[y]
#         for crop in range(len(irrigation_set_dry)):
#             if water_avail_dry_year > irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]:
#                 use = irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]
#             else:
#                 use = max(water_avail_dry_year, 0)
#             irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop] = use
                
                
#             water_avail_dry_year -= irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]
        
#         for crop in reversed(range(len(irrigation_set_dry))):
#             if muni_shortage_dry_year > irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop]:
#                 use = irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop]
#             else:
#                 use = max(muni_shortage_dry_year, 0)
#             irrigation_set_dry['POSSIBLEMUNIUSAGE'].iloc[crop] = use
                
                
#             water_avail_dry_year -= irrigation_set_dry['DELIVERED_TO_STRUCTURE'].iloc[crop]
#             muni_shortage_dry_year -= irrigation_set_dry['AVAILABLE_TO_MUNI'].iloc[crop]

#         year_df = pd.concat([year_df, irrigation_set_dry])
                          
#     uses_of_water_dry[y] = year_df  
 
TWO_Selection = {}
for y in range(1950,2013):
    two_year_df = pd.DataFrame()    
    irrigation_set_year = uses_of_water_dry[y].sort_values(by=['MARGINAL_VALUE_OF_WATER'], ascending=True)
    irrigation_set_year['TWO_USAGE'] = 0
    #march cbi test
    if y in [1951, 1955, 1956, 1957, 1964, 1965, 1967, 1977, 1978, 1979, 1982, 1990, 1991, 1992, 1993, 1995, 2002, 2003, 2004, 2005]:
    #if y in [1951,1954,1955,1956,1964,1977,1978,1982,1990,1991,1992,2002,2003,2004]:
        muni_shortage_dry_year = munihistoricalrights.Northern_Water_Muni_Shortages['Shortage'].loc[y]
    else:
        muni_shortage_dry_year = 0
    for crop in range(len(irrigation_set_year)):
        if muni_shortage_dry_year > irrigation_set_year['AVAILABLE_TO_MUNI'].iloc[crop]:
            use = irrigation_set_year['AVAILABLE_TO_MUNI'].iloc[crop]
        else:
            use = max(muni_shortage_dry_year, 0)
        irrigation_set_year['TWO_USAGE'].iloc[crop] = use
        muni_shortage_dry_year -= irrigation_set_year['AVAILABLE_TO_MUNI'].iloc[crop]

    TWO_Selection[y] = irrigation_set_year
    
    
###########################################################################################################
######## FOR WET YEAR IRRIGATOR SHORTAGES - ASSIGN SHORTAGE VALUE TO CROPS BASED ON INDIVIDUAL CROP CONSUMPTION #########
## subtract shortage value from lowest value crop up the list, when consumptive use is not fulfilled, that is the marginal value of water to this structure in a wet year ##
## 'USAGE' column also relates to potential revenue gains with the purchase of water from municipality in wet year (USAGE * MARGINAL VALUE OF WATER) ######
uses_of_water_wet = {}
#structure_revenue_wet = {}
#year_df = pd.DataFrame()
for y in range(1950,2013):
    year_df = pd.DataFrame()
    for i in irrigation_structure_ids_TWO:
        irrigation_set_wet = map_df_aggregated_crops_sorted.loc[map_df_aggregated_crops_sorted['StateMod_Structure'] == i]
        irrigation_set_wet['AG_DEMAND'] = 0
        #use_of_water = pd.DataFrame()
        water_avail_wet_year = Historical_Irrigation_Shortage_Sums[i][y]
        for crop in reversed(range(len(irrigation_set_wet))):
            if water_avail_wet_year > irrigation_set_wet['DELIVERED_TO_STRUCTURE'].iloc[crop]:
                use = irrigation_set_wet['DELIVERED_TO_STRUCTURE'].iloc[crop]
            else:
                use = max(water_avail_wet_year, 0)
            irrigation_set_wet['AG_DEMAND'].iloc[crop] = use
                 
                
            water_avail_wet_year -= irrigation_set_wet['DELIVERED_TO_STRUCTURE'].iloc[crop]
        #added irrigator rev here is = to the amount of revenue generated by fulfilling the shortage. since MNB is calculated
        #at completely fulfilled 2010 irrigator plot conditions, we subtract the shortage amount to get realized rev in any year
        # irrigation_set_wet['added_irr_rev'] = irrigation_set_wet['USAGE']*irrigation_set_wet['MARGINAL_VALUE_OF_WATER']
        # if y in Wet_Year_Triggers:
        #     irrigation_set_wet['rev'] = irrigation_set_wet['MNB'].sum() - irrigation_set_wet['added_irr_rev'].sum()
        #     irrigation_set_wet['revTWO'] = irrigation_set_wet['MNB'].sum() + irrigation_set_wet['added_irr_rev'].sum()
        # else:
        #     irrigation_set_wet['rev'] = irrigation_set_wet['MNB'].sum() - irrigation_set_wet['added_irr_rev'].sum()
        #     irrigation_set_wet['revTWO'] = irrigation_set_wet['rev']
        year_df = pd.concat([year_df, irrigation_set_wet])
                      
    uses_of_water_wet[y] = year_df     



value_of_water_wet = {}

for i in irrigation_structure_ids_TWO:
    irrigation_set_wet = map_df_aggregated_crops_sorted.loc[map_df_aggregated_crops_sorted['StateMod_Structure'] == i]
    values_of_water = pd.Series(index=years)
    for y in range(1950,2013):
        water_avail_wet_year = Historical_Irrigation_Shortage_Sums[i][y]
        for crop in reversed(range(len(irrigation_set_wet))):
            if water_avail_wet_year > 0:
                water_avail_wet_year -= irrigation_set_wet['DELIVERED_TO_STRUCTURE'].iloc[crop]
                if (water_avail_wet_year <= 0):
                    # print(water_avail)
                    # print(crop)
                    # print(irrigation_set['TOTAL_COST'].iloc[crop])
                    values_of_water[str(y)] = irrigation_set_wet['MARGINAL_VALUE_OF_WATER'].iloc[crop]
                else:
                    values_of_water[str(y)] = irrigation_set_wet['MARGINAL_VALUE_OF_WATER'].iloc[crop]                    
    #                 #print(y)
    #                 #print(crop)
    # value_of_water[i] = values_of_water.fillna(irrigation_set_wet['MARGINAL_VALUE_OF_WATER'].iloc[-1], inplace=True)               
    value_of_water_wet[i] = values_of_water

structures_by_yearly_water_value_wet = pd.DataFrame()
for i in irrigation_structure_ids_TWO:
    structures_by_yearly_water_value_wet[i] = value_of_water_wet[i]

yearly_water_values_forattach_wet = structures_by_yearly_water_value_wet.transpose()

max_value_wet = np.nanmax(yearly_water_values_forattach_wet.iloc[:, 1].values)

irrigation_uses_by_year_wet = {}
consumptive_use_by_structure_wet = {}
for y in range(1950,2013):
    irrigation_uses_by_year_wet[y] = uses_of_water_wet[y].groupby(['CROP_TYPE'], as_index=False)['AG_DEMAND'].sum()
    consumptive_use_by_structure_wet[y] = uses_of_water_wet[y].groupby(['StateMod_Structure'], as_index=False)['DELIVERED_TO_STRUCTURE'].sum()
    

### SPACE
columnnames = {}
count = 1949
for i in yearly_water_values_forattach_wet.columns:
    count +=1
    columnnames[i] = f"value_{count}_wet"
yearly_water_values_forattach_wet.rename(columns = columnnames, inplace = True)
yearly_water_values_forattach_wet.columns


yearly_water_values_forattach_wet['StateMod_Structure'] = yearly_water_values_forattach_wet.index
#map_df_update = map_df_update.drop('StateMod_Structure', axis=1)

from geopandas import GeoDataFrame

map_df_update_wet = pd.merge(map_df_update, yearly_water_values_forattach_wet, on="StateMod_Structure")

map_df_update_wet = gpd.GeoDataFrame(map_df_update_wet)
map_df_update_wet = map_df_update_wet.to_crs(epsg=3857)

for i in range(1950,2013):
    # create the colorbar
    norm2 = colors.Normalize(vmin=0, vmax=max_value_dry)
    cbar = plt.cm.ScalarMappable(norm=norm2, cmap='jet')

    fig, ax = plt.subplots(1, figsize =(24, 8))
    ax.set_ylim([4820000, 5050000])
    ax.set_xlim([-11780000, -11375000])
    # src_basemap = cx.providers.Stamen.Terrain
    # cx.add_basemap( ax, source=src_basemap,crs = map_df.crs )
    #cx.add_basemap( ax, source=src_basemap, alpha=0.6, zorder=8, crs = map_df.crs )
    map_df_update_wet.plot(column= f"value_{i}_wet", categorical=False, cmap='jet', linewidth=.2, edgecolor='0.4',
                legend=False, ax=ax)
    northern_water_boundary[['geometry']].plot(facecolor="none", edgecolor="black",linewidth=.8, ax=ax)
    ax_cbar = fig.colorbar(cbar, ax=ax)
    # add label for the colorbar
    ax_cbar.set_label(label='Marginal Value of Water ($/AF)',size = 15, weight='bold')
    ax.axis('off')
    ax.grid (False)
    ax.set_title('South Platte Two-Way Option Market in a Wet Hydrologic Year: ' + str(i) ,fontsize=20, weight='bold')
    # src_basemap = cx.providers.Stamen.Terrain
    # cx.add_basemap( ax, source=src_basemap, alpha=0.6, zorder=8, crs = map_df.crs )
    plt.tight_layout()
    
###################################################################################################################
## append values of water in wet year ####

for y in range(1950,2013):
    for crop in irrigation_uses_by_year_wet[y]['CROP_TYPE'].unique():
        irrigation_uses_by_year_wet[y].loc[irrigation_uses_by_year_wet[y]['CROP_TYPE'] == crop, 'MARGINAL_VALUE_OF_WATER'] = map_df_aggregated_crops_sorted.loc[map_df_aggregated_crops_sorted['CROP_TYPE'] == crop,'MARGINAL_VALUE_OF_WATER'].mean()

for i in range(1950,2013):
    irrigation_uses_by_year_wet[i] = irrigation_uses_by_year_wet[i].sort_values(by=['MARGINAL_VALUE_OF_WATER'], ascending=False)
    irrigation_uses_by_year_wet[i] = irrigation_uses_by_year_wet[i][irrigation_uses_by_year_wet[i]['AG_DEMAND'] != 0]


structure_uses_by_year_wet = {}
for y in range(1950,2013):
    structure_uses_by_year_wet [y] = uses_of_water_wet[y].sort_values(by=['MARGINAL_VALUE_OF_WATER'], ascending=False)

#################################################################################################################
# make demand functions #################

new_test = {}
#year_df = pd.DataFrame()
for i in range(1950,2013):
    irrigation_uses_by_year_wet[i]['CUM_AG_DEMAND'] = 0
        #use_of_water = pd.DataFrame()
    water = 0
    for crop in (range(len(irrigation_uses_by_year_wet[i]))):  
        water += irrigation_uses_by_year_wet[i]['AG_DEMAND'].iloc[crop]
        print(water)

        irrigation_uses_by_year_wet[i]['CUM_AG_DEMAND'].iloc[crop] = water 
        irrigation_uses_by_year_wet[i]['NUMERIC_CROP_TYPE'] = pd.factorize(irrigation_uses_by_year_wet[i]['CROP_TYPE'])[0]
    #plt.scatter(irrigation_uses_by_year_wet[i]['CUM_AG_DEMAND'],irrigation_uses_by_year_wet[i]['MARGINAL_VALUE_OF_WATER'])


    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(data=irrigation_uses_by_year_wet[i], x='CUM_AG_DEMAND', y='MARGINAL_VALUE_OF_WATER', c='NUMERIC_CROP_TYPE', cmap='Set3')
    ax.set(xlabel='Available Water Supply (AF)', ylabel='Marginal Value of Water ($)')
    #ax.vlines(structure_uses_by_year_dry[1954]['CUM_USAGE'], 0, structure_uses_by_year_dry[1954]['MARGINAL_VALUE_OF_WATER'], linestyle="solid", color = 'NUMERIC_CROP_TYPE')
    ax.set_ylim ([0, 300])
    
    
    plt.show()


for i in range(1950,2013):
    structure_uses_by_year_wet[i]['CUM_AG_DEMAND'] = 0
        #use_of_water = pd.DataFrame()
    water = 0
    for crop in (range(len(structure_uses_by_year_wet[i]))):  
        water += structure_uses_by_year_wet[i]['AG_DEMAND'].iloc[crop]
        print(water)

        structure_uses_by_year_wet[i]['CUM_AG_DEMAND'].iloc[crop] = water 
        structure_uses_by_year_wet[i]['NUMERIC_CROP_TYPE'] = pd.factorize(structure_uses_by_year_wet[i]['CROP_TYPE'])[0]
    #plt.scatter(irrigation_uses_by_year_wet[i]['CUM_AG_DEMAND'],irrigation_uses_by_year_wet[i]['MARGINAL_VALUE_OF_WATER'])


    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(data=structure_uses_by_year_wet[i], x='CUM_AG_DEMAND', y='MARGINAL_VALUE_OF_WATER', c='NUMERIC_CROP_TYPE', cmap='Set3')
    ax.set(xlabel='Irrigation Water Demand (AF)', ylabel='Marginal Value of Water ($)')
    #ax.vlines(structure_uses_by_year_dry[1954]['CUM_USAGE'], 0, structure_uses_by_year_dry[1954]['MARGINAL_VALUE_OF_WATER'], linestyle="solid", color = 'NUMERIC_CROP_TYPE')
    ax.set_ylim ([0, 300])
    
    
    plt.show()

for i in range(1950,2013):

     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'GRASS_PASTURE', 'color'] = 'lawngreen'
     
     ## CSU ag extension - alfalfa hay - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'ALFALFA', 'color'] = 'sandybrown'
     
     ## CSU ag extension - grass hay - western colorado - - 2020, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SNAP_BEANS', 'color'] = 'peru'
     
     
     ## CSU ag extension - irrigated corn - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'CORN_GRAIN', 'color'] = 'yellow'
     
     ## HB ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SMALL_GRAINS', 'color'] = 'burlywood'
     
     
     ## CSU ag extension - sorghum grain - southeastern colorado - - 2017, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SORGHUM_GRAIN', 'color'] = 'wheat'
         
     ## CSU ag extension - sugar beets - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SUGAR_BEETS', 'color'] = 'firebrick'
     
     ## CSU ag extension - soybeans - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'DRY_BEANS', 'color'] = 'brown'
     
     ## CSU ag extension - potatoes - san luis valley - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'POTATOES', 'color'] = 'rosybrown'
     
     ## CSU ag extension - sunflowers - northeastern colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SUNFLOWER', 'color'] = 'gold'
     
     ## CSU ag extension - onions - western colorado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'VEGETABLES', 'color'] = 'tomato'
     
     ## CSU ag extension - irrigated winter wheat - south platte valley - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SPRING_GRAIN', 'color'] = 'goldenrod'
    
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'WHEAT_FALL', 'color'] = 'blue'
     
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'ORCHARD_WO_COVER', 'color'] = 'purple'
    
## hatch styles ###    
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'GRASS_PASTURE', 'design'] = '+o'
     
     ## CSU ag extension - alfalfa hay - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'ALFALFA', 'design'] = 'o'
     
     ## CSU ag extension - grass hay - western hado - - 2020, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SNAP_BEANS', 'design'] = '|'
     
     
     ## CSU ag extension - irrigated corn - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'CORN_GRAIN', 'design'] = '|*'
     
     ## HB ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SMALL_GRAINS', 'design'] = '..'
     
     
     ## CSU ag extension - sorghum grain - southeastern hado - - 2017, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SORGHUM_GRAIN', 'design'] = 'o'
         
     ## CSU ag extension - sugar beets - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SUGAR_BEETS', 'design'] = 'O.'
     
     ## CSU ag extension - soybeans - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'DRY_BEANS', 'design'] = '.'
     
     ## CSU ag extension - potatoes - san luis valley - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'POTATOES', 'design'] = '**'
     
     ## CSU ag extension - sunflowers - northeastern hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SUNFLOWER', 'design'] = '++'
     
     ## CSU ag extension - onions - western hado - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'VEGETABLES', 'design'] = 'o'
     
     ## CSU ag extension - irrigated winter wheat - south platte valley - - 2021, net receipt before factor payments ###
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'SPRING_GRAIN', 'design'] = 'O'
    
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'WHEAT_FALL', 'design'] = 'x'
     
     irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == 'ORCHARD_WO_COVER', 'design'] = '/'
    
for i in range(1950,2013):
    fig, ax = plt.subplots(figsize = (20,12))
    figure_name = 'irrigable_land_demand_fn' + 'f{i}'+'.png'
   
    #make dictionary of total irrigated area for each crop
    #crop type is the dictionary key, crop acreage is the dictionary value
    overall_crop_areas = {}
    for index, row in irrigation_uses_by_year_wet[i].iterrows():
      #get total irrigated acreage (in thousand acres) for all listed crops
      if row['CROP_TYPE'] in overall_crop_areas:
        overall_crop_areas[row['CROP_TYPE']] += row['AG_DEMAND'] / 1000.0
      else:
        overall_crop_areas[row['CROP_TYPE']] = row['AG_DEMAND'] / 1000.0
        
    # #crop marginal net benefits ($/acre) come from enterpirse budget reports from the CSU ag extension
    marginal_value_of_water = {}
    for crop in range(len(irrigation_uses_by_year_wet[i]['CROP_TYPE'])):
        name_of_crop = irrigation_uses_by_year_wet[i]['CROP_TYPE'].iloc[crop]
        marginal_value_of_water[name_of_crop] = irrigation_uses_by_year_wet[i]['MARGINAL_VALUE_OF_WATER'].iloc[crop]

    #sort crops from highest cost to lowest cost
    water_costs = np.zeros(len(marginal_value_of_water))
    crop_list = []
    for crop_cnt, crop_name in enumerate(marginal_value_of_water):
      #MNB = $/ac; et = AF/ac; MNB/et = $/AF
      water_costs[crop_cnt] = marginal_value_of_water[crop_name] #et is ac-in, divide by 12 for ac-ft
      crop_list.append(crop_name)
    #sort crops by $/AF
    sorted_index = np.argsort(water_costs*(-1.0))
    crop_list_new = np.asarray(crop_list)
    sorted_crops = crop_list_new[sorted_index]
    
    running_area = 0.0#total crop acreage across all crop types
    for crop_name in sorted_crops:
      # color_val = range(len(sorted_crops))
      # color_of_crop = irrigation_uses_by_year_wet[i]['color'].iloc[color_val]
      # $/AF of fallowing for current crop type
      total_cost = marginal_value_of_water[crop_name]

      #acreage irrigated for this crop type      
      total_area = overall_crop_areas[crop_name]
      #plot value per acre foot vs. cumulative acreage, with crops ordered from highest value crop to lowest value crop
      #plots a single 'box' for each individual crop
      #color = irrigation_uses_by_year_dry[i].loc[irrigation_uses_by_year_dry[i]['CROP_TYPE'] == crop_name, 'color']
      hat = irrigation_uses_by_year_wet[i].loc[irrigation_uses_by_year_wet[i]['CROP_TYPE'] == crop_name, 'design']
      # ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = color, edgecolor = 'black', linewidth = 2.0, label = crop_name)
      ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = 'None', hatch = hat.values[0], linewidth = 2.0, label = crop_name)
      running_area += total_area
    
      #format plot
      #make sure y ticklabels line up with break points
      ax.set_yticks([0.0, 100.0, 200.0, 300.0, 400.0])
      ax.set_yticklabels(['$0', '$100', '$200', '$300', '$400'])
      ax.set_ylabel('Marginal Value of Water ($/AF)', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
      ax.set_xlabel('Irrigation Water Demand (tAF)', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
      ax.set_ylim([0.0, 200.0])
      ax.set_xlim([0, 370.0])
      ax.legend(prop={'size':20}, loc='upper right')
      #ax.set_xlim([0, running_area * 1.05])
    for item in (ax.get_xticklabels()):
      item.set_fontsize(32)
      item.set_fontname('Gill Sans MT')
    for item in (ax.get_yticklabels()):
      item.set_fontsize(32)
      item.set_fontname('Gill Sans MT')
    plt.show()
    plt.savefig(figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()

TWO_Selection_Wet = {}
for y in range(1950,2013):
    two_year_df_wet = pd.DataFrame()    
    irrigation_set_year_wet = uses_of_water_wet[y].sort_values(by=['MARGINAL_VALUE_OF_WATER'], ascending=False)
    irrigation_set_year_wet['TWO_USAGE'] = 0
    #march CBI test
    if y in [1950, 1952, 1958, 1962, 1969, 1970, 1971, 1972, 1973, 1974, 1980, 1984, 1985, 1986, 1988, 1989, 1996, 1997, 1998, 2000, 2006, 2011]:
    #if y in [1962,1970,1971,1972,1974,1980,1984,1985,1986,1988,1996,1997,2011]:
        muni_surplus_available = surplus.Lease_Amounts_by_CBI_H[y]
    else:
        muni_surplus_available = 0
    for crop in range(len(irrigation_set_year)):
        if muni_surplus_available > irrigation_set_year_wet['AG_DEMAND'].iloc[crop]:
            use = irrigation_set_year_wet['AG_DEMAND'].iloc[crop]
        else:
            use = max(muni_surplus_available, 0)
        irrigation_set_year_wet['TWO_USAGE'].iloc[crop] = use
        muni_surplus_available -= irrigation_set_year_wet['AG_DEMAND'].iloc[crop]

    TWO_Selection_Wet[y] = irrigation_set_year_wet
    