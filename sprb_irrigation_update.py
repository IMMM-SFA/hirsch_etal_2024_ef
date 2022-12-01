# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:01:50 2022

@author: zacha
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal
import rasterio
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
import fiona
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
from skimage import exposure
import seaborn as sns
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import geopandas as gpd
import seaborn as sns
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
#import shapefile as shp
import descartes

# irrigaton_nodes = []

# # Creating a dictionary
# d = {}

# # Using for loop for creating dataframes
# for i in l:
#     d[i] = pd.DataFrame()


fp = 'Div1_Irrig_2020.shp'
map_df = gpd.read_file(fp) 
map_df_copy = gpd.read_file(fp)
print(map_df)

test = map_df.loc[:,['CROP_TYPE', 'geometry']]
test['CROP_TYPE'] = int(10)
test['CROP_TYPE'] = pd.to_numeric(test.CROP_TYPE)
test.plot()

map_df[['CROP_TYPE','geometry']].plot()

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
        result = 1
    return result


 
print(map_df)

map_df['StateMod_Structure'] = map_df['SW_WDID1'].apply(assign_irrigation)
print(map_df['StateMod_Structure'])

map_df['StateMod_Structure'].unique()

# map_df['StateMod_Structure'] = 0
# if map_df['SW_WDID1'] == ['0100503', '0100504', '0100710'] :
#     map_df['StateMod_Structure'] = '0100503_I'

    
 



fig, ax = plt.subplots(1, figsize =(16, 8))
map_df.plot(ax = ax, color ='black')
#map_df.plot(ax = ax, column =['CROP_TYPE','geometry'], cmap ='Reds')


#map_df['CROP_NUM'] = 0
#if map_df['CROP_TYPE'] == "GRASS_PASTURE", map_df['CROP_NUM' == 1]




df = pd.DataFrame()
map_df['crop'] = 0


for c, crop in enumerate(map_df['CROP_TYPE'].unique()):
    # df_2 = pd.DataFrame({'crop' : crop, 'num' : c})
    # df_2['c'] = c
    map_df['crop'].loc[map_df['CROP_TYPE'] == crop] = c
    #df= pd.concat([df,df_2])
 
print(df)

print(map_df['crop'])

fig, ax = plt.subplots(1, figsize =(24, 8))
ax.set_ylim([4400000, 4550000])
ax.set_xlim([460000, 750000])
map_df.plot(ax = ax, cmap='rainbow')
plt.legend()


#legend_labels = ['Corn', 'Alfalfa', 'Barley', 'Dry Beans', 'Pasture', 'Potatoes', 'Small Grain', 'Sorghum Grain', 'Sugar Beets', 'Sunflowers', 'Vegetables', 'Wheat Spring']
# ax.set_axis_off()
fig, ax = plt.subplots(1, figsize =(24, 8))
ax.set_ylim([4400000, 4550000])
ax.set_xlim([460000, 750000])
map_df.plot(column='CROP_TYPE', cmap = 'jet', legend = True, 
            categorical=True, ax=ax)
#ax.legend(bbox_to_anchor=(1.35, 1), loc='upper right', borderaxespad=0)

# leg1 = ax.get_legend()
# leg1.set_title("Crop Type")
# new_legtxt = ['Corn', 'Alfalfa', 'Barley', 'Dry Beans', 'Pasture', 'Potatoes', 'Small Grain', 'Sorghum Grain', 'Sugar Beets', 'Sunflowers', 'Vegetables', 'Wheat Spring']
# for ix,eb in enumerate(leg1.get_texts()):
#     print(eb.get_text(), "-->", new_legtxt[ix])
#     eb.set_text(new_legtxt[ix])


# plt.show()




legend_labels = ['Corn', 'Alfalfa', 'Barley', 'Dry Beans', 'Pasture', 'Potatoes', 'Small Grain', 'Sorghum Grain', 'Sugar Beets', 'Sunflowers', 'Vegetables', 'Wheat Spring']
#gplt.choropleth(map_df, hue='crops', cmap='Blues', scheme='quantiles',
#                legend=True, legend_labels=legend_labels)

# leg1 = ax.get_legend()
# leg1.set_title("South Platte Irrigation")
# ax.title.set_text("South Platte Irrigation")
# plt.show()