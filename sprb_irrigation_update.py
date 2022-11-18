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
    if row == ['0100503', '0100504', '0100710']:
        result = '0100503_I'
    elif row == ['0100506', '0100507', '0100509']:
        result = '0100507_I'
    elif row == ['0100519', '0100521', '0100522', '0100523']:
        result = '0100519_D'
    elif row == ['0100687', '6400563', '6403794']:
        result = '0100687_I'
    elif row == ['0103817']:
        result = '0103817_I'
    elif row == ['0200805', '0200902']:
        result ='0200805_I'
    elif row == '0700547':
        result = '0700547_I'
    else:
        result = 1
    return result


 
print(map_df)

map_df['StateMod_Structure'] = map_df['SW_WDID1'].apply(assign_irrigation)
print(map_df['StateMod_Structure'])

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
gplt.choropleth(map_df, hue='crops', cmap='Blues', scheme='quantiles',
                legend=True, legend_labels=legend_labels)

# leg1 = ax.get_legend()
# leg1.set_title("South Platte Irrigation")
# ax.title.set_text("South Platte Irrigation")
# plt.show()