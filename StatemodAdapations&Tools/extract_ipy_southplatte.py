# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:37:52 2022

@author: zacha
"""

import os
import numpy as np
import pandas as pd
#import sp_irrigation_final as spirr

# set random seed for reproducibility
seed_value = 123

# directory where the data is stored
data_dir = 'C:/Users/zacha/Documents/UNC/SP2016_StateMod'

# template file as a source for modification
template_file = os.path.join(data_dir, "SP2016.ipy")

# directory to write modified files to
output_dir = "C:/Users/zacha/Documents/UNC"

# scenario name
scenario = "test"

# character indicating row is a comment
comment = "#"

# dictionary to hold values for each field
d = {"yr": [], 
     "id": [], 
     "maxsurf": [], 
     "floodeff": [], 
     "spr": [], 
     "acswfl": [], 
     "acswsp": [], 
     "acgwfl": [], 
     "acgwspr": [], 
     "pumpingmax": [],
     "gmode": [],
     "actotal": [],
     "acsw": [],
     "acgw": []}

# define the column widths for the output file
column_widths = {"yr": 4, 
                 "id": 9, 
                 "maxsurf": 10, 
                 "floodeff": 7, 
                 "spr": 7, 
                 "acswfl": 7, 
                 "acswsp": 7, 
                 "acgwfl": 7, 
                 "acgwspr": 7, 
                 "pumpingmax": 7,
                 "gmode": 7,
                 "actotal": 7,
                 "acsw": 7,
                 "acgw": 7}

# list of columns to process
column_list = ["yr", "id", "maxsurf", "floodeff", "spr", "acswfl", "acswsp", "acgwfl", "acgwspr", "pumpingmax", "gmode", "actotal", "acsw", "acgw"]

# list of value columns that may be modified
value_columns = ["maxsurf", "floodeff", "spr", "acswfl", "acswsp", "acgwfl", "acgwspr", "pumpingmax", "gmode", "actotal", "acsw", "acgw"]

#%%time

# empty string to hold header data
header = ""

capture = False
with open(template_file) as template:
    
    for idx, line in enumerate(template):
        
        if capture:
            
            # strip newline and split on spaces
            value_list = line.strip().split()

            # comprehension only used to build dict
            x = [d[column_list[idx]].append(i) for idx, i in enumerate(value_list)]

        else:
            
            # store any header and preliminary lines to use in restoration
            header += line
            
            # passes uncommented date range line before data start and all commented lines in header
            if line[0] != comment:
                capture = True
                
# convert dictionary to a pandas data frame  
historical_irrigation_efficiencies_acreage_table = pd.DataFrame(d)

historical_irrigation_efficiencies_acreage_table[value_columns] = historical_irrigation_efficiencies_acreage_table[value_columns].astype(np.float64)

historical_irrigation_efficiencies_acreage_table

historical_irrigation_efficiencies_acreage_table_selection = historical_irrigation_efficiencies_acreage_table.loc[historical_irrigation_efficiencies_acreage_table['yr'] == '2010']

# irrigation_efficiencies_2012 = {}
# irrigation_efficiencies = {}
# for i in spirr.irrigation_structure_ids_list:
#     irrigation_efficiencies_2012[i] = historical_irrigation_efficiencies_acreage_table_selection.loc[historical_irrigation_efficiencies_acreage_table_selection['id'] == i]
#     irrigation_efficiencies_2012[i]['yr'] = irrigation_efficiencies_2012[i]['yr'].astype(int)
#     irrigation_efficiencies[i] = irrigation_efficiencies_2012[i].iloc[:, 2:14]
#     irrigation_efficiencies[i] = pd.DataFrame(irrigation_efficiencies[i].values.ravel(),columns = ['demand'])
#     irrigation_efficiencies[i] = pd.concat([irrigation_efficiencies[i]]*63, ignore_index=True)
#     irrigation_efficiencies[i]['date'] = pd.date_range(start='01/1/1950', periods=len(irrigation_efficiencies[i]), freq='M')
#     irrigation_efficiencies[i].index = irrigation_efficiencies[i]['date']
#     irrigation_efficiencies[i]['month'] = irrigation_efficiencies[i].index.month
#     irrigation_efficiencies[i]['year'] = irrigation_efficiencies[i].index.year
#     irrigation_efficiencies[i]['index'] = range(0,756)
#     irrigation_efficiencies[i].index = irrigation_efficiencies[i]['index']

# irrigation_efficiencies_statemod = pd.DataFrame()

# for i in spirr.irrigation_structure_ids_list:
#     irrigation_efficiencies_statemod[f"{i}"] = irrigation_efficiencies[i]['demand']
# irrigation_efficiencies_statemod = irrigation_efficiencies_statemod.dropna(how='all', axis=1)
# print(irrigation_efficiencies_statemod)

# convert value column types to float
historical_irrigation_efficiencies_acreage_table[value_columns] = historical_irrigation_efficiencies_acreage_table[value_columns].astype(np.float64)

historical_irrigation_efficiencies_acreage_table

historical_irrigation_efficiencies_acreage_table_selection = historical_irrigation_efficiencies_acreage_table.loc[historical_irrigation_efficiencies_acreage_table['yr'] == '2010']

historical_irrigation_efficiencies_acreage_table_selection['pctsurfacewater'] = historical_irrigation_efficiencies_acreage_table_selection['acsw'] / historical_irrigation_efficiencies_acreage_table_selection['actotal']
historical_irrigation_efficiencies_acreage_table_selection['pctsurfacewater_flood'] = historical_irrigation_efficiencies_acreage_table_selection['acswfl']/ historical_irrigation_efficiencies_acreage_table_selection['acsw']
historical_irrigation_efficiencies_acreage_table_selection['pctsurfacewater_sprinkler'] = historical_irrigation_efficiencies_acreage_table_selection['acswsp']/ historical_irrigation_efficiencies_acreage_table_selection['acsw']
historical_irrigation_efficiencies_acreage_table_selection['StateMod_Structure'] = historical_irrigation_efficiencies_acreage_table_selection['id']
historical_irrigation_efficiencies_acreage_table_selection.index = historical_irrigation_efficiencies_acreage_table_selection['StateMod_Structure']

ipydata = pd.DataFrame()
ipydata['maxsurf'] = historical_irrigation_efficiencies_acreage_table_selection['maxsurf']
ipydata['floodeff'] = historical_irrigation_efficiencies_acreage_table_selection['floodeff']
ipydata['spr'] = historical_irrigation_efficiencies_acreage_table_selection['spr']
ipydata['pctsurfacewater'] = historical_irrigation_efficiencies_acreage_table_selection['pctsurfacewater']

#check 2012 acreages

statemod_acreages = historical_irrigation_efficiencies_acreage_table_selection['actotal']

