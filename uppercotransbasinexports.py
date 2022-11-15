# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 09:56:21 2022

@author: zacha
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:37:52 2022

@author: zacha
"""

import os

import numpy as np
import pandas as pd
from SALib.sample import latin
from joblib import Parallel, delayed

# set random seed for reproducibility
seed_value = 123

# directory where the data is stored
data_dir = 'C:/Users/zacha/Documents/UNC/cm2015_StateMod/StateMod'

# template file as a source for modification
template_file = os.path.join(data_dir, "cm2015B.ddm")

# directory to write modified files to
output_dir = "C:/Users/zacha/Documents/UNC"

# scenario name
scenario = "roberts tunnel"

# character indicating row is a comment
comment = "#"

# dictionary to hold values for each field
d = {"yr": [], 
     "id": [], 
     "oct": [], 
     "nov": [], 
     "dec": [], 
     "jan": [], 
     "feb": [], 
     "mar": [], 
     "apr": [], 
     "may": [],
     "jun": [],
     "jul": [],
     "aug": [],
     "sep": [],
     "total": []}

# define the column widths for the output file
column_widths = {"yr": 4, 
                 "id": 9, 
                 "oct": 10, 
                 "nov": 7, 
                 "dec": 7, 
                 "jan": 7, 
                 "feb": 7, 
                 "mar": 7, 
                 "apr": 7, 
                 "may": 7,
                 "jun": 7,
                 "jul": 7,
                 "aug": 7,
                 "sep": 7,
                 "total": 9}

# list of columns to process
column_list = ["yr", "id", "oct", "nov", "dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "total"]

# list of value columns that may be modified
value_columns = ["oct", "nov", "dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "total"]

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
df = pd.DataFrame(d)

# convert value column types to float
df[value_columns] = df[value_columns].astype(np.float64)

df

df.describe()


#ROBERTS TUNNEL
roberts_tunnel_df = df.loc[df['id'] == '3604684']
roberts_tunnel_df['yr'] = roberts_tunnel_df['yr'].astype(int)
roberts_tunnel_df_selection = roberts_tunnel_df.iloc[0:105, 2:14]
roberts_tunnel_df_update = pd.DataFrame(roberts_tunnel_df_selection.values.ravel(),columns = ['column'])
roberts_tunnel_df_update['date'] = pd.date_range(start='10/1/1909', periods=len(roberts_tunnel_df_update), freq='M')
roberts_tunnel_df_update.index = roberts_tunnel_df_update['date']
roberts_tunnel_df_update['month'] = roberts_tunnel_df_update.index.month
roberts_tunnel_df_update['year'] = roberts_tunnel_df_update.index.year

roberts_tunnel_sp_imports = roberts_tunnel_df_update.iloc[483:1239, 0:5]
roberts_tunnel_sp_imports['index'] = range(0,756)
roberts_tunnel_sp_imports.index = roberts_tunnel_sp_imports['index']


#MOFFAT TUNNEL
moffat_tunnel_df = df.loc[df['id'] == '5104655']
moffat_tunnel_df['yr'] = moffat_tunnel_df['yr'].astype(int)
moffat_tunnel_df_selection = moffat_tunnel_df.iloc[0:105, 2:14]
moffat_tunnel_df_update = pd.DataFrame(moffat_tunnel_df_selection.values.ravel(),columns = ['column'])
moffat_tunnel_df_update['date'] = pd.date_range(start='10/1/1909', periods=len(moffat_tunnel_df_update), freq='M')
moffat_tunnel_df_update.index = moffat_tunnel_df_update['date']
moffat_tunnel_df_update['month'] = moffat_tunnel_df_update.index.month
moffat_tunnel_df_update['year'] = moffat_tunnel_df_update.index.year

moffat_tunnel_sp_imports = moffat_tunnel_df_update.iloc[483:1239, 0:5]
moffat_tunnel_sp_imports['index'] = range(0,756)
moffat_tunnel_sp_imports.index = moffat_tunnel_sp_imports['index']


#ADAMS TUNNEL
adams_tunnel_df = df.loc[df['id'] == '5104634']
adams_tunnel_df['yr'] = adams_tunnel_df['yr'].astype(int)
adams_tunnel_df_selection = adams_tunnel_df.iloc[0:105, 2:14]
adams_tunnel_df_update = pd.DataFrame(adams_tunnel_df_selection.values.ravel(),columns = ['column'])
adams_tunnel_df_update['date'] = pd.date_range(start='10/1/1909', periods=len(adams_tunnel_df_update), freq='M')
adams_tunnel_df_update.index = adams_tunnel_df_update['date']
adams_tunnel_df_update['month'] = adams_tunnel_df_update.index.month
adams_tunnel_df_update['year'] = adams_tunnel_df_update.index.year

adams_tunnel_sp_imports = adams_tunnel_df_update.iloc[483:1239, 0:5]
adams_tunnel_sp_imports['index'] = range(0,756)
adams_tunnel_sp_imports.index = adams_tunnel_sp_imports['index']