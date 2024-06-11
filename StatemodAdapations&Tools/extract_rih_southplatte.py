# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:37:52 2022

@author: zacha
"""

import os
import re
import numpy as np
import pandas as pd
from SALib.sample import latin
from joblib import Parallel, delayed

# set random seed for reproducibility
seed_value = 123

# directory where the data is stored
data_dir = 'C:/Users/zacha/Documents/UNC/SP2016_StateMod'

# template file as a source for modification
template_file = os.path.join(data_dir, "SP2016.rih")

# directory to write modified files to
output_dir = "C:/Users/zacha/Documents/UNC"

# scenario name
scenario = "test"

# character indicating row is a comment
comment = "#"

# dictionary to hold values for each field
d = {"yr": [], 
     "id": [], 
     "jan": [], 
     "feb": [], 
     "mar": [], 
     "apr": [], 
     "may": [], 
     "jun": [], 
     "jul": [], 
     "aug": [],
     "sep": [],
     "oct": [],
     "nov": [],
     "dec": [],
     "total": []}

# define the column widths for the output file
column_widths = {"yr": 4, 
                 "id": 9, 
                 "jan": 10, 
                 "feb": 7, 
                 "mar": 7, 
                 "apr": 7, 
                 "may": 7, 
                 "jun": 7, 
                 "jul": 7, 
                 "aug": 7,
                 "sep": 7,
                 "oct": 7,
                 "nov": 7,
                 "dec": 7,
                 "total": 9}

# list of columns to process
column_list = ["yr", "id", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "total"]

# list of value columns that may be modified
value_columns = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "total"]

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

# #%%time

# # ids that have been identified as municipal
# municipal_ids = ["06_BOULDER_O", "06_BOULDER_I", "05LONG_IN"]

# municial_multiplier = 0.2

# # modify value columns associated with municipal ids
# df[value_columns] = (df[value_columns] * municial_multiplier).where(df["id"].isin(municipal_ids), df[value_columns])

# # apply precision adjustment function to match statemods format
# df[value_columns] = df[value_columns].apply(np.around)

# # convert all fields to str type
# df = df.astype(str)

# # if adjusting precision, remove trailing 0
# df[value_columns] = df[value_columns].apply(lambda x: x.str[:-1])

df
df.to_csv('south_platte_inflows.csv')

south_platte_tributaries = ['06724000', '06744000', '06727000', '06752500']
df2 = df[df['id'].isin(south_platte_tributaries)] 

df2.to_csv('south_platte_inflows.csv')

# #%%time

template_basename = os.path.basename(template_file)
template_name_parts = os.path.splitext(template_basename)

output_file = os.path.join(output_dir, f"{template_name_parts[0]}_{scenario}{template_name_parts[-1]}")

formatters={'id':'{{:<{}s}}'.format(df['id'].str.len().max()).format}

with open(output_file, "w") as out:
    
     # write header
     out.write(header)

     # write altered content
     df.to_string(buf=out,
                  col_space=column_widths,
                  header=False,
                  index=False,
                  formatters={'id':'{{:<{}s}}'.format(df['id'].str.len().max()).format})