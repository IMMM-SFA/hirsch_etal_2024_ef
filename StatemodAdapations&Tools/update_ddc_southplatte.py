# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:08:58 2023

@author: zacha
"""

import os
#import re
import numpy as np
import pandas as pd
#from SALib.sample import latin
#from joblib import Parallel, delayed
# import re
# import matplotlib.pyplot as plt
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
# import geopandas as gpd
#import sp_irrigation_final as spirr
import statemod_irrigation_structureid_extractor as ids



################################################################################

## EXTRACT .ddm file values for use in the updated .ddm #################

# set random seed for reproducibility
seed_value = 123

# directory where the data is stored
data_dir = 'C:/Users/zacha/Documents/UNC/SP2016_StateMod'

# template file as a source for modification
template_file = os.path.join(data_dir, "SP2016_Limited.ddc")

# directory to write modified files to
output_dir = "C:/Users/zacha/Documents/UNC/SP2016_StateMod"

# scenario name
scenario = "irr_test"

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
statemod_consumptive_use_df = pd.DataFrame(d)


# convert value column types to float
statemod_consumptive_use_df[value_columns] = statemod_consumptive_use_df[value_columns].astype(np.float64)

statemod_consumptive_use_df

statemod_consumptive_use_df_selection = statemod_consumptive_use_df.loc[statemod_consumptive_use_df['yr'] == '2010']

statemod_consumptive_use_2010 = {}
statemod_consumptive_use = {}
for i in ids.irrigation_structure_ids_list:
    statemod_consumptive_use_2010[i] = statemod_consumptive_use_df_selection.loc[statemod_consumptive_use_df_selection['id'] == i]
    statemod_consumptive_use_2010[i]['yr'] = statemod_consumptive_use_2010[i]['yr'].astype(int)
    statemod_consumptive_use[i] = statemod_consumptive_use_2010[i].iloc[:, 2:14]
    statemod_consumptive_use[i] = pd.DataFrame(statemod_consumptive_use[i].values.ravel(),columns = ['demand'])
    statemod_consumptive_use[i] = pd.concat([statemod_consumptive_use[i]]*63, ignore_index=True)
    statemod_consumptive_use[i]['date'] = pd.date_range(start='01/1/1950', periods=len(statemod_consumptive_use[i]), freq='M')
    statemod_consumptive_use[i].index = statemod_consumptive_use[i]['date']
    statemod_consumptive_use[i]['month'] = statemod_consumptive_use[i].index.month
    statemod_consumptive_use[i]['year'] = statemod_consumptive_use[i].index.year
    statemod_consumptive_use[i]['index'] = range(0,756)
    statemod_consumptive_use[i].index = statemod_consumptive_use[i]['index']

statemod_consumptive_use_adapted_df = pd.DataFrame()

for i in ids.irrigation_structure_ids_list:
    statemod_consumptive_use_adapted_df[f"{i}"] = statemod_consumptive_use[i]['demand']
statemod_consumptive_use_adapted_df = statemod_consumptive_use_adapted_df.dropna(how='all', axis=1)
print(statemod_consumptive_use_adapted_df)

def read_text_file(filename):
  with open(filename,'r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  return all_split_data

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test')

consumptive_use_data = read_text_file('SP2016_Limited.DDC')

start_year = 1950

def writenewDDC(consumptive_use_data, statemod_consumptive_use_adapted_df, start_year, scenario_name):    
  new_data = []
  use_value = 0
  start_loop = 0
  col_start = [0, 5, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113]
  for n in ids.irrigation_structure_ids_list: 
      for i in range(0, len(consumptive_use_data)):
        if use_value == 1:
          start_loop = 1
        if consumptive_use_data[i][0] != '#':
          use_value = 1
        if start_loop == 1:
          use_line = True
          row_data = []
          value_use = []
          for col_loc in range(0, len(col_start)):
            if col_loc == len(col_start) - 1:
              value_use.append(consumptive_use_data[i][col_start[col_loc]:].strip())
            else:          
              value_use.append(consumptive_use_data[i][col_start[col_loc]:(col_start[col_loc + 1])].strip())
          try:
            year_num = int(value_use[0])
            structure_name = str(value_use[1]).strip()
          except:
            use_line = False
          if structure_name in ids.irrigation_structure_ids_list:
            index_val = (year_num - start_year) * 12
            new_demands = np.zeros(13)
            for i in range(0,12):
                new_demands[i] = statemod_consumptive_use_adapted_df.loc[index_val + i, structure_name]
            new_demands[12] = np.sum(new_demands)
            row_data.append(str(year_num))
            row_data.append(structure_name)
            for i in range(0,13):
               row_data.append(str(int(float(new_demands[i]))))
          else:
             for i in range(0,len(value_use)):
                 if i == 1:
                     row_data.append(str(value_use[i]))
                 else:  
                     row_data.append(str(int(float(value_use[i]))))  
          new_data.append(row_data)  
        
        # write new data to file
      f = open('SP2016_' + scenario_name + '.ddc','w')
      # write firstLine # of rows as in initial file
      i = 0
      while consumptive_use_data[i][0] == '#':
        f.write(consumptive_use_data[i])
        i += 1
      f.write(consumptive_use_data[i])
      i+=1
      for i in range(len(new_data)):
        # write year, ID and first month of adjusted data
        structure_length = len(new_data[i][1])
        f.write(new_data[i][0] + ' ' + new_data[i][1] + (12-structure_length)*' ')
        # write all but last month of adjusted data
        for j in range(len(new_data[i])-3):
          f.write((8-len(new_data[i][j+2])-2)*' ' + new_data[i][j+2] + '.0')                
            # write last month of adjusted data
        f.write((10-len(new_data[i][j+3])-2)*' ' + new_data[i][j+3] + '.0' + '\n')             
      f.close()
        
      return None

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test')

writenewDDC(consumptive_use_data, statemod_consumptive_use_adapted_df, 1950, 'A')

