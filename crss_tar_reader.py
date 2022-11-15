# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 22:58:08 2022

@author: zacha
"""

import numpy as np 
import os
import re
import pandas as pd
from SALib.sample import latin
from joblib import Parallel, delayed
import re
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import scipy.stats as stats
import seaborn as sns
from datetime import datetime
from matplotlib.lines import Line2D
import os


# set random seed for reproducibility
seed_value = 123

# directory where the data is stored
data_dir = 'C:/Users/zacha/Documents/UNC/SP2016_StateMod'

# template file as a source for modification
template_file = os.path.join(data_dir, "SP2016_H.tar")

# directory to write modified files to
output_dir = "C:/Users/zacha/Documents/UNC/SP2016_StateMod"

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

###################################################################################################################################

chatfield_df = pd.DataFrame()
chatfield_df = df.loc[df['id'] == '0803514']
chatfield_df['yr'] = chatfield_df['yr'].astype(int)
chatfield_df_selection = chatfield_df.iloc[125:126, 2:14]
chatfield_df_selection_update = pd.DataFrame(chatfield_df_selection.values.ravel(),columns = ['0803514'])
chatfield_repeated = pd.concat([chatfield_df_selection_update]*63, ignore_index=True)
chatfield_repeated['index'] = range(0,756)
chatfield_repeated.index = chatfield_repeated['index']
print(chatfield_repeated)

strontia_df = pd.DataFrame()
strontia_df = df.loc[df['id'] == '0803983']
strontia_df['yr'] = strontia_df['yr'].astype(int)
strontia_df_selection = strontia_df.iloc[125:126, 2:14]
strontia_df_selection_update = pd.DataFrame(strontia_df_selection.values.ravel(),columns = ['0803983'])
strontia_repeated = pd.concat([strontia_df_selection_update]*63, ignore_index=True)
strontia_repeated['index'] = range(0,756)
strontia_repeated.index = strontia_repeated['index']
print(strontia_repeated)

carter_df = pd.DataFrame()
carter_df = df.loc[df['id'] == '0404513']
carter_df['yr'] = carter_df['yr'].astype(int)
carter_df_selection = carter_df.iloc[125:126, 2:14]
carter_df_selection_update = pd.DataFrame(carter_df_selection.values.ravel(),columns = ['0404513'])
carter_repeated = pd.concat([carter_df_selection_update]*63, ignore_index=True)
carter_repeated['index'] = range(0,756)
carter_repeated.index = carter_repeated['index']
print(carter_repeated)

buttonrock_df = pd.DataFrame()
buttonrock_df = df.loc[df['id'] == '0504010']
buttonrock_df['yr'] = buttonrock_df['yr'].astype(int)
buttonrock_df_selection = buttonrock_df.iloc[125:126, 2:14]
buttonrock_df_selection_update = pd.DataFrame(buttonrock_df_selection.values.ravel(),columns = ['0504010'])
buttonrock_repeated = pd.concat([buttonrock_df_selection_update]*63, ignore_index=True)
buttonrock_repeated['index'] = range(0,756)
buttonrock_repeated.index = buttonrock_repeated['index']
print(buttonrock_repeated)

boulderres_df = pd.DataFrame()
boulderres_df = df.loc[df['id'] == '0504515']
boulderres_df['yr'] = boulderres_df['yr'].astype(int)
boulderres_df_selection = boulderres_df.iloc[125:126, 2:14]
boulderres_df_selection_update = pd.DataFrame(boulderres_df_selection.values.ravel(),columns = ['0504515'])
boulderres_repeated = pd.concat([boulderres_df_selection_update]*63, ignore_index=True)
boulderres_repeated['index'] = range(0,756)
boulderres_repeated.index = boulderres_repeated['index']
print(boulderres_repeated)

gross_df = pd.DataFrame()
gross_df = df.loc[df['id'] == '0604199']
gross_df['yr'] = gross_df['yr'].astype(int)
gross_df_selection = gross_df.iloc[125:126, 2:14]
gross_df_selection_update = pd.DataFrame(gross_df_selection.values.ravel(),columns = ['0604199'])
gross_repeated = pd.concat([gross_df_selection_update]*63, ignore_index=True)
gross_repeated['index'] = range(0,756)
gross_repeated.index = gross_repeated['index']
print(gross_repeated)

bullres_df = pd.DataFrame()
bullres_df = df.loc[df['id'] == '0203351']
bullres_df['yr'] = bullres_df['yr'].astype(int)
bullres_df_selection = bullres_df.iloc[125:126, 2:14]
bullres_df_selection_update = pd.DataFrame(bullres_df_selection.values.ravel(),columns = ['0203351'])
bullres_repeated = pd.concat([bullres_df_selection_update]*63, ignore_index=True)
bullres_repeated['index'] = range(0,756)
bullres_repeated.index = bullres_repeated['index']
print(bullres_repeated)

aurorares_df = pd.DataFrame()
aurorares_df = df.loc[df['id'] == '0203379']
aurorares_df['yr'] = aurorares_df['yr'].astype(int)
aurorares_df_selection = aurorares_df.iloc[125:126, 2:14]
aurorares_df_selection_update = pd.DataFrame(aurorares_df_selection.values.ravel(),columns = ['0203379'])
aurorares_repeated = pd.concat([aurorares_df_selection_update]*63, ignore_index=True)
aurorares_repeated['index'] = range(0,756)
aurorares_repeated.index = aurorares_repeated['index']
print(aurorares_repeated)

westgravel_df = pd.DataFrame()
westgravel_df = df.loc[df['id'] == '0203699']
westgravel_df['yr'] = westgravel_df['yr'].astype(int)
westgravel_df_selection = westgravel_df.iloc[125:126, 2:14]
westgravel_df_selection_update = pd.DataFrame(westgravel_df_selection.values.ravel(),columns = ['0203699'])
westgravel_repeated = pd.concat([westgravel_df_selection_update]*63, ignore_index=True)
westgravel_repeated['index'] = range(0,756)
westgravel_repeated.index = westgravel_repeated['index']
print(westgravel_repeated)

eastgravel_df = pd.DataFrame()
eastgravel_df = df.loc[df['id'] == '0203700']
eastgravel_df['yr'] = eastgravel_df['yr'].astype(int)
eastgravel_df_selection = eastgravel_df.iloc[125:126, 2:14]
eastgravel_df_selection_update = pd.DataFrame(eastgravel_df_selection.values.ravel(),columns = ['0203700'])
eastgravel_repeated = pd.concat([eastgravel_df_selection_update]*63, ignore_index=True)
eastgravel_repeated['index'] = range(0,756)
eastgravel_repeated.index = eastgravel_repeated['index']
print(eastgravel_repeated)

greenridge_df = pd.DataFrame()
greenridge_df = df.loc[df['id'] == '0403659']
greenridge_df['yr'] = greenridge_df['yr'].astype(int)
greenridge_df_selection = greenridge_df.iloc[125:126, 2:14]
greenridge_df_selection_update = pd.DataFrame(greenridge_df_selection.values.ravel(),columns = ['0403659'])
greenridge_repeated = pd.concat([greenridge_df_selection_update]*63, ignore_index=True)
greenridge_repeated['index'] = range(0,756)
greenridge_repeated.index = greenridge_repeated['index']
print(greenridge_repeated)

foothills_df = pd.DataFrame()
foothills_df = df.loc[df['id'] == '0504071']
foothills_df['yr'] = foothills_df['yr'].astype(int)
foothills_df_selection = foothills_df.iloc[125:126, 2:14]
foothills_df_selection_update = pd.DataFrame(foothills_df_selection.values.ravel(),columns = ['0504071'])
foothills_repeated = pd.concat([foothills_df_selection_update]*63, ignore_index=True)
foothills_repeated['index'] = range(0,756)
foothills_repeated.index = foothills_repeated['index']
print(foothills_repeated)

coorsouth_df = pd.DataFrame()
coorsouth_df = df.loc[df['id'] == '0703010']
coorsouth_df['yr'] = coorsouth_df['yr'].astype(int)
coorsouth_df_selection = coorsouth_df.iloc[125:126, 2:14]
coorsouth_df_selection_update = pd.DataFrame(coorsouth_df_selection.values.ravel(),columns = ['0703010'])
coorsouth_repeated = pd.concat([coorsouth_df_selection_update]*63, ignore_index=True)
coorsouth_repeated['index'] = range(0,756)
coorsouth_repeated.index = coorsouth_repeated['index']
print(coorsouth_repeated)

arvadares_df = pd.DataFrame()
arvadares_df = df.loc[df['id'] == '0703308']
arvadares_df['yr'] = arvadares_df['yr'].astype(int)
arvadares_df_selection = arvadares_df.iloc[125:126, 2:14]
arvadares_df_selection_update = pd.DataFrame(arvadares_df_selection.values.ravel(),columns = ['0703308'])
arvadares_repeated = pd.concat([arvadares_df_selection_update]*63, ignore_index=True)
arvadares_repeated['index'] = range(0,756)
arvadares_repeated.index = arvadares_repeated['index']
print(arvadares_repeated)

jimbaker_df = pd.DataFrame()
jimbaker_df = df.loc[df['id'] == '0703336']
jimbaker_df['yr'] = jimbaker_df['yr'].astype(int)
jimbaker_df_selection = jimbaker_df.iloc[125:126, 2:14]
jimbaker_df_selection_update = pd.DataFrame(jimbaker_df_selection.values.ravel(),columns = ['0703336'])
jimbaker_repeated = pd.concat([jimbaker_df_selection_update]*63, ignore_index=True)
jimbaker_repeated['index'] = range(0,756)
jimbaker_repeated.index = jimbaker_repeated['index']
print(jimbaker_repeated)

coorsnorth_df = pd.DataFrame()
coorsnorth_df = df.loc[df['id'] == '0703389']
coorsnorth_df['yr'] = coorsnorth_df['yr'].astype(int)
coorsnorth_df_selection = coorsnorth_df.iloc[125:126, 2:14]
coorsnorth_df_selection_update = pd.DataFrame(coorsnorth_df_selection.values.ravel(),columns = ['0703389'])
coorsnorth_repeated = pd.concat([coorsnorth_df_selection_update]*63, ignore_index=True)
coorsnorth_repeated['index'] = range(0,756)
coorsnorth_repeated.index = coorsnorth_repeated['index']
print(coorsnorth_repeated)

guanella_df = pd.DataFrame()
guanella_df = df.loc[df['id'] == '0704030']
guanella_df['yr'] = guanella_df['yr'].astype(int)
guanella_df_selection = guanella_df.iloc[125:126, 2:14]
guanella_df_selection_update = pd.DataFrame(guanella_df_selection.values.ravel(),columns = ['0704030'])
guanella_repeated = pd.concat([guanella_df_selection_update]*63, ignore_index=True)
guanella_repeated['index'] = range(0,756)
guanella_repeated.index = guanella_repeated['index']
print(guanella_repeated)

cherrycreek_df = pd.DataFrame()
cherrycreek_df = df.loc[df['id'] == '0803532']
cherrycreek_df['yr'] = cherrycreek_df['yr'].astype(int)
cherrycreek_df_selection = cherrycreek_df.iloc[125:126, 2:14]
cherrycreek_df_selection_update = pd.DataFrame(cherrycreek_df_selection.values.ravel(),columns = ['0803532'])
cherrycreek_repeated = pd.concat([cherrycreek_df_selection_update]*63, ignore_index=True)
cherrycreek_repeated['index'] = range(0,756)
cherrycreek_repeated.index = cherrycreek_repeated['index']
print(cherrycreek_repeated)

mclellan_df = pd.DataFrame()
mclellan_df = df.loc[df['id'] == '0803832']
mclellan_df['yr'] = mclellan_df['yr'].astype(int)
mclellan_df_selection = mclellan_df.iloc[125:126, 2:14]
mclellan_df_selection_update = pd.DataFrame(mclellan_df_selection.values.ravel(),columns = ['0803832'])
mclellan_repeated = pd.concat([mclellan_df_selection_update]*63, ignore_index=True)
mclellan_repeated['index'] = range(0,756)
mclellan_repeated.index = mclellan_repeated['index']
print(mclellan_repeated)

southplatteres_df = pd.DataFrame()
southplatteres_df = df.loc[df['id'] == '0804097']
southplatteres_df['yr'] = southplatteres_df['yr'].astype(int)
southplatteres_df_selection = southplatteres_df.iloc[125:126, 2:14]
southplatteres_df_selection_update = pd.DataFrame(southplatteres_df_selection.values.ravel(),columns = ['0804097'])
southplatteres_repeated = pd.concat([southplatteres_df_selection_update]*63, ignore_index=True)
southplatteres_repeated['index'] = range(0,756)
southplatteres_repeated.index = southplatteres_repeated['index']
print(southplatteres_repeated)

bearcreek_df = pd.DataFrame()
bearcreek_df = df.loc[df['id'] == '0903999']
bearcreek_df['yr'] = bearcreek_df['yr'].astype(int)
bearcreek_df_selection = bearcreek_df.iloc[125:126, 2:14]
bearcreek_df_selection_update = pd.DataFrame(bearcreek_df_selection.values.ravel(),columns = ['0903999'])
bearcreek_repeated = pd.concat([bearcreek_df_selection_update]*63, ignore_index=True)
bearcreek_repeated['index'] = range(0,756)
bearcreek_repeated.index = bearcreek_repeated['index']
print(bearcreek_repeated)

spinneymtn_df = pd.DataFrame()
spinneymtn_df = df.loc[df['id'] == '2304013']
spinneymtn_df['yr'] = spinneymtn_df['yr'].astype(int)
spinneymtn_df_selection = spinneymtn_df.iloc[125:126, 2:14]
spinneymtn_df_selection_update = pd.DataFrame(spinneymtn_df_selection.values.ravel(),columns = ['2304013'])
spinneymtn_repeated = pd.concat([spinneymtn_df_selection_update]*63, ignore_index=True)
spinneymtn_repeated['index'] = range(0,756)
spinneymtn_repeated.index = spinneymtn_repeated['index']
print(spinneymtn_repeated)

conmutualagg_df = pd.DataFrame()
conmutualagg_df = df.loc[df['id'] == 'ConMutualAGG']
conmutualagg_df['yr'] = conmutualagg_df['yr'].astype(int)
conmutualagg_df_selection = conmutualagg_df.iloc[125:126, 2:14]
conmutualagg_df_selection_update = pd.DataFrame(conmutualagg_df_selection.values.ravel(),columns = ['ConMutualAGG'])
conmutualagg_repeated = pd.concat([conmutualagg_df_selection_update]*63, ignore_index=True)
conmutualagg_repeated['index'] = range(0,756)
conmutualagg_repeated.index = conmutualagg_repeated['index']
print(conmutualagg_repeated)

res_targets = pd.DataFrame()
res_targets['0803514'] = chatfield_repeated['0803514']
res_targets['0803983'] = strontia_repeated['0803983']
res_targets['0404513'] = carter_repeated['0404513']
res_targets['0504010'] = buttonrock_repeated['0504010']
res_targets['0504515'] = boulderres_repeated['0504515']
res_targets['0604199'] = gross_repeated['0604199']
res_targets['0203351'] = bullres_repeated['0203351']
res_targets['0203379'] = aurorares_repeated['0203379']
res_targets['0203699'] = westgravel_repeated['0203699']
res_targets['0203700'] = eastgravel_repeated['0203700']
res_targets['0403659'] = greenridge_repeated['0403659']
res_targets['0504071'] = foothills_repeated['0504071']
res_targets['0703010'] = coorsouth_repeated['0703010']
res_targets['0703308'] = arvadares_repeated['0703308']
res_targets['0703336'] = jimbaker_repeated['0703336']
res_targets['0703389'] = coorsnorth_repeated['0703389']
res_targets['0704030'] = guanella_repeated['0704030']
res_targets['0803532'] = cherrycreek_repeated['0803532']
res_targets['0803832'] = mclellan_repeated['0803832']
res_targets['0804097'] = southplatteres_repeated['0804097']
res_targets['0903999'] = bearcreek_repeated['0903999']
res_targets['2304013'] = spinneymtn_repeated['2304013']
res_targets['ConMutualAGG'] = conmutualagg_repeated['ConMutualAGG']


###################################################################################################################################


def read_text_file(filename):
  with open(filename,'r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  return all_split_data

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test')

reservoir_target_data = read_text_file('SP2016_H.tar')


res_of_interest = ['0404513', '0504515', '0604199', '0504010', '0803514', '0803983', '0203351', '0203379', '0203699',
                   '0203700', '0403659', '0504071', '0703010', '0703308', '0703336', '0703389', '0704030', '0803532',
                   '0803832', '0804097', '0903999', '2304013', 'ConMutualAGG']

def writenewTAR(reservoir_target_data, scenario_name, start_year):    
  new_data = []
  use_value = 0
  start_loop = 0
  col_start = [0, 5, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113]
#               , 121, 129, 137, 145]
  max_counter = 0
  for i in range(0, len(reservoir_target_data)):
    if use_value == 1:
      start_loop = 1
    if reservoir_target_data[i][0] != '#':
      use_value = 1
    if start_loop == 1:
      use_line = True
      row_data = []
      value_use = []
      for col_loc in range(0, len(col_start)):
        if col_loc == len(col_start) - 1:
          value_use.append(reservoir_target_data[i][col_start[col_loc]:].strip())
        else:          
          value_use.append(reservoir_target_data[i][col_start[col_loc]:(col_start[col_loc + 1])].strip())
      try:
        year_num = int(value_use[0])
        structure_name = str(value_use[1]).strip()
      except:
        use_line = False
      if structure_name in res_of_interest and max_counter == 1:
        index_val = (year_num - start_year) * 12
        new_targets = np.zeros(13)
        for i in range(0,12):
            new_targets[i] = res_targets.loc[index_val + i, structure_name]
        new_targets[12] = np.sum(new_targets)
        row_data.append(str(year_num))
        row_data.append(structure_name)
        for i in range(0,13):
            row_data.append(str(int(float(new_targets[i]))))
        max_counter = 0    
      elif structure_name in res_of_interest and max_counter == 0:
         for i in range(0,len(value_use)):
             if i == 1:
                 row_data.append(str(value_use[i]))
             else:  
                 row_data.append(str(int(float(value_use[i]))))  
         max_counter = 1
      # elif structure_name in strontia_springs and max_counter == 1:
      #    index_val = (year_num - start_year) * 12
      #    new_targets = np.zeros(13)
      #    for i in range(0,12):
      #      new_targets[i] = strontia_repeated.loc[index_val + i, '0803983']
      #    new_targets[12] = np.sum(new_targets)
      #    row_data.append(str(year_num))
      #    row_data.append(structure_name)
      #    for i in range(0,13):
      #     row_data.append(str(int(float(new_targets[i]))))
      #    max_counter = 0
      # elif structure_name in strontia_springs and max_counter == 0: 
      #    for i in range(0,len(value_use)):
      #        if i == 1:
      #            row_data.append(str(value_use[i]))
      #        else:  
      #            row_data.append(str(int(float(value_use[i]))))  
      #    max_counter = 1           
      else:
          for i in range(0,len(value_use)):
              if i == 1:
                  row_data.append(str(value_use[i]))
              else:  
                  row_data.append(str(int(float(value_use[i]))))  
      new_data.append(row_data)  

  # write new data to file
  f = open('SP2016_' + scenario_name + '.tar','w')
  # write firstLine # of rows as in initial file
  i = 0
  while reservoir_target_data[i][0] == '#':
    f.write(reservoir_target_data[i])
    i += 1
  f.write(reservoir_target_data[i])
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

writenewTAR(reservoir_target_data, 'A', 1950)      