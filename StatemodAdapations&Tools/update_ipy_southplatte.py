# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:37:47 2023

@author: zacha
"""

import os
# import re
import numpy as np
# import pandas as pd
# from SALib.sample import latin
# from joblib import Parallel, delayed
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
#import sp_irrigation_final as spirr
import extract_ipy_southplatte as exipy

def read_text_file(filename):
  with open(filename,'r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  return all_split_data

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/')

irrigation_efficiency_data = read_text_file('SP2016.ipy')

start_year = 1950

def writenewIPY(irrigation_efficiency_data, irrigation_efficiencies_statemod, start_year, scenario_name):    
  new_data = []
  use_value = 0
  start_loop = 0
  col_start = [0, 5, 17, 25, 31, 36, 45, 52, 61, 74, 78, 81, 83, 91, 99]
  for i in range(0, len(irrigation_efficiency_data)):
      if use_value == 1:
        start_loop = 1
      if irrigation_efficiency_data[i][0] != '#':
        use_value = 1
      if start_loop == 1:
        use_line = True
        row_data = []
        value_use = []
        for col_loc in range(0, len(col_start)):
          if col_loc == len(col_start) - 1:
            value_use.append(irrigation_efficiency_data[i][col_start[col_loc]:].strip())
          else:          
            value_use.append(irrigation_efficiency_data[i][col_start[col_loc]:(col_start[col_loc + 1])].strip())
        try:
          year_num = int(value_use[0])
          structure_name = str(value_use[1]).strip()
        except:
          use_line = False
        if structure_name in irrigation_efficiencies_statemod.columns:
          index_val = (year_num - start_year) * 12
          new_demands = np.zeros(13)
          for i in range(0,12):
              new_demands[i] = irrigation_efficiencies_statemod.loc[index_val + i, structure_name]
          #new_demands[12] = np.sum(new_demands)
          row_data.append(str(year_num))
          row_data.append(structure_name)
          for i in range(0,12):
              row_data.append(str(int(float(new_demands[i]))))
        else:
            for i in range(0,len(value_use)):
                if i == 1:
                    row_data.append(str(value_use[i]))
                else:  
                    row_data.append(str(int(float(value_use[i]))))  
        new_data.append(row_data)  
      
      # write new data to file
  f = open('SP2016_' + scenario_name + '.ipy','w')
# write firstLine # of rows as in initial file
  i = 0
  while irrigation_efficiency_data[i][0] == '#':
    f.write(irrigation_efficiency_data[i])
    i += 1
  f.write(irrigation_efficiency_data[i])
  i+=1
  for i in range(len(new_data)):
  # write year, ID and first month of adjusted data
    structure_length = len(new_data[i][1])
    f.write(new_data[i][0] + ' ' + new_data[i][1] + (12-structure_length)*' ')
  # write all but last month of adjusted data
    for j in range(len(new_data[i])-3):
      f.write((8-len(new_data[i][j+2])-2)*' ' + new_data[i][j+2] + '.0')                
      # write last month of adjusted data
    #f.write((10-len(new_data[i][j+3])-2)*' ' + new_data[i][j+3] + '.0' + '\n')             
  f.close()
      
  return None

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test')

writenewIPY(irrigation_efficiency_data, exipy.irrigation_efficiencies_statemod, 1950, 'C')
