# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 10:14:00 2022

@author: zacha
"""

import numpy as np 
# import matplotlib.pyplot as plt
# import pandas as pd
# #from osgeo import gdal
# import rasterio
# from shapely.geometry import Point, Polygon, LineString
# #import geopandas as gpd
# import fiona
# from matplotlib.colors import ListedColormap
# import matplotlib.pylab as pl
# from skimage import exposure
# import seaborn as sns
# import sys
# import scipy.stats as stats
# from datetime import datetime, timedelta
#import ddm_extraction_timeseries as ddm

def read_text_file(filename):
  with open(filename,'r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  return all_split_data

demand_data = read_text_file('SP2016_H.ddm')

demands_of_interest = ["0400691","0400692","05_LongCBT","05_SVCBT","05_BRCBT","060800_IMP","06_SWP_IMP","06_CBT_IMP","06_MOF_IMP","MoffatWTP","8000653","HOMSPICO","2304611","0704625","0704626","0400691_X","0400692_X","BCSC","0600800_SV"]

start_year = 1950

def writenewDDM(demand_data, demands_of_interest, start_year, scenario_name):    
  new_data = []
  use_value = 0
  start_loop = 0
  col_start = [0, 5, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113]
#               , 121, 129, 137, 145]
  for i in range(0, len(demand_data)):
    if use_value == 1:
      start_loop = 1
    if demand_data[i][0] != '#':
      use_value = 1
    if start_loop == 1:
      use_line = True
      row_data = []
      value_use = []
      for col_loc in range(0, len(col_start)):
        if col_loc == len(col_start) - 1:
          value_use.append(demand_data[i][col_start[col_loc]:].strip())
        else:          
          value_use.append(demand_data[i][col_start[col_loc]:(col_start[col_loc + 1])].strip())
      try:
        year_num = int(value_use[0])
        structure_name = str(value_use[1]).strip()
      except:
        use_line = False
      if structure_name in demands_of_interest:
        index_val = (year_num - start_year) * 12
        new_demands = np.zeros(13)
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
  f = open('SP2016_' + scenario_name + '.ddm','w')
  # write firstLine # of rows as in initial file
  i = 0
  while demand_data[i][0] == '#':
    f.write(demand_data[i])
    i += 1
  f.write(demand_data[i])
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

writenewDDM(demand_data, demands_of_interest, start_year, 'X')

#synthetic_demand.loc[index_val + I, structure_name] * scalar_demand[structure_name]
#scalar_demand.loc[year_num, structure_name]