import numpy as np 
import os


def read_text_file(filename):
  with open(filename,'r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  return all_split_data

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/historicalbaseline')

demand_data = read_text_file('SP2016_H.ddm')

demands_of_interest = ["04_Lovelnd_I","04_Lovelnd_O","05LONG_IN","05LONG_OUT","06BOULDER_I","06BOULDER_O","08_Denver_I","08_Denver_O","08_Aurora_I","08_Aurora_O","08_Englwd_I","08_Englwd_O","02_Nglenn_I",
                       "02_Nglenn_O","02_Westy_I","02_Westy_O","02_Thorn_I","02_Thorn_O","06LAFFYT_I","06LAFFYT_O","06LOUIS_I","06LOUIS_O","07_Arvada_I","07_Arvada_O","07_ConMut_I","07_ConMut_O","07_Golden_I",
                       "07_Golden_O", "0400518_I", "0400518_O", "0400702"]
aurora_water_rights = ["2302900","2302901","2302902","2302903","2302904","2302906","2302907","2302908","2302909","2302910","2302911","2302912","2302913","2302914","2302915","2302916","2302917","2302918"]
adams_tunnel = ["0404634"]
cbt_split_1 = ["0400691_X", "0400692_X"]
cbt_split_2 = ["05_LongCBT", "05LHCBT", "05_SVCBT",'06_SWSP_IMP']
cbt_split_3 = ["05_BRCBT", "060800_IMP", "06_CBT_IMP"]
boulder_infrastructure = ["BCSC", "0600800_SV"]
roberts_tunnel = ["8000653"]
moffat_tunnel = ["06_MOF_IMP"]
moffat_wtp = ['MoffatWTP']
start_year = 1950

def writenewDDM(demand_data, demands_of_interest, synthetic_demands, historical_means, adams_tunnel_sp_imports, transbasin_means, Adapted_0400692_X, Adapted_05_BRCBT, infrastructure_allocations, roberts_tunnel_sp_imports, moffat_tunnel_sp_imports, start_year, scenario_name):    
  new_data = []
  use_value = 0
  start_loop = 0
  col_start = [0, 5, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113]
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
        for i in range(0,12):
            new_demands[i] = synthetic_demands.loc[index_val + i, structure_name]* historical_means.loc[0,structure_name]
        new_demands[12] = np.sum(new_demands)
        row_data.append(str(year_num))
        row_data.append(structure_name)
        for i in range(0,13):
           row_data.append(str(int(float(new_demands[i]))))
      elif structure_name in adams_tunnel:
         index_val = (year_num - start_year) * 12
         new_demands = np.zeros(13)
         for i in range(0,12):
           new_demands[i] = adams_tunnel_sp_imports.loc[index_val + i, 'column']* -1
         new_demands[12] = np.sum(new_demands)
         row_data.append(str(year_num))
         row_data.append(structure_name)
         for i in range(0,13):
          row_data.append(str(int(float(new_demands[i])))) 
      elif structure_name in aurora_water_rights:
          index_val = (year_num - start_year) * 12
          new_demands = np.zeros(13)
          for i in range(0,12):
            new_demands[i] = 0
          new_demands[12] = np.sum(new_demands)
          row_data.append(str(year_num))
          row_data.append(structure_name)
          for i in range(0,13):
           row_data.append(str(int(float(new_demands[i]))))
      elif structure_name in cbt_split_1:
         index_val = (year_num - start_year) * 12
         new_demands = np.zeros(13)
         for i in range(0,12):
           new_demands[i] = adams_tunnel_sp_imports.loc[index_val + i, 'column']* transbasin_means.loc[0,structure_name]
         new_demands[12] = np.sum(new_demands)
         row_data.append(str(year_num))
         row_data.append(structure_name)
         for i in range(0,13):
          row_data.append(str(int(float(new_demands[i]))))    
      elif structure_name in cbt_split_2:
         index_val = (year_num - start_year) * 12
         new_demands = np.zeros(13)
         for i in range(0,12):
           new_demands[i] = Adapted_0400692_X.loc[index_val + i, 'carried']* infrastructure_allocations.loc[0,structure_name]
         new_demands[12] = np.sum(new_demands)
         row_data.append(str(year_num))
         row_data.append(structure_name)
         for i in range(0,13):
          row_data.append(str(int(float(new_demands[i])))) 
      elif structure_name in cbt_split_3:
         index_val = (year_num - start_year) * 12
         new_demands = np.zeros(13)
         for i in range(0,12):
           new_demands[i] = Adapted_05_BRCBT.loc[index_val + i, 'carried']* infrastructure_allocations.loc[0,structure_name]
         new_demands[12] = np.sum(new_demands)
         row_data.append(str(year_num))
         row_data.append(structure_name)
         for i in range(0,13):
          row_data.append(str(int(float(new_demands[i])))) 
      elif structure_name in boulder_infrastructure:
         index_val = (year_num - start_year) * 12
         new_demands = np.zeros(13)
         for i in range(0,12):
           new_demands[i] = Adapted_05_BRCBT.loc[index_val + i, 'carried']* infrastructure_allocations.loc[0,structure_name]
         new_demands[12] = np.sum(new_demands)
         row_data.append(str(year_num))
         row_data.append(structure_name) 
         for i in range(0,13):
          row_data.append(str(int(float(new_demands[i])))) 
      elif structure_name in roberts_tunnel:
         index_val = (year_num - start_year) * 12
         new_demands = np.zeros(13)
         for i in range(0,12):
           new_demands[i] = roberts_tunnel_sp_imports.loc[index_val + i, 'column']* -1
         new_demands[12] = np.sum(new_demands)
         row_data.append(str(year_num))
         row_data.append(structure_name) 
         for i in range(0,13):
          row_data.append(str(int(float(new_demands[i])))) 
      elif structure_name in moffat_tunnel:
         index_val = (year_num - start_year) * 12
         new_demands = np.zeros(13)
         for i in range(0,12):
           new_demands[i] = moffat_tunnel_sp_imports.loc[index_val + i, 'column']* -1
         new_demands[12] = np.sum(new_demands)
         row_data.append(str(year_num))
         row_data.append(structure_name) 
         for i in range(0,13):
          row_data.append(str(int(float(new_demands[i]))))
      elif structure_name in moffat_wtp:
         index_val = (year_num - start_year) * 12
         new_demands = np.zeros(13)
         for i in range(0,12):
           new_demands[i] = moffat_tunnel_sp_imports.loc[index_val + i, 'column']
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

os.chdir('C:/Users/zacha/Documents/UNC/SP2016_StateMod/SP_update_test')
