# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:09:14 2022

@author: zacha
"""

#update South Platte .ddm files

import statemodify as stm

# a dictionary to describe what you want to modify and the bounds for the LHS
setup_dict = {
    "names": ["municipal"],
    "ids": [["0404634"]],
    "bounds": [[2.0, 3.0]]
}

output_directory = "C:/Users/zacha/Documents/UNC/SP2016_StateMod"
scenario = "model_run"

# the number of samples you wish to generate
n_samples = 2

# seed value for reproducibility if so desired
seed_value = 124

# my template file.  If none passed into the `modify_ddm` function, the default file will be used.
template_file = "SP2016_U.ddm"

# the field that you want to use to query and modify the data
query_field = "id"

# generate a batch of files using generated LHS
stm.modify_ddm(modify_dict=setup_dict,
                output_dir=output_directory,
                scenario=scenario,
                n_samples=n_samples,
                seed_value=seed_value,
                query_field=query_field,
                template_file=template_file)

