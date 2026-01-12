##location: cd Github/dynamics_random_complex_systems

##to run: python3 plot_interventions_two_variables_heatmap.py

########################################################

##Part A: load in the libraries and functions for running the code

##libraries

import random
from random import randint
import numpy as np
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import networkx as nx

import pandas as pd

import time as time

import scipy as sp
from scipy.integrate import solve_ivp

###############################################################

##and load in the data

kick_type=3

save_int=957

df=pd.read_csv(f'vary_two_variables_data_intervention_type_{kick_type}_{save_int}.csv')

##and now plot it

plot_data=np.array(df)

print("plot_data")

print(plot_data)

##first PA and em supp

sel_plot_factor=0

factor_data_locs=np.where(plot_data[:, 2]==sel_plot_factor)[0]

print("factor_data_locs")

print(factor_data_locs)

factor_var_data_1=np.unique(plot_data[factor_data_locs, 0])

factor_var_data_2=np.unique(plot_data[factor_data_locs, 1])

##plot for the pd

factor_pd_data_long=plot_data[factor_data_locs, 3]

print("factor_pd_data_long")

print(factor_pd_data_long)

no_vars_1=len(factor_var_data_1)

no_vars_2=len(factor_var_data_2)

factor_pd_data=np.reshape(factor_pd_data_long, (no_vars_2, no_vars_1))

print("factor_pd_data")

print(factor_pd_data)

#im, cbar = #heatmap(factor_pd_data, all_variable_values_1, all_variable_values_2, ax=ax, cmap="YlGn", cbarlabel="pd")

fig, ax = plt.subplots()

im = ax.imshow(factor_pd_data, cmap="Blues")

ax.set_xticks(range(no_vars_1), labels=np.round(factor_var_data_1, 2))
ax.set_yticks(range(no_vars_2), labels=np.round(factor_var_data_2, 2))

fig.colorbar(im, ax=ax)

plt.show()
        
fig.savefig(f"heat_map_pd_PA_em_supp_intervention_type_{kick_type}_{save_int}.png")
        
plt.close()

####

##plot for the rope

factor_pd_data_long=plot_data[factor_data_locs, 4]

print("factor_pd_data_long")

print(factor_pd_data_long)

no_vars_1=len(factor_var_data_1)

no_vars_2=len(factor_var_data_2)

factor_pd_data=np.reshape(factor_pd_data_long, (no_vars_2, no_vars_1))

print("factor_pd_data")

print(factor_pd_data)

fig, ax = plt.subplots()

#im, cbar = #heatmap(factor_pd_data, all_variable_values_1, all_variable_values_2, ax=ax, cmap="YlGn", cbarlabel="pd")

im = ax.imshow(factor_pd_data, cmap="Blues")

ax.set_xticks(range(no_vars_1), labels=np.round(factor_var_data_1, 2))
ax.set_yticks(range(no_vars_2), labels=np.round(factor_var_data_2, 2))

fig.colorbar(im, ax=ax)

plt.show()
        
fig.savefig(f"heat_map_rope_PA_em_supp_intervention_type_{kick_type}_{save_int}.png")
        
plt.close()






























