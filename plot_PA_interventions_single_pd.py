##location: cd Github/dynamics_random_complex_systems

##to run: python plot_PA_interventions_single_pd.py

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

save_int=385

df=pd.read_csv(f'pd_data_{save_int}.csv')

##and now plot it

plot_data=np.array(df)

print("plot_data")

print(plot_data)

##first PA and em supp

fig, ax = plt.subplots(nrows=2)

plot_factors=[0, 20]

plot_row=0

for sel_plot_factor in plot_factors:
    
    factor_data_locs=np.where(plot_data[:,1]==sel_plot_factor)[0]
    
    print("factor_data_locs")

    print(factor_data_locs)
    
    factor_var_data=plot_data[factor_data_locs, 0]
    
    factor_pd_data=plot_data[factor_data_locs, 2]
    
    ax[plot_row].plot(factor_var_data, factor_pd_data, '.')

    plot_row=plot_row+1
    
plt.show()
        
fig.savefig(f"many_interventions_pd_PA_em_supp_{save_int}.png")
        
plt.close()

##and then the others

no_factors=int(np.max(plot_data[:, 1]))

half_remaining_factors=int(np.round((no_factors-2)/2)+1)

fig, ax = plt.subplots(nrows=half_remaining_factors+1, ncols=2)

all_factors_to_plot=np.arange(no_factors)
    
factors_to_plot=np.delete(all_factors_to_plot, plot_factors)

plot_row=0

plot_col=0

factor_count=0

for sel_plot_factor in factors_to_plot:
    
    if factor_count<half_remaining_factors/2:
        
        plot_row=0

        plot_col=1
    
    factor_data_locs=np.where(plot_data[:,1]==sel_plot_factor)[0]
    
    print("factor_data_locs")

    print(factor_data_locs)
    
    factor_var_data=plot_data[factor_data_locs, 0]
    
    factor_pd_data=plot_data[factor_data_locs, 2]
    
    ax[plot_row, plot_col].plot(factor_var_data, factor_pd_data, '.')

    plot_row=plot_row+1
    
    factor_count=factor_count+1
    
    
plt.show()
        
fig.savefig(f"many_interventions_pd_other_factors_{save_int}.png")
        
plt.close()

































