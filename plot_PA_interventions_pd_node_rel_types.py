##location: cd Github/dynamics_random_complex_systems

##to run: python3 plot_PA_interventions_pd_node_rel_types.py

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

save_int=774

df=pd.read_csv(f'pd_data_{save_int}.csv')

##load in the network so we can colour the points based on the relationship to the PA node

interactions_include=np.array(pd.read_csv("PA_network.csv"))#, header=None)

all_PA_interactions=(interactions_include[:,0])#+1)/3

PA_interactions=np.delete(all_PA_interactions, 0)

print("PA_interactions")

print(PA_interactions)

##and now plot it

plot_data=np.array(df)

#print("plot_data")

#print(plot_data)

##first PA and em supp

fig, ax = plt.subplots(nrows=2)

plot_factors=[0, 20]

plot_row=0

for sel_plot_factor in plot_factors:
    
    factor_data_locs=np.where(plot_data[:,1]==sel_plot_factor)[0]
    
#    print("factor_data_locs")

#    print(factor_data_locs)
    
    factor_var_data=plot_data[factor_data_locs, 0]
    
    print("factor_var_data")
    
    print(factor_var_data)
    
    factor_pd_data=plot_data[factor_data_locs, 2]
    
    print("factor_pd_data")
    
    print(factor_pd_data)
    
    scatter=ax[plot_row].scatter(factor_var_data, factor_pd_data, s=10, c=PA_interactions)

    if plot_row==1:
    
        # produce a legend with the unique colors from the scatter
        legend1 = ax[plot_row].legend(*scatter.legend_elements(), loc="lower right", title="Int. type")
        ax[plot_row].add_artist(legend1)
        
    plot_row=plot_row+1
    
plt.show()
        
fig.savefig(f"many_interventions_pd_PA_em_supp_{save_int}.png")
        
plt.close()

##and then the others

no_factors=int(np.max(plot_data[:, 1]))

half_remaining_factors=int(np.round((no_factors-2)/2)+1)

fig, ax = plt.subplots(nrows=half_remaining_factors-1, ncols=2)

all_factors_to_plot=np.arange(no_factors)
    
factors_to_plot=np.delete(all_factors_to_plot, plot_factors)

plot_col=0

plot_row=0

#factor_count=0

for sel_plot_factor_count in np.arange(int(np.round(len(factors_to_plot)/2))):
    
    print("sel_plot_factor_count = ",sel_plot_factor_count)
    
    sel_plot_factor=factors_to_plot[sel_plot_factor_count]
    
    factor_data_locs=np.where(plot_data[:,1]==sel_plot_factor)[0]
    
#    print("factor_data_locs")

#    print(factor_data_locs)
    
    factor_var_data=plot_data[factor_data_locs, 0]
    
    factor_pd_data=plot_data[factor_data_locs, 2]
    
    ax[plot_row, plot_col].scatter(factor_var_data, factor_pd_data, s=10, c=PA_interactions)

    plot_row=plot_row+1
    
#    factor_count=factor_count+1

plot_col=1

plot_row=0

#factor_count=0

for sel_plot_factor_count in np.arange(int(np.round(len(factors_to_plot)/2))):
    
#    print("sel_plot_factor_count = ",sel_plot_factor_count)
    
    sel_plot_factor=factors_to_plot[sel_plot_factor_count+int(np.round(len(factors_to_plot)/2))-1]
    
    factor_data_locs=np.where(plot_data[:,1]==sel_plot_factor)[0]
    
#    print("factor_data_locs")

#    print(factor_data_locs)
    
    factor_var_data=plot_data[factor_data_locs, 0]
    
    factor_pd_data=plot_data[factor_data_locs, 2]
    
    ax[plot_row, plot_col].scatter(factor_var_data, factor_pd_data, s=10, c=PA_interactions)

    plot_row=plot_row+1
    
#    factor_count=factor_count+1
    
    
plt.show()
        
fig.savefig(f"many_interventions_pd_other_factors_{save_int}.png")
        
plt.close()

































