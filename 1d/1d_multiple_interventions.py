##location: cd Github/dynamics_random_complex_systems

##source: source standard_venv/bin/activate

##to run: python3 linear_1d_multiple_interventions.py

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

#import networkx as nx

import time as time

import pandas as pd

import scipy as sp
from scipy.integrate import solve_ivp

###############################################################

##functions

def Calc_x_dot(x):

    max_indice=len(model_pars)
    
    f=0
    
    for i in np.arange(max_indice):

        f=f+model_pars[i]*x**i

    x_dot=x*(max_x-x)*f
            
    return(x_dot)

def Behaviour_Model_ODE(t, x):#, alpha, beta, gamma, delta, epsilon):

	x_dot=Calc_x_dot(x)
	
	return(x_dot)
	

################################################

##initialise general starting values

intervention_index=0

no_intervention_tests=100

max_x=2

max_power=3

max_t=25

no_t=1000

intervention_effect_data=np.zeros([no_intervention_tests, max_power+4])



for sel_intervention_test in np.arange(no_intervention_tests):

    intervention_size=np.random.random()#0.1

    model_pars=np.random.random(max_power+1)*2-1
    
    x_init=np.random.random(1)*max_x

    #print(x_init)

    x0_calc1=model_pars[0]/model_pars[1]

    #print("x0_calc1 = ", x0_calc1)

    ##run the model

    t_min=0

    t_max=int(max_t)

    t_sol1=np.linspace(t_min, t_max, no_t)
            
    sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol1), np.max(t_sol1)], x_init, dense_output=True, t_eval=t_sol1)

    z1=sol.sol(t_sol1)

    #print("z1 = ", z1)

    ##now intervene

#    model_pars[0]=model_pars[0]+intervention_size

    model_pars[intervention_index]=model_pars[intervention_index]+intervention_size

    x0_calc2=model_pars[0]/model_pars[1]

    x_init=[z1[0, no_t-1]]

    #print(x_init)

    t_min=t_max

    t_max=2*int(max_t)

    t_sol2=np.linspace(t_min, t_max, no_t)
            
    sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol2), np.max(t_sol2)], x_init, dense_output=True, t_eval=t_sol2)

    z2=sol.sol(t_sol2)

    z=np.hstack([z1, z2])

    t_sol=np.hstack([t_sol1, t_sol2])

    ##calculate the intervention effect

    before_intervention=[z1[0, no_t-1]]

    after_intervention=[z2[0, no_t-1]]

    intervention_effect=after_intervention[0]-before_intervention[0]

#    print("intervention_size = ", intervention_size, ", intervention_effect = ", intervention_effect)
    
    intervention_effect_data[sel_intervention_test, 0:(max_power+1)]=model_pars
#    intervention_effect_data[sel_intervention_test, 1]=model_pars[1]
    
    intervention_effect_data[sel_intervention_test, max_power+1]=before_intervention[0]
    
    intervention_effect_data[sel_intervention_test, max_power+2]=intervention_size
    intervention_effect_data[sel_intervention_test, max_power+3]=intervention_effect

print("intervention_effect_data")

print(intervention_effect_data)

##and substitute it

#print("z = ", z)

#print("t_sol = ", t_sol)

fig, ax = plt.subplots(nrows=1, ncols=max_power+2)


#ax[0, 0].scatter(intervention_effect_data[:,max_power+2], intervention_effect_data[:,max_power+3])

ax[0].hist(intervention_effect_data[:,max_power+3], bins=10)

for i in np.arange(max_power+1):

    ax[i+1].scatter(intervention_effect_data[:, i], intervention_effect_data[:,max_power+1])
    #ax[1].scatter(intervention_effect_data[:,1], intervention_effect_data[:,max_power+1])


##plot and save the results

plt.show()

np.random.seed(int(time.time()))

save_int=np.random.randint(low=100, high=999)
        
fig.savefig(f"model_plots/intervention_effects_{save_int}.png")
        
plt.close()





























