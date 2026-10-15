##location: cd Github/dynamics_random_complex_systems

##source: source standard_venv/bin/activate

##to run: python3 linear_1d.py

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

max_x=2

max_t=25

no_t=1000

model_pars=np.random.random(2)*2-1

x_init=np.random.random(1)*max_x

print(x_init)

x0_calc1=-model_pars[0]/model_pars[1]

print("x0_calc1 = ", x0_calc1)

##run the model

t_min=0

t_max=int(max_t)

t_sol1=np.linspace(t_min, t_max, no_t)
        
sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol1), np.max(t_sol1)], x_init, dense_output=True, t_eval=t_sol1)

z1=sol.sol(t_sol1)

print("z1 = ", z1)

##now intervene

#model_pars[0]=model_pars[0]+0.1

model_pars[1]=model_pars[1]+0.1

x0_calc2=-model_pars[0]/model_pars[1]

x_init=[z1[0, no_t-1]]

print(x_init)

t_min=t_max

t_max=2*int(max_t)

t_sol2=np.linspace(t_min, t_max, no_t)
        
sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol2), np.max(t_sol2)], x_init, dense_output=True, t_eval=t_sol2)

z2=sol.sol(t_sol2)

z=np.hstack([z1, z2])

t_sol=np.hstack([t_sol1, t_sol2])

##and substitute it

print("z = ", z)

print("t_sol = ", t_sol)

fig, ax = plt.subplots(nrows=1, ncols=1)
        
ax.hlines(y=x0_calc1, xmin=0, xmax=max_t, color="k", linestyle='--', linewidth=1)

ax.hlines(y=x0_calc2, xmin=max_t, xmax=2*max_t, color="k", linestyle='--', linewidth=1)

ax.plot(t_sol, z.T, linewidth=2)

##plot and save the results

plt.show()

np.random.seed(int(time.time()))

save_int=np.random.randint(low=100, high=999)
        
fig.savefig(f"model_plots/intervention_examples_{save_int}.png")
        
plt.close()





























