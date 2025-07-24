##location: cd Github/dynamics_random_complex_systems

##to run: python intervention_effects.py

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

import scipy as sp
from scipy.integrate import solve_ivp

###############################################################

##dyadic behaviour model

def Calc_x_dot(x):

	x_dot=np.zeros(no_factors)

	for sel_ind in np.arange(no_factors):

		x_growth=x[sel_ind]*growth_rate[sel_ind]

		x_logistic_growth=growth_to_max_rate[sel_ind]*max_resources[sel_ind]-x[sel_ind]

		for sel_other_ind in np.arange(no_factors):
		
			x_logistic_growth=x_logistic_growth+interactions[sel_other_ind, sel_ind]*x[sel_other_ind]
			
		x_dot[sel_ind]=x_growth*x_logistic_growth
			
	return(x_dot)

def Behaviour_Model_ODE(t, x):#, alpha, beta, gamma, delta, epsilon):

	x_dot=Calc_x_dot(x)
	
	return(x_dot)
    

def Calc_Intervention_Effect(no_factors, no_t, max_t, prop_interactions, kick_size, kick_type):
    
    sel_node=0

    sel_other_node=1

#    print("growth_rate = ", growth_rate)

 #   print("growth_to_max_rate = ", growth_to_max_rate)

  #  print("max_resources = ", max_resources)

   # print("interactions = ", interactions)

    #single_kick_data=Single_Behaviour_Kick(kick_size, no_factors, no_t, 1)

    single_kick_data=[]

    x_init=np.random.random(no_factors)*0.4
            
    full_z=np.reshape(x_init,(no_factors,1))

#    print(full_z)

    full_t=[0]

 #   print(full_t)

    nudge_behaviour=0
            
    t_min=0

    t_max=int(max_t/2)

    t_sol=np.linspace(t_min, t_max, no_t)
            
    sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

    intervention_effect=0
    
    if sol.success==1:
        
        z=sol.sol(t_sol)

        full_z=np.hstack([full_z,z])
                
        full_t=np.hstack([full_t,t_sol])

        L=len(z[0,:])

        x_init=z[:,L-1]

        single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])

        ##record the value before the kick_size

        before_intervention_value=x_init[0]

    #    print("before_intervention_value = ", before_intervention_value)

        ######

        ##nudge the system

        if kick_type==1:

            x_init[sel_node]=x_init[sel_node]+kick_size#np.random.random(2)*2
            
        if kick_type==2:

            max_resources[sel_node]=max_resources[sel_node]+kick_size#np.random.random(2)*2
            
        if kick_type==3:

            interactions[sel_other_node, sel_node]=interactions[sel_other_node, sel_node]+kick_size#np.random.random(2)*2

        #######

        ##run for another 10 seconds to see what happens

        t_min=int(max_t/2)

        t_max=int(max_t)

        t_sol=np.linspace(t_min, t_max, no_t)
                
        sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

        z=sol.sol(t_sol)

        full_z=np.hstack([full_z,z])
                
        full_t=np.hstack([full_t,t_sol])

        L=len(z[0,:])

        x_init=z[:,L-1]

        last_value=x_init[0]

    #    print("last_value = ", last_value)

        intervention_effect=last_value-before_intervention_value
    
    return(intervention_effect)
    
	
#######################################
	
##variables

no_runs=500

no_factors=5

no_t=250

max_t=20

prop_interactions=0.8

all_kick_sizes=np.random.random(no_runs)#np.ones(no_runs)

kick_type=1 ##1=value, 2=max, 3=interaction

all_intervention_effects=[]

for sel_run in np.arange(no_runs):
    
    kick_size=all_kick_sizes[sel_run]
    
    print("Sel run = ", sel_run)

    growth_rate=np.random.random(no_factors)*2

    growth_to_max_rate=np.random.random(no_factors)*2

    max_resources=np.random.random(no_factors)*2

    initial_interactions=np.random.random([no_factors, no_factors])*2-1
            
    ##create an array that tells us which interactions to include

    interactions_include=(np.random.choice([0, 1], no_factors*no_factors, p=[1-prop_interactions, prop_interactions])).reshape(no_factors, no_factors)

    interactions=initial_interactions*interactions_include

    for i in np.arange(no_factors):

        interactions[i,i]=0
        
    ##also, set the interaction between 1 and 0 to be 0.5 (always positive, and somewhere in the middle)

    interactions[1, 0]=0.5

    intervention_effect=Calc_Intervention_Effect(no_factors, no_t, max_t, prop_interactions, kick_size, kick_type)

#    print("intervention_effect = ", intervention_effect)

    all_intervention_effects=np.append(all_intervention_effects, intervention_effect)

print("all_intervention_effects")

print(all_intervention_effects)

all_kick_sizes=all_kick_sizes[all_intervention_effects<20]

all_intervention_effects=all_intervention_effects[all_intervention_effects<20]

all_kick_sizes=all_kick_sizes[all_intervention_effects>-20]

all_intervention_effects=all_intervention_effects[all_intervention_effects>-20]

fig, ax=plt.subplots(nrows=1, ncols=1)

ax.hist(all_intervention_effects, bins=80)

#ax.set_xlim([-10, 10])

#ax.set_ylim([0, 10])

plt.show()

fig.savefig("intervention_effects.png")

plt.close()

fig, ax=plt.subplots(nrows=1, ncols=1)

ax.scatter(all_kick_sizes, all_intervention_effects)

#ax.set_xlim([-10, 10])

#ax.set_ylim([0, 10])

plt.show()

fig.savefig("intervention_size_vs_intervention_effects.png")

plt.close()













































