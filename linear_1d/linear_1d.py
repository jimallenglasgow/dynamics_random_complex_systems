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

        x_dot=x*(max_x-x)*(model_pars[0]-model_pars[1]*x)
                
        return(x_dot)

def Behaviour_Model_ODE(t, x):#, alpha, beta, gamma, delta, epsilon):

	x_dot=Calc_x_dot(x)
	
	return(x_dot)
	
############################
	
def Single_Behaviour_Kick(kick_size, no_factors, no_t, max_t, plot_dynamics=0):
	
	t_max=0
	
	single_kick_data=[]

	#t=0

	#b_dot=Behaviour_Model_ODE(t, b, alpha, beta, gamma, delta, epsilon)
			
	#print("b_dot = ",b_dot)

	x_init=x_init0#np.random.random(no_factors)*0.4
		
	full_z=np.reshape(x_init,(no_factors,1))

	print(full_z)

	#full_t=[]#np.empty(shape=[1])

	full_t=[0]

	print(full_t)

	nudge_behaviour=0
		
	t_min=t_max

	t_max=t_max+10

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	x_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])


	######

	##nudge the system

	nudge_behaviour=1

	x_init[0]=x_init[0]+nudge_behaviour*kick_size#np.random.random(2)*2

	#alpha[[0,1]]=alpha[[0,1]]+nudge_behaviour*0.5#(np.random.random(2)*2)*0.5
		
	#beta=beta+nudge_behaviour*0.5#(np.random.random(2)*2)*0.5
		
	#delta[[0,1]]=delta[[0,1]]+nudge_behaviour*(np.random.random(2)*2)*0.5

	#######

	##run for a second, to see what happens

	t_min=t_max

	t_max=t_max+1

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	x_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])

	#########

	nudge_behaviour=0

	##run for another 9 seconds to see what happens

	t_min=t_max

	t_max=t_max+9

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	x_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])
	
	if plot_dynamics==1:
	
		fig, ax = plt.subplots(nrows=1, ncols=1)

#		ax[0].plot(full_t, full_z.T[:,[0,1]])
		#plt.ylim([0, 20])
		#ax[0].x_label('t')

		#	ax[1].plot(full_z.T[:,0],full_z.T[:,1])

		ax.plot(full_t, full_z.T)

		plt.show()
		
		fig.savefig("single_kick.png")
		
		
		plt.close()

	
	return(single_kick_data)
	
	
######################################################################################################
    
def Single_Model_Run(full_inputs, model_inputs):

    plot_output=model_inputs[0]
    no_factors=model_inputs[1]
    no_each_type_of_factor=model_inputs[2]
    no_t=model_inputs[3]
    max_t=model_inputs[4]
    prop_interactions=model_inputs[5]
    kick_size=model_inputs[6]
    kick_type=model_inputs[7]
    sel_target_node=model_inputs[8]
    sel_intervention_node=model_inputs[9]
    factor_order=model_inputs[10]
    plot_row=model_inputs[11]

    #print("Factor order")

    #print(factor_order)

    factor_assignment=np.ones(no_factors)*-1

    des_factors=factor_order[0:no_each_type_of_factor[0]]

#    factor_assignment[des_factors]=1

 #   neutral_factors=factor_order[no_each_type_of_factor[0]:(no_each_type_of_factor[0]+no_each_type_of_factor[1])]

  #  factor_assignment[neutral_factors]=0

    print("Factor assignment")

    print(factor_assignment)

    #print("full inputs = ", full_inputs)

    ##separate the full input into the actual inputs

    growth_rate=full_inputs[0:no_factors]

    growth_to_max_rate=full_inputs[no_factors:2*no_factors]

    max_resources=full_inputs[2*no_factors:3*no_factors]

    no_full_inputs=len(full_inputs)

    interactions_long=full_inputs[3*no_factors:(no_full_inputs+1)]

    interactions=np.reshape(interactions_long, (no_factors, no_factors))

    #print("growth_rate = ", growth_rate)

    #print("growth_to_max_rate = ", growth_to_max_rate)

    #print("max_resources = ", max_resources)

    #print("interactions = ", interactions)

    ###########

    ##run the dynamics

    single_kick_data=[]

    x_init=x_init0#np.random.random(no_factors)*0.2#(1/no_factors)
            
    full_z=np.reshape(x_init,(no_factors,1))

    print(full_z)

    full_t=[0]

    print(full_t)

    nudge_behaviour=0
            
    t_min=0

    t_max=int(max_t/2)

    t_sol=np.linspace(t_min, t_max, no_t)
            
    sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

    z=sol.sol(t_sol)

    full_z=np.hstack([full_z,z])
            
    full_t=np.hstack([full_t,t_sol])

    L=len(z[0,:])

    x_init=z[:,L-1]

    single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])

    ##record the value before the kick_size

    before_intervention_value=x_init[0]
    
    final_value=x_init
    
#    print("Key interaction value = ", set_interactions[20, 0])

    print("before_intervention_value = ", before_intervention_value)

    ######

    ##nudge the system

    if kick_type==1:

        x_init[sel_intervention_node]=x_init[sel_intervention_node]+kick_size#np.random.random(2)*2
        
        if x_init[sel_intervention_node]>0.99: ##make sure that the intervention doesn't lift the value above 1
        
                x_init[sel_intervention_node]=0.99
        
    if kick_type==2:

        interaction_type=interactions[sel_intervention_node, sel_target_node]
        
        #print("interaction_type = ", interaction_type)

        kick_sign=np.sign(interaction_type)
        
        if kick_sign==0:
        
                kick_sign=1

        #print("kick_sign = ", kick_sign)

        max_resources[sel_intervention_node]=max_resources[sel_intervention_node]+kick_size*kick_sign#np.random.random(2)*2
        
    if kick_type==3:
    
        interactions[sel_target_node, sel_intervention_node]=interactions[sel_target_node, sel_intervention_node]+kick_size#np.random.random(2)*2
        
    if kick_type==4:
        
        possible_structural_additions=np.where(interactions==0)
        
#        print("possible_structural_additions")
        
 #       print(possible_structural_additions)
        
        no_possible_structural_additions=len(possible_structural_additions[0])
        
        sel_structural_addition=np.random.permutation(np.arange(no_possible_structural_additions))[0]
        
  #      print("sel_structural_addition = ",sel_structural_addition)
        
        sel_i=possible_structural_additions[0][sel_structural_addition]
        
        sel_j=possible_structural_additions[1][sel_structural_addition]
        
   #     print("A to change = (",sel_i,",",sel_j,")")
        
    #    print(interactions[sel_i, sel_j])
        
        interactions[sel_i, sel_j]=np.random.normal(0, interaction_std)
        
        interactions[sel_other_node, sel_node]=interactions[sel_other_node, sel_node]+kick_size
        
     #   print("new interaction = ",interactions[sel_i, sel_j])
        
    if kick_type==5:
    
        all_possible_intervention_points=np.where(full_inputs!=0)[0]
        
        possible_intervention_points=all_possible_intervention_points[all_possible_intervention_points>2*no_factors]

        print("Poss intervention points")

        print(possible_intervention_points)

        sel_intervention_point=np.random.permutation(possible_intervention_points)[0]

        full_inputs[sel_intervention_point]=full_inputs[sel_intervention_point]+kick_size

        

    #######

    intervention_effect=0
    
#    print("Key interaction value = ", set_interactions[20, 0])
   
    if kick_type>0: ##only double the run if we have intervened

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

        print("last_value = ", last_value)

        intervention_effect=last_value-before_intervention_value

        print("intervention_effect = ", intervention_effect)

        single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])
        
        final_value=x_init
        
    ##plot the PA and em supp factors
    
#    if no_factors>20:
        
    main_factors_to_plot=[sel_target_node, sel_intervention_node]
        
 #   else:
        
  #      main_factors_to_plot=[0]
    
    for sel_factor in main_factors_to_plot:

        set_line_width=1
        
        set_line_type="solid"
        
        if sel_factor==0:
        
                set_line_width=3
                
#        if sel_assignment==-1:
        
 #               set_line_width=3
                
  #              set_line_type="dashed"

        ax[plot_row].plot(full_t, full_z.T[:, sel_factor], linewidth=set_line_width, linestyle=set_line_type, label=f"{sel_factor}")
        
        if plot_row==0:
            
            ax[plot_row].set_title("PA (thick) and Em. supp.")
            #ax[plot_row].set_title("All other constructs")
        
        if plot_row==2:
            
            ax[plot_row].set_xlabel("Time")
            #ax[plot_row].set_xlabel("Time")
        
#        ax[0].legend(bbox_to_anchor=(1, -0.1), ncol=no_factors)

    ##and plot all the others

    all_factors_to_plot=np.arange(no_factors)
    
    factors_to_plot=np.delete(all_factors_to_plot, main_factors_to_plot)
    
    print("factors_to_plot")
    
    print(factors_to_plot)
    
    for sel_factor in factors_to_plot:

        set_line_width=1
        
        set_line_type="solid"

#        ax[plot_row, 1].plot(full_t, full_z.T[:, sel_factor], linewidth=set_line_width, linestyle=set_line_type, label=f"{sel_factor}")
    
    print("single_kick_data")
    print(single_kick_data)

        #############################################################

        
    return(final_value)
    
    
######################################################################################################
    
def Intervention_Effect(full_inputs, model_inputs):

    plot_output=model_inputs[0]
    no_factors=model_inputs[1]
    no_each_type_of_factor=model_inputs[2]
    no_t=model_inputs[3]
    max_t=model_inputs[4]
    prop_interactions=model_inputs[5]
    kick_size=model_inputs[6]
    kick_type=model_inputs[7]
    sel_target_node=model_inputs[8]
    sel_intervention_node=model_inputs[9]
    factor_order=model_inputs[10]
    plot_row=model_inputs[11]

    #print("Factor order")

    #print(factor_order)

    factor_assignment=np.ones(no_factors)*-1

    des_factors=factor_order[0:no_each_type_of_factor[0]]

#    factor_assignment[des_factors]=1

 #   neutral_factors=factor_order[no_each_type_of_factor[0]:(no_each_type_of_factor[0]+no_each_type_of_factor[1])]

  #  factor_assignment[neutral_factors]=0

#    print("Factor assignment")
#
 #   print(factor_assignment)

    #print("full inputs = ", full_inputs)

    ##separate the full input into the actual inputs

    growth_rate=full_inputs[0:no_factors]

    growth_to_max_rate=full_inputs[no_factors:2*no_factors]

    max_resources=full_inputs[2*no_factors:3*no_factors]

    no_full_inputs=len(full_inputs)

    interactions_long=full_inputs[3*no_factors:(no_full_inputs+1)]

    interactions=np.reshape(interactions_long, (no_factors, no_factors))

    #print("growth_rate = ", growth_rate)

    #print("growth_to_max_rate = ", growth_to_max_rate)

    #print("max_resources = ", max_resources)

    #print("interactions = ", interactions)

    ###########

    ##run the dynamics

    single_kick_data=[]

    x_init=x_init0#np.random.random(no_factors)*0.2#(1/no_factors)
            
    full_z=np.reshape(x_init,(no_factors,1))

#    print(full_z)

    full_t=[0]

 #   print(full_t)

    nudge_behaviour=0
            
    t_min=0

    t_max=int(max_t/2)

    t_sol=np.linspace(t_min, t_max, no_t)
            
    sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

    z=sol.sol(t_sol)

    full_z=np.hstack([full_z,z])
            
    full_t=np.hstack([full_t,t_sol])

    L=len(z[0,:])

    x_init=z[:,L-1]

    single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])

    ##record the value before the kick_size

    before_intervention_value=x_init[0]
    
    x0=x_init[0]
    x1=x_init[1]
    
    final_value=x_init
    
#    print("Key interaction value = ", set_interactions[20, 0])

  #  print("before_intervention_value = ", before_intervention_value)

    ######

    ##nudge the system

    if kick_type==1:

        x_init[sel_intervention_node]=x_init[sel_intervention_node]+kick_size#np.random.random(2)*2
        
        if x_init[sel_intervention_node]>0.99: ##make sure that the intervention doesn't lift the value above 1
        
                x_init[sel_intervention_node]=0.99
        
    if kick_type==2:

        interaction_type=interactions[sel_intervention_node, sel_target_node]
        
        #print("interaction_type = ", interaction_type)

        kick_sign=np.sign(interaction_type)
        
        if kick_sign==0:
        
                kick_sign=1

        #print("kick_sign = ", kick_sign)

        max_resources[sel_intervention_node]=max_resources[sel_intervention_node]+kick_size*kick_sign#np.random.random(2)*2
        
    if kick_type==3:
    
        interactions[sel_target_node, sel_intervention_node]=interactions[sel_target_node, sel_intervention_node]+kick_size#np.random.random(2)*2
        
    if kick_type==4:
        
        possible_structural_additions=np.where(interactions==0)
        
#        print("possible_structural_additions")
        
 #       print(possible_structural_additions)
        
        no_possible_structural_additions=len(possible_structural_additions[0])
        
        sel_structural_addition=np.random.permutation(np.arange(no_possible_structural_additions))[0]
        
  #      print("sel_structural_addition = ",sel_structural_addition)
        
        sel_i=possible_structural_additions[0][sel_structural_addition]
        
        sel_j=possible_structural_additions[1][sel_structural_addition]
        
   #     print("A to change = (",sel_i,",",sel_j,")")
        
    #    print(interactions[sel_i, sel_j])
        
        interactions[sel_i, sel_j]=np.random.normal(0, interaction_std)
        
        interactions[sel_other_node, sel_node]=interactions[sel_other_node, sel_node]+kick_size
        
     #   print("new interaction = ",interactions[sel_i, sel_j])
        
    if kick_type==5:
    
        all_possible_intervention_points=np.where(full_inputs!=0)[0]
        
        possible_intervention_points=all_possible_intervention_points[all_possible_intervention_points>2*no_factors]

        print("Poss intervention points")

        print(possible_intervention_points)

        sel_intervention_point=np.random.permutation(possible_intervention_points)[0]

        full_inputs[sel_intervention_point]=full_inputs[sel_intervention_point]+kick_size

        

    #######

    intervention_effect=0
    
#    print("Key interaction value = ", set_interactions[20, 0])
   
    if kick_type>0: ##only double the run if we have intervened

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

#        print("last_value = ", last_value)

        intervention_effect=last_value-before_intervention_value

 #       print("intervention_effect = ", intervention_effect)

        single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])
        
        final_value=x_init
        
    ##plot the PA and em supp factors
    
#    if no_factors>20:
        

    
    print("single_kick_data")
    print(single_kick_data)

        #############################################################

    intervention_data=np.zeros(3)
    
    intervention_data=[np.round(intervention_effect, 2), np.round(x0, 2), np.round(x1, 2)]
        
    return(intervention_data)

#######################################################################################

##function to generate the interactions based on the normal distribution

def Normal_Dist_Interactions(no_factors, interaction_mean, interaction_std, self_regulation_level):

        set_interactions=np.zeros([no_factors, no_factors])

        for i in np.arange(no_factors):
            
            for j in np.arange(no_factors):
                
                sel_interaction=0
                
                sel_interaction_type=interactions_include[i, j]
                
                if sel_interaction_type==-1:
                    
                    sel_interaction=-abs(np.random.normal(interaction_mean, interaction_std))
                    
        #            sel_interaction=np.random.normal(-interaction_mean, interaction_std)
                    
                if sel_interaction_type==1:
                    
                    sel_interaction=abs(np.random.normal(interaction_mean, interaction_std))
                    
#                    sel_interaction=np.random.normal(interaction_mean, interaction_std)
                    
                if sel_interaction_type==2:
                    
                    sel_interaction=np.random.normal(0, interaction_std)
                    
                if sel_interaction_type==3:
                    
                    sel_interaction=-self_regulation_level#(abs(np.random.normal(0, 1))+self_regulation_level)
                    
                set_interactions[i, j]=sel_interaction
                
        return(set_interactions)
        
#######################################################################################

##function to generate the interactions based on one of two values

def Binomial_Dist_Interactions(no_factors, prob_large_connection, large_connection_value, small_connection_value, self_regulation_level):

        poss_connection_strengths=[small_connection_value, large_connection_value]

        set_interactions=np.zeros([no_factors, no_factors])

        for i in np.arange(no_factors):
            
            for j in np.arange(no_factors):
                
                sel_interaction=0
                
                selected_connection_strength=np.random.choice(2, 1, p=[1-prob_large_connection, prob_large_connection])
                
                print("selected_connection_strength")
                
                print(selected_connection_strength)

                connection_strength=poss_connection_strengths[int(selected_connection_strength)]
                
                sel_interaction_type=interactions_include[i, j]
                
                if sel_interaction_type==-1:
                    
        #            sel_interaction=-abs(np.random.normal(neg_interaction_mean, interaction_std))
                    
                    sel_interaction=-connection_strength
                    
                if sel_interaction_type==1:
                    
          #          sel_interaction=abs(np.random.normal(pos_interaction_mean, interaction_std))
                    
                    sel_interaction=connection_strength#np.random.normal(interaction_mean, interaction_std)
                    
                if sel_interaction_type==2:
                    
                    sel_interaction=np.random.permutation([connection_strength, -connection_strength])[0]
                    
                if sel_interaction_type==3:
                    
                    sel_interaction=-self_regulation_level#(abs(np.random.normal(0, 1))-self_regulation_level)
                    
                set_interactions[i, j]=sel_interaction
                
        return(set_interactions)

#######################################################################################

#np.random.seed(1214)

use_emp_network=2

plot_output=1

no_factors=21

no_each_type_of_factor=[1, 0, 0] ##must add up to the number of factors [desirable, neutral, undesirable]

no_t=1000

max_t=30

prop_interactions=0.5

interaction_mean=1

interaction_std=0.5#/no_factors ##standard deviation of the strength of the interactions

kick_size=0.2#/no_factors

##parameters for the binomial set

prob_large_connection=1

large_connection_value=1

small_connection_value=0.8

self_regulation_level=1.5

##decide which nodes are desirable and undesirable

factor_order=np.random.permutation(np.arange(no_factors))

##select the intervention node

sel_intervention_node=1#20#np.random.permutation(np.arange(1, no_factors))[0]

##set variables

R=3

no_factors=2

################################################

##initialise general starting values

max_x=2

max_t=25

no_t=1000

model_pars=np.random.random(2)

x_init=np.random.random(1)*max_x

print(x_init)

##run the model

t_min=0

t_max=int(max_t)

t_sol=np.linspace(t_min, t_max, no_t)
        
sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

z=sol.sol(t_sol)

x0_calc=model_pars[0]/model_pars[1]

print("x0_calc = ", x0_calc)

##and substitute it

print("z = ", z)

fig, ax = plt.subplots(nrows=1, ncols=1)
        
ax.hlines(y=x0_calc, xmin=0, xmax=max_t, color="k", linestyle='--', linewidth=0.7)

ax.plot(t_sol, z.T, linewidth=2)

##plot and save the results

plt.show()

np.random.seed(int(time.time()))

save_int=np.random.randint(low=100, high=999)
        
fig.savefig(f"model_plots/intervention_examples_{save_int}.png")
        
plt.close()





























