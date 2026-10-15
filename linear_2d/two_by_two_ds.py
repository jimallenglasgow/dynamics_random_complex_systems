##location: cd Github/dynamics_random_complex_systems/linear_2d

##to run: python two_by_two_ds.py

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

import time as time

import pandas as pd

import scipy as sp
from scipy.integrate import solve_ivp

###############################################################

##functions

def Calc_x_dot(x):

        x_dot=np.zeros(no_factors)
        
        for sel_ind in np.arange(no_factors):

                x_growth=1#x[sel_ind]*(1-x[sel_ind])#*growth_rate[sel_ind]*(1-x[sel_ind])

                #x_logistic_growth=growth_to_max_rate[sel_ind]*max_resources[sel_ind]#-x[sel_ind]
                
                x_logistic_growth=set_max_resources[sel_ind]#-x[sel_ind]

                for sel_other_ind in np.arange(no_factors):
		
                        #x_logistic_growth=x_logistic_growth+interactions[sel_other_ind, sel_ind]*x[sel_other_ind]
                        
                        x_logistic_growth=x_logistic_growth+set_interactions[sel_ind, sel_other_ind]*x[sel_other_ind]
			
                x_dot[sel_ind]=x_growth*x_logistic_growth
                
#        for sel_ind in np.arange(no_factors):
            
 #               x_tmp=x[sel_ind]
            
  #              if x_tmp>100:
                
   #                 x_dot[sel_ind]=0
                
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

##calculate the equilibrium point

def Calc_Equilibrium_Point(set_interactions, set_max_resources, max_x):

    #inv_interactions=np.linalg.inv(set_interactions)

    #print("Inverse interactions")

    #print(inv_interactions)

    #equilibrium_solution=np.matmul(inv_interactions, -set_max_resources)#np.random.random(no_factors)

    #print("equilibrium_solution")

    #print(equilibrium_solution)

    ##calculate the equilibrium solution for x0

    det_A=np.linalg.det(set_interactions)

    #print("det_A = ", det_A)

    delta0=set_interactions.copy()

    delta0[:,0]=-set_max_resources

    #print("delta0")

    #print(delta0)

    det_delta0=np.linalg.det(delta0)

    #print("det_delta0 = ", det_delta0)

    x0_sol=det_delta0/det_A
    
 #   if x0_sol>max_x:
  #      
   #     x0_sol=max_x
        
    #if x0_sol<0:
        
     #   x0_sol=0
        
    return(x0_sol)

#######################################################################################

##set variables

R=1

no_factors=2

max_x=1

max_t=40

no_t=1000

set_interactions=np.zeros([2, 2])

c=np.random.random()*4-2

print("c = ", c)

set_interactions[0, 0]=-1
set_interactions[0, 1]=1#np.random.random()*4-2

set_interactions[1, 0]=c
set_interactions[1, 1]=-1

print("set_interactions")

print(set_interactions)

set_max_resources=np.ones(2)

print("set_max_resources")

print(set_max_resources)

x_init=np.random.random(2)#*max_x

#print(x_init)

x0_calc1=Calc_Equilibrium_Point(set_interactions, set_max_resources, max_x)

##run the model

t_min=0

t_max=int(max_t)

t_sol1=np.linspace(t_min, t_max, no_t)
        
sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol1), np.max(t_sol1)], x_init, dense_output=True, t_eval=t_sol1)

z1=sol.sol(t_sol1)

print("z1 = ", z1)

##now intervene

set_interactions[0, 1]=set_interactions[0, 1]+0.5

x0_calc2=Calc_Equilibrium_Point(set_interactions, set_max_resources, max_x)

x_init=[z1[0, no_t-1], z1[1, no_t-1]]

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
