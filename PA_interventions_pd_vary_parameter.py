##location: cd Github/dynamics_random_complex_systems

##to run: python PA_interventions_pd_vary_parameter.py

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

##dyadic behaviour model

def Calc_x_dot(x):

        growth_rate=full_inputs[0:no_factors]

        growth_to_max_rate=full_inputs[no_factors:2*no_factors]

        max_resources=full_inputs[2*no_factors:3*no_factors]

        no_full_inputs=len(full_inputs)

        interactions_long=full_inputs[3*no_factors:(no_full_inputs+1)]

        interactions=np.reshape(interactions_long, (no_factors, no_factors))
        
        x_dot=np.zeros(no_factors)
        
        for sel_ind in np.arange(no_factors):
        
                #print(interactions[sel_ind, sel_ind])

                x_growth=x[sel_ind]*growth_rate[sel_ind]

#                x_logistic_growth=growth_to_max_rate[sel_ind]*max_resources[sel_ind]-interactions[sel_ind, sel_ind]*x[sel_ind]

                x_logistic_growth=max_resources[sel_ind]

                for sel_other_ind in np.arange(no_factors):
		
                        x_logistic_growth=x_logistic_growth+interactions[sel_other_ind, sel_ind]*x[sel_other_ind]
			
                x_dot[sel_ind]=x_growth*x_logistic_growth
                
        for sel_ind in np.arange(no_factors):
            
            x_tmp=x[sel_ind]
            
            if x_tmp>50:
                
                x_dot[sel_ind]=0
			
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

	x_init=np.random.random(no_factors)*0.4
		
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
    
def Single_Model_Run(full_inputs, model_inputs):

    plot_output=model_inputs[0]
    no_factors=model_inputs[1]
    no_each_type_of_factor=model_inputs[2]
    no_t=model_inputs[3]
    max_t=model_inputs[4]
    prop_interactions=model_inputs[5]
    kick_size=model_inputs[6]
    kick_type=model_inputs[7]
    sel_node=model_inputs[8]
    sel_other_node=model_inputs[9]
    factor_order=model_inputs[10]
    plot_row=model_inputs[11]

    #print("Factor order")

    #print(factor_order)

    factor_assignment=np.ones(no_factors)*-1

    des_factors=factor_order[0:no_each_type_of_factor[0]]

    factor_assignment[des_factors]=1

    neutral_factors=factor_order[no_each_type_of_factor[0]:(no_each_type_of_factor[0]+no_each_type_of_factor[1])]

    factor_assignment[neutral_factors]=0

#    print("Factor assignment")

#    print(factor_assignment)

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

    x_init=x_init0#np.random.random(no_factors)*0.4
            
    full_z=np.reshape(x_init,(no_factors,1))

#    print(full_z)

    full_t=[0]

#    print(full_t)

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

    before_intervention_values=x_init.copy()
    
    final_value=x_init
    
#    print("Key interaction value = ", set_interactions[20, 0])

#    print("before_intervention_value = ", before_intervention_values)

    ######

    ##nudge the system

    if kick_type==1:

        x_init[sel_node]=x_init[sel_node]+kick_size#np.random.random(2)*2
        
    if kick_type==2:

        interaction_type=interactions[sel_node, 0]
        
        #print("interaction_type = ", interaction_type)

        kick_sign=np.sign(interaction_type)
        
        if kick_sign==0:
        
                kick_sign=1

        #print("kick_sign = ", kick_sign)

        max_resources[sel_node]=max_resources[sel_node]+kick_size*kick_sign#np.random.random(2)*2
        
    if kick_type==3:

        interactions[sel_other_node, sel_node]=interactions[sel_other_node, sel_node]+kick_size#np.random.random(2)*2
        
    if kick_type==4:
    
        all_possible_intervention_points=np.where(full_inputs!=0)[0]
        
        possible_intervention_points=all_possible_intervention_points[all_possible_intervention_points>2*no_factors]

#        print("Poss intervention points")

#        print(possible_intervention_points)

        sel_intervention_point=np.random.permutation(possible_intervention_points)[0]

        full_inputs[sel_intervention_point]=full_inputs[sel_intervention_point]+kick_size

        

    #######

    intervention_effects=0
    
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

        last_values=x_init

#        print("last_value = ", last_values)

        intervention_effects=last_values-before_intervention_values

#        print("intervention_effect = ", intervention_effects)

        single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])
        
        final_value=x_init
        
    ##plot the PA and em supp factors
    
#    if no_factors>20:
        
 #       main_factors_to_plot=[0, sel_intervention_node]
        
  #  else:
        
   #     main_factors_to_plot=[0]
    
#    for sel_factor in main_factors_to_plot:

 #       set_line_width=1
        
  #      set_line_type="solid"
        
 #       if sel_factor==0:
        
   #             set_line_width=3
                
#        if sel_assignment==-1:
        
 #               set_line_width=3
                
  #              set_line_type="dashed"

#        ax[plot_row, 0].plot(full_t, full_z.T[:, sel_factor], linewidth=set_line_width, linestyle=set_line_type, label=f"{sel_factor}")
        
 #       if plot_row==0:
            
  ##          ax[plot_row, 0].set_title("PA (thick) and Em. supp.")
    #        ax[plot_row, 1].set_title("All other constructs")
        
     #   if plot_row==2:
            
      #      ax[plot_row, 0].set_xlabel("Time")
        #    ax[plot_row, 1].set_xlabel("Time")
       # 
#        ax[0].legend(bbox_to_anchor=(1, -0.1), ncol=no_factors)

    ##and plot all the others

#    all_factors_to_plot=np.arange(no_factors)
    
#    factors_to_plot=np.delete(all_factors_to_plot, main_factors_to_plot)
    
#    print("factors_to_plot")
    
 #   print(factors_to_plot)
    
#    for sel_factor in factors_to_plot:

 #       set_line_width=1
        
  #      set_line_type="solid"

   #     ax[plot_row, 1].plot(full_t, full_z.T[:, sel_factor], linewidth=set_line_width, linestyle=set_line_type, label=f"{sel_factor}")
    
#    print("single_kick_data")
 #   print(single_kick_data)

        #############################################################

        
    return(intervention_effects)


#######################################################################################

np.random.seed(1214)

use_emp_network=1

plot_output=1

no_factors=10

no_each_type_of_factor=[1, 0, 0] ##must add up to the number of factors [desirable, neutral, undesirable]

no_t=50

max_t=5

prop_interactions=0.5

interaction_mean=1/no_factors ##average strength of the interactions

interaction_std=1/no_factors ##standard deviation of the strength of the interactions

kick_size=4/no_factors

##decide which nodes are desirable and undesirable

factor_order=np.random.permutation(np.arange(no_factors))


################################################

##run a single instantiation of the dynamic system
        
##create an array that tells us which interactions to include

##choose whether to use this set of interactions, or the empirical ones

interactions_include=(np.random.choice([0, -1, 1, 2], no_factors*no_factors, p=[1-prop_interactions, prop_interactions/3, prop_interactions/3, prop_interactions/3])).reshape(no_factors, no_factors)

if use_emp_network==1:

        interactions_include=np.array(pd.read_csv("PA_network.csv"))#, header=None)
        
        no_factors=len(interactions_include[:,0])
        
        factor_order=np.arange(no_factors)
        
print("no_factors = ",no_factors)
        
print("Interactions to include")

print(interactions_include)

##include negative self loops

for i in np.arange(no_factors):

    interactions_include[i,i]=3

##generate some random interactions
    
set_interactions=np.zeros([no_factors, no_factors])

for i in np.arange(no_factors):
    
    for j in np.arange(no_factors):
        
        sel_interaction=0
        
        sel_interaction_type=interactions_include[i, j]
        
        if sel_interaction_type==-1:
            
            sel_interaction=-abs(np.random.normal(-interaction_mean, interaction_std))
            
        if sel_interaction_type==1:
            
            sel_interaction=abs(np.random.normal(interaction_mean, interaction_std))
            
        if sel_interaction_type==2:
            
            sel_interaction=np.random.normal(0, interaction_std)
            
        if sel_interaction_type==3:
            
            sel_interaction=-1#abs(np.random.normal(0.5, interaction_std))
                    
        set_interactions[i, j]=sel_interaction

print("Connections")

print(interactions_include)

print("Interaction strength")

print(set_interactions)

##add in a self-interaction, which will take a different value from the other interactions

for i in np.arange(no_factors):

    interactions_include[i,i]=3#0


    
##also, set the interaction between 1 and 0 to be 0.5 (always positive, and somewhere in the middle)

#interactions[1, 0]=0.5

##set other random inputs

set_growth_rate=np.random.random(no_factors)*2

set_growth_to_max_rate=np.random.random(no_factors)*2

set_max_resources=np.random.random(no_factors)*2

##and put all inputs into own long vector

full_interactions_long=np.reshape(set_interactions, (1,-1))

set_interactions_long=full_interactions_long[0, :]

print("growth_rate = ", set_growth_rate)

print("growth_to_max_rate = ", set_growth_to_max_rate)

print("max_resources = ", set_max_resources)

print("interactions = ", set_interactions)

#print("interactions long = ", interactions_long)

full_interactions_long=np.reshape(set_interactions, (1,-1))

set_interactions_long=full_interactions_long[0, :]

full_inputs=np.append(set_growth_rate, set_growth_to_max_rate)

full_inputs=np.append(full_inputs, set_max_resources)

full_inputs=np.append(full_inputs, set_interactions_long)

##and plot the intervention effects

no_repeats=500

##choose a node to change

x_init0=np.random.random(no_factors)*0.4

sel_intervention_node=5#20#np.random.permutation(np.arange(1, no_factors))[0]

###############################################################################

##interaction intervention

intervention_array=np.zeros([no_factors, no_repeats])

intervention_factors=np.zeros([no_factors, no_repeats])

kick_type=3 ##1=value, 2=max, 3=interaction, 4=random

sel_node=0

sel_other_node=sel_intervention_node#20

all_variable_values=np.arange(0, 0.4, 0.05)

no_vars=len(all_variable_values)

all_data=np.zeros([int(no_vars*no_factors), 3])

data_count=0

for sel_variable_value in all_variable_values:
    
    interaction_mean=sel_variable_value
    
    for intervention_count in np.arange(no_repeats):
    
        print("variable value = ", sel_variable_value, ", no. = ", intervention_count)
        
        set_interactions=np.zeros([no_factors, no_factors])

        for i in np.arange(no_factors):
            
            for j in np.arange(no_factors):
                
                sel_interaction=0
                
                sel_interaction_type=interactions_include[i, j]
                
                if sel_interaction_type==-1:
                    
                    #sel_interaction=-abs(np.random.normal(-interaction_mean, interaction_std))
                    
                    sel_interaction=np.random.normal(-interaction_mean, interaction_std)
                    
                if sel_interaction_type==1:
                    
                    #sel_interaction=abs(np.random.normal(interaction_mean, interaction_std))
                    
                    sel_interaction=np.random.normal(interaction_mean, interaction_std)
                    
                if sel_interaction_type==2:
                    
                    sel_interaction=np.random.normal(0, interaction_std)
                    
                if sel_interaction_type==3:
            
                    sel_interaction=-1#abs(np.random.normal(0.5, interaction_std))
                    
                set_interactions[i, j]=sel_interaction
                
        full_interactions_long=np.reshape(set_interactions, (1,-1))

        set_interactions_long=full_interactions_long[0, :]

        full_inputs=np.append(set_growth_rate, set_growth_to_max_rate)

        full_inputs=np.append(full_inputs, set_max_resources)

        full_inputs=np.append(full_inputs, set_interactions_long)

        model_inputs=[plot_output, no_factors, no_each_type_of_factor, no_t, max_t, prop_interactions, kick_size, kick_type, sel_node, sel_other_node, factor_order, 0]

        intervention_effects=Single_Model_Run(full_inputs, model_inputs)

        intervention_array[:, intervention_count]=intervention_effects
        
        intervention_factors[:, intervention_count]=np.arange(no_factors)

#    print("Intervention factors")

#    print(intervention_factors)

#    print("Intervention effects")

#    print(intervention_array)

    ##check the pd for each node from this set of interventions

    intervention_array_sign=intervention_array>0

    pd_calc=np.sum(intervention_array_sign, axis=1)/no_repeats

#    print("intervention_array_sign")

#    print(intervention_array_sign)

    print("pd...")

    print(pd_calc)
    
    all_data[int(data_count*no_factors):int((data_count+1)*no_factors), 0]=sel_variable_value
    
    all_data[int(data_count*no_factors):int((data_count+1)*no_factors), 1]=np.arange(no_factors)
    
    all_data[int(data_count*no_factors):int((data_count+1)*no_factors), 2]=pd_calc
    
    data_count=data_count+1

print("All data")

print(all_data)

##create the data frame

df=pd.DataFrame(data=all_data, columns=["Var", "Factor", "pd_value"])

##and save it

np.random.seed(int(time.time()))

save_int=np.random.randint(low=100, high=999)

df.to_csv(f'pd_data_{save_int}.csv', index=False)

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
    
    print("factor_data_locs")

    print(factor_data_locs)
    
    factor_var_data=plot_data[factor_data_locs, 0]
    
    factor_pd_data=plot_data[factor_data_locs, 2]
    
    ax[plot_row, plot_col].plot(factor_var_data, factor_pd_data, '.')

    plot_row=plot_row+1
    
#    factor_count=factor_count+1

plot_col=1

plot_row=0

#factor_count=0

for sel_plot_factor_count in np.arange(int(np.round(len(factors_to_plot)/2))):
    
    print("sel_plot_factor_count = ",sel_plot_factor_count)
    
    sel_plot_factor=factors_to_plot[sel_plot_factor_count+int(np.round(len(factors_to_plot)/2))-1]
    
    factor_data_locs=np.where(plot_data[:,1]==sel_plot_factor)[0]
    
    print("factor_data_locs")

    print(factor_data_locs)
    
    factor_var_data=plot_data[factor_data_locs, 0]
    
    factor_pd_data=plot_data[factor_data_locs, 2]
    
    ax[plot_row, plot_col].plot(factor_var_data, factor_pd_data, '.')

    plot_row=plot_row+1
    
#    factor_count=factor_count+1
    
    
plt.show()
        
fig.savefig(f"many_interventions_pd_other_factors_{save_int}.png")
        
plt.close()





































