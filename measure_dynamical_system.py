##location: cd Github/dynamics_random_complex_systems

##to run: python3 measure_dynamical_system.py

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

	growth_rate=full_inputs[0:no_factors]
	
	growth_to_max_rate=full_inputs[no_factors:2*no_factors]
	
	max_resources=full_inputs[2*no_factors:3*no_factors]
	
	no_full_inputs=len(full_inputs)
	
	interactions_long=full_inputs[3*no_factors:(no_full_inputs+1)]
	
	interactions=np.reshape(interactions_long, (no_factors, no_factors))
	
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

    x_init=np.random.random(no_factors)*0.4
            
    full_z=np.reshape(x_init,(no_factors,1))

  #  print(full_z)

    full_t=[0]

   # print(full_t)

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

    intervention_effect=0

    if kick_type>0: ##only double the run if we have intervened
    
        if kick_type==4:
        
                full_inputs[int(best_intervention_point)]=best_intervention_value

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

 #       print("last_value = ", last_value)

        intervention_effect=last_value-before_intervention_value

  #      print("intervention_effect = ", intervention_effect)

        single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])
        
        final_value=x_init
        
    if plot_output==1:
        
        fig, ax = plt.subplots(nrows=1, ncols=2)

        for sel_factor in np.arange(no_factors):

            set_line_width=1
            
            set_line_type="solid"
            
            sel_assignment=factor_assignment[sel_factor]
            
            if sel_assignment==1:
            
                    set_line_width=3
                    
            if sel_assignment==-1:
            
                    set_line_width=3
                    
                    set_line_type="dashed"

            ax[0].plot(full_t, full_z.T[:, sel_factor], linewidth=set_line_width, linestyle=set_line_type, label=f"{sel_factor}")
            
        ax[0].legend(bbox_to_anchor=(1, -0.1), ncol=no_factors)



        print("single_kick_data")

        print(single_kick_data)

        #############################################################

        ##plot the connecting networkx

        G = nx.DiGraph(set_interactions)

        seed = 13648  # Seed random number generators for reproducibility
        #G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
        #pos = nx.spring_layout(G, seed=seed)

        pos = nx.circular_layout(G, scale=2)

        node_sizes = 200#*(1+max_resources/np.sum(max_resources))
        M = G.number_of_edges()

        all_edge_colors = np.reshape(set_interactions, (len(set_interactions[:,0])*len(set_interactions[:,0]), 1))

        edge_colors=all_edge_colors[all_edge_colors!=0]

        cmap = plt.cm.plasma

        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="white", edgecolors="black")

        edges = nx.draw_networkx_edges(
            G,
            pos,
            node_size=node_sizes,
            arrowstyle="->",
            arrowsize=10,
            edge_color=edge_colors,
            edge_cmap=cmap,
            width=2,
            connectionstyle='arc3,rad=0.1'
        )

        labels=nx.draw_networkx_labels(G, pos=pos)

        pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        pc.set_array(edge_colors)

        ax[1] = plt.gca()
        ax[1].set_axis_off()
        plt.colorbar(pc, ax=ax[1])

        plt.show()
                
        fig.savefig(f"single_model_run_{sel_run}.png")
                
        plt.close()
        
    return(final_value)
    
    
################################

##function to calculate the final score

def Calc_Final_State_Score(full_inputs, model_inputs):

    final_values=Single_Model_Run(full_inputs, model_inputs)

 #   print("Final values")

  #  print(final_values)

    goal_state=np.ones(no_factors)*10

    factor_assignment=np.ones(no_factors)*-1

    des_factors=factor_order[0:no_each_type_of_factor[0]]

    factor_assignment[des_factors]=1

    neutral_factors=factor_order[no_each_type_of_factor[0]:(no_each_type_of_factor[0]+no_each_type_of_factor[1])]

    factor_assignment[neutral_factors]=0

    #print("Factor assignment")

    #print(factor_assignment)

    goal_state[np.where(factor_assignment==-1)[0]]=0

    goal_state[np.where(factor_assignment==0)[0]]='nan'

#    print("Goal state")

 #   print(goal_state)

    factor_fit=abs(goal_state-final_values)

  #  print("Factor fit")

   # print(factor_fit)

    total_score=np.nansum(factor_fit)
    
    return(total_score)

##############################################################################################################

##run the model to calculate the score

noise_to_add=0.1

plot_output=1 ##1=yes

no_factors=5

no_each_type_of_factor=[2, 1, 2] ##must add up to the number of factors [desirable, neutral, undesirable]

no_t=250

max_t=30

prop_interactions=0.5

kick_size=1

kick_type=-1 ##1=value, 2=max, 3=interaction, -1=no kick

sel_node=1

sel_other_node=1

##decide which nodes are desirable and undesirable

factor_order=np.random.permutation(np.arange(no_factors))


################################################

##run a single instantiation of the dynamic system

set_growth_rate=np.random.random(no_factors)*2

set_growth_to_max_rate=np.random.random(no_factors)*2

set_max_resources=np.random.random(no_factors)*2

initial_interactions=np.random.random([no_factors, no_factors])*2-1
        
##create an array that tells us which interactions to include

interactions_include=(np.random.choice([0, 1], no_factors*no_factors, p=[1-prop_interactions, prop_interactions])).reshape(no_factors, no_factors)

set_interactions=initial_interactions*interactions_include

for i in np.arange(no_factors):

	set_interactions[i,i]=0.001 ##not quite zero to make it easier to not use these later
    
##also, set the interaction between 1 and 0 to be 0.5 (always positive, and somewhere in the middle)

#interactions[1, 0]=0.5

full_interactions_long=np.reshape(set_interactions, (1,-1))

set_interactions_long=full_interactions_long[0, :]

print("growth_rate = ", set_growth_rate)

print("growth_to_max_rate = ", set_growth_to_max_rate)

print("max_resources = ", set_max_resources)

print("interactions = ", set_interactions)

#print("interactions long = ", interactions_long)

#full_inputs=np.append(set_growth_rate, set_growth_to_max_rate)

#full_inputs=np.append(full_inputs, set_max_resources)

#full_inputs=np.append(full_inputs, set_interactions_long)

#print("Full inputs")

#print(full_inputs)

##################################################################################

##run some noisy sytems and look at the final outputs

no_runs=10

for sel_run in np.arange(no_runs):

    updated_growth_rate=set_growth_rate+np.random.random(no_factors)*2*noise_to_add-noise_to_add

    updated_growth_rate[updated_growth_rate<0]=0

    updated_growth_to_max_rate=set_growth_to_max_rate+np.random.random(no_factors)*2*noise_to_add-noise_to_add

    updated_growth_to_max_rate[updated_growth_to_max_rate<0]=0

    updated_max_resources=set_max_resources+np.random.random(no_factors)*2*noise_to_add-noise_to_add

    updated_max_resources[updated_max_resources<0]=0
    
    updated_interactions_long=set_interactions_long.copy()

    positive_interactions=np.where(set_interactions_long>0.01)[0]

    no_positive_interactions=len(positive_interactions)

    updated_interactions_long[positive_interactions]=set_interactions_long[positive_interactions]+np.random.random(no_positive_interactions)*2*noise_to_add-noise_to_add

    negative_interactions=np.where(set_interactions_long>0.01)[0]

    no_negative_interactions=len(negative_interactions)

    updated_interactions_long[negative_interactions]=set_interactions_long[negative_interactions]+np.random.random(no_negative_interactions)*2*noise_to_add-noise_to_add

    full_inputs=np.append(updated_growth_rate, updated_growth_to_max_rate)

    full_inputs=np.append(full_inputs, updated_max_resources)

    full_inputs=np.append(full_inputs, updated_interactions_long)
    
    print("Full inputs")

    print(full_inputs)

    model_inputs=[plot_output, no_factors, no_each_type_of_factor, no_t, max_t, prop_interactions, kick_size, kick_type, sel_node, sel_other_node, factor_order]

    ##calculate the score

    final_value=Single_Model_Run(full_inputs, model_inputs)
    
    if sel_run==0:
        
        all_runs_full_inputs=full_inputs
        
        all_outputs=final_value
        
    else:
        
        all_runs_full_inputs=np.vstack([all_runs_full_inputs, full_inputs])
        
        all_outputs=np.vstack([all_outputs, final_value])

    print("Final value")

    print(final_value)

print("All runs full inputs")

print(all_runs_full_inputs)

print("All outputs")

print(all_outputs)

correlation_array=all_outputs[:,[0,1]]

print("Correlations")

print(correlation_array)

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.scatter(correlation_array[:,0], correlation_array[:,1])
            
plt.show()
        
fig.savefig(f"dynamic_correlations.png")
        
plt.close()

correlations_between_factors=np.zeros([no_factors, no_factors])

for sel_output_1 in np.arange(no_factors):

        for sel_output_2 in np.arange(no_factors):

                correlation_array=all_outputs[:,[sel_output_1, sel_output_2]]

                correlations_value=np.corrcoef(correlation_array[:,0], correlation_array[:,1])
                
                correlations_between_factors[sel_output_1, sel_output_2]=correlations_value[0,1]

print("Correlation values")

print(correlations_between_factors)

print("And the original interaction")

print(set_interactions)

##################################################

##based on these what intervention should we try?

full_inputs=np.append(set_growth_rate, set_growth_to_max_rate)

full_inputs=np.append(full_inputs, set_max_resources)

full_inputs=np.append(full_inputs, set_interactions_long)

print("Full inputs")

print(full_inputs)

kick_size=1

kick_type=int(input("What type of kick? 1=value, 2=max, 3=interaction.... "))#-1 ##1=value, 2=max, 3=interaction, -1=no kick

sel_node=int(input("Which node? = "))

if kick_type==3:

        sel_other_node=int(input("Which node should the interaction come from?... "))

model_inputs=[plot_output, no_factors, no_each_type_of_factor, no_t, max_t, prop_interactions, kick_size, kick_type, sel_node, sel_other_node, factor_order]

##calculate the score

final_value=Single_Model_Run(full_inputs, model_inputs)

























