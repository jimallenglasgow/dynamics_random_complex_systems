##location: cd Github/dynamics_random_complex_systems

##to run: python random_resources_ODE.py

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


no_factors=5

no_t=250

max_t=20

prop_interactions=0.5

kick_size=1

kick_type=2 ##1=value, 2=max, 3=interaction

sel_node=1

sel_other_node=1

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

print("growth_rate = ", growth_rate)

print("growth_to_max_rate = ", growth_to_max_rate)

print("max_resources = ", max_resources)

print("interactions = ", interactions)

#single_kick_data=Single_Behaviour_Kick(kick_size, no_factors, no_t, 1)

single_kick_data=[]

x_init=np.random.random(no_factors)*0.4
		
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

print("before_intervention_value = ", before_intervention_value)

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

print("last_value = ", last_value)

intervention_effect=last_value-before_intervention_value

print("intervention_effect = ", intervention_effect)

single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])
	
fig, ax = plt.subplots(nrows=1, ncols=2)

for sel_factor in np.arange(no_factors):

    set_line_width=1
    
    if sel_factor==0:
    
            set_line_width=3

    ax[0].plot(full_t, full_z.T[:, sel_factor], linewidth=set_line_width, label=f"{sel_factor}")
    
ax[0].legend(bbox_to_anchor=(1, -0.1), ncol=no_factors)



print("single_kick_data")

print(single_kick_data)

#############################################################

##plot the connecting networkx

G = nx.DiGraph(interactions)

seed = 13648  # Seed random number generators for reproducibility
#G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
#pos = nx.spring_layout(G, seed=seed)

pos = nx.circular_layout(G, scale=2)

node_sizes = 200#*(1+max_resources/np.sum(max_resources))
M = G.number_of_edges()

all_edge_colors = np.reshape(interactions, (len(interactions[:,0])*len(interactions[:,0]), 1))

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
		
fig.savefig("single_kick.png")
		
plt.close()

#nx.draw(G, edge_color=interactions+np.min(interactions)+0.01)

#plt.show()

#plt.close()





















































