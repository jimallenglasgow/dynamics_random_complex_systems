#to run: python3 asymmetric_coord_game_fair.py

########################################################

##libraries

import random
from random import randint
import numpy as np
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy as sp
from scipy.integrate import solve_ivp

###############################################################

class Agent():

	def __init__(self, agent_id, no_strats, init_prob_play_0, init_prob_play_left_0, init_prob_play_left_1, init_prob_imi_own_play_group):
	
		self.agent_id=agent_id
		
#		self.group=poss_group_members[agent_id]
		
		self.left_members=-1
		
		self.other_group_members=-1
		
		self.play_group=-1
		
		##
		
		self.prob_playing_0=init_prob_play_0[agent_id]
		
		self.prob_play_left_0=init_prob_play_left_0[agent_id]
		
		self.prob_play_left_1=init_prob_play_left_1[agent_id]

		self.prob_imi_own_play_group=init_prob_imi_own_play_group[agent_id]
		
		##
	
		self.strategy=-1#np.random.randint(no_strats)
		
		self.play_slot=-1
		
		self.opponent=-1
		
		self.payoff=-1
		
		##
		
		self.agent_to_imitate=-1
		
	def Select_Strategy(self):
	
		p=self.prob_playing_0
		
		r=np.random.random()
		
		self.strategy=1
		
		if r<p:
		
			self.strategy=0
		
		
	def Calc_Payoff(self, all_agents, payoff_matrix):
	
		s1=self.strategy
		
		s2=all_agents[self.opponent].strategy
		
		self.payoff=payoff_matrix[s1, s2]
		
			
	def Update_Strat_Prob(self, all_agents, mut):
	
		p1=self.prob_playing_0
		
		p2=all_agents[self.agent_to_imitate].prob_playing_0
		
		q1_0=self.prob_play_left_0
		
		q2_0=all_agents[self.agent_to_imitate].prob_play_left_0
		
		q1_1=self.prob_play_left_1
		
		q2_1=all_agents[self.agent_to_imitate].prob_play_left_1
		
		w1=self.prob_imi_own_play_group
		
		w2=all_agents[self.agent_to_imitate].prob_imi_own_play_group
		
		P1=self.payoff
	
		P2=all_agents[self.agent_to_imitate].payoff
		
		p1_new=p1
		
		q1_0_new=q1_0
		
		q1_1_new=q1_1
		
		w1_new=w1
		
		if P2>P1:
		
			p1_new=p2+np.random.normal(scale=mut[0])
			
			q1_0_new=q2_0+np.random.normal(scale=mut[1])
			
			q1_1_new=q2_1+np.random.normal(scale=mut[2])
			
			w1_new=w2+np.random.normal(scale=mut[3])
		
		
		if p1_new<0:
		
			p1_new=0
			
		if p1_new>1:
		
			p1_new=1
			
		if q1_0_new<0:
		
			q1_0_new=0
			
		if q1_0_new>1:
		
			q1_0_new=1
			
		if q1_1_new<0:
		
			q1_1_new=0
			
		if q1_1_new>1:
		
			q1_1_new=1
			
		if w1_new<0:
	
			w1_new=0
			
		if w1_new>1:
		
			w1_new=1
		
		
		self.prob_playing_0=p1_new
		
		self.prob_play_left_0=q1_0_new
		
		self.prob_play_left_1=q1_1_new
		
		self.prob_imi_left=w1_new

###############################################################

##input pars

display_output=-1

no_agents=1000 ##needs to be even

no_time_steps=500

mut=[0.05, 0.05, 0.05, 0.05]

####################################################

payoff_matrix=np.array([[0,2],[1,0]])

print("Payoff matrix")

print(payoff_matrix)

no_strats=len(payoff_matrix[:,0])

all_p_data=np.zeros([no_time_steps, 3])
all_payoff_data=np.zeros([no_time_steps, 3])
all_q0_data=np.zeros([no_time_steps, 3])
all_q1_data=np.zeros([no_time_steps, 3])
all_w_data=np.zeros([no_time_steps, 3])

####################################################

##initialise the agents

init_prob_play_left_0_tmp=0.8

init_prob_play_left_1_tmp=0.2

init_prob_imi_own_play_group_tmp=0.1

init_prob_play_0_tmp=0.5

init_prob_play_0=np.random.random(no_agents)#np.ones(no_agents)*init_prob_play_0_tmp

init_prob_play_left_0=np.random.random(no_agents)#np.random.choice([0, 1], size=no_agents, p=[init_prob_play_left_tmp, 1-init_prob_play_left_tmp])#

init_prob_play_left_1=np.random.random(no_agents)#np.random.choice([0, 1], size=no_agents, p=[init_prob_play_left_tmp, 1-init_prob_play_left_tmp])#

init_prob_imi_own_play_group=np.random.random(no_agents)#np.ones(no_agents)*init_prob_imi_own_play_group_tmp#

all_agents=[]

for sel_agent in np.arange(no_agents):

	all_agents.append(Agent(sel_agent, no_strats, init_prob_play_0, init_prob_play_left_0, init_prob_play_left_1, init_prob_imi_own_play_group))
	

##############

##run the dynamics
	
	
for time_step in np.arange(no_time_steps):

	print("time step = ",time_step)
	
	##select a strategy

	for sel_agent in np.arange(no_agents):

		all_agents[sel_agent].Select_Strategy()

	##generate opponents to play

	remaining_left_play_slots=np.arange(int(no_agents/2))
	
	remaining_right_play_slots=np.arange(int(no_agents/2))
	
	agent_order=np.random.permutation(np.arange(no_agents))
	
	sel_play_slot=0
	
	no_pairs=int(no_agents/2)
	
	all_pairs=np.ones([no_pairs, 2])*-1
	
	for sel_agent in agent_order:
	
		q=-1
	
		agent_strat=all_agents[sel_agent].strategy
		
		if agent_strat==0:
	
			q=all_agents[sel_agent].prob_play_left_0
			
			
		if agent_strat==1:
	
			q=all_agents[sel_agent].prob_play_left_1
		
		r=np.random.random()
			
		if r<q:
		
			L=len(remaining_left_play_slots)
			
			if L>0:
		
				sel_play_slot=np.random.permutation(remaining_left_play_slots)[0]
				
				all_pairs[sel_play_slot, 0]=int(sel_agent)
				
				remaining_left_play_slots=np.delete(remaining_left_play_slots, np.where(remaining_left_play_slots==sel_play_slot)[0])
				
				all_agents[sel_agent].play_group=0
				
			else:
			
				sel_play_slot=np.random.permutation(remaining_right_play_slots)[0]
			
				all_pairs[sel_play_slot, 1]=int(sel_agent)
				
				remaining_right_play_slots=np.delete(remaining_right_play_slots, np.where(remaining_right_play_slots==sel_play_slot)[0])
				
				all_agents[sel_agent].play_group=1
			
		else:
		
			L=len(remaining_right_play_slots)
			
			if L>0:
		
				sel_play_slot=np.random.permutation(remaining_right_play_slots)[0]
				
				all_pairs[sel_play_slot, 1]=int(sel_agent)
				
				remaining_right_play_slots=np.delete(remaining_right_play_slots, np.where(remaining_right_play_slots==sel_play_slot)[0])
				
				all_agents[sel_agent].play_group=1
				
			else:
			
				sel_play_slot=np.random.permutation(remaining_left_play_slots)[0]
				
				all_pairs[sel_play_slot, 0]=int(sel_agent)
				
				remaining_left_play_slots=np.delete(remaining_left_play_slots, np.where(remaining_left_play_slots==sel_play_slot)[0])
				
				all_agents[sel_agent].play_group=0
		
		
	#################
	
	##assign opponents to play
	
	no_pairs=int(no_agents/2)

#	print("Pairs of opponents")

#	print(all_pairs)

	for sel_pair in np.arange(no_pairs):

		agent1=int(all_pairs[sel_pair, 0])
		agent2=int(all_pairs[sel_pair, 1])

		all_agents[agent1].opponent=agent2
		all_agents[agent2].opponent=agent1

	##calculate the payoff

	for sel_agent in np.arange(no_agents):

		all_agents[sel_agent].Calc_Payoff(all_agents, payoff_matrix)
		
	##update the strategy

	agents_to_imitate=np.random.permutation(no_agents)

#	sel_agent=np.random.permutation(no_agents)[0]

	for sel_agent in np.arange(no_agents):
	
		w=all_agents[sel_agent].prob_imi_own_play_group
		
		r=np.random.random()
		
		if r<w:
		
			group_to_play=int(all_agents[sel_agent].play_group)
		
			poss_agents_to_imitate=all_pairs[:,group_to_play]
			
		else:
		
			group_to_play=int(all_agents[sel_agent].play_group)
		
			poss_agents_to_imitate=all_pairs[:,1-group_to_play]
		
		sel_agent_to_imitate=int(np.random.permutation(poss_agents_to_imitate)[0])

#		print("sel_agent_to_imitate = ",sel_agent_to_imitate)

		all_agents[sel_agent].agent_to_imitate=sel_agent_to_imitate
		
	for sel_agent in np.arange(no_agents):

		all_agents[sel_agent].Update_Strat_Prob(all_agents, mut)
		
	##average the mean strategy

	all_payoffs=np.zeros(no_agents)
	all_ps=np.zeros(no_agents)
	all_q0s=np.zeros(no_agents)
	all_q1s=np.zeros(no_agents)
	all_ws=np.zeros(no_agents)

	for sel_agent in np.arange(no_agents):

		all_payoffs[sel_agent]=all_agents[sel_agent].payoff
		all_ps[sel_agent]=all_agents[sel_agent].prob_playing_0
		all_q0s[sel_agent]=all_agents[sel_agent].prob_play_left_0
		all_q1s[sel_agent]=all_agents[sel_agent].prob_play_left_1
		all_ws[sel_agent]=all_agents[sel_agent].prob_imi_own_play_group
		
	left_group_inds=all_pairs[:,0].astype(int)
	right_group_inds=all_pairs[:,1].astype(int)
		
#	print("left_group_inds")
	
#	print(left_group_inds)
	
	mean_payoff=np.mean(all_payoffs)
	mean_payoff_L=np.mean(all_payoffs[left_group_inds])
	mean_payoff_R=np.mean(all_payoffs[right_group_inds])
	
	mean_p=np.mean(all_ps)
	mean_p_L=np.mean(all_ps[left_group_inds])
	mean_p_R=np.mean(all_ps[right_group_inds])
	
	mean_q0=np.mean(all_q0s)
	mean_q0_L=np.mean(all_q0s[left_group_inds])
	mean_q0_R=np.mean(all_q0s[right_group_inds])
	
	mean_q1=np.mean(all_q1s)
	mean_q1_L=np.mean(all_q1s[left_group_inds])
	mean_q1_R=np.mean(all_q1s[right_group_inds])
	
	mean_w=np.mean(all_ws)
	mean_w_L=np.mean(all_ws[left_group_inds])
	mean_w_R=np.mean(all_ws[right_group_inds])

	single_time_output=np.round([mean_payoff, mean_payoff_L, mean_payoff_R],2)

	all_payoff_data[time_step,:]=single_time_output
	
	single_time_output=np.round([mean_p, mean_p_L, mean_p_R],2)

	all_p_data[time_step,:]=single_time_output
	
	single_time_output=np.round([mean_q0, mean_q0_L, mean_q0_R],2)

	all_q0_data[time_step,:]=single_time_output
	
	single_time_output=np.round([mean_q1, mean_q1_L, mean_q1_R],2)

	all_q1_data[time_step,:]=single_time_output
	
	single_time_output=np.round([mean_w, mean_w_L, mean_w_R],2)

	all_w_data[time_step,:]=single_time_output
	
	

	##########################

	##check that everything worked

	if display_output==1:

		for sel_agent in np.arange(no_agents):

			print("Agent = ",all_agents[sel_agent].agent_id)
			
			print("Own strat = ",all_agents[sel_agent].strategy)
			
			print("Opp = ",all_agents[sel_agent].opponent)
			
			print("Opp strat = ",all_agents[all_agents[sel_agent].opponent].strategy)

			print("Payoff = ",all_agents[sel_agent].payoff)
			
			print("New strategy = ",all_agents[sel_agent].prob_playing_0)
			
			print("##################################")

		print("mean p = ",single_time_output)



if display_output==1:

	print("All time p")

	print(all_p_data)


fig, ax = plt.subplots(nrows=2, ncols=3)

ax[0,0].plot(np.arange(no_time_steps), all_payoff_data)
ax[0,0].set_title("Payoffs")

ax[1,0].plot(np.arange(no_time_steps), all_p_data)
ax[1,0].set_title("Prob playing A")

ax[0,1].plot(np.arange(no_time_steps), all_q0_data)
ax[0,1].set_title("Prob playing left given A")

ax[1,1].plot(np.arange(no_time_steps), all_q1_data)
ax[1,1].set_title("Prob playing left given B")

ax[0,2].hist(all_ps)
ax[0,2].set_title("Prob playing A dist")

ax[1,2].plot(np.arange(no_time_steps), all_w_data)
ax[1,2].set_title("Prob imitating own group")


plt.xlabel('t')




plt.show()
plt.close()


#fig, ax = plt.subplots(nrows=1, ncols=2)

#ax[0].plot(full_t, full_z.T[:,[0,1]])

#fig.savefig("single_kick.png")



















