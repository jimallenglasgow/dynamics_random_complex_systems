#to run: python3 asymmetric_coord_game_groups.py

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

	def __init__(self, agent_id, no_strats, poss_group_members, init_prob_play_0, init_prob_play_own_group, init_prob_imi_own_group):
	
		self.agent_id=agent_id
		
		self.group=poss_group_members[agent_id]
		
		self.own_group_members=-1
		
		self.other_group_members=-1
		
		##
		
		self.prob_playing_0=init_prob_play_0[agent_id]
		
#		self.prob_play_in_even_slot=np.random.random()

		self.prob_play_own_group=init_prob_play_own_group[agent_id]
		
		if poss_group_members[agent_id]==1:
		
			self.prob_play_in_even_slot=0
			
		self.prob_imi_own_group=init_prob_imi_own_group[agent_id]
		
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
		
		q1=self.prob_play_own_group
		
		q2=all_agents[self.agent_to_imitate].prob_play_own_group
		
		w1=self.prob_imi_own_group
		
		w2=all_agents[self.agent_to_imitate].prob_imi_own_group
		
		P1=self.payoff
	
		P2=all_agents[self.agent_to_imitate].payoff
		
		p1_new=p1
		
		q1_new=q1
		
		w1_new=w1
		
		if P2>P1:
		
			p1_new=p2+np.random.normal(scale=mut[0])
			
			q1_new=q2+np.random.normal(scale=mut[1])
			
			w1_new=w2+np.random.normal(scale=mut[2])
		
		
		if p1_new<0:
		
			p1_new=0
			
		if p1_new>1:
		
			p1_new=1
			
		if q1_new<0:
		
			q1_new=0
			
		if q1_new>1:
		
			q1_new=1
			
		if w1_new<0:
	
			w1_new=0
			
		if w1_new>1:
		
			w1_new=1
		
		
		self.prob_playing_0=p1_new
		
		self.prob_play_own_group=q1_new
		
		self.prob_imi_own_group=w1_new

###############################################################

##input pars

display_output=-1

no_agents=1000 ##needs to be even

no_time_steps=500

mut=[0.05, 0.05, 0]

####################################################

payoff_matrix=np.array([[0,2],[1,0]])

print("Payoff matrix")

print(payoff_matrix)

no_strats=len(payoff_matrix[:,0])

all_p_data=np.zeros([no_time_steps, 3])
all_payoff_data=np.zeros([no_time_steps, 3])
all_q_data=np.zeros([no_time_steps, 3])
all_w_data=np.zeros([no_time_steps, 3])

####################################################

##initialise the agents

init_prob_play_own_group_tmp=0.1

init_prob_imi_own_group_tmp=0.9

init_prob_play_0=np.random.random(no_agents)

init_prob_play_own_group=np.random.random(no_agents)#np.ones(no_agents)*init_prob_play_own_group_tmp

init_prob_imi_own_group=np.ones(no_agents)*init_prob_imi_own_group_tmp

all_agents=[]

poss_group_members_tmp=np.hstack([np.zeros(int(no_agents/2)), np.ones(int(no_agents/2))])

poss_group_members=np.random.permutation(poss_group_members_tmp)

print("poss_group_members")

print(poss_group_members)

all_agent_ids=np.arange(no_agents)

group_0_members=all_agent_ids[np.where(poss_group_members==0)[0]]

print("group_0_members")

print(group_0_members)

group_1_members=all_agent_ids[np.where(poss_group_members==1)[0]]

print("group_1_members")

print(group_1_members)

for sel_agent in np.arange(no_agents):

	all_agents.append(Agent(sel_agent, no_strats, poss_group_members, init_prob_play_0, init_prob_play_own_group, init_prob_imi_own_group))
	

for sel_agent in np.arange(no_agents):

	sel_own_group=all_agents[sel_agent].group
	
	if sel_own_group==0:
	
		all_agents[sel_agent].own_group_members=group_0_members
		all_agents[sel_agent].other_group_members=group_1_members
		
	else:
	
		all_agents[sel_agent].own_group_members=group_1_members
		all_agents[sel_agent].other_group_members=group_0_members

##############

##run the dynamics
	
	
for time_step in np.arange(no_time_steps):

	print("time step = ",time_step)
	
	##select a strategy

	for sel_agent in np.arange(no_agents):

		all_agents[sel_agent].Select_Strategy()

	##generate opponents to play

	all_possible_play_slots=np.arange(no_agents)
	
	remaining_agents=np.arange(no_agents)
	
	remaining_group_0_agents=group_0_members
	
	remaining_group_1_agents=group_1_members
	
	sel_play_slot=0
	
	for sel_agent_tmp in np.arange(int(no_agents/2)):
	
		sel_agent=np.random.permutation(remaining_agents)[0]
		
		sel_agent_group=all_agents[sel_agent].group
	
		q=all_agents[sel_agent].prob_play_own_group
		
		r=np.random.random()
			
		if r<q:
		
			if sel_agent_group==0:
		
				possible_opponents=remaining_group_0_agents
				
				opponent_group=0
				
				L=len(possible_opponents)
				
				if L==0:
				
					possible_opponents=remaining_group_1_agents
					
					opponent_group=1
				
			else:
	
				possible_opponents=remaining_group_1_agents
				
				opponent_group=1
				
				L=len(possible_opponents)
				
				if L==0:
				
					possible_opponents=remaining_group_0_agents
					
					opponent_group=0
				
		else:
	
			if sel_agent_group==0:
		
				possible_opponents=remaining_group_1_agents
				
				opponent_group=1
				
				L=len(possible_opponents)
				
				if L==0:
				
					possible_opponents=remaining_group_0_agents
					
					opponent_group=0
				
			else:
	
				possible_opponents=remaining_group_0_agents
				
				opponent_group=0
				
				L=len(possible_opponents)
				
				if L==0:
				
					possible_opponents=remaining_group_1_agents
					
					opponent_group=1
					
		
		sel_opponent=np.random.permutation(possible_opponents)[0]
		
		##assign the opponents
		
		all_agents[sel_agent].opponent=sel_opponent
		all_agents[sel_opponent].opponent=sel_agent
		
		##assign the correct play slots
		
		all_agents[sel_agent].play_slot=sel_play_slot
		
		sel_play_slot=sel_play_slot+1
		
		all_agents[sel_opponent].play_slot=sel_play_slot
		
		sel_play_slot=sel_play_slot+1
		
		##and then delete all the required individuals from the list
		
		remaining_agents=np.delete(remaining_agents, np.where(remaining_agents==sel_agent)[0])
		remaining_agents=np.delete(remaining_agents, np.where(remaining_agents==sel_opponent)[0])
		
		if sel_agent_group==0:
		
			remaining_group_0_agents=np.delete(remaining_group_0_agents, np.where(remaining_group_0_agents==sel_agent)[0])
			
		if sel_agent_group==1:
		
			remaining_group_1_agents=np.delete(remaining_group_1_agents, np.where(remaining_group_1_agents==sel_agent)[0])
			
		if opponent_group==0:
		
			remaining_group_0_agents=np.delete(remaining_group_0_agents, np.where(remaining_group_0_agents==sel_opponent)[0])
			
		if opponent_group==1:
		
			remaining_group_1_agents=np.delete(remaining_group_1_agents, np.where(remaining_group_1_agents==sel_opponent)[0])
		
	
	all_opponents=np.ones(no_agents)*-1
	
	for sel_agent in np.arange(no_agents):
	
		all_opponents[all_agents[sel_agent].play_slot]=sel_agent	

	
	no_pairs=int(no_agents/2)

	all_pairs=np.reshape(all_opponents, [no_pairs, 2])

#	print("Pairs of opponents")
#
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
	
		w=all_agents[sel_agent].prob_imi_own_group
		
		r=np.random.random()
		
		if r<w:
		
			poss_agents_to_imitate=all_agents[sel_agent].own_group_members
			
		else:
		
			poss_agents_to_imitate=all_agents[sel_agent].other_group_members
		
		sel_agent_to_imitate=np.random.permutation(poss_agents_to_imitate)[0]

		all_agents[sel_agent].agent_to_imitate=sel_agent_to_imitate
		
	for sel_agent in np.arange(no_agents):

		all_agents[sel_agent].Update_Strat_Prob(all_agents, mut)
		
	##average the mean strategy

	all_payoffs=np.zeros(no_agents)
	all_ps=np.zeros(no_agents)
	all_qs=np.zeros(no_agents)
	all_ws=np.zeros(no_agents)

	for sel_agent in np.arange(no_agents):

		all_payoffs[sel_agent]=all_agents[sel_agent].payoff
		all_ps[sel_agent]=all_agents[sel_agent].prob_playing_0
		all_qs[sel_agent]=all_agents[sel_agent].prob_play_own_group
		all_ws[sel_agent]=all_agents[sel_agent].prob_imi_own_group
		
	
	mean_payoff=np.mean(all_payoffs)
	mean_payoff_0=np.mean(all_payoffs[group_0_members])
	mean_payoff_1=np.mean(all_payoffs[group_1_members])
	
	mean_p=np.mean(all_ps)
	mean_p_0=np.mean(all_ps[group_0_members])
	mean_p_1=np.mean(all_ps[group_1_members])
	
	mean_q=np.mean(all_qs)
	mean_q_0=np.mean(all_qs[group_0_members])
	mean_q_1=np.mean(all_qs[group_1_members])
	
	mean_w=np.mean(all_ws)
	mean_w_0=np.mean(all_ws[group_0_members])
	mean_w_1=np.mean(all_ws[group_1_members])

	single_time_output=np.round([mean_payoff, mean_payoff_0, mean_payoff_1],2)

	all_payoff_data[time_step,:]=single_time_output
	
	single_time_output=np.round([mean_p, mean_p_0, mean_p_1],2)

	all_p_data[time_step,:]=single_time_output
	
	single_time_output=np.round([mean_q, mean_q_0, mean_q_1],2)

	all_q_data[time_step,:]=single_time_output
	
	single_time_output=np.round([mean_w, mean_w_0, mean_w_1],2)

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


fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0,0].plot(np.arange(no_time_steps), all_payoff_data)
ax[0,0].set_title("Payoffs")

ax[0,1].plot(np.arange(no_time_steps), all_p_data)
ax[0,1].set_title("Prob playing A")

ax[1,0].plot(np.arange(no_time_steps), all_q_data)
ax[1,0].set_title("Prob playing own group")

ax[1,1].plot(np.arange(no_time_steps), all_w_data)
ax[1,1].set_title("Prob imitating own group")


plt.xlabel('t')




plt.show()
plt.close()


#fig, ax = plt.subplots(nrows=1, ncols=2)

#ax[0].plot(full_t, full_z.T[:,[0,1]])

#fig.savefig("single_kick.png")



















