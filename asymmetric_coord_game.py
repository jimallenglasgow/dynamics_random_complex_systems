#to run: python3 asymmetric_coord_game.py

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

	def __init__(self, agent_id, no_strats, poss_group_members):
	
		self.agent_id=agent_id
		
		self.group=poss_group_members[agent_id]
		
		##
		
		self.prob_playing_0=np.random.random()
		
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
		
		P1=self.payoff
	
		P2=all_agents[self.agent_to_imitate].payoff
		
		p1_new=p1
		
		if P2>P1:
		
			p1_new=p2+np.random.normal(scale=mut)
		
		
		if p1_new<0:
		
			p1_new=0
			
		if p1_new>1:
		
			p1_new=1
		
		
		self.prob_playing_0=p1_new

###############################################################

##input pars

display_output=1

no_agents=8 ##needs to be even

no_time_steps=1

mut=0.05

####################################################

payoff_matrix=np.array([[0,2],[1,0]])

print("Payoff matrix")

print(payoff_matrix)

no_strats=len(payoff_matrix[:,0])

all_p_data=np.zeros([no_time_steps, 3])

####################################################

##initialise the agents

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

	all_agents.append(Agent(sel_agent, no_strats, poss_group_members))
	
	
for time_step in np.arange(no_time_steps):
	
	##select a strategy

	for sel_agent in np.arange(no_agents):

		all_agents[sel_agent].Select_Strategy()

	##generate opponents to play

	all_possible_play_slots=np.arange(no_agents)
	
	for sel_agent in np.arange(no_agents):
	
		sel_play_slot=np.random.permutation(all_possible_play_slots)[0]
		
		all_agents[sel_agent].play_slot=sel_play_slot
		
		all_possible_play_slots=np.delete(all_possible_play_slots, np.where(all_possible_play_slots==sel_play_slot)[0])
		
		print("all_possible_play_slots")
		
		print(all_possible_play_slots)
	
	
	all_opponents=np.ones(no_agents)*-1
	
	for sel_agent in np.arange(no_agents):
	
		all_opponents[all_agents[sel_agent].play_slot]=sel_agent	

	
	no_pairs=int(no_agents/2)

	all_pairs=np.reshape(all_opponents, [no_pairs, 2])

	print("Pairs of opponents")

	print(all_pairs)

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

	for sel_agent in np.arange(no_agents):

		all_agents[sel_agent].agent_to_imitate=agents_to_imitate[sel_agent]
		
	for sel_agent in np.arange(no_agents):

		all_agents[sel_agent].Update_Strat_Prob(all_agents, mut)
		
	##average the mean strategy

	all_ps=np.zeros(no_agents)

	for sel_agent in np.arange(no_agents):

		all_ps[sel_agent]=all_agents[sel_agent].prob_playing_0
		
	mean_p=np.mean(all_ps)

	mean_p_0=np.mean(all_ps[group_0_members])

	mean_p_1=np.mean(all_ps[group_1_members])

	single_time_output=np.round([mean_p, mean_p_0, mean_p_1],2)

	all_p_data[time_step,:]=single_time_output

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


plt.plot(np.arange(no_time_steps), all_p_data)
plt.xlabel('t')
plt.show()
plt.close()


#fig, ax = plt.subplots(nrows=1, ncols=2)

#ax[0].plot(full_t, full_z.T[:,[0,1]])

#fig.savefig("single_kick.png")



















