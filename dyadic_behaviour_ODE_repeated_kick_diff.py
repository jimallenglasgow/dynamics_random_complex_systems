#to run: python3 dyadic_behaviour_ODE_repeated_kick_diff.py

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

import scipy as sp
from scipy.integrate import solve_ivp

###############################################################


##dyadic behaviour model

def Calc_b_dot(b):

	b_dot=np.zeros(no_eqs)

	for sel_ind in np.arange(no_eqs):

#		b_dot[sel_ind]=b[sel_ind]*(alpha[sel_ind]*(b[sel_ind])-beta[sel_ind]*(b[sel_ind]**2)+gamma[sel_ind]*b[1-sel_ind]+epsilon[sel_ind]-delta[sel_ind]*(b[sel_ind]+b[1-sel_ind]))+(np.random.random()*2-1)*0.1
		
		#b_dot[sel_ind]=b[sel_ind]*(delta[sel_ind]-alpha[sel_ind]*(b[sel_ind]**2)-gamma[sel_ind]*b[1-sel_ind])+(np.random.random()*2-1)*0.1
		
		#b_dot[sel_ind]=b[sel_ind]*(5-b[sel_ind])*(alpha[sel_ind]*b[sel_ind]+beta[sel_ind]*b[1-sel_ind]+delta[sel_ind])+(np.random.random()*2-1)*0.2
		
		b_dot_tmp=alpha[sel_ind]*b[sel_ind]
		
		for sel_other_ind in np.arange(no_eqs):
		
			b_dot_tmp=b_dot_tmp+beta[sel_ind, sel_other_ind]*b[sel_other_ind]
			
#		if sel_ind<2:
		
		b_dot[sel_ind]=(b[sel_ind]-epsilon[sel_ind])*(delta[sel_ind]-b[sel_ind])*b_dot_tmp#+(np.random.random()*2-1)*0.2
			
#		else:
		
#			b_dot[sel_ind]=b_dot_tmp

	return(b_dot)

def Behaviour_Model_ODE(t, b):#, alpha, beta, gamma, delta, epsilon):
		
	for sel_ind in np.arange(2):
			
		if b[sel_ind]<0:
		
			b[sel_ind]=0

	b_dot=Calc_b_dot(b)
	
	for sel_ind in np.arange(no_eqs):
	
		if b[sel_ind]>1000:
		
			b_dot[sel_ind]=0
			
		if b[sel_ind]<-1000:
		
			b[sel_ind]=0
		
	return(b_dot)
	
############################
	
def Single_Behaviour_Kick(kick_size, no_eqs, no_t, plot_dynamics=0):
	
	t_max=0
	
	single_kick_data=[]

	#t=0

	#b_dot=Behaviour_Model_ODE(t, b, alpha, beta, gamma, delta, epsilon)
			
	#print("b_dot = ",b_dot)

	b_init=np.random.random(no_eqs)*0.4
		
	full_z=np.reshape(b_init,(no_eqs,1))

	print(full_z)

	#full_t=[]#np.empty(shape=[1])

	full_t=[0]

	print(full_t)

	nudge_behaviour=0
		
	t_min=t_max

	t_max=t_max+10

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], b_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	b_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, b_init[[0, 1]]])


	######

	##nudge the system

	nudge_behaviour=1

	b_init[0]=b_init[0]+nudge_behaviour*kick_size#np.random.random(2)*2

	#alpha[[0,1]]=alpha[[0,1]]+nudge_behaviour*0.5#(np.random.random(2)*2)*0.5
		
	#beta=beta+nudge_behaviour*0.5#(np.random.random(2)*2)*0.5
		
	#delta[[0,1]]=delta[[0,1]]+nudge_behaviour*(np.random.random(2)*2)*0.5

	#######

	##run for a second, to see what happens

	t_min=t_max

	t_max=t_max+1

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], b_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	b_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, b_init[[0, 1]]])

	#########

	nudge_behaviour=0

	##run for another 9 seconds to see what happens

	t_min=t_max

	t_max=t_max+9

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], b_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	b_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, b_init[[0, 1]]])
	
	if plot_dynamics==1:
	
		fig, ax = plt.subplots(nrows=1, ncols=2)

		ax[0].plot(full_t, full_z.T[:,[0,1]])
		#plt.ylim([0, 20])
		#ax[0].x_label('t')

		#	ax[1].plot(full_z.T[:,0],full_z.T[:,1])

		ax[1].plot(full_t, full_z.T[:,np.arange(2,no_eqs)])

		plt.show()
		plt.close()

	
	return(single_kick_data)
	
##################################################################################################################

def Single_Alpha_Kick(kick_size, no_eqs, no_t, plot_dynamics=0):
	
	t_max=0
	
	single_kick_data=[]

	#t=0

	#b_dot=Behaviour_Model_ODE(t, b, alpha, beta, gamma, delta, epsilon)
			
	#print("b_dot = ",b_dot)

	b_init=np.random.random(no_eqs)*0.4
		
	full_z=np.reshape(b_init,(no_eqs,1))

	print(full_z)

	#full_t=[]#np.empty(shape=[1])

	full_t=[0]

	print(full_t)

	nudge_behaviour=0
		
	t_min=t_max

	t_max=t_max+10

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], b_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	b_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, b_init[[0, 1]]])


	######

	##nudge the system

	nudge_behaviour=1

	#b_init[[0,1]]=b_init[[0,1]]+nudge_behaviour*kick_size#np.random.random(2)*2

	alpha[0]=alpha[0]+nudge_behaviour*kick_size#(np.random.random(2)*2)*0.5
		
	#beta=beta+nudge_behaviour*0.5#(np.random.random(2)*2)*0.5
		
	#delta[[0,1]]=delta[[0,1]]+nudge_behaviour*(np.random.random(2)*2)*0.5

	#######

	##run for a second, to see what happens

	t_min=t_max

	t_max=t_max+1

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], b_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	b_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, b_init[[0, 1]]])

	#########

	nudge_behaviour=0

	##run for another 9 seconds to see what happens

	t_min=t_max

	t_max=t_max+9

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], b_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	b_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, b_init[[0, 1]]])
	
	if plot_dynamics==1:
	
		fig, ax = plt.subplots(nrows=1, ncols=2)

		ax[0].plot(full_t, full_z.T[:,[0,1]])
		#plt.ylim([0, 20])
		#ax[0].x_label('t')

		#	ax[1].plot(full_z.T[:,0],full_z.T[:,1])

		ax[1].plot(full_t, full_z.T[:,np.arange(2,no_eqs)])

		plt.show()
		plt.close()

	
	return(single_kick_data)







##################################################################################################################


no_trials=50

kick_size=0.5
	
no_eqs=5

no_t=250

all_kick_data=[]

for sel_trial in np.arange(no_trials):

	print("sel trial = ",sel_trial)

	alpha=np.random.random(no_eqs)*2-1

	#print("alpha = ",alpha)

	beta=np.random.random([no_eqs,no_eqs])*2-1

	for i in np.arange(no_eqs):

		beta[i,i]=0

	#print("beta = ",beta)

	gamma=np.random.random(no_eqs)*2-1

	#print("gamma = ",gamma)

	delta=np.random.random(no_eqs)*7

	#print("delta = ",delta)

	epsilon=np.random.random(no_eqs)*2-1

	epsilon[0]=0
	epsilon[1]=0

	#print("epsilon = ",epsilon)

	single_kick_data=Single_Behaviour_Kick(kick_size, no_eqs, no_t)

#	print("single_kick_data")

#	print(single_kick_data)
	
	all_kick_data=np.hstack([all_kick_data, single_kick_data])


all_kick_data=np.reshape(all_kick_data, (no_trials, 6))

print("all kick data")

print(all_kick_data)

improvements=np.zeros([no_trials, 4])

improvements[:, 0]=all_kick_data[:, 2]-all_kick_data[:, 0]
improvements[:, 1]=all_kick_data[:, 4]-all_kick_data[:, 0]

improvements[:, 2]=all_kick_data[:, 3]-all_kick_data[:, 1]
improvements[:, 3]=all_kick_data[:, 5]-all_kick_data[:, 1]

improvements[improvements>20]=20

print("Improvements")

print(improvements)

fig, ax=plt.subplots(nrows=2, ncols=2)

ax[0,0].hist(improvements[:, 0], bins=40)

ax[0,1].hist(improvements[:, 1], bins=40)

ax[1,0].hist(improvements[:, 2], bins=40)

ax[1,1].hist(improvements[:, 3], bins=40)

plt.show()

fig.savefig("behaviour_kick.png")

plt.close()

#####################################################################################


#no_trials=10

#kick_size=0.5
	
#no_eqs=5

#no_t=250

all_kick_data=[]

for sel_trial in np.arange(no_trials):

	print("sel trial = ",sel_trial)

	alpha=np.random.random(no_eqs)*2-1

	#print("alpha = ",alpha)

	beta=np.random.random([no_eqs,no_eqs])*2-1

	for i in np.arange(no_eqs):

		beta[i,i]=0

	#print("beta = ",beta)

	gamma=np.random.random(no_eqs)*2-1

	#print("gamma = ",gamma)

	delta=np.random.random(no_eqs)*7

	#print("delta = ",delta)

	epsilon=np.random.random(no_eqs)*2-1

	epsilon[0]=0
	epsilon[1]=0

	#print("epsilon = ",epsilon)

	single_kick_data=Single_Alpha_Kick(kick_size, no_eqs, no_t)

#	print("single_kick_data")

#	print(single_kick_data)
	
	all_kick_data=np.hstack([all_kick_data, single_kick_data])


all_kick_data=np.reshape(all_kick_data, (no_trials, 6))

print("all kick data")

print(all_kick_data)

improvements=np.zeros([no_trials, 4])

improvements[:, 0]=all_kick_data[:, 2]-all_kick_data[:, 0]
improvements[:, 1]=all_kick_data[:, 4]-all_kick_data[:, 0]

improvements[:, 2]=all_kick_data[:, 3]-all_kick_data[:, 1]
improvements[:, 3]=all_kick_data[:, 5]-all_kick_data[:, 1]

improvements[improvements>20]=20

print("Improvements")

print(improvements)

fig, ax=plt.subplots(nrows=2, ncols=2)

ax[0,0].hist(improvements[:, 0], bins=40)

ax[0,1].hist(improvements[:, 1], bins=40)

ax[1,0].hist(improvements[:, 2], bins=40)

ax[1,1].hist(improvements[:, 3], bins=40)

plt.show()

fig.savefig("alpha_kick.png")

plt.close()
















































