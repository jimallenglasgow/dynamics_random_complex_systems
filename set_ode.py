#to run: python3 set_ode.py

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
	
		#b_dot_tmp=0
	
		for sel_exponent in np.arange((max_power+1)):
		
			for sel_other_ind in np.arange(no_eqs):
			
#				print("sel_ind = ", sel_ind, ", sel_exponent = ",sel_exponent, ", sel_other_ind = ", sel_other_ind)
				
#				print("coeffs = ", coeffs[sel_exponent, sel_ind, sel_other_ind])

#				b_dot_tmp=b_dot_tmp+coeffs[sel_exponent, sel_ind, sel_other_ind]*(b[sel_other_ind]**sel_exponent)
		
				b_dot[sel_ind]=b_dot[sel_ind]+coeffs[sel_exponent, sel_ind, sel_other_ind]*(b[sel_other_ind]**sel_exponent)
				
#				print("exponented = ", (b[sel_other_ind]**sel_exponent))
					
#		b_dot[sel_ind]=b_dot_tmp

	return(b_dot)

def Behaviour_Model_ODE(t, b):#, alpha, beta, gamma, delta, epsilon):
		
	b_dot=Calc_b_dot(b)
	
	for sel_ind in np.arange(no_eqs):
	
		if b[sel_ind]>20:
		
			b_dot[sel_ind]=0#9.99
			
		if b[sel_ind]<-20:
		
			b_dot[sel_ind]=0#-9.99
		
		
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

	t_max=t_max+20

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
	
	print("b_init")
	
	print(b_init)

#	b_init[[0,1]]=b_init[[0,1]]+nudge_behaviour*kick_size#np.random.random(2)*2
	
	b_init[0]=b_init[0]+nudge_behaviour*kick_size#np.random.random(2)*2
	
	print("b_init")
	
	print(b_init)

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

	t_max=t_max+20

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], b_init, dense_output=True, t_eval=t_sol)

	z=sol.sol(t_sol)

	full_z=np.hstack([full_z,z])
		
	full_t=np.hstack([full_t,t_sol])

	L=len(z[0,:])

	b_init=z[:,L-1]

	single_kick_data=np.hstack([single_kick_data, b_init[[0, 1]]])
	
	if plot_dynamics==1:
	
		fig, ax = plt.subplots(nrows=1, ncols=1)

		#ax[0].plot(full_t, full_z.T[:,[0,1]])
		#plt.ylim([0, 20])
		#ax[0].x_label('t')

		#	ax[1].plot(full_z.T[:,0],full_z.T[:,1])

		#ax[0].plot(full_t, full_z.T[:,np.arange(2,no_eqs)])
		
		plt.plot(full_t, full_z.T)

		plt.show()
		
		fig.savefig("single_kick.png")
		
		
		plt.close()

	
	return(single_kick_data)


kick_size=np.random.random()#*2-1

max_power=3
	
no_eqs=2

no_t=250

coeffs=np.zeros([max_power+1, no_eqs, no_eqs])

coeffs[3,0,0]=-1
coeffs[2,0,0]=4.5
coeffs[1,0,0]=-10.5
coeffs[0,0,0]=3

coeffs[1,0,1]=2

coeffs[1,1,1]=1
coeffs[1,1,0]=-2

print("coeffs")

print(coeffs)

single_kick_data=Single_Behaviour_Kick(kick_size, no_eqs, no_t, 1)

print("single_kick_data")

print(single_kick_data)


#########

##and draw the vector plot

full_vector_plot=[]

b_init=np.random.random(no_eqs)*0.4

for b1 in np.arange(0,10.04,0.25):

	for b2 in np.arange(0,10.04,0.25):
	
		b=b_init
		
		b[0]=b1
		
		b[1]=b2
	
		b_dot=Calc_b_dot(b)
		
#		print("b1 = ",b[0], "b2 = ",b[1], "b dot = ",b_dot[[0, 1]])
		
		r=np.sqrt(b_dot[0]**2+b_dot[1]**2)
		
		if r==0:
		
			r=1

		full_vector_plot=np.hstack([full_vector_plot, [b[0], b[1], b_dot[0], b_dot[1], r]])

full_vector_plot=np.reshape(full_vector_plot, (int(len(full_vector_plot)/5), 5))

#print("full_vector_plot")

#print(full_vector_plot)



fig, ax = plt.subplots(nrows=1, ncols=1)

plt.quiver(full_vector_plot[:,0], full_vector_plot[:,1], full_vector_plot[:,2]/full_vector_plot[:,4], full_vector_plot[:,3]/full_vector_plot[:,4])


plt.show()

#fig.savefig("alpha_kick_vector_plot.png")

plt.close()
























































