#to run: python3 dyadic_behaviour_ODE.py

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

def Behaviour_Model_ODE(t, b):#, alpha, beta, gamma, delta, epsilon):

	for sel_ind in np.arange(2):
	
		if b[sel_ind]<0:
		
			b[sel_ind]=0

	b_dot=np.zeros(2)

	for sel_ind in np.arange(2):

		b_dot[sel_ind]=alpha[sel_ind]*(b[sel_ind])-beta[sel_ind]*(b[sel_ind]**2)+gamma[sel_ind]*b[1-sel_ind]+epsilon[sel_ind]-delta[sel_ind]*(b[sel_ind]+b[1-sel_ind])
		
	return(b_dot)
	

alpha=np.random.random(2)*2

print("alpha = ",alpha)

beta=np.random.random(2)*2

print("beta = ",beta)

gamma=np.random.random(2)*2

print("gamma = ",gamma)

delta=np.random.random(2)*2

print("delta = ",delta)

epsilon=np.random.random(2)*2

print("epsilon = ",epsilon)

#t=0

#b_dot=Behaviour_Model_ODE(t, b, alpha, beta, gamma, delta, epsilon)
		
#print("b_dot = ",b_dot)

b_init=np.random.random(2)*3
	
t_max=0

no_t=100

for i in np.arange(3):

	t_min=t_max

	t_max=t_max+10

	t_sol=np.linspace(t_min, t_max, no_t)
		
	sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], b_init, dense_output=True, t_eval=t_sol)

	z = sol.sol(t_sol)

	L=len(z[0,:])

	b_init=z[:,L-1]

	#print("new b_init = ",b_init)

	fig, ax = plt.subplots(nrows=1, ncols=2)

	ax[0].plot(t_sol, z.T)
	#plt.ylim([0, 20])
	#ax[0].x_label('t')


	ax[1].plot(z.T[:,0],z.T[:,1])

	plt.show()
	plt.close()















