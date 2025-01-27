#to run: python3 dyadic_behaviour_ODE_single_equation.py

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

	b_dot=np.zeros(1)
	
	b_dot_tmp=alpha[0]
	
	for i in np.arange(1,no_eqs):
	
		b_dot_tmp=b_dot_tmp+alpha[i]*b[0]**i
		
#		if sel_ind<2:
	
	b_dot[0]=b[0]*(1-b[0])*b_dot_tmp#+(np.random.random()*2-1)*0.2
	
#	b_dot[0]=b_dot_tmp#+(np.random.random()*2-1)*0.2
		
	return(b_dot)
	
no_eqs=5

alpha=10*(np.random.random(no_eqs)*2-1)

alpha_sign=np.zeros(no_eqs)

current_sign=-1

for i in np.arange(no_eqs):

	alpha_sign[i]=current_sign

	current_sign=current_sign*-1
	
#alpha=alpha*alpha_sign

print("alpha = ",alpha)


#########

##and draw the vector plot

full_vector_plot=[]

#b_init=np.random.random(no_eqs)*0.4

for b1 in np.arange(0,1.1,0.02):

	b=np.zeros(1)
	
	b[0]=b1
	
	b_dot=Calc_b_dot(b)
	
#		print("b1 = ",b[0], "b2 = ",b[1], "b dot = ",b_dot[[0, 1]])
	
	full_vector_plot=np.hstack([full_vector_plot, [b[0], b_dot[0]]])

full_vector_plot=np.reshape(full_vector_plot, (int(len(full_vector_plot)/2), 2))

#print("full_vector_plot")

#print(full_vector_plot)



fig, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(full_vector_plot[:,0], full_vector_plot[:,1])

#ax.set_ylim([-1, 1])

#ax.set_xlim([-0.1, 1.1])

plt.show()

plt.close()




























































