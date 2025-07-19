#to run: python3 good_bad_ODE_vector_plot.py

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


##dyadic behaviour model

def Calc_gb_dot(x):

	g=x[0]
	
	b=x[1]

	g_dot=g*(1-g)*(alpha*g+beta*b+gamma)
	
	b_dot=b*(1-b)*(delta*g+epsilon*b+mu)

	return([g_dot, b_dot])


alpha=0.6#2*(np.random.random()*2-1)

beta=1#2*(np.random.random()*2-1)

gamma=-1#2*(np.random.random()*2-1)

delta=0.35#2*(np.random.random()*2-1)

epsilon=1#2*(np.random.random()*2-1)

mu=-0.5#2*(np.random.random()*2-1)

#########

##and draw the vector plot

full_vector_plot=[]

for g in np.arange(0,1.04,0.02):

	for b in np.arange(0,1.04,0.02):
	
		[g_dot, b_dot]=Calc_gb_dot([g, b])
		
#		print("b1 = ",b[0], "b2 = ",b[1], "b dot = ",b_dot[[0, 1]])
		
		r=np.sqrt(g_dot**2+b_dot**2)
		
		if r==0:
		
			r=1

		full_vector_plot=np.hstack([full_vector_plot, [g, b, g_dot, b_dot, r]])

full_vector_plot=np.reshape(full_vector_plot, (int(len(full_vector_plot)/5), 5))

#print("full_vector_plot")

#print(full_vector_plot)



fig, ax = plt.subplots()

ax.quiver(full_vector_plot[:,0], full_vector_plot[:,1], full_vector_plot[:,2]/full_vector_plot[:,4], full_vector_plot[:,3]/full_vector_plot[:,4])

plt.show()

#fig.savefig("alpha_kick_vector_plot.png")

plt.close()




























































