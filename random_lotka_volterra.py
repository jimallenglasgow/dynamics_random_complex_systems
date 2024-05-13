#to run: python3 random_lotka_volterra.py

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


##lotka volterra

def lotkavolterra(t, z, a, b, c, d):

	x, y=z
	return(a*x-b*x*y,-c*y+d*x*y)
	
def deriv_vec(t, y):

    return B @ y

num_eqs=10

A=np.random.random([num_eqs,num_eqs])*2-1

print("A")

print(A)

y_init=np.random.random(num_eqs)

print("y_init")

print(y_init)

y=y_init

def General_LV(t, y):

	N=len(A[:,0])
	
	y_new=np.zeros(N)
	
	y_array=[y]
		
	print("y = ",y_array[0,0])
	
	for i in np.arange(N):
	
		yi_new=0
	
		for j in np.arange(N):
	
			yi_new=yi_new+A[i,j]#*y[i]*y[j]
			
		y_new[i]=yi_new
		
	return y_new
	
y_new=General_LV(y, 0.01)
	
print("y_new")

print(y_new)

t_sol=np.linspace(0, 10, 50)
	
sol=solve_ivp(General_LV, [0, 10], y_init, dense_output=True, t_eval=t_sol)

z = sol.sol(t_sol)

plt.plot(t_sol, z.T)
plt.xlabel('t')
plt.show()
plt.close()




























