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

A=np.random.normal(loc=0, scale=1/num_eqs, size=[num_eqs,num_eqs])

print("A")

print(A)

y_init=np.random.random(num_eqs)

print("y_init")

print(y_init)

y=y_init

K=np.ones(num_eqs)

r=np.ones(num_eqs)


def Calc_ydot(y):

	N=len(A[:,0])
	
	y_new=np.zeros(N)
	
	#print("y = ",y_array[0,0])
	
	for i in np.arange(N):
	
		yi_new=0
	
		for j in np.arange(N):
	
			yi_new=yi_new+A[i,j]*y[j]
			
		y_new[i]=(r[i]/K[i])*y[i]*(K[i]-y[i]-yi_new)
		
	return y_new

	

def General_LV(t, y):

	y_dot=Calc_ydot(y)
		
	return y_dot
	
#y_new=General_LV(y, 0.01)
	
#print("y_new")

#print(y_new)

full_z=np.reshape(y_init,(num_eqs,1))

print(full_z)

#full_t=[]#np.empty(shape=[1])

full_t=[0]

print(full_t)

t_min=0

t_max=20

t_sol=np.linspace(t_min, t_max, 50)
	
sol=solve_ivp(General_LV, [t_min, t_max], y_init, dense_output=True, t_eval=t_sol)

z = sol.sol(t_sol)

full_z=np.hstack([full_z,z])
	
full_t=np.hstack([full_t,t_sol])

#plt.plot(t_sol, z.T)
plt.plot(full_t, full_z.T[:,np.arange(2,num_eqs)])
plt.xlabel('t')
plt.show()
plt.close()


L=len(z[0,:])

y_init=z[:,L-1]

##and nudge it

y_init=y_init+np.random.normal(scale=0.1, size=num_eqs)

t_min=20

t_max=40

t_sol=np.linspace(t_min, t_max, 50)
	
sol=solve_ivp(General_LV, [t_min, t_max], y_init, dense_output=True, t_eval=t_sol)

z = sol.sol(t_sol)

full_z=np.hstack([full_z,z])
	
full_t=np.hstack([full_t,t_sol])

plt.plot(full_t, full_z.T[:,np.arange(2,num_eqs)])
plt.xlabel('t')
plt.show()
plt.close()

























