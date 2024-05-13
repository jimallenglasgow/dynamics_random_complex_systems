#to run: python3 random_complex_systems.py

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
	
#sol=solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1),dense_output=True)

#t = np.linspace(0, 15, 300)

#z = sol.sol(t)

#plt.plot(t, z.T)
#plt.xlabel('t')
#plt.show()
#plt.close()

###############################################

##input matrix A

#A = np.array([[-0.25 + 0.14j, 0, 0.33 + 0.44j],
#
 #             [0.25 + 0.58j, -0.2 + 0.14j, 0],
#
 #             [0, 0.2 + 0.4j, -0.1 + 0.97j]])

num_eqs=10
   
neg_ident=np.identity(num_eqs)*-1

#print("neg_ident =",neg_ident)
              
A=np.random.random([num_eqs,num_eqs])*2-1

A_density=np.random.random()

for i in np.arange(num_eqs):
	for j in np.arange(num_eqs):
	
		r=np.random.random()
	
		if r>A_density:
		
			A[i,j]=0
	
	A[i,i]=0

#print("A = ",A)

B=A+neg_ident

print("B = ",B)

lambdas=np.linalg.eig(B)

print("lambdas = ",lambdas.eigenvalues)

init_conds=np.random.random(num_eqs)*2-1

#print("init_conds = ",init_conds)

def deriv_vec(t, y):

    return B @ y

t_sol=np.linspace(0, 10, 50)

result = solve_ivp(deriv_vec, [0, 25], init_conds, t_eval=t_sol)

#print(result.y)

plt.plot(t_sol, result.y.T)
plt.xlabel('t')
plt.show()
plt.close()





















