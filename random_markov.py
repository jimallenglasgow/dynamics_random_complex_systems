#to run: python3 random_markov.py

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

N=10

M=np.random.random([N, N])

print("M")

print(M)

max_time=10

p_tmp=np.random.random([N, 1])

p_tot=np.sum(p_tmp)

p=p_tmp/p_tot

print("p")

print(p)

#for time in np.arange(max_time):

#	p1=np.matmul(M,p)
	
#	p=p1
	
#	print("p")
	
#	print(p)























