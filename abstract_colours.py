##location: cd Github/...

##to run: python3 abstract_colours.py

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
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

import networkx as nx

import time as time

import pandas as pd

import scipy as sp
from scipy.integrate import solve_ivp

##############################################################

def Generate_An_Initial_Picture(grid_width, grid_height):

        standard_deviation=np.random.random()#*5

        all_color_maps=list(colormaps)

        sel_color_map=np.random.permutation(len(all_color_maps))[0]

        no_pixels=grid_width*grid_height

        colour_values=np.zeros(no_pixels)

        pixel_value=np.random.random()

        for sel_pixel in np.arange(no_pixels):

                pixel_mean=pixel_value
                
                pixel_value=np.random.normal(pixel_mean, standard_deviation)
                
                colour_values[sel_pixel]=pixel_value

        print("colour_values")

        print(colour_values)

        ##scale between 0 and 1

        min_value=np.min(colour_values)

        max_value=np.max(colour_values)

        scaled_colour_values=(colour_values-min_value)/(max_value-min_value)-10
        
        ##vertical or horizontal stripes?
        
        vert_prob=np.random.random()

        if vert_prob<0.5:
        
                stripes=0
                
        else:
        
                stripes=1
        
        picture_values=np.append([sel_color_map, stripes], scaled_colour_values)
        
        return(picture_values)

###############################################################

no_figures=20

grid_width=150
grid_height=100

all_color_maps=list(colormaps)

no_rows=int(np.sqrt(no_figures))

no_cols=int(np.ceil(no_figures/no_rows))

##generate an initial population

initial_population=np.empty([grid_width*grid_height+2, no_figures])

for sel_figure in np.arange(no_figures):

        sel_picture_values=Generate_An_Initial_Picture(grid_width, grid_height)
        
        initial_population[:, sel_figure]=sel_picture_values

#hex_labels=

colors = list(mcolors.CSS4_COLORS)#["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff"]

custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

fig, ax = plt.subplots(nrows=no_rows, ncols=no_cols)

for sel_figure in np.arange(no_figures):

        option_count=sel_figure

        row_id=int(option_count/no_cols)

        col_id=int(np.mod(option_count, no_cols))

        print("row_id = ", row_id)
        print("col_id = ", col_id)

        picture_values=initial_population[:, sel_figure]

        sel_color_map=custom_cmap#all_color_maps[int(picture_values[0])]
        stripes=picture_values[1]
        scaled_colour_values=np.array(picture_values[2:(grid_width*grid_height+2)]).astype(float)

        print("scaled_colour_values")

        print(scaled_colour_values)

        ##and plot them

        #print("stripes = ", stripes)

        if stripes==0:

                scaled_colour_matrix=np.resize(scaled_colour_values, (grid_height, grid_width))

                ax[row_id, col_id].imshow(scaled_colour_matrix, cmap=sel_color_map)
                
                #ax[row_id, col_id].matshow(scaled_colour_matrix)
                
        if stripes==1:

                scaled_colour_matrix=np.resize(scaled_colour_values, (grid_width, grid_height))

                ax[row_id, col_id].imshow(scaled_colour_matrix.T, cmap=sel_color_map)

#                ax[row_id, col_id].matshow(scaled_colour_matrix.T)

        ax[row_id, col_id].set_title(f"Option {option_count+1}")

        ax[row_id, col_id].tick_params(bottom=False, top=False, left=False, right=False)

        ax[row_id, col_id].axis("off")

plt.show()

plt.close()



print(hex_labels)



















