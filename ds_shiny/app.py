##shiny run --reload --launch-browser GitHub/dynamics_random_complex_systems/ds_shiny/app.py

########################################################

##Part A: load in the libraries and functions for running the code

##libraries

import random
from random import randint
import numpy as np
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt

import networkx as nx

import scipy as sp
from scipy.integrate import solve_ivp

from shiny import reactive, Inputs, ui, App, render

###############################################################

app_ui = ui.page_sidebar(

    ui.sidebar(
    
    ui.div(
        ui.input_action_button(
            "run", "Run this model", class_="btn-primary"
        ),
        
    ),
    
    ui.div(
        
        ui.input_action_button(
            "reset_model", "Reset the model", class_="btn-primary"
        ),
        
    ),
    
    ui.input_radio_buttons("run_type", "Type of run", choices=["Random", "Intuitive", "Counter-intuitive"]),
    
    ui.input_radio_buttons("parameter_to_update", "Parameter to update", choices=["None", "Value", "Max level", "Interaction"]),
    
    ui.input_slider("sel_node", "Selected node", 0, 15, 0, step = 1),
    
    ui.input_slider("sel_other_node", "Selected other node", 0, 15, 1, step = 1),
    
    ui.input_slider("no_factors", "No. factors", 1, 15, 5, step = 1),
    
    ui.input_slider("kick_size", "Kick size", -2, 2, 0.1, step = 0.1),
    
    ui.input_slider("max_time", "Maximum time", 0, 50, 20, step = 1),
    
    ),
    
    ui.output_plot("Plot_Model_Output"),
    
)

###############################################################


def server(input, output, session):
    
        
    @render.plot
    # ignore_none=False is used to instruct Shiny to render this plot even before the
    # input.run button is clicked for the first time. We do this because we want to
    # render the empty 3D space on app startup, to give the user a sense of what's about
    # to happen when they run the simulation.
    @reactive.event(input.run, ignore_none=True)

    def Plot_Model_Output():
        
        ##initialise the factors
        
        kick_size=input.kick_size()
        
        no_t=250
        
        no_factors=input.no_factors()
        
        sel_node=int(input.sel_node())
        
        sel_other_node=int(input.sel_other_node())
        
        np.random.seed(100*input.reset_model())
        
        growth_rate=np.random.random(no_factors)*2

        growth_to_max_rate=np.random.random(no_factors)*2

        max_resources=np.random.random(no_factors)*2

        interactions=np.random.random([no_factors, no_factors])*2-1

        for i in np.arange(no_factors):

            interactions[i,i]=0
        
        def Calc_x_dot(x):

            x_dot=np.zeros(no_factors)

            for sel_ind in np.arange(no_factors):

                x_growth=x[sel_ind]*growth_rate[sel_ind]

                x_logistic_growth=growth_to_max_rate[sel_ind]*max_resources[sel_ind]-x[sel_ind]

                for sel_other_ind in np.arange(no_factors):
                
                    x_logistic_growth=x_logistic_growth+interactions[sel_ind, sel_other_ind]*x[sel_other_ind]
                    
                x_dot[sel_ind]=x_growth*x_logistic_growth
                    
            return(x_dot)

        def Behaviour_Model_ODE(t, x):#, alpha, beta, gamma, delta, epsilon):

            x_dot=Calc_x_dot(x)
            
            return(x_dot)
            


        #single_kick_data=Single_Behaviour_Kick(kick_size, no_factors, no_t, 1)

        t_max=0
            
        single_kick_data=[]

        x_init=np.random.random(no_factors)*0.4
                
        full_z=np.reshape(x_init,(no_factors,1))

        print(full_z)

        full_t=[0]

        print(full_t)

        nudge_behaviour=0
                
        t_min=0

        t_max=int(input.max_time()/2)

        t_sol=np.linspace(t_min, t_max, no_t)
                
        sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

        z=sol.sol(t_sol)

        full_z=np.hstack([full_z,z])
                
        full_t=np.hstack([full_t,t_sol])

        L=len(z[0,:])

        x_init=z[:,L-1]

        single_kick_data=np.hstack([single_kick_data, x_init[[0, 1]]])

        ######

        ##nudge the system

        if input.parameter_to_update()=="Value":

            x_init[sel_node]=x_init[sel_node]+kick_size#np.random.random(2)*2
            
        if input.parameter_to_update()=="Max level":

            max_resources[sel_node]=max_resources[sel_node]+kick_size#np.random.random(2)*2
            
        if input.parameter_to_update()=="Interaction":

            interactions[sel_other_node, sel_node]=interactions[sel_other_node, sel_node]+kick_size#np.random.random(2)*2

        #######

        ##run for another half of the time to see what happens

        t_min=int(input.max_time()/2)

        t_max=int(input.max_time())

        t_sol=np.linspace(t_min, t_max, no_t)
                
        sol=solve_ivp(Behaviour_Model_ODE, [np.min(t_sol), np.max(t_sol)], x_init, dense_output=True, t_eval=t_sol)

        z=sol.sol(t_sol)

        full_z=np.hstack([full_z,z])
                
        full_t=np.hstack([full_t,t_sol])
     
        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].plot(full_t, full_z.T)

        #############################################################

        ##plot the connecting networkx

        seed = 13648  # Seed random number generators for reproducibility
        #G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)

        G = nx.DiGraph(interactions)

        pos = nx.spring_layout(G, seed=seed)

        node_sizes = 200*(1+max_resources/np.sum(max_resources))
        M = G.number_of_edges()

        all_edge_colors = np.reshape(interactions, (len(interactions[:,0])*len(interactions[:,0]), 1))

        edge_colors=all_edge_colors[all_edge_colors!=0]

        cmap = plt.cm.plasma

        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="white", edgecolors="black")

        edges = nx.draw_networkx_edges(
            G,
            pos,
            node_size=node_sizes,
            arrowstyle="->",
            arrowsize=10,
            edge_color=edge_colors,
            edge_cmap=cmap,
            width=2,
            connectionstyle='arc3,rad=0.1'
        )

        labels=nx.draw_networkx_labels(G, pos=pos)

        pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        pc.set_array(edge_colors)

        ax[1] = plt.gca()
        ax[1].set_axis_off()
        plt.colorbar(pc, ax=ax[1])

        #plt.show()
                
        #fig.savefig("single_kick.png")
                
        #plt.close()

        #nx.draw(G, edge_color=interactions+np.min(interactions)+0.01)

        #plt.show()

        #plt.close()
        
        return fig
     
     
app = App(app_ui, server)
    





















































