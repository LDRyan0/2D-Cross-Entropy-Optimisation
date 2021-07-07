# Copyright 2021, Liam Ryan, Maria Kovaleva, David Bulger
# All rights reserved.

# Adapted from MATLAB to Python by Liam Ryan (2021)

import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.stats import beta

# ------------------- ADJUST PARAMETERS HERE -------------------
x_range = (-5, 5)
y_range = (-5, 5)

# Set Cross Entropy parameters
genSize = 50 # generation size
qElite = 20 # elite size (inclusive)
smoothing = 0.5 # smoothing parameter
fBestAntenna = -100000 # initial best
N_it = 15 # number of iterations

# Fitness function (use np.cos(), np.exp() etc as x and y are vectors)
# Vectorisation is used to speed up the algorithm
def func(x,y): # x and y are vectors
    # Booth's
    # return -(np.square(x+2*y-7)+np.square(2*x+y-5))
    # Paraboloid
    # return -(np.square(x-2) + np.square(y+1))
    # Ackley's
    return -(-20*np.exp(-0.2*np.sqrt(0.5*(np.square(x)+np.square(y))))-np.exp(0.5*(np.cos(2*math.pi*x)+np.cos(2*math.pi*y)))+20+math.exp(1))

# --------------------------------------------------------------

# Allow for interactivity in matplotlib plots
plt.ion()

# Creating arrays to hold fitness information over generations
average_fitness_plot = np.zeros(N_it)
fBestAntenna_plot = np.zeros(N_it)

# Set initial distribution parameters
beta_alpha = np.ones((2,1)) # x and y values for alpha
beta_beta = np.ones((2,1)) # x and y values for beta

# Creating the grid
x_vec = np.linspace(x_range[0], x_range[1], 100)
y_vec = np.linspace(y_range[0], y_range[1], 100) 

# Obtain the 2D function values
f = func(x_vec[:, None], y_vec[None, :])

## Start Cross-Entropy Optimization Algorithm
for i in range(N_it):
    # Creating and formatting the subplots
    fig, (main_ax, fit_ax) = plt.subplots(1,2, figsize=(8,4), num=1)
    plt.subplots_adjust(left=0.1, wspace = 0.35, right=0.92, top=0.85) # neater formatting
    if (i==N_it-1): # identify last generation in title
        fig.suptitle("Generation " + str(i+1) + " (Final)", fontsize=18)
    else:
        fig.suptitle("Generation " + str(i+1), fontsize=18)

    main_ax.contourf(x_vec, y_vec, f, 20) # adding contour plot to main image
    
    # Sampling of the beta distribution
    x = np.random.beta(beta_alpha[0], beta_beta[0], genSize)
    x_real = x * (x_range[1]-x_range[0]) + x_range[0] 
    y = np.random.beta(beta_alpha[1], beta_beta[1], genSize)
    y_real = y * (y_range[1]-y_range[0]) + y_range[0]

    # Obtain fitness values (utilising vectorisation)
    fitness = func(x_real, y_real)

    # After fitness evaulation is done, we sort the best parameters (quicksort...O(n*logn))
    # More efficient implementation possible
    sortOrder = np.argsort(fitness)[::-1] # reverse array for descending order
    fitness = np.sort(fitness)[::-1] # reverse array for descending order

    # only take qElite number of the total generation
    sortOrder = sortOrder[0:qElite]

    # the elite, in descending order of fitness:
    elite_x = x[sortOrder]
    elite_x_real = x_real[sortOrder]
    elite_y = y[sortOrder]
    elite_y_real = y_real[sortOrder]

    # has this generation produced a new best antenna?
    if fitness[0] > fBestAntenna:
        fBestAntenna = fitness[0]
        best_x = elite_x[0]
        best_y = elite_y[0]
        best_x_real = elite_x_real[0]
        best_y_real = elite_y_real[0]

    
    # plot vs iterations
    average_fitness_plot[i] = np.mean(fitness)
    fBestAntenna_plot[i] = fBestAntenna
    
    fit_ax.plot(np.arange(1, i+2), fBestAntenna_plot[0:i+1])
    fit_ax.plot(np.arange(1, i+2), average_fitness_plot[0:i+1])
    fit_ax.set_title('Fitness over generations')
    fit_ax.set_xlabel('Number of generations')
    fit_ax.set_ylabel('Fitness function')
    fit_ax.legend(['best fitness', 'average fitness'], loc='lower right', framealpha=1,
        fontsize=6)
    fit_ax.grid()

    # update beta distribution parameters
    if i < N_it: # don't bother in the last generation
        xa, xb, xloc, xscale = beta.fit(elite_x , floc=0, fscale=1) # returns alpha, beta, location and scale
        ya, yb, yloc, yscale = beta.fit(elite_y, floc=0, fscale=1) # returns alpha, beta, location and scale
        new_beta_alpha = np.array([[xa], [ya]]) # column vector of alpha values (x,y)
        new_beta_beta = np.array([[xb], [yb]]) # column vector of beta values (x,y)
        # Smoothing is optional (defined in a conventional way)
    
        beta_alpha = beta_alpha + smoothing*(new_beta_alpha-beta_alpha)
        beta_beta = beta_beta + smoothing*(new_beta_beta-beta_beta)
    
    # Create x and y beta distributions
    dist_x = beta.pdf(x_vec, beta_alpha[0,0], beta_beta[0,0], loc=x_range[0], scale=(x_range[1]-x_range[0]))
    dist_y = beta.pdf(y_vec, beta_alpha[1,0], beta_beta[1,0], loc=y_range[0], scale=(y_range[1]-y_range[0]))

    # Create the 2D distribution (x, y no correlation)
    d = np.outer(dist_y, dist_x)
    
    # Formatting for sampling results visualisation
    main_ax.set_title('Sampling Results')
    main_ax.set_xlabel('x')
    main_ax.set_ylabel('y')
    sa = main_ax.scatter(x_real, y_real, c='#000000')
    se = main_ax.scatter(elite_x_real, elite_y_real, c='#ff0000', marker='x') # elite values
    sb = main_ax.scatter(elite_x_real[0], elite_y_real[0], c='#11FFEE', marker='+') # best value
    dist = main_ax.contour(x_vec, y_vec, d, 5, colors='blue', linestyles='dashed') 
    main_ax.legend((sa, se, sb), ('All', 'Elite', 'Best'), fontsize=6, framealpha=1, 
        loc='lower right')

    # Give terminal output of the results of the current generation
    print("Generation " + str(i+1))
    print("\t Best point (x,y) = ({},{})".format(best_x_real, best_y_real))
    print("\t     Best fitness = " + str(fBestAntenna))
    print("\t  Average fitness = " + str(np.mean(fitness)))
    print("\t(alpha x, beta x) = ({},{})".format(beta_alpha[0,0], beta_beta[0,0]))
    print("\t(alpha y, beta y) = ({},{})\n".format(beta_alpha[1,0], beta_beta[1,0]))
       
    # Show plot, wait for key press before moving to next generation
    plt.show()
    plt.waitforbuttonpress()
    plt.clf()

# -------------------------------------------------------------------------------------------------------------------------------------

plt.close() # close sample figure window

# plot the results
fig, fit_ax = plt.subplots()
fit_ax.plot(np.arange(1, N_it+1), fBestAntenna_plot)
fit_ax.plot(np.arange(1, N_it+1), average_fitness_plot)
fit_ax.set_title('Fitness over generations')
fit_ax.set_xlabel('Number of generations')
fit_ax.set_ylabel('Fitness function')
fit_ax.legend(['best fitness', 'average fitness'])
fit_ax.grid()

# output the best x and y value, corresponding fitness
print("FINAL RESULTS")
print("\tBest point (x,y) = ({:.10f}, {:.10f})".format(best_x_real, best_y_real))
print("\t    Best fitness = {:.10f}\n".format(fBestAntenna))

# Show plot and wait for key press before finishing
plt.show()
plt.waitforbuttonpress()
plt.close()
