# Copyright 2021, Liam Ryan, Maria Kovaleva, David Bulger
# All rights reserved

# Adapted from MATLAB to Python by Liam Ryan (2021)

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.stats import beta

# ------------------- ADJUST PARAMETERS HERE -------------------
# Set design parameters - Here we have only X and Y
x_range = (-5, 5)
y_range = (-5, 5)

# Set Cross Entropy parameters
genSize = 30 # generation size
qElite = 12 # elite size (inclusive)
smoothing = 0.5 # smoothing parameter
fBestAntenna = -100000 # initial best
N_it = 20 # number of iterations

# Fitness function
def func(x,y):
    return -(-20*np.exp(-0.2*np.sqrt(0.5*(np.square(x)+np.square(y))))-np.exp(0.5*(np.cos(2*math.pi*x)+np.cos(2*math.pi*y)))+20+math.exp(1))
# --------------------------------------------------------------

start = time.perf_counter() # start timer

# Creating arrays 
average_fitness_plot = np.zeros(N_it)
fBestAntenna_plot = np.zeros(N_it)

## Set initial distribution parameters
beta_alpha = np.ones((2,1))
beta_beta = np.ones((2,1))

# Start Cross-Entropy Optimization Algorithm
for i in range(N_it):
    
    # Sampling of the beta distribution
    x = np.random.beta(beta_alpha[0], beta_beta[0], genSize)
    x_real = x * (x_range[1]-x_range[0]) + x_range[0] 
    y = np.random.beta(beta_alpha[1], beta_beta[1], genSize)
    y_real = y * (y_range[1]-y_range[0]) + y_range[0]

    # Obtain fitness values (utilising vectorisation)
    fitness = func(x_real, y_real)

    # After fitness evaulation is done, we sort the best parameters (quicksort...O(n*logn))
    sortOrder = np.argsort(fitness)[::-1] # reverse for descending order
    fitness = np.sort(fitness)[::-1] # reverse for descending order

    # only take qElite number of the total generation
    sortOrder = sortOrder[0:qElite]
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

    if i < N_it: # don't bother in the last generation
        xa, xb, xloc, xscale = beta.fit(elite_x, floc=0, fscale=1) # returns alpha, beta, location and scale
        ya, yb, yloc, yscale = beta.fit(elite_y, floc=0, fscale=1) # returns alpha, beta, location and scale
        new_beta_alpha = np.array(xa, ya)
        new_beta_beta = np.array(xb, yb)
        # Smoothing in optional (defined in a conventional way)
        beta_alpha = beta_alpha + smoothing*(new_beta_alpha-beta_alpha)
        beta_beta = beta_beta + smoothing*(new_beta_beta-beta_beta)

# -------------------------------------------------------------------------------------------------------------------------------------
fin = time.perf_counter() # stop timer
total = fin - start

# plot the results
fig, ax = plt.subplots()
ax.plot(np.arange(1, N_it+1), fBestAntenna_plot)
ax.plot(np.arange(1, N_it+1), average_fitness_plot)
ax.set_title('Fitness over generations')
ax.set_xlabel('Number of generations')
ax.set_ylabel('Fitness function')
ax.legend(['best fitness', 'average fitness'])
ax.grid()
plt.show()

# output the best x and y value, corresponding fitness
print("\nBest point (x,y) = ({}, {})".format(best_x_real, best_y_real))
print("    Best fitness = {}".format(fBestAntenna))
print("Computation time = {:.4f} ms\n".format(total*1000))
# -------------------------------------------------------------------------------------------------------------------------------------
