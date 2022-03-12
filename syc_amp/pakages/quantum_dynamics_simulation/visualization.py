"""
This module contains functions for ploting the time evolution.
"""
from qutip import *
import matplotlib.pyplot as plt
import numpy as np



def individual_population(states, spin_number):
    density_list = [[state.ptrace(j) for state in states] for j in np.arange(spin_number).tolist()]
    population_list = []
    for densities in density_list:
        population = [[np.abs(density[i][0][i]) for density in densities] for i in np.arange(2)] #中间那个[0]是qutip奇怪的密度矩阵定义导致的
        population_list = population_list + [population]
    return population_list

def individual_plot(tlist, states, spin_number):
    population = individual_population(states, spin_number)
    fig, axs = plt.subplots(spin_number, 1)
    for i in np.arange(spin_number):
        axs[i].plot(tlist, population[i][0], tlist, population[i][1])
        axs[i].set_ylabel('Population')
        axs[i].grid(True)
        axs[i].set_xlabel('time')
    fig.tight_layout()
    plt.show()
    
def correlated_population(states, spin_number):
    spin_density_matrix = [state.ptrace(np.arange(spin_number)) for state in states]
    population_correlated = [[np.abs(density[i,i]) for density in spin_density_matrix] for i in np.arange(2**spin_number)]
    return population_correlated

def correlated_plot(tlist, states, spin_number):
    population_correlated = correlated_population(states, spin_number)
    for i in np.arange(2**spin_number):
        plt.plot(tlist,population_correlated[i], label=('{0:0'+str(spin_number)+'b}').format(i))
    plt.legend()
    plt.show()



def phonon_two_level_individual_population(states, spin_number, N_max):
    density_list = [state.ptrace([0,spin_number]) for state in states]
    population_list = []
    for densities in density_list:
        population = [[np.abs(densities[i][0][i]) for i in np.arange(N_max)],[np.abs(densities[i][0][i]) for i in np.arange(N_max,2*N_max)]]  #中间那个[0]是qutip奇怪的密度矩阵定义导致的
        population_list = population_list + [population]
    return population_list



#   计算 nbar 的均值，等效为alpha的 Re 部分
def MS2_phonon_modes_nbar(states, spin_number, N_max):
    density_list = [[state.ptrace(j) for state in states] for j in np.arange(spin_number,2*spin_number).tolist()]
    population_list = []
    for densities in density_list:
        # population = [ (  ( create(N_max) * destroy(N_max) ) * density ).tr() for density in densities]
        population = [expect((create(N_max) * destroy(N_max)), density) for density in densities]
        population_list = population_list + [population]
    return population_list



# #   计算 nbar 的均值，等效为alpha的 Re 部分, 不采用 trace
# def MS2_phonon_modes_nbar(states, spin_number, N_max):
#     density_list = [[state for state in states] for j in np.arange(spin_number,2*spin_number).tolist()]
#     population_list = []
#     for densities in density_list:
#         # population = [ (  ( create(N_max) * destroy(N_max) ) * density ).tr() for density in densities]
#         population = [expect( tensor( identity(2), identity(2), identity(N_max), (create(N_max) * destroy(N_max))), density) for density in densities]
#         population_list = population_list + [population]
#     return population_list







#   计算 X 的均值，等效为alpha的 Re 部分
def MS2_phonon_alpha_x(states, spin_number, N_max):
    density_list = [[state for state in states] for j in np.arange(spin_number,2*spin_number).tolist()]
    population_list = []
    for densities in density_list:
        population = [abs(expect( tensor( identity(2), identity(2), (create(N_max) + destroy(N_max)), identity(N_max)), density) ) for density in densities]
        population_list = population_list + [population]
    return population_list

#   计算 P 的均值，等效为alpha的 Im 部分
def MS2_phonon_alpha_p(states, spin_number, N_max):
    density_list = [[state for state in states] for j in np.arange(spin_number,2*spin_number).tolist()]
    population_list = []
    for densities in density_list:
        population = [abs(expect( tensor( identity(2), identity(2), (create(N_max) - destroy(N_max)), identity(N_max)), density) )for density in densities]
        population_list = population_list + [population]
    return population_list
