"""
This module contains functions for simulate segmented pulses.

segmented_sequence format: [[sequence1, duration1],[sequence2, duration2],...]
"""


import numpy as np
from qutip import *
from .pulse_generator import *
from .hamiltonian_generator import *
#segmented_sequence format: [[sequence1, duration1],[sequence2, duration2],...]

def run_once(H, psi0, tlist):
    opts   = Options(store_states=True)
    output = mesolve(H, psi0, tlist, e_ops=None, args=None, options=opts)
    return output.states
    

def segmented_run(H_para, psi0, segmented_sequence, nstep=100):
    spin_number      = H_para['spin_number']
    mode_frequencies = H_para['mode_frequencies']
    motional_pattern = H_para['motional_pattern']
    Lamb_Dick_para   = H_para['Lamb_Dick_para']
    cut_off          = H_para['cut_off']
    total_tlist      = []
    t_start          = 0
    
    state_list       = []
    for sequence in segmented_sequence:
        laser_fields = sequence[0]
        fields_func  = pulse_function_generator(laser_fields)
        duration     = sequence[1]
        tlist        = np.linspace(t_start, t_start + duration, nstep)
        total_tlist  = total_tlist + tlist.tolist()
        H            = H_Ion(fields_func, spin_number, mode_frequencies, motional_pattern, Lamb_Dick_para, cut_off)

        state_recent = run_once(H, psi0, tlist)
        state_list   = state_list + state_recent
        psi0         = state_recent[-1]
        t_start      = t_start + duration
    return [total_tlist, state_list]
        
        
    