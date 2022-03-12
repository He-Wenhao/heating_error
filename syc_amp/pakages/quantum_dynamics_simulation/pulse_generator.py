"""
This module contains functions for generating time-dependent function based on input pulses.

Format of laser_setting [Ion1's laser, Ion2's laser, Ion3's laser, Ion4's laser......]
Ion's laser = [[Rabi_freqeuence, detuning, phase],[Rabi_freqeuence, detuning, phase],...]
return: [Ion1 functions, Ion2 functions, Ion3 functions]
Ion functions = [[posi1, posi2, ...], [nega1, nega2, ...]]
注意这里应该是 Omega/2
"""
import numpy as np

def make_pulse_posi(pulse):
    def posi(t, args):
        return pulse[0]/2*np.exp(1j*pulse[1]*t+1j*pulse[2])
    return posi

def make_pulse_nega(pulse):
    def nega(t, args):
        return pulse[0]/2*np.exp(-1j*pulse[1]*t-1j*pulse[2])
    return nega

def pulse_function_generator(sequence):
    pulse_function = []
    for ion_index in np.arange(len(sequence)):
        pulse_function_posi = []
        pulse_function_nega = []
        for pulse in sequence[ion_index]:
            pulse_function_posi.append(make_pulse_posi(pulse))
            pulse_function_nega.append(make_pulse_nega(pulse))
        pulse_function.append([pulse_function_posi,pulse_function_nega])
    return pulse_function