from qutip import *
import numpy as np

from pakages.quantum_dynamics_simulation.segmented_simulation import *
from pakages.quantum_dynamics_simulation.visualization import *

def MS2_calculate( initial_state_spin, initial_phonon_num, N_max, eta, v1_trap, v2_trap, Omega_1b, detu_1b, phi_1b, Omega_1r, detu_1r, phi_1r, Omega_2b, detu_2b, phi_2b, Omega_2r, detu_2r, phi_2r, duration ):
    H_para = {'spin_number': 2
        , 'mode_frequencies': np.array([v1_trap, v2_trap])
        ,'motional_pattern':  [[1,1],[1,-1]] / np.sqrt(2)
        , 'Lamb_Dick_para': eta
        , 'cut_off': N_max}

    MS_sequence = np.array([[[Omega_1b, detu_1b, phi_1b], [Omega_1r, detu_1r, phi_1r]], \
                            [[Omega_2b, detu_2b, phi_2b], [Omega_2r, detu_2r, phi_2r]]])

    segmented_sequence = np.array([[MS_sequence, duration ]],dtype=object)
    nstep = 1000

    psi0 = tensor(initial_state_spin[0], initial_state_spin[1], basis(H_para['cut_off'], initial_phonon_num), basis(H_para['cut_off'], initial_phonon_num))
    output = segmented_run(H_para, psi0, segmented_sequence, nstep)
    return output




