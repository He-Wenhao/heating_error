from qutip import *
import numpy as np

from pakages.quantum_dynamics_simulation.segmented_simulation import *
from pakages.quantum_dynamics_simulation.visualization import *



# 纯粹二能级计算
def two_level_calculate(initial_state, Omega, detu, phi, tlist):  ##计算二能级系统在外场的作用下演化过程
    H0 = 0 * identity(2);
    H1 = sigmap();
    H2 = sigmam()  ##  对角项 和 非对角项（2个）

    def H1_coeff(t, args):
        nu_1 = args['nu_1']
        return 1 / 2 * Omega * np.exp(-1.0j * (nu_1 * t + phi))

    def H2_coeff(t, args):
        nu_2 = args['nu_2']
        return 1 / 2 * Omega * np.exp(1.0j * (nu_2 * t + phi))

    H = [H0, [H1, H1_coeff], [H2, H2_coeff]]
    psi0 = initial_state  ##  设置初态为 |up>
    # psi0=(basis(2,0)+basis(2,1)).unit()   #此处可以设置初态为 0>+|1> 的叠加态

    output = mesolve(H, psi0, tlist, [], [qdiags([1, 0], 0), qdiags([0, 1], 0)], args={'nu_1': detu, 'nu_2': detu})
    return output


# 纯粹二能级计算，bloch球
def two_level_calculate_bloch(initial_state, Omega, detu, phi, tlist):  ##上述函数在Bloch球上展示
    H0 = 0 * identity(2);
    H1 = sigmap();
    H2 = sigmam()  ##  对角项 和 非对角项（2个）

    def H1_coeff(t, args):
        nu_1 = args['nu_1']
        return 1 / 2 * Omega * np.exp(-1.0j * (nu_1 * t + phi))

    def H2_coeff(t, args):
        nu_2 = args['nu_2']
        return 1 / 2 * Omega * np.exp(1.0j * (nu_2 * t + phi))

    H = [H0, [H1, H1_coeff], [H2, H2_coeff]]
    psi0 = initial_state  ##  设置初态为 |up>
    # psi0=(basis(2,0)+basis(2,1)).unit()   #此处可以设置初态为 0>+|1> 的叠加态

    output = mesolve(H, psi0, tlist, [], [], args={'nu_1': detu, 'nu_2': detu})
    return output




# 二能级 + 声子 计算
def two_level_with_phonon_calculate( Omega_2_level, tao, detu, phi, initial_state_twolevel, eta, N_max, v_trap,initial_phonon ):
    H_para = {'spin_number': 2
        , 'mode_frequencies': np.array([v_trap, 0])     #   注意这里 trap frequency已经有2 pi
        , 'motional_pattern': [[1, 0], [0, 0]]
        , 'Lamb_Dick_para': eta
        , 'cut_off': N_max}

    Carrier_sequence = np.array([[[Omega_2_level, detu, phi]], [[0, 0, 0]]])  # 三个参数分别为 强度，频率，相位
    segmented_sequence = np.array([[Carrier_sequence, tao]],dtype=object)
    nstep = np.int(tao / (2 * np.pi * Omega_2_level) * 20)      #  设置步长
    psi0 = tensor(initial_state_twolevel, basis(2, 0), basis(H_para['cut_off'], initial_phonon), basis(H_para['cut_off'], initial_phonon))
    output = segmented_run(H_para, psi0, segmented_sequence,nstep)
    tlist  = output[0]
    states = output[1]
    two_level_population = individual_population(states, 2)

    phonon_population = phonon_two_level_individual_population(states, 2, N_max)
    return tlist, two_level_population, phonon_population


# SDF 计算
def SDF_calculate( Omega_2_level, tao, detu, phi, initial_state_twolevel, eta, N_max, v_trap , SDF_amp, SDF_detu,initial_phonon ):
    H_para = {'spin_number': 2
        , 'mode_frequencies': np.array([v_trap, 0])     #   注意这里 trap frequency已经有2 pi
        , 'motional_pattern': [[1, 0], [0, 0]]
        , 'Lamb_Dick_para': eta
        , 'cut_off': N_max}

    blue_and_red_sequence = [[[SDF_amp, v_trap + SDF_detu, phi],[1.3*SDF_amp, -v_trap-SDF_detu, phi]], [[0, 0, 0]]]  # 三个参数分别为 强度，频率，相位
    segmented_sequence = np.array([[blue_and_red_sequence, tao]],dtype=object)
    nstep = np.int(tao / (2 * np.pi / SDF_detu) * 1000)      #  设置步长
    psi0 = tensor(initial_state_twolevel, basis(2, 0), basis(H_para['cut_off'], initial_phonon), basis(H_para['cut_off'], initial_phonon))
    output = segmented_run(H_para, psi0, segmented_sequence,nstep)
    tlist  = output[0]
    states = output[1]
    two_level_population = individual_population(states, 2)

    phonon_population = phonon_two_level_individual_population(states, 2, N_max)

    return tlist, two_level_population, phonon_population
    # return states






#   dreft

# Omega_2_level, tao, detu, phi, initial_state_twolevel, eta, N_max, v_trap , SDF_amp, SDF_detu = 0.0, 10, 0, 0, basis(2,0), 0.1, 3, 12, 3, 0.2
# output = SDF_calculate( Omega_2_level, tao, detu, phi, initial_state_twolevel, eta, N_max, v_trap , SDF_amp, SDF_detu )
# print(output[2])
#
# print(output[2].ptrace([2]))

# spin1=basis(2,0)
# spin2=basis(2,1)
# adag=create(3,2)
#
# statess = ket2dm(tensor(spin1,spin2))
# print(statess)
#
# statess.ptrace([0,1])

