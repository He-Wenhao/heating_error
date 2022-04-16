"""
This module contains functions for generating the Hamiltonian of trapped ions under laser fields.
"""

from qutip import *
import numpy as np

def rotate_right(list_in, n):
    return list_in[-n:]+list_in[:-n]

def list_sum(list_in):
    total = list_in[0]
    for i in np.arange(1,len(list_in)):
        total = total + list_in[i]
    return total

def H_Ion(laser_fields, spin_number, mode_frequencies, motional_pattern, Lamb_Dick_para, cut_off):
    #   Hamiltonian的构造是参考note中的章节：通过Magnus 展开推导2-qbits MS门
    spin_part = [sigmap()] + [qeye(2) for i in np.arange(spin_number-1)] #rotate sigma_plus to generate different spin interaction
    motional_part = [create(cut_off)*destroy(cut_off)] + [qeye(cut_off) for i in np.arange(spin_number-1)]  # rotate n to generate different motional H0
    motion_ladder_part = [create(cut_off)+destroy(cut_off)] + [qeye(cut_off) for i in np.arange(spin_number-1)] #create a+adag part
    
    H_motion = [mode_frequencies[i]*tensor([qeye(2) for i in np.arange(spin_number)] + rotate_right(motional_part,i)) for i in np.arange(spin_number)]
    #   motional_pattern[m][j]中，m表示motional modes的编号，j表示离子的编号

    H_int = [tensor(tensor(rotate_right(spin_part, j)), (1j*sum([Lamb_Dick_para*motional_pattern[m][j]*tensor(rotate_right(motion_ladder_part,m)) for m in np.arange(spin_number)])).expm()) for j in np.arange(spin_number)]

    H_with_laser = [[[H_int[j], laser] for laser in laser_fields[j][0]] for j in np.arange(spin_number)]
    H_with_laser_dag = [[[H_int[j].dag(), laser] for laser in laser_fields[j][1]] for j in np.arange(spin_number)]

    return list_sum(H_with_laser) + list_sum(H_with_laser_dag) + [sum(H_motion)]


#   H_Ion 将 exp( 1.0j * eta( a + a^dag ) ) 展开成  I + 1.0j * eta( a + a^dag )
def H_Ion_new(laser_fields, spin_number, mode_frequencies, motional_pattern, Lamb_Dick_para, cut_off):
    #   Hamiltonian的构造是参考note中的章节：通过Magnus 展开推导2-qbits MS门
    spin_part = [sigmap()] + [qeye(2) for i in
                              np.arange(spin_number - 1)]  # rotate sigma_plus to generate different spin interaction
    motional_part = [create(cut_off) * destroy(cut_off)] + [qeye(cut_off) for i in np.arange(
        spin_number - 1)]  # rotate n to generate different motional H0
    motion_ladder_part = [create(cut_off) + destroy(cut_off)] + [qeye(cut_off) for i in
                                                                 np.arange(spin_number - 1)]  # create a+adag part

    H_motion = [mode_frequencies[i] * tensor([qeye(2) for i in np.arange(spin_number)] + rotate_right(motional_part, i))
                for i in np.arange(spin_number)]

    #   motional_pattern[m][j]中，m表示motional modes的编号，j表示离子的编号
    H_int = []
    for ions in np.arange(spin_number):
        eta_a_part = 1
        for m in np.arange(spin_number):
           temp =  tensor([qeye(cut_off) for i in np.arange(spin_number)]) + 1j * (Lamb_Dick_para * motional_pattern[m][ions] * tensor(rotate_right(motion_ladder_part, m)) )
           eta_a_part = eta_a_part * temp
        H_int.append(tensor(tensor(rotate_right(spin_part, ions)), eta_a_part))

    H_with_laser = [[[H_int[j], laser] for laser in laser_fields[j][0]] for j in np.arange(spin_number)]
    H_with_laser_dag = [[[H_int[j].dag(), laser] for laser in laser_fields[j][1]] for j in np.arange(spin_number)]

    return list_sum(H_with_laser) + list_sum(H_with_laser_dag) + [sum(H_motion)]




