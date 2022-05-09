from multiprocessing.pool import Pool
import functools
import numpy as np
from scipy import integrate

#   seperate e^{i\delta_m*t} into cos\delta_m*t+i*sin\delta_m*t and then integrate seperately
#   used for calculate the cost function
def calculator_P(detuning_list, duration, number_of_segments, if_restrict = False,restrictNum = None):
    k_list = [i for i in np.arange(len(detuning_list))]
    res_c = restrictNum
    if if_restrict == True and len(detuning_list) > res_c:
        print('hh')
        num = len(detuning_list)
        while num > res_c:
            if num // 2 == 0:
                k_list = k_list[1:]
            else:
                k_list = k_list[:-1]
            num -= 1
    print('k_list',k_list)
    dur_seg = duration / number_of_segments  # duration of one segment

    P_real = []
    P_imag = []
    for m in np.arange(len(detuning_list)):
        if m in k_list:
            P_real.append([0.5 * integrate.quad(lambda x: np.cos(detuning_list[m] * x), k * dur_seg, (k + 1) * dur_seg)[0] \
               for k in np.arange(number_of_segments)])
            P_imag.append([0.5 * integrate.quad(lambda x: np.sin(detuning_list[m] * x), k * dur_seg, (k + 1) * dur_seg)[0] \
               for k in np.arange(number_of_segments)])
        else:
            P_real.append([0.]*number_of_segments)
            P_imag.append([0.]*number_of_segments)

    #P_real = [[0.5 * integrate.quad(lambda x: np.cos(detuning_list[m] * x), k * dur_seg, (k + 1) * dur_seg)[0] \
               #for k in np.arange(number_of_segments)] for m in k_list]
    #P_imag = [[0.5 * integrate.quad(lambda x: np.sin(detuning_list[m] * x), k * dur_seg, (k + 1) * dur_seg)[0] \
               #for k in np.arange(number_of_segments)] for m in k_list]
    # return np.array(P_real) + 1.0j * np.array(P_imag)
    return np.matrix(P_real,dtype = complex) + 1.0j * np.matrix(P_imag,dtype = complex)


'''

def calculator_G_ijspecified_real(detuning_list, duration, number_of_segments, eta, ij):
    i, j = ij
    dur_seg = duration / number_of_segments  # duration of one segment
    G_real = []
    for k in np.arange(number_of_segments):
        _middle_real = []

        for l in np.arange(k):
            middle = sum([0.25 * eta[m][i] * eta[m][j] * integrate.dblquad(lambda t1, t2: \
                                                                               np.cos(detuning_list[m] * (t1 - t2)),
                                                                           k * dur_seg, (k + 1) * dur_seg, \
                                                                           lambda t1: l * dur_seg,
                                                                           lambda t1: (l + 1) * dur_seg)[0] for m in
                          np.arange(len(detuning_list))])
            _middle_real.append(middle)

        _middle_real.append(sum([0.25 * eta[m][i] * eta[m][j] * integrate.dblquad(lambda t1, t2: \
                                                                                      np.cos(
                                                                                          detuning_list[m] * (t1 - t2)),
                                                                                  k * dur_seg, (k + 1) * dur_seg, \
                                                                                  lambda t1: k * dur_seg,
                                                                                  lambda t1: t1)[0] for m in
                                 np.arange(len(detuning_list))]))
        [_middle_real.append(0) for pad in np.arange(number_of_segments - k - 1)]
        G_real.append(_middle_real)
    return G_real



    
def calculator_G_ijspecified_imag(detuning_list, duration, number_of_segments, eta, ij):
    i,j = ij
    dur_seg = duration / number_of_segments  # duration of one segment
    G_imag = []
    for k in np.arange(number_of_segments):
        _middle_imag = []

        for l in np.arange(k):
            middle = sum([0.25 * eta[m][i] * eta[m][j] * integrate.dblquad(lambda t1, t2: \
                                                                               np.sin(detuning_list[m] * (t1 - t2)),
                                                                           k * dur_seg, (k + 1) * dur_seg, \
                                                                           lambda t1: l * dur_seg,
                                                                           lambda t1: (l + 1) * dur_seg)[0] for m in
                          np.arange(len(detuning_list))])
            _middle_imag.append(middle)

        _middle_imag.append(sum([0.25 * eta[m][i] * eta[m][j] * integrate.dblquad(lambda t1, t2: \
                                                                                      np.sin(
                                                                                          detuning_list[m] * (t1 - t2)),
                                                                                  k * dur_seg, (k + 1) * dur_seg, \
                                                                                  lambda t1: k * dur_seg,
                                                                                  lambda t1: t1)[0] for m in
                                 np.arange(len(detuning_list))]))
        [_middle_imag.append(0) for pad in np.arange(number_of_segments - k - 1)]
        G_imag.append(_middle_imag)
    return G_imag

'''



def func_real(detuning_list, duration, number_of_segments, eta, ij):
    return (ij,calculator_G_ijspecified_real(detuning_list, duration, number_of_segments, eta, ij))

def func_img(detuning_list, duration, number_of_segments, eta, ij):
    return (ij,calculator_G_ijspecified_imag(detuning_list, duration, number_of_segments, eta, ij))



def calculator_G(detuning_list, duration, number_of_segments, eta):
    spin_number = len(detuning_list)
    pool = Pool()
    
    G_real_para = pool.map(functools.partial(func_real,detuning_list, duration, number_of_segments, eta), [(i,j) for i in np.arange(spin_number) for j in np.arange(spin_number)])
    G_real_para_dict = {}
    for i in G_real_para:
        G_real_para_dict[i[0]] = i[1]


    G_img_para = pool.map(functools.partial(func_img,detuning_list, duration, number_of_segments, eta), [(i,j) for i in np.arange(spin_number) for j in np.arange(spin_number)])
    G_img_para_dict = {}
    for i in G_img_para:
        G_img_para_dict[i[0]] = i[1]

    G_real = [[G_real_para_dict[(i,j)] for i in np.arange(spin_number)]for j in np.arange(spin_number)]
    G_imag =  [[G_img_para_dict[(i,j)] for i in np.arange(spin_number)]for j in np.arange(spin_number)]
    return np.matrix(G_real,dtype=complex)+1.j*np.matrix(G_imag,dtype=complex)