##simulation
cut_off = 10


import numpy as np
import qutip
import time
import scipy.integrate as integrate
import sympy as sp
from functools import lru_cache
import os
import re
import configparser

import sys
sys.path.append('../..')
import sim_method_2L
from my_constants import *
import trap_config

prefix = './configurations'
all_dir = os.listdir(prefix)
all_dir.sort(key=lambda x:int(re.sub(u"([^\u0030-\u0039])", "", x)))



def infidelity(a,b):
    return (1-qutip.fidelity(qutip.Qobj(a), qutip.Qobj(b)),qutip.tracedist(1e4*qutip.Qobj(a), 1e4*qutip.Qobj(b))/1e4)

@lru_cache(1000)
def _integrate(*args):
    return sp.integrate(*args)

def seg_integrate(f_l,x,interval_l,eval_flag):
    if eval_flag == False:
        res = 0
        for i in range(len(f_l)-1):
            res += _integrate(f_l[i],(x,interval_l[i][0],interval_l[i][1])).evalf()
        i = len(f_l)-1
        res += _integrate(f_l[i],(x,interval_l[i][0],interval_l[i][1]))
        #res = sum([_integrate(f_l[i],(x,interval_l[i][0],interval_l[i][1])) for i in range(len(f_l))])
        return res
    else:
        return sum([_integrate(f_l[i],(x,interval_l[i][0],interval_l[i][1])).evalf() for i in range(len(f_l))])

def bound_eq26():
    #iterate through all configurations
    for folder in all_dir:
        new_prefix = prefix+'/'+folder+'/'
        #read trap.ini
        config = configparser.ConfigParser()
        config.read(new_prefix+'trap.ini')
        N_ions = eval(config['trap']['N_ions'])
        lamb_dicke = eval(config['trap']['lamb_dicke'])
        heating_rate = eval(config['trap']['heating_rate'])
        if N_ions != 3:
            continue

        #read phonon.ini
        config = configparser.ConfigParser()
        config.read(new_prefix+'phonon.ini')
        Z_freqcal  = eval(config['phonon']['Z_freqcal'])
        Z_modes = eval(config['phonon']['Z_modes'])

        # set some laser parameters
        config.read(new_prefix+'laser.ini')
        delta_t = eval(config['amplitude']['delta_t'])/us
        mu = eval(config['amplitude']['mu'])
        j_ions = eval(config['shined']['shined'])
        j0,j1 = j_ions
        amp_list = [eval(config['amplitude']['amp_list0']),eval(config['amplitude']['amp_list1'])]
        amp_list = np.array(amp_list)*0.5
        segNum = len(amp_list[0])
        detuning = np.array([i - mu for i in Z_freqcal])/MHz

        #compute
        for j in j_ions[0:1]:
            t = sp.Symbol('t',real = True)
            result0 = 0
            result1 = 0
            result01 = 0
            for k in range(N_ions):
                t1 = time.time()
                print('k = ',k)
                t0 = sp.Symbol('t0',real = True)
                f_l = []
                interval_l = [(i*delta_t,(i+1)*delta_t) for i in range(segNum)]
                for p in range(segNum):
                    alpha_int = seg_integrate(np.array(amp_list[0][0:p+1])*sp.exp(-sp.I*sp.S(detuning[k])*t0),t0,[(i*delta_t,(i+1)*delta_t) for i in range(p)]+[(p*delta_t,t)],eval_flag=False)
                    alpha_int = sp.expand(alpha_int)  #此处还有优化空间,因为同样的式子被simplify了好多次
                    alpha_int = alpha_int * sp.conjugate(alpha_int)
                    alpha_int = sp.expand(alpha_int)
                    f_l.append(alpha_int)
                t2 = time.time()
                print('median',t2-t1)
                temp_result = seg_integrate(f_l, t, interval_l,eval_flag = True)
                #result = max(result,sp.re(temp_result.evalf()))
                result0 += temp_result*lamb_dicke**2*Z_modes[j0][k]**2
                result1 += temp_result*lamb_dicke**2*Z_modes[j1][k]**2
                result01 += temp_result*lamb_dicke**2*Z_modes[j0][k]*Z_modes[j0][k]
                t3 = time.time()
                print('end',t3-t2)
            result = 2*heating_rate*(result0+result1+2*result01)*1e-6
            result = sp.re(result)
            print('---------------N =',N_ions,'my bound',result)
            config = configparser.ConfigParser()
            config.read(new_prefix+'result.ini')
            if 'result' not in config.sections():
                config['result'] = {}
            config['result']['bound_eq26'] = str(result)
            with open(new_prefix+'result.ini','w') as configfile:
                config.write(configfile)

def num_heating_error():
    #iterate through all configurations
    for folder in all_dir:
        para = trap_config.trap_para()
        new_prefix = prefix+'/'+folder+'/'
        #
        para.cut_off = cut_off

        # set initial state
        para._ini_state = (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))
        para._phonon_ini = (qutip.fock_dm(para.cut_off,0))

        #read trap.ini
        config = configparser.ConfigParser()
        config.read(new_prefix+'trap.ini')
        para.N_ions = eval(config['trap']['N_ions'])
        para.lamb_dicke = eval(config['trap']['lamb_dicke'])
        para.heating_rate = eval(config['trap']['heating_rate'])
        if para.N_ions != 3:
            continue

        #read phonon.ini
        config = configparser.ConfigParser()
        config.read(new_prefix+'phonon.ini')
        para.Z_freqcal  = eval(config['phonon']['Z_freqcal'])
        para.Z_modes = eval(config['phonon']['Z_modes'])

        # set some laser parameters
        config.read(new_prefix+'laser.ini')
        para.delta_t = eval(config['amplitude']['delta_t'])
        para.mu = eval(config['amplitude']['mu'])
        para.j_ions = eval(config['shined']['shined'])
        j0,j1 = para.j_ions
        para.Z_freqcal = np.array([i - para.mu for i in para.Z_freqcal])

        #initialize pulse
        amp_list = [eval(config['amplitude']['amp_list0']),eval(config['amplitude']['amp_list1'])]
        amp_list = np.array(amp_list)*0.5*MHz
        segNum = len(amp_list[0])
        para.t_tol = segNum*para.delta_t

        #
        para.partialpos = None


        def result_func(t,i):
            ion_i_amp_list = np.real(amp_list[i])
            ind = min(int(t/para.delta_t),segNum-1)
            return ion_i_amp_list[ind]
        
        para.amplitude = {j0:lambda t: result_func(t,0)  , j1:lambda t:result_func(t,1)}

        para.init_2(ifTheta=False)   


        #simulate
        para.lind_type = ['aH','a']
        t1 = time.time()
        alg1_lind = sim_method_2L.alg1(para).sim()
        print('alg1:',alg1_lind)
        t2 = time.time()
        print('with lind','N_ions =',para.N_ions, 'time alg1 =',t2-t1)
        para.lind_type = []
        t1 = time.time()
        alg1 = sim_method_2L.alg1(para).sim()
        print('alg1:',alg1)
        t2 = time.time()
        print('no lind','N_ions =',para.N_ions, 'time alg1 =',t2-t1)

        infid1 = infidelity(alg1, alg1_lind)
        print('--------')
        print('N_ions',para.N_ions,'cut_off',para.cut_off,'(infid,tracedist)',infid1)

        #save file
        config = configparser.ConfigParser()
        config.read(new_prefix+'result.ini')
        if 'result' not in config.sections():
            config['result'] = {}
        config['result']['infid'] = str(infid1[0])
        config['result']['tracedist'] = str(infid1[1])

        #compute infid against ideal solution
        ideal_mat = np.zeros((4,4),dtype=complex)
        ideal_mat[0][0] = 0.5
        ideal_mat[3][3] = 0.5
        ideal_mat[3][0] = -0.5j
        ideal_mat[0][3] = 0.5j
        infid_ideal = infidelity(ideal_mat, alg1)
        print(infid_ideal)
        config['result']['infid_ideal'] = str(infid_ideal[0])

        with open(new_prefix+'result.ini','w') as configfile:
            config.write(configfile)

if __name__ == '__main__':  
    #num_heating_error()
    bound_eq26()