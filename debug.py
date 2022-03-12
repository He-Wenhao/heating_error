import numpy as np
import qutip
import sim_method_2L
from my_constants import *
import trap_config
import time
from hzx_amp.wave_pulse import hzx_amp

def infidelity(a,b):
    return (1-qutip.fidelity(qutip.Qobj(a), qutip.Qobj(b)),qutip.tracedist(qutip.Qobj(a), qutip.Qobj(b)))

def check_alg1_alg2():

    #construct para
    cut_off = 8
    Omega_R = 2*np.pi*30*MHz
    ini_state= (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))    # init condition
    phonon_ini = (qutip.fock_dm(cut_off,0)) # init of phonon
    detuning = 20*kHz
    amplitude = {0:lambda t: Omega_R * np.sin(detuning*t),1:lambda t: Omega_R * np.sin(detuning*t)}
    para = trap_config.trap_para(ion_number = 3, laser_ion_list = [0,1],omega_ax = 0.32*2*np.pi*MHz,omega_ra = 2.18*2*np.pi*MHz, cut_off = cut_off,heating_rate = 1e0,t_tol = None,lamb_dicke = 0.1,_ini_state=ini_state,_phonon_ini=phonon_ini,basis = 'Gellman',amplitude = amplitude, lind_type = [],if_testXX=True)



    #simulation with 5 different methods
    
    t1 = time.time()
    
    ideal = sim_method_2L.ideal(para).sim()
    t2 = time.time()
    print('ideal:','time =',t2-t1,'\n',ideal)

    #brute = sim_method_2L.brute(para).sim()
    #t3 = time.time()
    #print('brute:','time =',t3-t2,'\n',brute)
    
    alg1 = sim_method_2L.alg1(para).sim()
    t4 = time.time()
    print('alg1:','time =',t4-t2,'\n',alg1)
    
    alg2_noLind = sim_method_2L.alg2_noLind(para).sim()
    t5 = time.time()
    print('alg2_noLind:','time =',t5-t4,'\n',alg2_noLind)
    
    alg2 = sim_method_2L.alg2(para).sim()
    t6 = time.time()
    print('alg2:','time =',t6-t5,'\n',alg2)

    print('cut_off =',cut_off,'total time = ',t6-t1)

    #results

def test_cutoff():

    #construct para
    def p(cut_off):
        Omega_R = 2*np.pi*30*MHz
        ini_state= (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))    # init condition
        phonon_ini = (qutip.fock_dm(cut_off,0)) # init of phonon
        detuning = 20*kHz
        amplitude = {0:lambda t: Omega_R * np.sin(detuning*t),1:lambda t: Omega_R * np.sin(detuning*t)}
        para = trap_config.trap_para(ion_number = 3, laser_ion_list = [0,1],omega_ax = 0.32*2*np.pi*MHz,omega_ra = 2.18*2*np.pi*MHz, cut_off = cut_off,heating_rate = 1e0,t_tol = None,lamb_dicke = 0.1,_ini_state=ini_state,_phonon_ini=phonon_ini,basis = 'Gellman',amplitude = amplitude, lind_type = [],if_testXX=True)
        return para

    
    for cut_off in [8,9,10,11,12]:
        print('----------------------')
        print('cut_off =',cut_off)
        t4 = time.time()
        para = p(cut_off)
        alg1 = sim_method_2L.alg1(para).sim()
        print('alg1 =',alg1)
        alg2_noLind = sim_method_2L.alg2_noLind(para).sim()
        print('alg2_noLind:',alg2_noLind)
        t5 = time.time()
        print('difference =',infidelity(alg1, alg2_noLind))
        print('time =',t5-t4)


def test_N_ions_hzx():

    #construct para
    def p(N_ions,lind_type):
        cut_off = 10
        Omega_R = 2*np.pi*30*MHz
        heating_rate = 1e2
        laser_ion_list = [0,1]
        ini_state= (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))    # init condition
        phonon_ini = (qutip.fock_dm(cut_off,0)) # init of phonon
        detuning = 20*kHz
        tau = 100*us
        a = hzx_amp(N_ions = N_ions,Omega_ax = 0.32*2*np.pi*MHz,Omega_ra = 2.18*2*np.pi*MHz,target_phase = np.pi/4,j_list = laser_ion_list,N_seg = 7,tau = tau)
        a.optimize()
        amp = a.get_amp()
        amplitude = {0:amp,1:amp}
        #amplitude = {0:lambda t: Omega_R * np.sin(detuning*t),1:lambda t: Omega_R * np.sin(detuning*t)}
        para = trap_config.trap_para()
        para .my_init(ion_number = N_ions, laser_ion_list = laser_ion_list,omega_ax = 0.32*2*np.pi*MHz,omega_ra = 2.18*2*np.pi*MHz, cut_off = cut_off,heating_rate = heating_rate,t_tol = tau,lamb_dicke = 0.1,_ini_state=ini_state,_phonon_ini=phonon_ini,basis = 'Gellman',amplitude = amplitude, lind_type = lind_type,if_testXX=False)
        return para

    
    for N_ions in range(2,10):
        
        para = p(N_ions,['aH','a'])
        t1 = time.time()
        alg1_lind = sim_method_2L.alg1(para).sim()
        print('alg1:',alg1_lind)
        t2 = time.time()
        print('with lind','N_ions =',N_ions, 'time alg1 =',t2-t1)
        para.lind_type = []
        t1 = time.time()
        alg1 = sim_method_2L.alg1(para).sim()
        print('alg1:',alg1)
        t2 = time.time()
        print('no lind','N_ions =',N_ions, 'time alg1 =',t2-t1)

        infid1 = infidelity(alg1, alg1_lind)
        print('--------')
        print(N_ions,infid1)


def para0(c):
    #construct para
    cut_off = c
    Omega_R = 2*np.pi*30*MHz
    ini_state= (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))    # init condition
    phonon_ini = (qutip.fock_dm(cut_off,0)) # init of phonon
    detuning = 20*kHz
    amplitude = {0:lambda t: Omega_R * np.sin(detuning*t),1:lambda t: Omega_R* np.sin(detuning*t)}
    para = trap_config.trap_para(ion_number = 3, laser_ion_list = [0,1],omega_ax = 0.32*2*np.pi*MHz,omega_ra = 2.18*2*np.pi*MHz, cut_off = cut_off,heating_rate = 1e0,t_tol = None,lamb_dicke = 0.1,_ini_state=ini_state,_phonon_ini=phonon_ini,basis = None,amplitude = amplitude, lind_type = [],if_testXX=True)
    return para


def para1(c):
    #construct para
    cut_off = c
    Omega_R = 2*np.pi*30*MHz
    ini_state= (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))    # init condition
    phonon_ini = (qutip.fock_dm(cut_off,0)) # init of phonon
    detuning = 20*kHz
    amplitude = {0:lambda t: Omega_R* np.sin(detuning*t),1:lambda t: Omega_R* np.sin(detuning*t)}
    para = trap_config.trap_para(ion_number = 3, laser_ion_list = [0,1],omega_ax = 0.32*2*np.pi*MHz,omega_ra = 2.18*2*np.pi*MHz, cut_off = cut_off,heating_rate = 1e0,t_tol = None,lamb_dicke = 0.1,_ini_state=ini_state,_phonon_ini=phonon_ini,basis = 'Gellman',amplitude = amplitude, lind_type = [],if_testXX=True)
 
    return para




if __name__ == '__main__':
    
    test_N_ions_hzx()



