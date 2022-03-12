import numpy as np
import qutip
import sim_method_2L
from my_constants import *
import trap_config
import time
from hzx_amp.wave_pulse import hzx_amp
from syc_amp.wave_pulse import syc_amp
def infidelity(a,b):
    return (1-qutip.fidelity(qutip.Qobj(a), qutip.Qobj(b)),qutip.tracedist(1e4*qutip.Qobj(a), 1e4*qutip.Qobj(b))/1e4)



def test_N_ions_constant():

    #construct para
    def p(N_ions,lind_type):
    
        para = trap_config.trap_para()

        # basic parameters
        para.N_ions = N_ions
        para.j_ions = [0,1]
        para.omega_ax = 0.32*2*np.pi*MHz
        para.omega_ra = 2.18*2*np.pi*MHz
        para.cut_off = 8
        para.heating_rate = 1e0
        para.lamb_dicke = 0.1
        para.mu = 5*MHz


        # set initial state
        para._ini_state = (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))
        para._phonon_ini = (qutip.fock_dm(para.cut_off,0))
        para.lind_type = lind_type

        # compute phonon frequency and bij
        phonons = trap_config.Ion_Trap_phonon(para.N_ions, para.omega_ax, para.omega_ra)
        para.Z_freqcal, para.Z_modes = phonons.w, phonons.b
        para.partialpos = phonons.equ_pos()


        # compute amplitute: chi_j1 chi_j2
        amplitude = {para.j_ions[0]:lambda t:10*MHz*np.sin(para.mu*t),para.j_ions[1]:lambda t:10*MHz*np.sin(para.mu*t)}
        para.amplitude = amplitude

        para.t_tol = para.get_tau_XX()

        para.init_2()      

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


def test_N_ions_syc():

    #construct para
    def p(N_ions,lind_type):
    
        para = trap_config.trap_para()

        # basic parameters
        para.N_ions = N_ions
        para.j_ions = [0,1]
        para.omega_ax = 0.32*2*np.pi*MHz
        para.omega_ra = 2.18*2*np.pi*MHz
        para.cut_off = 8
        para.heating_rate = 1e0
        para.lamb_dicke = 0.1
        para.mu = 5*MHz
        para.t_tol = 100*us

        # set initial state
        para._ini_state = (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))
        para._phonon_ini = (qutip.fock_dm(para.cut_off,0))
        para.lind_type = lind_type

        # compute phonon frequency and bij
        phonons = trap_config.Ion_Trap_phonon(para.N_ions, para.omega_ax, para.omega_ra)
        para.Z_freqcal, para.Z_modes = phonons.w, phonons.b
        para.partialpos = phonons.equ_pos()


        # compute amplitute: chi_j1 chi_j2
        syc = syc_amp(ion_number=para.N_ions,j_list=para.j_ions,omega=[x/MHz for x in para.Z_freqcal],bij=para.Z_modes,detuning=para.mu/MHz,tau=para.t_tol/us,segment_num=15,lamb_dicke=para.lamb_dicke)

        syc.func2_optimize_process_save_data(plotfig=True, pulse_symmetry = False, ions_same_amps = False)

        syc.import_amp()

        para.amplitude = syc.get_amp()

        

        para.init_2()      

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


def test_N_ions_hzx():

    #construct para
    def p(N_ions,lind_type):
    
        para = trap_config.trap_para()

        # basic parameters
        para.N_ions = N_ions
        para.j_ions = [0,1]
        para.omega_ax = 0.32*2*np.pi*MHz
        para.omega_ra = 2.18*2*np.pi*MHz
        para.cut_off = 10
        para.heating_rate = 1e2
        para.lamb_dicke = 0.1
        para.mu = 10*kHz
        para.t_tol = 100*us

        # set initial state
        para._ini_state = (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))
        para._phonon_ini = (qutip.fock_dm(para.cut_off,0))
        para.lind_type = lind_type

        # compute phonon frequency and bij
        phonons = trap_config.Ion_Trap_phonon(para.N_ions, para.omega_ax, para.omega_ra)
        para.Z_freqcal, para.Z_modes = phonons.w, phonons.b
        para.partialpos = phonons.equ_pos()


        # compute amplitute: chi_j1 chi_j2
        a = hzx_amp(N_ions = N_ions,Omega_ax = para.omega_ax,Omega_ra = para.omega_ra,target_phase = np.pi/4,j_list = para.j_ions,N_seg = 7,tau = para.t_tol)
        a.optimize()
        amp = a.get_amp()
        amplitude = {0:amp,1:amp}
        para.amplitude = amplitude

        

        para.init_2()      

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


def test_N_ions_hzx2():

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




if __name__ == '__main__':
    
    test_N_ions_hzx()



