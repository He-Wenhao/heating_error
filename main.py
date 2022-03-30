import numpy as np
import qutip
import sim_method_2L
from my_constants import *
import trap_config
import time
from hzx_amp.wave_pulse import hzx_amp
from syc_amp.wave_pulse import syc_amp
from zxm_freq.Tab1_ion_pos_spec_cal import  radial_mode_spectrum
import read_amp
import scipy.integrate as integrate

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
        para.heating_rate = 1e2
        para.lamb_dicke = 0.1
        
        #para.t_tol = 200*us

        # set initial state
        para._ini_state = (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))
        para._phonon_ini = (qutip.fock_dm(para.cut_off,0))
        para.lind_type = lind_type

        # compute phonon frequency and bij
        #phonons = trap_config.Ion_Trap_phonon(para.N_ions, para.omega_ax, para.omega_ra)
        para.partialpos = np.array(range(para.N_ions))*5
        y=np.array(range(para.N_ions))*5*us
        para.Z_freqcal, para.Z_modes = radial_mode_spectrum( para.N_ions, para.omega_ax,para.omega_ra,y)
        #para.partialpos = phonons.equ_pos()
        para.mu = sum(para.Z_freqcal)/len(para.Z_freqcal)
        print('Z_freq',para.Z_freqcal)
        # compute amplitute: chi_j1 chi_j2
        '''
        syc = syc_amp(ion_number=para.N_ions,j_list=para.j_ions,omega=[x - para.mu for x in para.Z_freqcal],bij=para.Z_modes,tau=para.t_tol/us,segment_num=15,lamb_dicke=para.lamb_dicke,mu = para.mu)

        syc.func2_optimize_process_save_data(plotfig=False, pulse_symmetry = False, ions_same_amps = True)

        #syc.func3_import_saved_data(plotfig = False, pulse_symmetry = False, ions_same_amps = True)

        syc.import_amp()

        para.amplitude = syc.get_amp()
        '''
      
        para.t_tol, para.amplitude = read_amp.get(para.N_ions,para.j_ions)

        para.Z_freqcal = [(x - para.mu)*MHz for x in para.Z_freqcal]
        

        para.init_2()      

        return para

    
    for N_ions in range(2,13):
        
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
        print('N_ions',N_ions,'cut_off',para.cut_off,'(infid,tracedist)',infid1)


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

def test_optimize_syc():
    #construct para
    def p(N_ions,seg_num):
        lind_type = []
        para = trap_config.trap_para()

        # basic parameters
        para.N_ions = N_ions
        para.j_ions = [0,1]
        para.omega_ax = 0.32*2*np.pi*MHz
        para.omega_ra = 2.18*2*np.pi*MHz
        para.cut_off = 9
        para.heating_rate = 1e2
        para.lamb_dicke = 0.1
        
        para.t_tol = 15*us*seg_num

        # set initial state
        para._ini_state = (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))
        para._phonon_ini = (qutip.fock_dm(para.cut_off,0))
        para.lind_type = lind_type

        # compute phonon frequency and bij
        #phonons = trap_config.Ion_Trap_phonon(para.N_ions, para.omega_ax, para.omega_ra)
        para.partialpos = np.array(range(para.N_ions))*5
        y=np.array(range(para.N_ions))*5*us
        para.Z_freqcal, para.Z_modes = radial_mode_spectrum( para.N_ions, para.omega_ax,para.omega_ra,y)
        #para.partialpos = phonons.equ_pos()
        para.mu = sum(para.Z_freqcal)/len(para.Z_freqcal)
        print('Z_freq',para.Z_freqcal)
        # compute amplitute: chi_j1 chi_j2
        syc = syc_amp(ion_number=para.N_ions,j_list=para.j_ions,omega=[x - para.mu for x in para.Z_freqcal],bij=np.matrix(para.Z_modes).transpose(),tau=para.t_tol/us,segment_num=seg_num,lamb_dicke=para.lamb_dicke,mu = para.mu)

        syc.func2_optimize_process_save_data(plotfig=False, pulse_symmetry = False, ions_same_amps = True)

        syc.import_amp()
        syc.print_amp()

        #syc.func3_import_saved_data(plotfig = True, pulse_symmetry = False, ions_same_amps = True)
        para.amplitude = syc.get_amp()
        para.Z_freqcal = [(x - para.mu)*MHz for x in para.Z_freqcal]
        #para.init_2() 
        #return para

        return syc.error
        #syc.import_amp()

        

      
        #para.Z_freqcal = [(x - para.mu)*MHz for x in para.Z_freqcal]
        

        #para.init_2()      

        #return para

    #para = p()
    #alg1 = sim_method_2L.alg1(para).sim()
    #print('alg1:',alg1)
    #alg1 = sim_method_2L.brute(para).sim()
    #print('brute:',alg1)
    evaluate = {2:10,3:18,4:20,5:20,6:24,7:32,8:36,9:36,10:38,11:48,12:52}
    for N in range(2,14):
        segNum = evaluate[N]
        while 1:
            print('-----','N',N,'segNum',segNum,'-----')
            error = p(N,segNum)
            if error < 1e-5:
                print('----- break -----')
                break
            else:
                segNum+=1
       
def test_N_ions_syc3():

    #construct para
    def p(N_ions,lind_type,cut_off):
    
        para = trap_config.trap_para()

        # basic parameters
        para.N_ions = N_ions
        para.j_ions = [0,1]
        para.omega_ax = 0.32*2*np.pi*MHz
        para.omega_ra = 2.18*2*np.pi*MHz
        para.cut_off = cut_off
        para.heating_rate = 1e2
        para.lamb_dicke = 0.1

        # set initial state
        para._ini_state = (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))
        para._phonon_ini = (qutip.fock_dm(para.cut_off,0))
        para.lind_type = lind_type

        # compute phonon frequency and bij
        #phonons = trap_config.Ion_Trap_phonon(para.N_ions, para.omega_ax, para.omega_ra)
        para.partialpos = np.array(range(para.N_ions))*5
        y=np.array(range(para.N_ions))*5*us
        para.Z_freqcal, para.Z_modes = radial_mode_spectrum( para.N_ions, para.omega_ax,para.omega_ra,y)
        #para.partialpos = phonons.equ_pos()
        para.mu = sum(para.Z_freqcal)/len(para.Z_freqcal)
        print('Z_freq',para.Z_freqcal)
        # compute amplitute: chi_j1 chi_j2
        '''
        syc = syc_amp(ion_number=para.N_ions,j_list=para.j_ions,omega=[x - para.mu for x in para.Z_freqcal],bij=np.matrix(para.Z_modes).transpose(),tau=para.t_tol/us,segment_num=para.segNum,lamb_dicke=para.lamb_dicke,mu = para.mu)

        syc.func2_optimize_process_save_data(plotfig=False, pulse_symmetry = False, ions_same_amps = True)

        #syc.func3_import_saved_data(plotfig = False, pulse_symmetry = False, ions_same_amps = True)

        syc.import_amp()

        para.amplitude = syc.get_amp()
        '''
      
        para.t_tol, para.amplitude = read_amp.get(para.N_ions,para.j_ions)

        para.Z_freqcal = [(x - para.mu)*MHz for x in para.Z_freqcal]
        

        para.init_2()      

        return para

    
    for cut_off in range(12,17):

        N_ions= 3
        para = p(N_ions,[],cut_off)
        para.cut_off = cut_off
        t1 = time.time()
        #alg1_lind = sim_method_2L.alg1(para).sim()
        #print('alg1:',alg1_lind)
        t2 = time.time()
        #print('with lind','N_ions =',N_ions, 'time alg1 =',t2-t1)
        #para.lind_type = []
        t1 = time.time()
        alg1 = sim_method_2L.alg1(para).sim()
        print('alg1:',alg1)
        t2 = time.time()
        print('no lind','N_ions =',N_ions, 'time alg1 =',t2-t1)

        #infid1 = infidelity(alg1, alg1_lind)
        print('--------')
        print('N_ions',N_ions,'cut_off',para.cut_off)

def compute_my_bound():

    #construct para
    def p(N_ions,lind_type):
    
        para = trap_config.trap_para()

        # basic parameters
        para.N_ions = N_ions
        para.j_ions = [0,1]
        para.omega_ax = 0.32*2*np.pi*MHz
        para.omega_ra = 2.18*2*np.pi*MHz
        para.cut_off = 8
        para.heating_rate = 1e2
        para.lamb_dicke = 0.1
        
        #para.t_tol = 200*us

        # set initial state
        para._ini_state = (qutip.tensor(qutip.fock_dm(2,0),qutip.fock_dm(2,0)))
        para._phonon_ini = (qutip.fock_dm(para.cut_off,0))
        para.lind_type = lind_type

        # compute phonon frequency and bij
        #phonons = trap_config.Ion_Trap_phonon(para.N_ions, para.omega_ax, para.omega_ra)
        para.partialpos = np.array(range(para.N_ions))*5
        y=np.array(range(para.N_ions))*5*us
        para.Z_freqcal, para.Z_modes = radial_mode_spectrum( para.N_ions, para.omega_ax,para.omega_ra,y)
        #para.partialpos = phonons.equ_pos()
        para.mu = sum(para.Z_freqcal)/len(para.Z_freqcal)
        print('Z_freq',para.Z_freqcal)
        # compute amplitute: chi_j1 chi_j2
        '''
        syc = syc_amp(ion_number=para.N_ions,j_list=para.j_ions,omega=[x - para.mu for x in para.Z_freqcal],bij=para.Z_modes,tau=para.t_tol/us,segment_num=15,lamb_dicke=para.lamb_dicke,mu = para.mu)

        syc.func2_optimize_process_save_data(plotfig=False, pulse_symmetry = False, ions_same_amps = True)

        #syc.func3_import_saved_data(plotfig = False, pulse_symmetry = False, ions_same_amps = True)

        syc.import_amp()

        para.amplitude = syc.get_amp()
        '''
      
        para.t_tol, para.amplitude, para.segNum = read_amp.get(para.N_ions,para.j_ions)

        para.Z_freqcal = [(x - para.mu)*MHz for x in para.Z_freqcal]
        

        para.init_2()      

        return para

    for N_ions in range(2,13):
        para = p(N_ions,['aH','a'])
        for j in para.j_ions:
            
            def int1_sqr(t,k):
                real_int = integrate.quad(lambda t0: para.amplitude[j](t0)*np.cos(para.Z_freqcal[k]*t0),0,t,limit = para.segNum*20)
                complex_int = integrate.quad(lambda t0: para.amplitude[j](t0)*np.sin(para.Z_freqcal[k]*t0),0,t,limit = para.segNum*20)
                return real_int[0]**2 + complex_int[0]**2
            result = 0
            for k in range(para.N_ions):
                temp_result = integrate.quad(lambda t: int1_sqr(t,k),0,para.t_tol,limit = para.segNum*20)
                result = max(result,temp_result[0])
            result = result* 4*2*para.heating_rate*para.lamb_dicke**2
            print('N =',N_ions,'my bound',result)
if __name__ == '__main__':
    
    #test_N_ions_syc()
    #test_optimize_syc()
    #test_N_ions_syc3()
    compute_my_bound()


