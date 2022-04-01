import numpy as np
import qutip
import sim_method_2L
from my_constants import *
import trap_config
import time
from syc_amp.wave_pulse import syc_amp
from zxm_freq.Tab1_ion_pos_spec_cal import  radial_mode_spectrum
import read_data
import scipy.integrate as integrate
import sympy as sp
from functools import lru_cache

pathj01_ex = './results/j01_ex/amp_opt.csv'


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



def compute_infid_syc():

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
        
        my_amp = read_data.read_data(pathj01)
        para.t_tol, para.amplitude,_ , __= my_amp.get(para.N_ions,para.j_ions)
        para.t_tol *= us
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


def optimize_amp_syc():
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

        # cast into ten or less phonons
        

        syc = syc_amp(ion_number=para.N_ions,j_list=para.j_ions,omega=[x - para.mu for x in para.Z_freqcal],bij=np.matrix(para.Z_modes).transpose(),tau=para.t_tol/us,segment_num=seg_num,lamb_dicke=para.lamb_dicke,mu = para.mu)

        syc.func2_optimize_process_save_data(plotfig=False, pulse_symmetry = False, ions_same_amps = True)

        syc.import_amp()
        syc.print_amp()

        #syc.func3_import_saved_data(plotfig = True, pulse_symmetry = False, ions_same_amps = True)
        para.amplitude = syc.get_amp()
        para.Z_freqcal = [(x - para.mu)*MHz for x in para.Z_freqcal]
        para.init_2() 
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
    for N in range(11,12):
        segNum = evaluate[N]
        while 1:
            print('-----','N',N,'segNum',segNum,'-----')
            error = p(N,segNum)
            if error < 1e-5:
                print('----- break -----')
                break
            else:
                segNum+=1
     
def compute_my_bound_26():

    #construct para
    def my_p(N_ions,lind_type):
    
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
      
        my_amp = read_data.read_data(pathj01)
        para.t_tol, para.amplitude, para.segNum, para.amp_list, para.detuning  = my_amp.get(para.N_ions,para.j_ions)

        

        #para.init_2()      

        return para

    for N_ions in range(12,13):
        para = my_p(N_ions,['aH','a'])
        ###### for test
        #para.segNum = 2
        #para.amp_list = [[1,-1],[1,-1]]
        #para.detuning = [0,0]
        #para.t_tol = 2
        print('N_ions',para.N_ions)
        print('segNum',para.segNum)
        print('amp_list',para.amp_list)
        print('detuning',para.detuning)
        print('t_tol',para.t_tol)
        #######
        j0, j1 = para.j_ions
        for j in para.j_ions[0:1]:
            delta_t = para.t_tol/para.segNum
            t = sp.Symbol('t',real = True)
            result0 = 0
            result1 = 0
            result01 = 0
            for k in range(para.N_ions):
                t1 = time.time()
                print('k = ',k)
                t0 = sp.Symbol('t0',real = True)
                f_l = []
                interval_l = [(i*delta_t,(i+1)*delta_t) for i in range(para.segNum)]
                for p in range(para.segNum):
                    alpha_int = seg_integrate(np.array(para.amp_list[0][0:p+1])*sp.exp(-sp.I*sp.S(para.detuning[k])*t0),t0,[(i*delta_t,(i+1)*delta_t) for i in range(p)]+[(p*delta_t,t)],eval_flag=False)
                    alpha_int = sp.expand(alpha_int)  #此处还有优化空间,因为同样的式子被simplify了好多次
                    alpha_int = alpha_int * sp.conjugate(alpha_int)
                    alpha_int = sp.expand(alpha_int)
                    f_l.append(alpha_int)
                t2 = time.time()
                print('median',t2-t1)
                temp_result = seg_integrate(f_l, t, interval_l,eval_flag = True)
                #result = max(result,sp.re(temp_result.evalf()))
                result0 += temp_result*para.lamb_dicke**2*para.Z_modes[j0][k]**2
                result1 += temp_result*para.lamb_dicke**2*para.Z_modes[j1][k]**2
                result01 += temp_result*para.lamb_dicke**2*para.Z_modes[j0][k]*para.Z_modes[j0][k]
                t3 = time.time()
                print('end',t3-t2)
            result = 2*para.heating_rate*(result0+result1+2*result01)*1e-6
            print('---------------N =',N_ions,'my bound',result)

def testify():

    #construct para
    def p(N_ions,lind_type):
    
        para = trap_config.trap_para()

        # basic parameters
        para.N_ions = N_ions
        para.j_ions = [0,1]
        para.omega_ax = 0.32*2*np.pi*MHz
        para.omega_ra = 2.18*2*np.pi*MHz
        para.cut_off = 12
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
        
        my_amp = read_data.read_data(pathj01_ex)
        para.t_tol, para.amplitude,_ , _, _= my_amp.get(2,para.j_ions)
        #para.t_tol = 300.0
        #para.amplitude = [(5.759732540148568e-08+7.701533567170219e-08j), (5.615525097310147e-08+7.666643842925619e-08j), (-9.264100606632453e-08+1.5910761887927112e-07j), (-3.35866153769091e-08-3.797213502095387e-09j), (0.15694613808351096-0.12942797948534585j)] [(-1.6793377933506672e-07-2.2454994734484871e-07j), (-9.647008574826375e-08-1.3170661266970459e-07j), (3.887603204473474e-08-6.676819642571257e-08j), (-1.9550773705633517e-08-2.2103585329554287e-09j), (0.15694613808351113-0.12942797948534598j)]
        para.t_tol *= us
        para.Z_freqcal = [(x - para.mu)*MHz for x in para.Z_freqcal]
        

        para.init_2()      

        return para

    
    for N_ions in range(11,12):
        
        para = p(N_ions,[])
        t1 = time.time()
        alg1_lind = sim_method_2L.alg1(para).sim()
        print('alg1:',alg1_lind)
        t2 = time.time()
        print('no lind','N_ions =',N_ions, 'time alg1 =',t2-t1)


if __name__ == '__main__':
    
    #compute_infid_syc()
    #optimize_amp_syc()
    #compute_my_bound_26()
    testify()


