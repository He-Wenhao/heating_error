import os
import configparser
import re

import sys
sys.path.append('../..')
from my_constants import *
from syc_amp.wave_pulse import syc_amp
from solve_eq_hwh import hwh_optimize

prefix = './configurations'
all_dir = os.listdir(prefix)
all_dir.sort(key=lambda x:int(re.sub(u"([^\u0030-\u0039])", "", x)))

segNumdict = {}

#evaluate = {2:2,3:18,4:20,5:20,6:24,7:32,8:36,9:36,10:38,11:48,12:52}
evaluate = {2:2}
def optimize_pulse():
    #iterate through all configurations
    for folder in all_dir:
        new_prefix = prefix+'/'+folder+'/'
        #read trap.ini
        config = configparser.ConfigParser()
        config.read(new_prefix+'trap.ini')
        N_ions = eval(config['trap']['N_ions'])
        lamb_dicke = eval(config['trap']['lamb_dicke'])

        #read phonon.ini
        config = configparser.ConfigParser()
        config.read(new_prefix+'phonon.ini')
        Z_freqcal  = eval(config['phonon']['Z_freqcal'])
        print(config['phonon']['Z_modes'])
        Z_modes = eval(config['phonon']['Z_modes'])

        # set some laser parameters
        delta_t = 15*us
        sortZ_freqcal = Z_freqcal.copy()
        sortZ_freqcal.sort()
        a = N_ions//2
        b = a-1
        mu = (sortZ_freqcal[a]+sortZ_freqcal[b])/2
        config = configparser.ConfigParser()
        config.read(new_prefix+'laser.ini')
        config['amplitude']['mu'] = str(mu)
        config['amplitude']['delta_t'] = str(delta_t)
        with open(new_prefix+'laser.ini','w') as configfile:
            config.write(configfile)
        j_ions = eval(config['shined']['shined'])
        #do optimization
        segNum = evaluate[N_ions]
        while 1:
            print('-----','N',N_ions,'segNum',segNum,'-----')
            #error = p(N,segNum)
            #in syc_amp MHz = us = 1
            #syc = syc_amp(ion_number=N_ions,j_list=j_ions,omega=[(x - mu)/MHz for x in Z_freqcal],bij=np.matrix(Z_modes).transpose(),tau=delta_t*segNum/us,segment_num=segNum,lamb_dicke=lamb_dicke,mu = mu/MHz)
            try:
                #syc.func2_optimize_process_save_data(plotfig=False, pulse_symmetry = False, ions_same_amps = True,if_restrict=True,restrictNum=10)
                pass
            except ValueError:
                #segNum += 2
                continue

            #syc.import_amp()
            #ampList0,ampList1 = syc.print_amp()
            hwh = hwh_optimize(unit = 'SI',detuning_list = [(x - mu) for x in Z_freqcal], duration = delta_t*segNum, number_of_segments = segNum,bij=np.matrix(Z_modes).transpose(),lamb_dicke = lamb_dicke)
            ampList,bnd = hwh.optimize_solve()

            if bnd < 0.5:
                print('----- break -----')
                config['amplitude']['segNum'] = str(segNum)
                config['amplitude']['amp_list0'] = str(ampList0)
                config['amplitude']['amp_list1'] = str(ampList1)
                with open(new_prefix+'laser.ini','w') as configfile:
                    config.write(configfile)
                
                evaluate[N_ions+1] = segNum
                
                break
            else:
                segNum+=2


if __name__ == '__main__':
    optimize_pulse()