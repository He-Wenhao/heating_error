'''
here we set partial position manually and compute frequency and mode by zxm_freq package
'''


import os
import configparser
import numpy as np
import sys
sys.path.append('../..')
from my_constants import *
from zxm_freq.Tab1_ion_pos_spec_cal import  radial_mode_spectrum

prefix = './configurations'
all_dir = os.listdir(prefix)

def compute_phonon():
    #iterate through all configurations
    for folder in all_dir:
        #read parameters
        new_prefix = prefix+'/'+folder+'/'
        config = configparser.ConfigParser()
        config.read(new_prefix+'trap.ini')
        N_ions = eval(config['trap']['N_ions'])
        omega_ax = eval(config['trap']['omega_ax'])
        omega_ra = eval(config['trap']['omega_ra'])
        partialpos = np.array(range(N_ions))*5*um

        #compute phonon freq and modes
        Z_freqcal, Z_modes = radial_mode_spectrum( N_ions, omega_ax,omega_ra,partialpos)
        Z_freqcal = [i*MHz for i in Z_freqcal]

        #convert np.array to list
        Z_modes = [list(i) for i in Z_modes]

        #write phonon freq and modes
        config = configparser.ConfigParser()
        config['phonon'] = {
            'partialpos': str(partialpos),
            'Z_freqcal':str(list(Z_freqcal)),
            'Z_modes':str(list(Z_modes))
        }
        with open(new_prefix+'phonon.ini','w') as configfile:
            config.write(configfile)
        print('---','N =',N_ions,'finished','---')

if __name__ == '__main__':
    compute_phonon()