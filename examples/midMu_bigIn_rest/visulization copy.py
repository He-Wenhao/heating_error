import matplotlib.pyplot as plt
import os
import re
import configparser
prefix = './configurations'
all_dir = os.listdir(prefix)
all_dir.sort(key=lambda x:int(re.sub(u"([^\u0030-\u0039])", "", x)))
heating_error = dict()
my_bound_26 = dict()
tau_lst = dict()
#iterate through all configurations
for folder in all_dir:
    new_prefix = prefix+'/'+folder+'/'
    #read trap.ini
    config = configparser.ConfigParser()
    config.read(new_prefix+'trap.ini')
    N_ions = eval(config['trap']['N_ions'])
    #read trap.ini
    config = configparser.ConfigParser()
    config.read(new_prefix+'phonon.ini')
    z_freqcal = eval(config['phonon']['z_freqcal'])
    
    plt.plot(list(range(N_ions)),z_freqcal)

    plt.xlabel('N')
    plt.ylabel('infid/t($\mu s$)')
    plt.title('infid-N (t normalized) (0 1 ions illuminated)')
    plt.legend(loc = 'upper left')
    plt.show()



