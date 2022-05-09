'''
in all_optimized settings, we set basic parameters as following, and iterate N from 2 to 12
'''
trap_type = 'Paul linear trap'
N_ions_list = list(range(11,31))
omega_ax = '0.32*2*np.pi*MHz'
omega_ra = '2.18*2*np.pi*MHz'
heating_rate = 1e2
lamb_dicke = 0.1
shined = [0,1]
pulse_type = 'AM'
shape = 'rectangular'


import os

prefix = './configurations/'

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder",path," ---")
 
	else:
		raise ValueError('folder already exists')

import configparser
def create_config():
    for N_ions in N_ions_list:
        mkdir(prefix+'N'+str(N_ions))
        new_prefix = prefix+'N'+str(N_ions)+'/'
        # create file trap.ini
        config = configparser.ConfigParser()
        config['trap'] = {
            'N_ions': str(N_ions),
            'omega_ax': str(omega_ax),
            'omega_ra': str(omega_ra),
            'heating_rate': str(heating_rate),
            'lamb_dicke': str(lamb_dicke)
        }
        with open(new_prefix+'trap.ini','w') as configfile:
            config.write(configfile)

        # create file laser.ini
        config = configparser.ConfigParser()
        config['shined'] = {
            'shined': str(shined)
        }
        config['amplitude'] = {
            'pulse_type' : pulse_type,
            'shape' : shape
        }
        with open(new_prefix+'laser.ini','w') as configfile:
            config.write(configfile)

        # create file phonon.ini
        config = configparser.ConfigParser()
        with open(new_prefix+'phonon.ini','w') as configfile:
            config.write(configfile)

        # create file result.ini
        config = configparser.ConfigParser()
        with open(new_prefix+'result.ini','w') as configfile:
            config.write(configfile)

if __name__ == '__main__':
    create_config()

