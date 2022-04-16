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
    if N_ions > 10:
        continue
    
    #read laser.ini
    config = configparser.ConfigParser()
    config.read(new_prefix+'laser.ini')
    delta_t = eval(config['amplitude']['delta_t'])
    segNum = eval(config['amplitude']['segNum'])
    tau = delta_t*segNum
    tau_lst[N_ions] = tau

    #read result.ini
    config = configparser.ConfigParser()
    config.read(new_prefix+'result.ini')
    try:
        bound_eq26 = eval(config['result']['bound_eq26'])
        my_bound_26[N_ions] = bound_eq26
    except:
        pass
    try:
        infid = eval(config['result']['infid'])
        heating_error[N_ions] = infid
    except:
        pass




plt.plot(list(range(2,11)),[2*n*1e2 for n in range(2,11)],label = 'N$\Gamma t$')


N = list(heating_error.keys())
N.sort()
print(heating_error)
plt.plot(N,[heating_error[n]/tau_lst[n] for n in N],label = 'numerical')



N = list(my_bound_26.keys())
N.sort()
print(my_bound_26)
plt.plot(N,[my_bound_26[n]/tau_lst[n] for n in N],label = 'mybound_eq26')



plt.xlabel('N')
plt.ylabel('infid/t($\mu s$)')
plt.title('infid-N (t normalized) (first & last ions illuminated)')
plt.legend(loc = 'upper left')
plt.savefig('j0-1.png')