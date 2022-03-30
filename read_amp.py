import pandas as pd
import numpy as np
from my_constants import *
import sympy as sp
import matplotlib.pyplot as plt

df = pd.read_csv('./results/j01/amp_opt_28.csv',sep = ';',skipinitialspace=True)
tau_l = df['tau']
segNum_l = df['segNum']
amp_l = [eval(i) for i in df['amp']]
detuning_l = df['detuning']

def get(N,j_list):
    # 注意,这里指标是从N=2开始的,所以要-2
    tau = tau_l[N-2]
    segNum = segNum_l[N-2]
    amp = amp_l[N-2]
    
    delta_t = tau / segNum
    def result_func(t,i):
        ion_i_amp_list = np.real(amp[i])
        ind = min(int(t/delta_t),segNum-1)
        return ion_i_amp_list[ind]
    j0 = j_list[0]
    j1 = j_list[1]
    res = {j0:lambda t: result_func(t/us,0)*MHz/2  , j1:lambda t:result_func(t/us,1)*MHz/2}########
    '''
    x = np.linspace(0, 200*us, 1000)
    y = [(res[j0](i)) for i in x]
    plt.plot(x,y)
    plt.xlabel('t ')
    plt.ylabel('$chi$')
    plt.title('amp')
    plt.show()
    '''
    return tau*us, res, segNum


