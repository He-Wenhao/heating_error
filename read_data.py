import pandas as pd
import numpy as np
from my_constants import *
import sympy as sp
import matplotlib.pyplot as plt



class read_data():
    def __init__(self,path):
        df = pd.read_csv(path,sep = ';',skipinitialspace=True)
        self.tau_l = df['tau']
        self.segNum_l = df['segNum']
        self.amp_l = [eval(i) for i in df['amp']]
        self.detuning_l = [eval(i) for i in df['detuning']]

    def get(self,N,j_list):
        # 注意,这里指标是从N=2开始的,所以要-2
        tau = self.tau_l[N-2]
        segNum = self.segNum_l[N-2]
        amp = self.amp_l[N-2]
        #convert Omega to chi
        amp = np.array(amp)*0.5
        detuning = self.detuning_l[N-2]
        
        delta_t = tau / segNum
        def result_func(t,i):
            ion_i_amp_list = np.real(amp[i])
            ind = min(int(t/delta_t),segNum-1)
            return ion_i_amp_list[ind]
        j0 = j_list[0]
        j1 = j_list[1]
        res = {j0:lambda t: result_func(t/us,0)*MHz  , j1:lambda t:result_func(t/us,1)*MHz}########
        '''
        x = np.linspace(0, 200*us, 1000)
        y = [(res[j0](i)) for i in x]
        plt.plot(x,y)
        plt.xlabel('t ')
        plt.ylabel('$chi$')
        plt.title('amp')
        plt.show()
        '''
        return tau, res, segNum,amp,detuning


