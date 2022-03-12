import numpy as np
from scipy.integrate import quad
from .constants import *
from .ion_trap import *
from .gate import *

def mu(arg_mu, tau, t):
    N_pulse = (len(arg_mu)-1)
    tau = tau / N_pulse / 2
    delta_0 = 0
    delta_1 = 0
    if int(t / tau) < N_pulse:
        i = int(t/tau)
        delta_0 = 0.5*(arg_mu[i]+arg_mu[i+1])
        delta_1 = 0.5*(arg_mu[i]-arg_mu[i+1])
    else:
        i = 2 * N_pulse - int(t/tau)-1
        delta_0 = 0.5*(arg_mu[i]+arg_mu[i+1])
        delta_1 = 0.5*(arg_mu[i+1]-arg_mu[i])
    return (delta_0+delta_1*np.cos(np.pi*(t/tau-int(t/tau)))) * 1e6


def wave(arg_amp, arg_mu, tau, t):
    amp = arg_amp[0]
    phase = quad(lambda x: mu(arg_mu, tau, x), 0, t)[0]
    return amp * np.sin(phase) * MHz


class hzx_amp():
    def __init__(self,N_ions,Omega_ax,Omega_ra,target_phase,j_list,N_seg,tau):
        self.N_ions = N_ions
        self.Omega_ax = Omega_ax
        self.Omega_ra = Omega_ra
        self.target_phase = target_phase
        self.j_list = j_list
        self.N_seg = N_seg
        self.trap = Ion_Trap(self.N_ions, self.Omega_ax, self.Omega_ra)
        self.tau = tau
        self.gate = XX_Gate(self.trap, self.target_phase, self.j_list, self.N_seg, self.tau)
        #print('tau:', tau)

    def optimize(self):
        gate = self.gate
        gate.optimize()
        gate.regularization()
        #gate.plot_mu()
        #gate.plot_amp()

        alphas = gate.get_alphas()
        sum_alphas = 0
        for a_list in alphas:
            for a in a_list:
                sum_alphas += (a*a.conjugate()).real
        print('alphas:', alphas, sum_alphas)

    def get_amp(self):
        return lambda t: wave(self.gate.arg_amp, self.gate.arg_mu, self.tau, t)

if __name__ == '__main__':
    tau = 100*us
    a = hzx_amp(N_ions = 5,Omega_ax = 0.32*2*np.pi*MHz,Omega_ra = 2.18*2*np.pi*MHz,target_phase = np.pi/4,j_list = [2, 3],N_seg = 7,tau = tau)
    a.optimize()
    amp = a.get_amp()
    import matplotlib.pyplot as plt
    #draw f(x) from x = 0 to x = 10 , cut into 1000 steps
    x = np.linspace(0, tau, 1000)
    y = [amp(i) for i in x]
    
    plt.plot(x,y,label = 'amp')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Histogram of IQ, greece $\\alpha$')
    plt.legend(loc = 'upper left')
    plt.show()