import matplotlib.pyplot as plt
import numpy as np
import sympy as sy

from numpy import *
from scipy.integrate import *
from scipy.optimize import minimize




import qutip 
import scipy.integrate as integrate
from functools import reduce
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import fsolve
import time

import warnings
warnings.filterwarnings(action="error", category=np.ComplexWarning)

from my_constants import *

from scipy import constants
m = 171 * constants.m_p

class Ion_Trap_phonon:
    def __init__(self, N_ions, Omega_ax, Omega_ra) -> None:
        r"""
        Paraments
        ---------
        N_ions : int
            The number of ions in the trap
        Omega_ax : float
            The angular frequency in the axial direction
        Omega_ra : float
            The angular frequency in the radial direction
        """
        self.N_ions = N_ions
        self.Omega_ax = Omega_ax
        self.Omega_ra = Omega_ra

        self.pos = self.equ_pos()
        #print('equal position:', self.pos)
        # The 1st subscript marks different ions, and the 2rd marks modes
        self.w, self.b = self.phonon_mode(self.pos)
        print('phonon frequency:', self.w)
        self.eta = np.array([0.1]*N_ions)

    def potential(self, x):
        r"""
        Return the potential energy when ions is at x

        Parameters
        ----------
        x : (N_ions) array
            The position of ions

        Returns
        -------
        v : float
            The potential energy
        """
        v = 0
        for i in range(0, self.N_ions):
            for j in range(0, i):
                v = v + k_cou / ((x[i] - x[j]) ** 2) ** 0.5
        axial = m * self.Omega_ax ** 2 / 2
        for i in range(0, self.N_ions):
            v = v + axial * x[i] ** 2
        return v

    def equ_pos(self):
        x_name = []
        for i in range(self.N_ions):
            x_name.append('z' + np.str(i))
        x_symbols = sy.symbols(x_name)

        grad = [0] * self.N_ions
        initialguess = [0] * self.N_ions
        for i in range(0, self.N_ions):
            grad[i] = sy.diff(self.potential(x_symbols), x_symbols[i])
            initialguess[i] = 5e-6 * (i - self.N_ions / 2)
        result = sy.simplify(sy.nsolve(grad, x_symbols, initialguess))
        return result

    def phonon_mode(self, pos):
        x_name = []
        for i in range(self.N_ions):
            x_name.append('z' + np.str(i))
        x_symbols = sy.symbols(x_name)

        jacob = [[0]*self.N_ions for i in range(self.N_ions)]
        for i in range(self.N_ions):
            for j in range(self.N_ions):
                jacob[i][j] = sy.diff(self.potential(
                    x_symbols), x_symbols[i], x_symbols[j])
        jacob = sy.matrix2numpy(sy.Matrix(jacob).subs(
            zip(x_symbols, pos)), dtype=float)

        freq, modes = linalg.eigh(jacob)
        freq = (freq / m) ** 0.5
        modes = np.array(modes)
        return freq, modes

    def plot_modes(self):
        plt.rc('font', size=12)
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        ax.set_title('Sideband frequencies (MHz)')
        for i in range(self.N_ions):
            ax.plot(linspace(0, self.tau, 200)*1e6,
                    linspace(self.w[i], self.w[i], 200)/1e6, 'r')
        ax.set_xticks([])

        plt.show()



class trap_para():

    def __init__(self):
        print('init')

    def my_init(self,ion_number,laser_ion_list,omega_ax,omega_ra,cut_off,heating_rate,t_tol,lamb_dicke,_ini_state,_phonon_ini,amplitude,basis,lind_type,if_testXX):
        self.N_ions = ion_number
        print('initialize:',self.N_ions,'ions')
        self.j_ions = laser_ion_list
        self.omega_ax = omega_ax
        self.omega_ra = omega_ra
        self.cut_off = cut_off
        self.heating_rate = heating_rate
        self.lamb_dicke = lamb_dicke
        phonons = Ion_Trap_phonon(self.N_ions, self.omega_ax, self.omega_ra)
        self.partialpos = phonons.equ_pos
        self.Z_freqcal, self.Z_modes = phonons.w, phonons.b
        self.amplitude = amplitude

        #self.tau_XX = self.get_tau_XX()    ##############
        if if_testXX == True and t_tol == None:
            self.t_tol = self.tau_XX
        elif if_testXX == False:
            self.t_tol = t_tol
        else:
            raise ValueError('not a t_tol')

        # alpha:
        self.alpha = self.get_alpha(self.t_tol)
        self._ini_state = _ini_state
        self._phonon_ini = _phonon_ini
        
        self.lind_type = lind_type
        # time for XX gate:
        if basis == 'Gellman':
            self.basis = self.Gellman_Basis(self.dim_matForm)
        elif basis == None:
            pass
        else:
            raise ValueError('not a basis')
        

        self.spinTerms = [[ qutip.identity(2) for ind2 in range(self.N_lasers)]for ind1 in range(self.N_lasers)]
        for ind in range(self.N_lasers):
            self.spinTerms[ind][ind] = qutip.sigmax()
            self.spinTerms[ind] = reduce(lambda x, y: qutip.tensor(x,y),self.spinTerms[ind])
        if len(self.spinTerms) != len(self.j_ions):
            raise ValueError('spinTerms not match')
        if set(self.amplitude.keys()) != set(self.j_ions):
            raise ValueError('amplitude numbers not true')
        if self.t_tol == None:
            raise ValueError("not a t_tol")

    
    def init_2(self):
        self.spinTerms = [[ qutip.identity(2) for ind2 in range(self.N_lasers)]for ind1 in range(self.N_lasers)]
        for ind in range(self.N_lasers):
            self.spinTerms[ind][ind] = qutip.sigmax()
            self.spinTerms[ind] = reduce(lambda x, y: qutip.tensor(x,y),self.spinTerms[ind])
        if len(self.spinTerms) != len(self.j_ions):
            raise ValueError('spinTerms not match')
        if set(self.amplitude.keys()) != set(self.j_ions):
            raise ValueError('amplitude numbers not true')
        if self.t_tol == None:
            raise ValueError("not a t_tol")  
        self.alpha = self.get_alpha(self.t_tol)
        #print('alpha:',self.alpha)
        #print('Theta:',self.Theta(self.t_tol))
    def Theta(self,tau):
        amp0 = self.amplitude[self.j_ions[0]]
        amp1 = self.amplitude[self.j_ions[1]]
        eta = self.lamb_dicke
        i,j = self.j_ions
        omega = self.Z_freqcal
        b = self.Z_modes
        result = []
        result = [eta**2*b[i][k_mode]*b[j][k_mode]*integrate.nquad(lambda t1,t2: (amp0(t1)*amp1(t2)+amp0(t2)*amp1(t1))*np.sin((t1-t2)*int(t1>t2)*omega[k_mode]) ,[[0,tau],[0,tau]],opts = [{'limit':2000},{'limit':2000}])[0] for k_mode in range(self.N_ions)]
        return sum(result)

    def Theta_draw(self):
       # x = [0.,1.,2.]
        x = np.linspace(0, self.tau_XX, 100)
        y = [self.Theta(i) for i in x]
        plt.plot(x,y)
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.title('Histogram of IQ, greece $\\alpha$')
        plt.show()

    def get_tau_XX(self):
        #result = least_squares(lambda tau:(self.Theta(tau)**2-(np.pi/4)**2)**2,x0 = 0.,bounds = (0,np.inf))
        result = fsolve(lambda tau:abs(self.Theta(tau*us))-np.pi/4,x0 = 10)
        print('root of Theta = pi/4:',result[0],'us; error:Delta Theta =',abs(self.Theta(result[0]*us))-np.pi/4)
        return result[0]*us

    #(16) in yukai wu
    def get_alpha(self,tau):
        b = self.Z_modes
        N = self.N_ions
        eta = self.lamb_dicke
        omega = self.Z_freqcal
        j0,j1 = self.j_ions
        result0,result1 = [],[]
        amp0 = self.amplitude[self.j_ions[0]]
        amp1 = self.amplitude[self.j_ions[1]]
        for k_mod in range(N):
            real0 = integrate.quad(lambda t:(amp0(t)*np.exp(1j*omega[k_mod]*t)).real,0,tau,limit = 1000)[0] 
            img0 = integrate.quad(lambda t:(amp0(t)*np.exp(1j*omega[k_mod]*t)).imag,0,tau,limit = 1000)[0] 
        
            result0.append(-1.j*b[j0][k_mod]*eta*(real0+1j*img0) )
            real1 = integrate.quad(lambda t:(amp1(t)*np.exp(1j*omega[k_mod]*t)).real,0,tau,limit = 1000)[0] 
            img1 = integrate.quad(lambda t:(amp1(t)*np.exp(1j*omega[k_mod]*t)).imag,0,tau,limit = 1000)[0] 
            result1.append(-1.j*b[j1][k_mod]*eta*(real1+1j*img1))
        print('alpha0,1:',result0,result1)
        return result0, result1

    # normalized basis, Gellman basis
    # 
    def Gellman_Basis(self,dim:int):
        result = []
        # non-diagonalized terms
        for i in range(dim):
            for j in range(i+1,dim):
                # ( |i><j|+|j><i| ) / sqrt(2)
                ele1 = np.matrix(np.zeros((dim,dim),dtype=complex))
                ele1[i,j] = 1/np.sqrt(2)
                ele1[j,i] = 1/np.sqrt(2)
                result.append(ele1)
                # i (|i><j|-|j><i|) / sqrt(2)
                ele2 = np.matrix(np.zeros((dim,dim),dtype=complex))
                ele2[i,j] = 1.j/np.sqrt(2)
                ele2[j,i] = -1.j/np.sqrt(2)
                result.append(ele2)
        #diagonalized terms
        for i in range(1,dim):
            ele = np.matrix(np.zeros((dim,dim),dtype = complex))
            for k in range(0,i):
                ele[k,k] = 1.
            ele[i,i] = -i
            ele = ele/np.sqrt(i+i**2)
            result.append(ele)
        
        if 1:    # test number of generators
            if len(result) != dim**2 - 1:
                raise ValueError("basis generation error")
        return result


    @property
    def ini_state(self):
        return self._ini_state
    @property
    def phonon_ini(self):
        return self._phonon_ini

    @property
    def N_lasers(self):
        return len(self.j_ions)

    @property
    def dim_matForm(self):
        return 2*2*self.cut_off

    @property
    def vec_matForm(self):
        return self.dim_matForm**2-1


