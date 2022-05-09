# convert Hermitian matrix to vector, on a basis
#所有函数输出均为np.array
import numpy as np
from numpy import mat
from typing import List, Tuple
import qutip 
import scipy.integrate as integrate
from scipy.linalg import expm
import sympy as sp
from functools import reduce, lru_cache
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import fsolve
import time

import warnings
warnings.filterwarnings(action="error", category=np.ComplexWarning)

from my_constants import *



class ideal():
       
    def __init__(self,para):
        self.para = para
        self.amp0 = self.para.amplitude[self.para.j_ions[0]]
        self.amp1 = self.para.amplitude[self.para.j_ions[1]]
        if self.para.N_lasers != 2:
            raise ValueError('not 2 lasers')

    def Theta(self,tau):
        return self.para.Theta(tau)
         



    def sim(self):
        t_tol = self.para.t_tol
        lnU = np.matrix(qutip.tensor(qutip.sigmax(),qutip.sigmax()))*self.para.Theta(t_tol) *1j
        U = np.matrix(expm(lnU))
        init = np.matrix(self.para.ini_state)
        return U*init*U.H

    def sim_XX(self):
        t_tol = self.para.tau_XX
        lnU = np.matrix(qutip.tensor(qutip.sigmax(),qutip.sigmax()))*self.para.Theta(t_tol) *1j
        U = np.matrix(expm(lnU))
        init = np.matrix(self.para.ini_state)
        return U*init*U.H



class alg2_noLind(ideal):
    def __init__(self,para):
        super(alg2_noLind,self).__init__(para)



    def sim(self):
        rho_ideal = super(alg2_noLind,self).sim()
        t_tol = self.para.t_tol
        a_k = qutip.destroy(self.para.cut_off)
        a_k_dagger = qutip.create(self.para.cut_off)
        rho = qutip.Qobj(rho_ideal,dims = [[2,2],[2,2]])
        alpha0, alpha1 = self.para.alpha
        for k in range(self.para.N_ions):
            rho = qutip.tensor(rho,self.para.phonon_ini)
            lnU = qutip.tensor(qutip.sigmax(),qutip.qeye(2),alpha0[k]*a_k_dagger - alpha0[k].conjugate()*a_k) + qutip.tensor( qutip.qeye(2),qutip.sigmax(),alpha1[k]*a_k_dagger - alpha1[k].conjugate()*a_k)
            U = np.matrix(expm(np.matrix(lnU)))
            rho = U*np.matrix(rho)*U.H
            rho = qutip.Qobj(rho,dims = [[2,2,self.para.cut_off],[2,2,self.para.cut_off]])
            rho = rho.ptrace([0,1])
        return np.matrix(rho)

# notice: must keep matrix form and vecForm up to date
# my density matrix
class my_matrix():
    def __init__(self,matrix,type_operator,type_Form):
        self.type_operator = type_operator
        if type_Form == 'matForm':
            self.matForm = matrix
            if self.matForm.shape[0] != self.matForm.shape[1]:
                raise TypeError('not a square matrix')
            self.dim_matForm = self.matForm.shape[1]
            self.dim_vecForm = self.dim_matForm**2 - 1  # dimension of vector form (-1 for throwing away identity matrix)
            self.vecForm = None
        elif type_Form == 'vecForm':
            self.vecForm = matrix
            if type_operator == 'density_matrix':
                s = matrix.shape
                if s[1] != 1:
                    raise TypeError('not a square matrix')
                self.dim_vecForm = self.vecForm.shape[0]
            else:
                if self.vecForm.shape[0] != self.vecForm.shape[1]:
                    raise TypeError('not a square matrix')
                self.dim_vecForm = self.vecForm.shape[0]
            self.dim_matForm = sp.sqrt(1+self.dim_vecForm)  # dimension of vector form (-1 for throwing away identity matrix)
            self.matForm = None
        else:
            raise TypeError('not a operator_type')
        
    def my_trace(self,c):
        return c.trace()

    # convert function into vecForm
    def map2matrix(self,basis,mymap) -> np.array:
        #dimension of result
        result = np.matrix(np.zeros((self.dim_vecForm,self.dim_vecForm)))
        #generate a map matrix
        for j in range(self.dim_vecForm):
            res_of_map = mymap(basis[j])
            for i in range(self.dim_vecForm):
                result[i,j] = (res_of_map*mat(basis[i])).trace()[0,0].real
        return result

    # convert matrix form into vector form
    def matForm_to_vecForm(self,basis):
        # convert a density matrix into a vector
        if self.type_operator == 'density_matrix':
            result = np.matrix(np.zeros((self.dim_vecForm,1)))
            for i in range(self.dim_vecForm):
                val = self.my_trace(self.matForm*basis[i])
                if abs(val.imag) > 1e-10:
                    raise ValueError('not real')
                result[i] = val.real
            if result.shape != (self.dim_vecForm,1):
                raise TypeError('dimension error')
        elif self.type_operator == 'Hamiltonian':
            H = self.matForm
            def mymap(ele):
                return -1.j* (H*ele-ele*H)
            result = self.map2matrix(basis,mymap)
        elif self.type_operator == 'Lindbladian':
            L = self.matForm
            def mymap(ele):
                return L*ele*L.getH()-1/2*L.getH()*L*ele-1/2*ele*L.getH()*L
            result = self.map2matrix(basis,mymap)
        else:
            raise TypeError('not a type')
        self.vecForm = result
        return result
    # convert vecForm back to matrix form
    def vecForm_to_matForm(self,basis):
        if self.type_operator == 'density_matrix':
            result = sum([basis[i]*self.vecForm[i,0] for i in range(self.dim_vecForm)])+np.matrix(np.eye(self.dim_matForm))/self.dim_matForm
        elif self.type_operator == 'Hamiltonian':
            raise TypeError('not support')
        elif self.type_operator == 'Lindbladian':
            raise TypeError('not support')
        else:
            raise TypeError('not a type')
        self.matForm = result
        return result



   
class alg2(ideal):
    def __init__(self,para):
        super(alg2,self).__init__(para)




    def lindbladian_a_vecForm(self,k_mod):
        ak = qutip.destroy(N=self.para.cut_off)
        ak = np.matrix(qutip.tensor(qutip.tensor(qutip.qeye(2),qutip.qeye(2)),ak))
        ak = np.sqrt(self.para.heating_rate)*ak
        transform = my_matrix(ak, 'Lindbladian', 'matForm')
        return transform.matForm_to_vecForm(self.para.basis)
    def lindbladian_aH_vecForm(self,k_mod):
        ak = qutip.create(N=self.para.cut_off)
        ak = np.matrix(qutip.tensor(qutip.tensor(qutip.qeye(2),qutip.qeye(2)),ak))
        ak = np.sqrt(self.para.heating_rate)*ak
        transform = my_matrix(ak, 'Lindbladian', 'matForm')
        return transform.matForm_to_vecForm(self.para.basis)
    def lindbladian_n_vecForm(self,k_mod):
        raise ValueError('not completed program')
        pass
    def integral_Hamiltonian_vecForm(self,k_mod):
        a_k = qutip.destroy(N=self.para.cut_off)
        a_k_dagger = qutip.create(N=self.para.cut_off)
        real_func = lambda t:(self.amp0(t) * np.exp(-1.j * self.para.Z_freqcal[k_mod]*t)).real
        imag_func = lambda t:(self.amp1(t) * np.exp(-1.j * self.para.Z_freqcal[k_mod]*t)).imag
        real_int = integrate.quad(real_func, 0., self.para.t_tol)[0]
        imag_int = integrate.quad(imag_func, 0., self.para.t_tol)[0]
        integral_1 = real_int + imag_int*1.j
        integral_2 = real_int - imag_int*1.j
        out = sum([ self.para.lamb_dicke * self.para.Z_modes[j][k_mod] * qutip.tensor(self.para.spinTerms[j],(a_k *integral_1 + a_k_dagger *integral_2)) for j in self.para.j_ions])
        transform = my_matrix(out, 'Hamiltonian', 'matForm')
        #print(self.amp0(0),self.para.Z_freqcal[k_mod],self.para.t_tol)
        return transform.matForm_to_vecForm(self.para.basis)
    def integral_2Dt2_Hamiltonian_vecForm(self,k_mod):
        a_k = qutip.destroy(N=self.para.cut_off)
        a_k_dagger = qutip.create(N=self.para.cut_off)
        real_func = lambda t:(self.amp0(t) * np.exp(-1.j * self.para.Z_freqcal[k_mod]*t)).real
        imag_func = lambda t:(self.amp1(t) * np.exp(-1.j * self.para.Z_freqcal[k_mod]*t)).imag
        real_int = integrate.dblquad(lambda t2,t1: real_func(t2), 0., self.para.t_tol,lambda x:0.,lambda x: x)[0]
        imag_int = integrate.dblquad(lambda t2,t1: imag_func(t2), 0., self.para.t_tol,lambda x:0.,lambda x: x)[0]

        integral_1 = real_int + imag_int*1.j
        integral_2 = real_int - imag_int*1.j
        out = sum([ self.para.lamb_dicke * self.para.Z_modes[j][k_mod] * qutip.tensor(self.para.spinTerms[j],(a_k *integral_1 + a_k_dagger *integral_2)) for j in self.para.j_ions])
        transform = my_matrix(out, 'Hamiltonian', 'matForm')
        return transform.matForm_to_vecForm(self.para.basis)
    def integral_2Dt1_Hamiltonian_vecForm(self,k_mod):
        a_k = qutip.destroy(N=self.para.cut_off)
        a_k_dagger = qutip.create(N=self.para.cut_off)
        real_func = lambda t:t*(self.amp0(t) * np.exp(-1.j * self.para.Z_freqcal[k_mod]*t)).real
        imag_func = lambda t:t*(self.amp1(t) * np.exp(-1.j * self.para.Z_freqcal[k_mod]*t)).imag
        
        real_int = integrate.quad(real_func, 0., self.para.t_tol)[0]
        imag_int = integrate.quad(imag_func, 0., self.para.t_tol)[0]

        integral_1 = real_int + imag_int*1.j
        integral_2 = real_int - imag_int*1.j
        out = sum([ self.para.lamb_dicke * self.para.Z_modes[j][k_mod] * qutip.tensor(self.para.spinTerms[j],(a_k *integral_1 + a_k_dagger *integral_2)) for j in self.para.j_ions])
        transform = my_matrix(out, 'Hamiltonian', 'matForm')
        return transform.matForm_to_vecForm(self.para.basis)

    def sim(self):
        rho = qutip.Qobj(super(alg2,self).sim(),dims = [[2,2],[2,2]])
        for k_mod in range(self.para.N_ions):
            #print('rho =',rho,k_mod)
            #print(k_mod)
            #print(rho)
            #construct unitary operator 
            L = []
            if 'a' in self.para.lind_type:
                L += [self.lindbladian_a_vecForm(k_mod)]
            if 'aH' in self.para.lind_type:
                L += [self.lindbladian_aH_vecForm(k_mod)]
            if 'n' in self.para.lind_type:
                L += [self.lindbladian_n_vecForm(k_mod)]
            
            Ht = self.integral_Hamiltonian_vecForm(k_mod)
            if len(L) != 0:
                L = sum(L)
                Ht1 = self.integral_2Dt1_Hamiltonian_vecForm(k_mod)
                Ht2 = self.integral_2Dt2_Hamiltonian_vecForm(k_mod)
                #print('norm of Ht1',np.linalg.norm(Ht1))##########
                #print('norm of Ht2',np.linalg.norm(Ht2))##########
                ln_U = Ht + L*self.para.t_tol + 0.5*(L*(Ht2-Ht1)-(Ht2-Ht1)*L)
                #ln_U = Ht + L*self.para.t_tol
            else:
                ln_U = Ht
            U = np.matrix(expm(ln_U))
            #do evolution
            rho = qutip.tensor(rho,self.para.phonon_ini)
            rho = np.matrix(rho)
            transform = my_matrix(rho, 'density_matrix', 'matForm')
            rho = transform.matForm_to_vecForm(self.para.basis)
            rho = U*rho

            transform = my_matrix(rho, 'density_matrix', 'vecForm')

            rho = transform.vecForm_to_matForm(self.para.basis)
            rho = qutip.Qobj(rho,dims = [[2,2,self.para.cut_off],[2,2,self.para.cut_off]])
            rho = rho.ptrace([0,1])
        #print(rho)
        return np.matrix(rho)




class brute():
    def __init__(self,para):
        print('maybe problematic, details in optimized pulse of N=2')
        self.para = para
        self.amp0 =lambda t: self.para.amplitude[self.para.j_ions[0]](t)* np.sin(self.para.mu * t)
        self.amp1 = lambda t: self.para.amplitude[self.para.j_ions[1]](t)* np.sin(self.para.mu * t)
        if self.para.N_lasers != 2:
            raise ValueError('not 2 lasers')

    @lru_cache(128)
    def _a(self,k):
        N_ions = self.para.N_ions
        cut_off = self.para.cut_off
        qeye, destroy,create,tensor = qutip.qeye, qutip.destroy,qutip.create,qutip.tensor
        a_k = [qeye(cut_off) for i in range(k)]+[destroy(N=cut_off)]+[qeye(cut_off) for i in range(k+1,N_ions)]
        a_k = reduce(lambda x, y: tensor(x,y), a_k)
        a_k = tensor(tensor(qeye(2), qeye(2)),a_k)
        return a_k

    @lru_cache(128)
    def _a_dagger(self,k):
        N_ions = self.para.N_ions
        cut_off = self.para.cut_off
        qeye, destroy,create,tensor = qutip.qeye, qutip.destroy,qutip.create,qutip.tensor
        a_k_dagger = [qeye(cut_off) for i in range(k)]+[create(N=cut_off)]+[qeye(cut_off) for i in range(k+1,N_ions)]
        a_k_dagger = reduce(lambda x, y: tensor(x,y), a_k_dagger)
        a_k_dagger = tensor(tensor(qeye(2), qeye(2)),a_k_dagger)
        return a_k_dagger

    def lindbladian(self):
        N_ions = self.para.N_ions
        a_list = []
        aH_list = []
        n_list = []
        for k in range(N_ions):
            a_k = self._a(k)
            a_k_dagger = self._a_dagger(k)
            n_k = a_k_dagger*a_k
            a_list.append(np.sqrt(self.para.heating_rate)*a_k)
            aH_list.append(np.sqrt(self.para.heating_rate)*a_k_dagger)
            n_list.append(np.sqrt(self.para.heating_rate)*n_k)
        return a_list, aH_list, n_list

    # construct Hamiltonian
    def Ham_eff_one_ion(self, j_ion, t):
        N_ions = self.para.N_ions
        cut_off = self.para.cut_off
        qeye, destroy,create,tensor = qutip.qeye, qutip.destroy,qutip.create,qutip.tensor
        partialpos = self.para.partialpos
        Z_freqcal, Z_modes = self.para.Z_freqcal, self.para.Z_modes
        result = [0 for i in range(N_ions)]
        lamb_dicke = self.para.lamb_dicke
        for k in range(N_ions):
            a_k = self._a(k)
            a_k_dagger = self._a_dagger(k)
            result[k] = lamb_dicke * Z_modes[j_ion][k] * (a_k *np.exp(-1.j * Z_freqcal[k]*t) + a_k_dagger *np.exp(1.j * Z_freqcal[k]*t)) 
            #print('t =',t,result[k].ptrace([0,1]))
        
        
        result = sum(result)
        return result

    def Ham_eff(self,t):
        j1, j2 = self.para.j_ions
        qeye, destroy,create,tensor = qutip.qeye, qutip.destroy,qutip.create,qutip.tensor
        N_ions = self.para.N_ions
        sigmax = qutip.sigmax()
        h0 = self.amp0(t)*self.Ham_eff_one_ion(j1, t)*tensor([sigmax,qeye(2)]+[qeye(self.para.cut_off)]*N_ions)
        h1 = self.amp1(t)*self.Ham_eff_one_ion(j2, t)*tensor([qeye(2),sigmax]+[qeye(self.para.cut_off)]*N_ions)
        return h0 + h1

    def sim(self):
        N_ions = self.para.N_ions
        cut_off = self.para.cut_off
        rho0 = qutip.tensor(self.para.ini_state,reduce(lambda x, y: qutip.tensor(x,y), [self.para.phonon_ini for i in range(N_ions)]))
        a_list, aH_list, n_list = self.lindbladian()
        c_list = []
        if 'a' in self.para.lind_type:
            c_list += a_list
        if 'aH' in self.para.lind_type:
            c_list += aH_list
        if 'n' in self.para.lind_type:
            c_list += n_list
        tlist = np.linspace(0, self.para.t_tol, 100000)
        output = qutip.mesolve(lambda x,args: self.Ham_eff(x), rho0, tlist, c_list,[])
        rho=(output.states[-1]).ptrace([0,1])
        return np.matrix(rho)


class alg1():
    def __init__(self,para):
        self.para = para
        self.amp0 = self.para.amplitude[self.para.j_ions[0]]
        self.amp1 = self.para.amplitude[self.para.j_ions[1]]
        if self.para.N_lasers != 2:
            raise ValueError('not 2 lasers')

    # construct Hamiltonian
    def Ham_sub_one_ion(self, j_ion, k, t):
        N_ions = self.para.N_ions
        cut_off = self.para.cut_off
        partialpos = self.para.partialpos
        Z_freqcal, Z_modes = self.para.Z_freqcal, self.para.Z_modes
        lamb_dicke = self.para.lamb_dicke
        
        a_k = qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.destroy(cut_off))
        a_k_dagger = qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.create(cut_off))
        result= lamb_dicke * Z_modes[j_ion][k] * (a_k *np.exp(-1.j * Z_freqcal[k]*t) + a_k_dagger *np.exp(1.j * Z_freqcal[k]*t)) 
            
        return result

    def Ham_sub(self,k_mod,t):
        j1, j2 = self.para.j_ions
        qeye, destroy,create,tensor = qutip.qeye, qutip.destroy,qutip.create,qutip.tensor
        N_ions = self.para.N_ions
        sigmax = qutip.sigmax()
        h0 = self.amp0(t)*self.Ham_sub_one_ion(j1,k_mod, t)*tensor([sigmax,qeye(2)]+[qeye(self.para.cut_off)])
        h1 = self.amp1(t)*self.Ham_sub_one_ion(j2,k_mod, t)*tensor([qeye(2),sigmax]+[qeye(self.para.cut_off)])
        return h0 + h1


    def sim(self):
        cut_off = self.para.cut_off
        j1, j2 = self.para.j_ions
        qeye, destroy,create,tensor = qutip.qeye, qutip.destroy,qutip.create,qutip.tensor
        N_ions = self.para.N_ions
        rho_sub =self.para.ini_state
        #演化
        for k_mod in range(N_ions):
            rho_sub = tensor(rho_sub,self.para.phonon_ini)
            a_k = np.sqrt(self.para.heating_rate)*tensor(qeye(2),qeye(2),qutip.destroy(N=cut_off))
            a_k_dagger = np.sqrt(self.para.heating_rate)*tensor(qeye(2),qeye(2),qutip.create(N=cut_off))
            n_k = a_k_dagger*a_k
            c_list = []
            if 'a' in self.para.lind_type:
                c_list.append(a_k)
            if 'aH' in self.para.lind_type:
                c_list.append(a_k_dagger)
            if 'n' in self.para.lind_type:
                c_list.append(n_k)
            tlist = np.linspace(0, self.para.t_tol, 100000)
            output_sub = qutip.mesolve(lambda x,args: self.Ham_sub(k_mod,x), rho0 = rho_sub, tlist = tlist, c_ops = c_list)
            rho_sub=(output_sub.states[-1]).ptrace([0,1])
        return np.matrix(rho_sub)

