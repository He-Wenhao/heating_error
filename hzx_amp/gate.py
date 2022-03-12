import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import time

from numpy import *
from scipy.integrate import *
from scipy.optimize import minimize
from scipy.integrate import dblquad

from .constants import *
from .ion_trap import *


class XX_Gate:
    def __init__(self, trap: Ion_Trap, target_phase, target_qubits, seg, tau, N_step=1000) -> None:
        r"""
        Paraments
        ---------
        trap : Ion_trap
            The target system of the gate
        target_phase : float
            The target phase of rotation
        target_qubits : list
            The number of ions acted by the gate. Only two-qubit gates are supported
        seg : int
            The number of segments
        tau : float
            The total time of the gate
        N_step : int
            Calculation accuracy
        """
        self.trap = trap
        self.target_phase = target_phase
        self.target_qubits = target_qubits
        self.tau = tau
        self.N_step = N_step
        self.seg = seg

        self.para_theta = 0.3
        self.para_psi = 0.01

        self.arg_mu = [0] * seg
        self.arg_amp = [2 * pi] * (seg - 1)

        self.costlist = []

    def __arg_mu2mu(self, arg_mu):
        mu = []
        dt = self.tau / self.N_step

        arg_mu = np.asarray(arg_mu)
        limits = [int(i*self.N_step/2/(len(arg_mu)-1))
                  for i in range(len(arg_mu))]

        for i in range(len(limits)-1):
            delta_0 = 0.5*(arg_mu[i]+arg_mu[i+1])
            delta_1 = 0.5*(arg_mu[i]-arg_mu[i+1])
            # delta_0 = arg_mu[i]
            # delta_1 = 0
            t = 0
            T = (limits[i+1]-limits[i]) * dt * 2
            for j in range(limits[i], limits[i+1]):
                det = delta_0 + delta_1*np.cos(2*pi*t/T)
                mu.append(det*1e6)
                t += dt
        mu.append(arg_mu[-1]*1e6)
        mu = mu + mu[-2::-1]
        return np.asarray(mu)

    def __arg_mu2partial_mu(self, arg_mu, k):
        partial_mu = []
        dt = self.tau / self.N_step

        limits = [int(i*self.N_step/2/(len(arg_mu)-1))
                  for i in range(len(arg_mu))]

        for i in range(len(limits)-1):
            delta_0 = 0
            delta_1 = 0
            if k == i:
                delta_0 = 0.5
                delta_1 = 0.5
            elif k == i+1:
                delta_0 = 0.5
                delta_1 = -0.5
            # if k == i:
            #     delta_0 = 1
            t = 0
            T = (limits[i+1]-limits[i]) * dt * 2
            for j in range(limits[i], limits[i+1]):
                det = delta_0 + delta_1*cos(2*pi*t/T)
                partial_mu.append(det*1e6)
                t += dt
        if k == len(arg_mu) - 1:
            partial_mu.append(1e6)
        else:
            partial_mu.append(0)
        partial_mu = partial_mu + partial_mu[-2::-1]
        return np.asarray(partial_mu)

    def plot_mu(self):
        r"""
        Plot the current waveform of $\mu$
        """
        mu = self.__arg_mu2mu(self.arg_mu)

        plt.rcParams['figure.figsize'] = (12, 3)
        plt.rcParams.update({'font.size': 15})

        plt.title('Detuning vs time')
        plt.plot([i*self.tau/self.N_step for i in range(self.N_step+1)], mu/1e3)
        plt.xlabel('Time ($\\mu$s)')
        plt.ylabel('Detuning. (2*$\\pi$*kHz)')
        plt.show()

    def __arg_amp2amp(self, arg_amp):
        amp = []

        limits = [int(i*self.N_step/2/(len(arg_amp)))
                  for i in range(len(arg_amp)+1)]
        for i in range(len(limits)-1):
            for j in range(limits[i], limits[i+1]):
                amp.append(arg_amp[i]*1e6)
        amp.append(arg_amp[-1]*1e6)
        amp = amp + amp[-2::-1]
        return np.asarray(amp)

    def __arg_amp2partial_amp(self, arg_amp, k):
        partial_amp = []

        limits = [int(i*self.N_step/2/(len(arg_amp)))
                  for i in range(len(arg_amp)+1)]
        for i in range(len(limits)-1):
            for j in range(limits[i], limits[i+1]):
                if i == k:
                    partial_amp.append(1e6)
                else:
                    partial_amp.append(0)
        if k == len(arg_amp) - 1:
            partial_amp.append(1e6)
        else:
            partial_amp.append(0)
        partial_amp = partial_amp + partial_amp[-2::-1]
        return np.asarray(partial_amp)

    def plot_amp(self):
        r"""
        Plot the current waveform of $\Omega$
        """
        amp = self.__arg_amp2amp(self.arg_amp)

        plt.rcParams['figure.figsize'] = (8, 4)
        plt.rcParams.update({'font.size': 15})
        plt.title('Rabi frequency vs time')

        plt.plot([i*self.tau/self.N_step for i in range(self.N_step+1)], amp/1e3)
        plt.ylabel('Rabi freq')
        plt.xlabel('Time ($\\mu s$)')

        plt.show()

    def __integrate(self, f):
        assert len(f) == self.N_step+1
        dt = self.tau/self.N_step

        result = np.zeros(self.N_step+1, dtype=f.dtype)
        total = 0.0
        for i in range(self.N_step):
            total += (f[i]+f[i+1])*dt/2
            result[i + 1] = total
        return np.asarray(result)

    # @jit
    def __double_integrate(self, f):
        # start = time.perf_counter()
        dt = self.tau / self.N_step

        # int1 = np.zeros(self.N_step+1)
        # for i in range(self.N_step+1):
        #     total = 0.0
        #     for j in range(i):
        #         total += (f[i, j]+f[i, j+1])*dt/2
        #     int1[i] = total

        int1 = np.tril(f, -1)
        int2 = np.sum(int1) * dt**2

        # int2 = 0.0
        # for i in range(self.N_step):
        #     int2 += (int1[i]+int1[i+1])*dt/2

        # end = time.perf_counter()
        # print("double int time:", end-start)
        return int2

    def __costfunction(self, arg_mu, arg_amp):
        mu = self.__arg_mu2mu(arg_mu)
        int_mu = self.__integrate(mu)
        amp = self.__arg_amp2amp(arg_amp)
        Chi = amp*np.sin(int_mu)
        theta = amp*np.cos(int_mu)
        t = np.asarray([i*self.tau/self.N_step for i in range(self.N_step+1)])

        # Theta = 0
        # for i in range(self.trap.N_ions):
        #     f = 2 * np.dot(Chi.reshape([-1, 1]), Chi.reshape([1, -1]))
        #     f = f * np.sin(self.trap.w[i] *
        #                    (t.reshape([-1, 1]) - t.reshape([1, -1])))
        #     Theta += self.trap.eta[i]**2 * \
        #         self.trap.b[self.target_qubits[0]][i] * \
        #         self.trap.b[self.target_qubits[1]][i] * \
        #         self.__double_integrate(f)
        # print('Theta:', Theta)

        # disturbance_Theta = 0
        # disturbance_amp = self.__disturbance_amp(amp)
        # for i in range(self.trap.N_ions):
        #     f = np.dot(
        #         (disturbance_amp * np.sin(int_mu)).reshape([-1, 1]), Chi.reshape([1, -1]))
        #     f += np.dot(Chi.reshape([-1, 1]), (disturbance_amp *
        #                 np.sin(int_mu)).reshape([1, -1]))
        #     f = f * np.sin(self.trap.w[i] *
        #                    (t.reshape([-1, 1]) - t.reshape([1, -1])))
        #     disturbance_Theta += self.trap.eta[i]**2 * \
        #         self.trap.b[self.target_qubits[0]][i] * \
        #         self.trap.b[self.target_qubits[1]][i] * \
        #         self.__double_integrate(f)
        # print('disturbance Theta:', disturbance_Theta)

        avg_alphas = []
        for i in range(self.trap.N_ions):
            for j in self.target_qubits:
                a = self.__integrate(Chi * np.exp(1j*t*self.trap.w[i]))
                a = self.__integrate(a) / self.tau
                avg_alphas.append(-1j*a[-1]*self.trap.b[j][i]*self.trap.eta[i])
        # print('avg alphas:', avg_alphas)

        psi = self.__integrate(theta)[-1]

        cost = 0
        # cost = self.para_theta * \
        #     (Theta-self.target_phase)**2
        cost += self.para_psi*psi**2
        # cost = self.para_dThetea * disturbance_Theta**2
        for a in avg_alphas:
            cost += (a*a.conjugate()).real

        jac = []
        for k in range(len(arg_mu)):
            partial_sum = 0
            partial_mu = self.__arg_mu2partial_mu(arg_mu, k)
            int_partial_mu = self.__integrate(partial_mu)
            partial_Chi = amp*np.cos(int_mu)*int_partial_mu
            partial_theta = -amp*np.sin(int_mu)*int_partial_mu

            # partial_Theta = 0
            # for i in range(self.trap.N_ions):
            #     f = 2 * np.dot(Chi.reshape([-1, 1]),
            #                    partial_Chi.reshape([1, -1]))
            #     f += 2 * np.dot(partial_Chi.reshape([-1, 1]),
            #                     Chi.reshape([1, -1]))
            #     f = f * np.sin(self.trap.w[i] *
            #                    (t.reshape([-1, 1]) - t.reshape([1, -1])))
            #     partial_Theta += self.trap.eta[i]**2 * \
            #         self.trap.b[self.target_qubits[0]][i] * \
            #         self.trap.b[self.target_qubits[1]][i] * \
            #         self.__double_integrate(f)
            # partial_sum += self.para_theta * 2 * \
            #     (Theta-self.target_phase)*partial_Theta

            # partial_disturbance_Theta = 0
            # for i in range(self.trap.N_ions):
            #     f = np.dot(
            #         (disturbance_amp*np.cos(int_mu)*int_partial_mu).reshape([-1, 1]), Chi.reshape([1, -1]))
            #     f += np.dot((disturbance_amp*np.sin(int_mu)).reshape([-1, 1]),
            #                 partial_Chi.reshape([1, -1]))
            #     f += np.dot(Chi.reshape([-1, 1]), (disturbance_amp*np.cos(int_mu)
            #                 * int_partial_mu).reshape([1, -1]))
            #     f += np.dot(partial_Chi.reshape([-1, 1]),
            #                 (disturbance_amp*np.sin(int_mu)).reshape([1, -1]))
            #     f = f * np.sin(self.trap.w[i] *
            #                    (t.reshape([-1, 1]) - t.reshape([1, -1])))
            #     partial_disturbance_Theta += self.trap.eta[i]**2 * \
            #         self.trap.b[self.target_qubits[0]][i] * \
            #         self.trap.b[self.target_qubits[1]][i] * \
            #         self.__double_integrate(f)
            # partial_sum += self.para_dThetea * 2*disturbance_Theta*partial_disturbance_Theta

            partial_avg_alphas = []
            for i in range(self.trap.N_ions):
                for j in self.target_qubits:
                    a = self.__integrate(
                        partial_Chi * np.exp(1j*t*self.trap.w[i]))
                    a = self.__integrate(a) / self.tau
                    partial_avg_alphas.append(-1j*a[-1] *
                                              self.trap.b[j][i]*self.trap.eta[i])
            for a, partial_a in zip(avg_alphas, partial_avg_alphas):
                partial_sum += 2 * (partial_a * a.conjugate()).real

            partial_psi = self.__integrate(partial_theta)[-1]
            partial_sum += 2 * self.para_psi * psi * partial_psi

            jac.append(partial_sum)

        for k in range(len(arg_amp)):
            partial_sum = 0
            # partial_amp = self.__arg_amp2partial_amp(arg_amp, k)
            # partial_Chi = partial_amp * np.sin(int_mu)
            # partial_psi = partial_amp * np.cos(int_mu)

            # partial_Theta = 0
            # for i in range(self.trap.N_ions):
            #     f = 2 * np.dot(Chi.reshape([-1, 1]),
            #                    partial_Chi.reshape([1, -1]))
            #     f += 2 * np.dot(partial_Chi.reshape([-1, 1]),
            #                     Chi.reshape([1, -1]))
            #     f = f * np.sin(self.trap.w[i] *
            #                    (t.reshape([-1, 1]) - t.reshape([1, -1])))
            #     partial_Theta += self.trap.eta[i]**2 * \
            #         self.trap.b[self.target_qubits[0]][i] * \
            #         self.trap.b[self.target_qubits[1]][i] * \
            #         self.__double_integrate(f)
            # partial_sum += self.para_theta * 2 * \
            #     (Theta-self.target_phase)*partial_Theta

            # # partial_disturbance_Theta = 0
            # # for i in range(self.trap.N_ions):
            # #     f = np.dot((disturbance_amp*np.sin(int_mu)).reshape([-1, 1]),
            # #                partial_Chi.reshape([1, -1]))
            # #     f += np.dot(partial_Chi.reshape([-1, 1]),
            # #                 (disturbance_amp*np.sin(int_mu)).reshape([1, -1]))
            # #     f = f * np.sin(self.trap.w[i] *
            # #                    (t.reshape([-1, 1]) - t.reshape([1, -1])))
            # #     partial_disturbance_Theta += self.trap.eta[i]**2 * \
            # #         self.trap.b[self.target_qubits[0]][i] * \
            # #         self.trap.b[self.target_qubits[1]][i] * \
            # #         self.__double_integrate(f)
            # # partial_sum += self.para_dThetea * 2*disturbance_Theta*partial_disturbance_Theta

            # partial_avg_alphas = []
            # for i in range(self.trap.N_ions):
            #     for j in self.target_qubits:
            #         a = self.__integrate(
            #             partial_Chi * np.exp(1j*t*self.trap.w[i]))
            #         a = self.__integrate(a) / self.tau
            #         partial_avg_alphas.append(-1j*a[-1] *
            #                                   self.trap.b[j][i]*self.trap.eta[i])
            # for a, partial_a in zip(avg_alphas, partial_avg_alphas):
            #     partial_sum += 2 * (partial_a * a.conjugate()).real

            # partial_psi = self.__integrate(partial_theta)[-1]
            # partial_sum += 2 * self.para_psi * psi * partial_psi

            jac.append(partial_sum)

        print("cost:", cost)
        # print("jac:", jac)
        self.costlist.append(cost)

        return cost, np.array(jac)

    def optimize(self, maxiter=50):
        r"""
        Optimize the gate, plot the cost function during optimization

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations to perform
        """
        self.costlist = []
        self.arg_mu = np.absolute(np.random.randn(self.seg)).tolist()
        self.arg_amp = [2 * pi] * (self.seg - 1)
        print('init mu:', self.arg_mu)
        print('init amp:', self.arg_amp)
        def costf(x): return self.__costfunction(x[:self.seg], x[self.seg:])
        solution = minimize(costf, self.arg_mu + self.arg_amp, method="BFGS",
                            jac=True,  options={'disp': True, 'maxiter': maxiter})
        self.arg_mu = solution.x[:self.seg]
        self.arg_amp = solution.x[self.seg:]

        # print("cost:", self.__costfunction(self.arg_mu, self.arg_amp)[0])
        # print('arg mu:', self.arg_mu)
        # print('arg amp:', self.arg_amp)
        #plt.plot(self.costlist)
        #plt.title("cost")
        #plt.show()

    def evolution(self):
        pass

    def regularization(self):
        r"""
        Regularized the Rabi freq to be the same as the target phase
        """
        mu = self.__arg_mu2mu(self.arg_mu)
        int_mu = self.__integrate(mu)
        amp = self.__arg_amp2amp(self.arg_amp)
        Chi = amp*np.sin(int_mu)
        t = np.asarray(
            [i*self.tau/self.N_step for i in range(self.N_step+1)])
        Theta = 0
        for i in range(self.trap.N_ions):
            f = 2 * np.dot(Chi.reshape([-1, 1]), Chi.reshape([1, -1]))
            f = f * np.sin(self.trap.w[i] *
                           (t.reshape([-1, 1]) - t.reshape([1, -1])))
            Theta += self.trap.eta[i]**2 * \
                self.trap.b[self.target_qubits[0]][i] * \
                self.trap.b[self.target_qubits[1]][i] * \
                self.__double_integrate(f)
        self.arg_amp = [
            *map(lambda x: x*np.sqrt(self.target_phase / Theta), self.arg_amp)]
        print(self.__costfunction(self.arg_mu, self.arg_amp)[0])
        print('arg mu:', self.arg_mu.tolist())
        print('arg amp:', self.arg_amp)

    def get_theta(self):
        mu = self.__arg_mu2mu(self.arg_mu)
        int_mu = self.__integrate(mu)
        amp = self.__arg_amp2amp(self.arg_amp)
        Chi = amp*np.sin(int_mu)
        t = np.asarray([i*self.tau/self.N_step for i in range(self.N_step+1)])

        Theta = 0
        for i in range(self.trap.N_ions):
            f = 2 * np.dot(Chi.reshape([-1, 1]), Chi.reshape([1, -1]))
            f = f * np.sin(self.trap.w[i] *
                           (t.reshape([-1, 1]) - t.reshape([1, -1])))
            Theta += self.trap.eta[i]**2 * \
                self.trap.b[self.target_qubits[0]][i] * \
                self.trap.b[self.target_qubits[1]][i] * \
                self.__double_integrate(f)
        return Theta

    def get_alphas(self):
        mu = self.__arg_mu2mu(self.arg_mu)
        int_mu = self.__integrate(mu)
        amp = self.__arg_amp2amp(self.arg_amp)
        Chi = amp*np.sin(int_mu)
        t = np.asarray([i*self.tau/self.N_step for i in range(self.N_step+1)])

        alphas = []
        for j in self.target_qubits:
            alphas.append([])
            for i in range(self.trap.N_ions):
                a = self.__integrate(Chi * np.exp(1j*t*self.trap.w[i]))
                alphas[-1].append(-1j*a[-1]*self.trap.b[j][i]*self.trap.eta[i])
        return alphas

    def get_psi(self):
        mu = self.__arg_mu2mu(self.arg_mu)
        int_mu = self.__integrate(mu)
        amp = self.__arg_amp2amp(self.arg_amp)
        theta = amp*np.cos(int_mu)
        t = np.asarray([i*self.tau/self.N_step for i in range(self.N_step+1)])

        f = self.__integrate(theta)[-1] / 2
        psi = []
        for ion in self.target_qubits:
            psi.append(0)
            for k in range(self.trap.N_ions):
                psi[-1] += (self.trap.eta[k] * self.trap.b[ion][k])**2
            psi[-1] *= f
        return psi

    def get_Upsilon_1(self):
        mu = self.__arg_mu2mu(self.arg_mu)
        int_mu = self.__integrate(mu)
        amp = self.__arg_amp2amp(self.arg_amp)
        theta = amp*np.cos(int_mu)
        t = np.asarray([i*self.tau/self.N_step for i in range(self.N_step+1)])

        f = -self.__integrate(theta)[-1] / 8
        Upsilon_1 = []
        for ion in self.target_qubits:
            Upsilon_1.append(0)
            for k in range(self.trap.N_ions):
                Upsilon_1[-1] += (self.trap.eta[k] * self.trap.b[ion][k])**2
            Upsilon_1[-1] = f * Upsilon_1[-1]**2
        return Upsilon_1

    def get_Upsilon_2(self):
        mu = self.__arg_mu2mu(self.arg_mu)
        int_mu = self.__integrate(mu)
        amp = self.__arg_amp2amp(self.arg_amp)
        Chi = amp*np.sin(int_mu)
        theta = amp*np.cos(int_mu)
        t = np.asarray([i*self.tau/self.N_step for i in range(self.N_step+1)])

        Upsilon_2 = 0
        tmp = 0
        for i in range(self.trap.N_ions):
            tmp += self.trap.eta[i]**2 * \
                (self.trap.b[self.target_qubits[0]][i]**2 +
                 self.trap.b[self.target_qubits[1]][i]**2)
        for i in range(self.trap.N_ions):
            f = np.dot(Chi.reshape([-1, 1]), Chi.reshape([1, -1]))
            f = f * np.sin(self.trap.w[i] *
                           (t.reshape([-1, 1]) - t.reshape([1, -1])))
            Upsilon_2 += -tmp * self.trap.eta[i]**2 * \
                self.trap.b[self.target_qubits[0]][i] * \
                self.trap.b[self.target_qubits[1]][i] * \
                self.__double_integrate(f)
        # tmp = 0
        # for i in range(self.trap.N_ions):
        #     tmp += self.trap.eta[i]**2 * \
        #         self.trap.b[self.target_qubits[0]][i] * \
        #         self.trap.b[self.target_qubits[1]][i]
        # for i in range(self.trap.N_ions):
        #     f = np.dot(theta.reshape([-1, 1]), theta.reshape([1, -1]))
        #     f = f * np.sin(2 * self.trap.w[i] *
        #                    (t.reshape([-1, 1]) - t.reshape([1, -1])))
        #     Upsilon_2 += tmp * self.trap.eta[i]**2 * \
        #         self.trap.b[self.target_qubits[0]][i] * \
        #         self.trap.b[self.target_qubits[1]][i] * \
        #         self.__double_integrate(f)
        return Upsilon_2

    def get_Upsilon_3(self):
        mu = self.__arg_mu2mu(self.arg_mu)
        int_mu = self.__integrate(mu)
        amp = self.__arg_amp2amp(self.arg_amp)
        Chi = amp*np.sin(int_mu)
        theta = amp*np.cos(int_mu)
        t = np.asarray([i*self.tau/self.N_step for i in range(self.N_step+1)])
        dt = self.tau / self.N_step

        tri_int = np.zeros((self.trap.N_ions, self.trap.N_ions))
        for k in range(self.trap.N_ions):
            for l in range(self.trap.N_ions):
                f = Chi.reshape(-1, 1, 1) * Chi.reshape(1, -
                                                        1, 1) * theta.reshape(1, 1, -1)
                f = f * np.sin(self.trap.w[k] * (t.reshape(-1, 1, 1)-t.reshape(1, 1, -1))) * np.sin(
                    self.trap.w[l] * (t.reshape(1, -1, 1)-t.reshape(1, 1, -1)))
                f = np.sum(np.tril(f, -1), 2)
                f = np.sum(np.tril(f, -1))
                tri_int[k][l] = f * dt**3

        Upsilon_3 = []
        for j in range(2):
            Upsilon_3.append(0)
            for k in range(self.trap.N_ions):
                for l in range(self.trap.N_ions):
                    tmp = self.trap.b[self.target_qubits[j]][k]**2 * \
                        self.trap.b[self.target_qubits[j]][l]**2 + \
                        self.trap.b[self.target_qubits[j]][k] * \
                        self.trap.b[self.target_qubits[j]][l] * \
                        self.trap.b[self.target_qubits[1-j]][k] * \
                        self.trap.b[self.target_qubits[1-j]][l] + \
                        self.trap.b[self.target_qubits[j]][k] * \
                        self.trap.b[self.target_qubits[1-j]][k] * \
                        self.trap.b[self.target_qubits[1-j]][l]**2 * 2
                    Upsilon_3[-1] += 2 / 3 * self.trap.eta[k]**2 * \
                        self.trap.eta[k]**2 * tmp * tri_int[k][l]
        return Upsilon_3
