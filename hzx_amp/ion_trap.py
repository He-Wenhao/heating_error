import matplotlib.pyplot as plt
import numpy as np
import sympy as sy

from numpy import *
from scipy.integrate import *
from scipy.optimize import minimize

from .constants import *


class Ion_Trap:
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
        print('equal position:', self.pos)
        # The 1st subscript marks different ions, and the 2rd marks modes
        self.w, self.b = self.phonon_mode(self.pos)
        print('phonon mode:', self.w)
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
