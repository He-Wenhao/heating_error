import numpy as np
from scipy import constants

MHz = 1e6
us = 1e-6
k_cou = constants.e ** 2 / (4 * np.pi * constants.epsilon_0)
m = 171 * constants.m_p
hbar = constants.hbar

s = [np.array([[1, 0], [0, 1]]),
     np.array([[0, 1], [1, 0]]),
     np.array([[0, -1j], [1j, 0]]),
     np.array([[1, 0], [0, -1]])]
