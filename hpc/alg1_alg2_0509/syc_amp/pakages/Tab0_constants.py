import numpy as np
from scipy import constants

MHz = 1e6
us = 1e-6
k_cou = constants.e ** 2 / (4 * np.pi * constants.epsilon_0)
m = 171 * constants.m_p

la_390=0.39*1e-6
k = 2*np.pi/la_390
eta_x = 2 * k *np.sqrt(constants.hbar/(2*m*2*np.pi*8*0.23423*MHz)) / np.sqrt(2)
eta_y = 2 * k *np.sqrt(constants.hbar/(2*m*2*np.pi*8*0.25933*MHz)) / np.sqrt(2)


# print('eta_x:',eta_x)
# print('eta_y:',eta_y)

# eta_x: 0.09015715679924886
# eta_y: 0.08568308336692779

# # ###################################################  微波作用下，两个离子的 Rabi 震荡    #############################################
# from matplotlib import pyplot as plt
# omega = 0.5
# t = np.linspace(0,20,100)
# plt.plot(t, np.cos(1/2*omega*t)**2)
# plt.plot(t, np.cos(1/2*omega*t)**4)
# plt.plot(t, (np.sin(1/2*omega*t)*np.cos(1/2*omega*t))**2)
# plt.plot(t, np.sin(1/2*omega*t)**6)
# plt.legend(['1','2','3','4'])
# plt.show()
# # ###################################################  微波作用下，两个离子的 Rabi 震荡    #############################################