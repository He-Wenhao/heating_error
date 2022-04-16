import numpy as np
from Tab0_constants import  *
from Tab1_ion_pos_spec_cal import  *





y=np.array([-10,-5,0,5,10])*us
#omega_k,b=radial_mode_spectrum(len(y), 0.32*2*np.pi*MHz,2.18*2*np.pi*MHz,y)
omega_k,b=radial_mode_spectrum( 5, 0.32*2*np.pi*MHz,2.18*2*np.pi*MHz,y)
omega_k = omega_k*2*np.pi*MHz

print('frequency in Hz:',omega_k)
print('frequency in MHz/2pi:',omega_k/(2*np.pi*10**6))
print('b',b)
