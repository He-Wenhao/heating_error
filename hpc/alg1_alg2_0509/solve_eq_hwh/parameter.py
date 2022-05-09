us = 1e-6
MHz = 1e6
import numpy as np
from .paraPG import calculator_P, calculator_G
from .solve_optimize import get_amp_list
class hwh_optimize():
    def __init__(self,unit,detuning_list, duration, number_of_segments,bij,lamb_dicke):
        self.number_of_segments = number_of_segments
        self.eta  = np.array(bij)*lamb_dicke
        if unit == 'SI':
            self.detuning_list = np.array(detuning_list)/MHz
            self.duration = duration/us
        elif unit == 'MHz':
            self.detuning_list = np.array(detuning_list)
            self.duration = duration
        else:
            raise TypeError('not a unit')
    
    def _compute_P(self):
        self.P = calculator_P(self.detuning_list, self.duration, self.number_of_segments)

    def _compute_G(self):
        self.G = calculator_G(self.detuning_list, self.duration, self.number_of_segments,self.eta)

    def optimize_solve(self):
        self._compute_P()
        self._compute_G()
        print('getP and G')
        self.x = get_amp_list(self.P, self.G, self.ideal_Theta)
        self.x = list(self.x.T)[0]
        sorted_x = self.x.copy()
        sorted_x.sort()
        self.bnd = max(abs(sorted_x[0]),abs(sorted_x[-1]))
        return self.x, self.bnd

