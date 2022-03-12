import numpy as np
import qutip
def infidelity(a,b):
    return (1-qutip.fidelity(qutip.Qobj(a), qutip.Qobj(b)),qutip.tracedist(10000*qutip.Qobj(a), 10000*qutip.Qobj(b))/10000)
alg1 = np.array([[ 0.28088266+8.81589727e-19j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.02705866+1.70481058e-01j],
 [ 0.        +0.00000000e+00j,  0.21912761+1.72385281e-19j,
  -0.02705866-1.18424195e-01j,  0.        +0.00000000e+00j],
 [ 0.        +0.00000000e+00j, -0.02705866+1.18424195e-01j,
   0.21911723+1.79930940e-18j,  0.        +0.00000000e+00j],
 [ 0.02705866-1.70481058e-01j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.28087251-2.44163356e-18j]])
   
alg2 = np.array([[ 0.28088289+2.02413008e-18j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.02705884+1.70481739e-01j],
 [ 0.        +0.00000000e+00j,  0.21912737-5.81540305e-19j,
  -0.02705884-1.18424491e-01j,  0.        +0.00000000e+00j],
 [ 0.        +0.00000000e+00j, -0.02705884+1.18424491e-01j,
   0.219117  -3.77822678e-19j,  0.        +0.00000000e+00j],
 [ 0.02705884-1.70481739e-01j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.28087274-9.10890333e-19j]])

print(np.linalg.norm(alg1-alg2))
print(infidelity(alg1, alg2))