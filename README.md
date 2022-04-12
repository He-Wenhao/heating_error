#heating error
this is the code for notes 
https://www.overleaf.com/5296295776bttfbkqgwhmb

#usage
taking ./examples as example
1.solve the environment with anaconda and ./env/xxx.yaml
2.run set_configs.py(set trap parameters by editing trap.ini and [shined] in laser.ini)
3.run compute_phonon.py(or ther methods) and get phonon paras. Data stored in phonon.ini
4.run optimize_pulse.py(or other optimized method) and get laser amplitude. Data stored in laser.ini
5.run simu_heating.py and get the some print on terminal
6.extract heating error manually into and draw

#attention
sympy must be updated to 1.10(conda will only install 1.5), otherwise error will be raised