# heating error
this is the code for notes 
https://www.overleaf.com/5296295776bttfbkqgwhmb

# usage

以./examples/simple/all.ipynb为例,可以先尝试跑通这个例子.

1.solve the environment with anaconda and ./env/alg1_alg2.yaml

taking ./examples/simple/all.ipynb as example

2.run set_configs.py(set trap parameters by editing trap.ini and [shined] in laser.ini)

3.run compute_phonon.py(or ther methods) and get phonon paras. Data stored in phonon.ini

4.run optimize_pulse.py(or other optimized method) and get laser amplitude. Data stored in laser.ini

5.run simu_heating.py and get the some print on terminal

6.run visulization.py and generate a picture

# attention
sympy must be updated to 1.10(conda will only install 1.5), otherwise error will be raised
