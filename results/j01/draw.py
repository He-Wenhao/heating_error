import matplotlib.pyplot as plt
#by default, .plot has line, not mark;.scatter has mark not line
N = [2,3,4,5,6,7,8,9,10,11,12]
heating_error = [0.004497865787334798,0.007936377532300032,0.005723306241093851,0.0045856509409013535,0.006050634291640988,0.017035091592401685,0.008318783619748582,0.004568849766719119,0.003949038258522886,0.03109336752259062,0.033921590429074944]
my_bound_32 = [0.018103753598240485,0.04407204644914977,0.04676122614286645,0.0462891573015059,0.0666125770363947,0.128566577650382,0.0787168502853161,0.0554046786881606,0.0498593459494195,0.426612935618896,0.512697444458948]
my_bound_26 = [0.018103753598240485,0.0283341805970892,0.0196718798978372,0.0163880086067742,0.0226755735951089,0.0697778958273321, 0.0333805751140382,0.018016090104686 ,0.015656946670235,0.135790631697009,0.148300948624974]
tau = [150,270,300,300,375,480,540,540,570,720,780]
plt.plot(N,[heating_error[i]/tau[i] for i in range(len(heating_error))],label = 'numerical result')
#plt.plot(N,[2*N[i]*1e2*1e-6*tau[i]/tau[i] for i in range(len(heating_error))],label = 'N$\Gamma t$')
plt.plot(N,[my_bound_32[i]/tau[i] for i in range(len(N))],label = 'our new bound eq.(32)')
plt.plot(N,[my_bound_26[i]/tau[i] for i in range(len(N))],label = 'our new bound eq.(26)')
#plt.plot(N,[heating_error[i] for i in range(len(heating_error))],label = 'numerical result')
#plt.plot(N,[2*N[i]*1e2*1e-6*tau[i] for i in range(len(heating_error))],label = 'N$\Gamma t$')
#plt.plot(N[0:len(my_bound)],[my_bound[i] for i in range(len(my_bound))],label = 'our new bound eq.(32)')
plt.xlabel('N')
plt.ylabel('infid/t($\mu s$)')
plt.title('infid-N (t normalized) (0 1 ions illuminated)')
plt.legend(loc = 'upper left')
plt.show()