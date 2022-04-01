import matplotlib.pyplot as plt
#by default, .plot has line, not mark;.scatter has mark not line
amp = [ 0.0287868 ,  0.30032735,  0.07264575, -0.05146555,  0.17184141,        0.29405068,  0.08671917, -0.13184138, -0.04461831,  0.17587219,        0.15860196, -0.08766853, -0.19345489,  0.00343705,  0.20678409,        0.10223854, -0.16454427, -0.18926199,  0.08250558,  0.23531121,       -0.02887981, -0.40799996, -0.28680877,  0.01223105,  0.01223756,       -0.28685568, -0.40819666, -0.02918696,  0.23512318,  0.08253774,       -0.18921696, -0.16453939,  0.10222827,  0.20663799,  0.00342114,       -0.19320495, -0.08767899,  0.15832718,  0.17579567, -0.0445792 ,       -0.13181015,  0.08668283,  0.29374253,  0.17153619, -0.05125279,        0.07369998,  0.29860844,  0.03005784]
N = [i for i in range(len(amp))]
plt.scatter(N, amp)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Histogram of IQ, greece $\\alpha$')
plt.show()