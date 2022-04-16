import numpy as np
from scipy.linalg import null_space

def get_amp_list(P,G,ideal_Theta):
    #找出nullspace
    null_of_P = np.matrix(null_space(P))
    # 变换

    G_trans = null_of_P.H * (np.matrix(G) + np.matrix(G).H) * null_of_P
    #找出本征值最大的本征态
    eiValue,eiVector=np.linalg.eig(G_trans)
    eiValue_sort = list(eiValue.copy())
    eiValue_sort.sort(key = lambda x: abs(x))
    maxValue = eiValue_sort[-1]
    ind = list(eiValue).index(maxValue)
    max_vec = eiVector[ind]
    #变换回去
    x = null_of_P*np.matrix(max_vec).H * np.sqrt(ideal_Theta/abs(maxValue))


    #验证
    print('Px',P*x)
    print('xGx',x.H*G*x)

    return x