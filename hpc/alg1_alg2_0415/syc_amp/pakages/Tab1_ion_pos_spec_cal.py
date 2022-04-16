import sympy as sy
from pakages.Tab0_constants import  *
from scipy import linalg



def potential_ax(x, N_ions, omega_ax):      ##  计算囚禁势场
    axial = m * omega_ax ** 2 / 2
    v = 0
    for i in range(0, N_ions):
        for j in range(0, i):
            v = v + k_cou / ((x[i] - x[j]) ** 2) ** 0.5     ##  这里只计算 xj < xi
    for i in range(0, N_ions):
        v = v + axial * x[i] ** 2
    return v



def equ_pos(N_ions, omega_ax):      ##	计算均衡位置
    xyzname = [0] * N_ions
    for i in range(0, N_ions):
        xyzname[i] = 'z' + np.str(i);
    xyzsymbols = sy.symbols(xyzname)

    fff = [0] * N_ions
    initialguess = [0] * N_ions
    for i in range(0, N_ions):
        fff[i] = sy.diff(potential_ax(xyzsymbols, N_ions, omega_ax), xyzsymbols[i])
        initialguess[i] = 5e-6 * (i - N_ions / 2)
    result = sy.simplify(sy.nsolve(fff, xyzsymbols, initialguess))
    return result


def pos_var(pos_list, N_ions):
    distance = [0] * (N_ions - 1)
    for i in range(0, N_ions - 1):
        distance[i] = abs(pos_list[i + 1] - pos_list[i])  # the distance unit is um
        distance = np.array(distance, dtype=float)
    return np.std(distance) / np.mean(distance) * 100



def radial_mode_spectrum(N_ions, omega_ax,omega_ra,partialpos):     #  计算X方向振动模式
 #   partialpos = equ_pos(N_ions, omega_ax)
    hessian_radial = [[0 for j in range(0, N_ions)] for i in range(0, N_ions)]

    def X_matrix_11(i):
        s = 0
        for ii in range(0, N_ions):
            if ii != i:
                s = s + 1 / abs(partialpos[i] - partialpos[ii]) ** 3
        return -s * k_cou + m * omega_ra ** 2
    def X_matrix_12(i, j):
        if i == j:
            return 0
        return k_cou * 1 / abs(partialpos[i] - partialpos[j]) ** 3

    for i in range(0, N_ions):
        hessian_radial[i][i] = X_matrix_11(i) / m / omega_ra**2
    for i in range(0, N_ions):
        for j in range(0, N_ions):
            if i != j:
                hessian_radial[i][j] = X_matrix_12(i, j) / m / omega_ra**2

    X_freq, X_modes = linalg.eigh(  np.array(hessian_radial,dtype=float)  )
    X_freqcal = (X_freq ** 0.5) * omega_ra / (2 * np.pi * MHz)
    X_modes = np.array(X_modes)     #   这里的X_modes[i,j] 即为b-matrix，其中i表示离子的编号，j表示模式的编号
    # print('radial b_j^m matrix is:\n', X_modes)
    return X_freqcal, X_modes


def axial_mode_spectrum(N_ions, omega_ax,omega_ra,partialpos):      #  计算X方向振动模式
  #  partialpos = equ_pos(N_ions, omega_ax)
    hessian_axial = [[0 for j in range(0, N_ions)] for i in range(0, N_ions)]
    def Z_matrix_11(i):
        s = 0
        for ii in range(0, N_ions):
            if ii != i:
                s = s + 2 / abs(partialpos[i] - partialpos[ii]) ** 3
        return s * k_cou + m * omega_ax ** 2
    def Z_matrix_12(i, j):
        if i == j:
            return 0
        return -2 * k_cou / abs(partialpos[i] - partialpos[j]) ** 3

    for i in range(0, N_ions):
        hessian_axial[i][i] = Z_matrix_11(i) / m / omega_ax**2
    for i in range(0, N_ions):
        for j in range(0, N_ions):
            if i != j:
                hessian_axial[i][j] = Z_matrix_12(i, j) / m / omega_ax**2

    Z_freq, Z_modes = linalg.eigh(  np.array(hessian_axial,dtype=float)  )
    Z_freqcal = (Z_freq ** 0.5) * omega_ax /(2*np.pi * MHz)
    Z_modes = np.array(Z_modes)     #   这里的Z_modes[i,j] 即为b-matrix，其中i表示离子的编号，j表示模式的编号
    # print(Z_freqcal)
    # print('radial b_j^m matrix is:\n', Z_modes)
    return Z_freqcal, Z_modes






