from pakages.Tab1_ion_pos_spec_cal import *

from scipy import linalg
import matplotlib.pyplot as plt
import datetime
import sympy as sy
import time


class uniform_spacing_duan(object):     #   theory paper: Large Scale Quantum Computation in an Anharmonic Linear Ion Trap
    def __init__(self, N_ions=10, f_ax=0.1, f_ra=2.18, gama4=1.0):
        self.N_ions = N_ions  # 总的离子数目
        self.omega_ax = 2 * np.pi * f_ax * MHz  # axial confinement
        self.omega_ra = 0.5 * 2 * np.pi * f_ra * MHz  # radial confinement
        self.gama4 = gama4  # potential 四阶系数的

    def potential_ax(self, z):  ##  计算囚禁势场
        self.axial = m * self.omega_ax ** 2  # 计算二次项系数
        l0 = (k_cou / self.axial) ** (1 / 3)    #   定义 unit distance
        self.anharmo4 = self.gama4 * self.axial / l0**2  # 计算四此项系数
        v = 0
        for i in range(0, self.N_ions):
            for j in range(0, i):
                v = v + k_cou / ((z[i] - z[j]) ** 2) ** 0.5  ##  这里只计算 xj < xi
        for i in range(0, self.N_ions):
            v = v - self.axial * z[i] ** 2 / 2 + self.anharmo4 * z[i] ** 4 / 4
        return v

    def equ_pos(self):  ##	计算均衡位置
        xyzname = [0] * self.N_ions
        for i in range(0, self.N_ions):
            xyzname[i] = 'z' + np.str(i)
        xyzsymbols = sy.symbols(xyzname)

        fff = [0] * self.N_ions
        initialguess = [0] * self.N_ions
        for i in range(0, self.N_ions):
            fff[i] = sy.diff(self.potential_ax(xyzsymbols), xyzsymbols[i])
            initialguess[i] = 5e-6 * (i - self.N_ions / 2)
        self.pos_list = sy.simplify(sy.nsolve(fff, xyzsymbols, initialguess))
        return self.pos_list

    def draw_ions_potential(self):
        pos_list = np.squeeze(np.asarray(self.pos_list)) / 1e-6
        x_max = 1.5 * int(abs(pos_list[-1]))
        x = np.linspace(-x_max, x_max, 1000)
        potential = - self.axial * x ** 2 + self.anharmo4 * x ** 4
        ions_y = int(max(potential) * (1 - 2 / (1 + np.sqrt(5))))  # 用黄金分割比例算出 离子链的摆放位置
        plt.plot(x, potential)
        plt.plot(pos_list, [ions_y] * self.N_ions, 'bo')
        plt.title('uniform spaced '+str(self.N_ions) +'-ions equilibrium position, and trapping potential')
        plt.show()

    def pos_var(self):
        distance = [0] * (self.N_ions - 1)
        for i in range(0, self.N_ions - 1):
            distance[i] = abs(self.pos_list[i + 1] - self.pos_list[i])  # the distance unit is um
            distance = np.array(distance, dtype=float)
        print('\\distances are:', distance * 1e6)
        distance = distance[1:-1]
        print('mean distance is:', np.mean(distance) * 1e6, ' um')
        return np.std(distance) / np.mean(distance) * 100

    def radial_mode_spectrum(self):  # 计算X方向振动模式
        hessian_radial = [[0 for j in range(0, self.N_ions)] for i in range(0, self.N_ions)]

        def X_matrix_11(i):
            s = 0
            for ii in range(0, self.N_ions):
                if ii != i:
                    s = s + 1 / abs(self.pos_list[i] - self.pos_list[ii]) ** 3
            return -s * k_cou + m * self.omega_ra ** 2

        def X_matrix_12(i, j):
            if i == j:
                return 0
            return k_cou * 1 / abs(self.pos_list[i] - self.pos_list[j]) ** 3

        for i in range(0, self.N_ions):
            hessian_radial[i][i] = X_matrix_11(i) / m / self.omega_ra ** 2
        for i in range(0, self.N_ions):
            for j in range(0, self.N_ions):
                if i != j:
                    hessian_radial[i][j] = X_matrix_12(i, j) / m / self.omega_ra ** 2

        X_freq, X_modes = linalg.eigh(np.array(hessian_radial, dtype=float))
        X_freqcal = (X_freq ** 0.5) * self.omega_ra / (2 * np.pi * MHz)
        X_modes = np.array(X_modes)  # 这里的X_modes[i,j] 即为b-matrix，其中i表示离子的编号，j表示模式的编号
        # print('radial b_j^m matrix is:\n', X_modes)
        return X_freqcal, X_modes



uni_spac = uniform_spacing_duan(N_ions=19, f_ax=0.08, f_ra=3, gama4=4.3)  # 初始化 class


#       求解均衡位置
start_time = time.time()
pos_list = uni_spac.equ_pos()

#       求解 modes 和 spectrum
X_freqcal, X_modes = uni_spac.radial_mode_spectrum()
end_time = time.time()
print('time used for calculation mode and spactrum is:', round(end_time-start_time,2), ' seconds')

#   画出离子排布和 potential
print('ions position are: ', np.squeeze(np.asarray(pos_list)) / 1e-6)
print('the relative standard deveration (RSD) of ions\' positions is:', round(uni_spac.pos_var(), 2), '%')
uni_spac.draw_ions_potential()

#   画出振动谱线
freq_gap = abs(X_freqcal[0] - X_freqcal[-1]) / (uni_spac.N_ions-1)      #   假设频谱是均匀的，计算频谱之间的间隔
for i in range(0,len(X_freqcal)):
    plt.vlines(X_freqcal[i], 0, 1.0*(i+1) , colors = "r", linestyles = "solid")     #   画出频谱，竖线
    if i < len(X_freqcal) - 1:      #   分段数，只有 N-1
        plt.hlines(0.5*(i+1), X_freqcal[i], X_freqcal[i+1], colors = "b", linestyles = "solid", label = 'aa')     #   画出频谱间距，竖线
        plt.text( (X_freqcal[i]+X_freqcal[i+1])/2, 0.6*(i+1), str(round(X_freqcal[i+1] - X_freqcal[i],3)), ha='center', va='center', color='b')
    plt.plot(X_freqcal[0]+i*freq_gap, 0, 'kv')       #   假设频谱是均匀排布的
plt.show()

#   打印出 b_matrix
print(X_modes)
sum_X_modes = np.sum(X_modes, axis=1)
print(sum_X_modes)


#   保存数据到文件
folder_path = "C:\\Users\\s00456606\\Desktop\\trapped_ions_simulator\\2. Make an Ion simulator\\test_scripts\\data_res\\"
#   保存 model_pulse_op_X 到npy文件
np.save(folder_path + "X_modes.npy", X_modes)
np.save(folder_path + "X_freqcal.npy", X_freqcal)
print("X_modes.npy", X_modes)
print("X_freqcal.npy", X_freqcal)