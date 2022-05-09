"""
This module contains functions for optimizing the pulse for entangling gate based on input pulses.
"""
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.linalg import null_space
import time
import sympy as sp
# #   来限定只使用 CPU 进行运算。
# cpus = tf.config.list_physical_devices(device_type='CPU')
# tf.config.set_visible_devices(devices=cpus)




#   seperate e^{i\delta_m*t} into cos\delta_m*t+i*sin\delta_m*t and then integrate seperately
#   used for calculate the cost function
def calculator_P(detuning_list, duration, number_of_segments):
    dur_seg = duration / number_of_segments  # duration of one segment
    P_real = [[0.5 * integrate.quad(lambda x: np.cos(detuning_list[m] * x), k * dur_seg, (k + 1) * dur_seg)[0] \
               for k in np.arange(number_of_segments)] for m in np.arange(len(detuning_list))]
    P_imag = [[0.5 * integrate.quad(lambda x: np.sin(detuning_list[m] * x), k * dur_seg, (k + 1) * dur_seg)[0] \
               for k in np.arange(number_of_segments)] for m in np.arange(len(detuning_list))]
    # 全部采用实数矩阵，将P分写为两部分（行数加倍）
    P=[]
    for p_r in P_real:
        P.append(p_r)
    for p_i in P_imag:
        P.append(p_i)
    return P
    # return np.array(P_real) + 1.0j * np.array(P_imag)
    # return tf.complex(np.array(P_real), np.array(P_imag))
    # return tf.complex(np.array(P_imag), np.array([[0.0] * number_of_segments] * len(detuning_list)))

# 这一积分与离子序号无关，对于不同离子对唯一的区别是不同的lamb-dicke系数来作为这些积分项的权重进行（对于声子态的）求和。指标顺序为k, l, m
# 因此我们对积分项进行预处理，仅进行一遍积分操作，避免(n^2)次重复计算
def pre_process_G_integral(detuning_list, duration, number_of_segments, eta):
    dur_seg = duration / number_of_segments  # duration of one segment
    G_int_terms_real=[]
    G_int_terms_imag=[]
    for k in np.arange(number_of_segments):
        _G_mid_real=[]
        _G_mid_imag=[]
        # print(k)
        for l in np.arange(k):
            _G_mid_real.append([integrate.dblquad(lambda t1, t2: \
                                   np.cos(detuning_list[m] * (t1 - t2)),
                                   k * dur_seg, (k + 1) * dur_seg, \
                                   lambda t1: l * dur_seg,
                                   lambda t1: (l + 1) * dur_seg)[0] for m in np.arange(len(detuning_list))])
            _G_mid_imag.append([integrate.dblquad(lambda t1, t2: \
                                   np.sin(detuning_list[m] * (t1 - t2)),
                                   k * dur_seg, (k + 1) * dur_seg, \
                                   lambda t1: l * dur_seg,
                                   lambda t1: (l + 1) * dur_seg)[0] for m in np.arange(len(detuning_list))])
        _G_mid_real.append([integrate.dblquad(lambda t1, t2: \
                                            np.cos(detuning_list[m] * (t1 - t2)),
                                            k * dur_seg, (k + 1) * dur_seg, \
                                            lambda t1: k * dur_seg,
                                            lambda t1: t1)[0] for m in np.arange(len(detuning_list))])
        _G_mid_imag.append([integrate.dblquad(lambda t1, t2: \
                                            np.sin(detuning_list[m] * (t1 - t2)),
                                            k * dur_seg, (k + 1) * dur_seg, \
                                            lambda t1: k * dur_seg,
                                            lambda t1: t1)[0] for m in np.arange(len(detuning_list))])
        [_G_mid_real.append(0) for pad in np.arange(number_of_segments - k - 1)]
        [_G_mid_imag.append(0) for pad in np.arange(number_of_segments - k - 1)]
        G_int_terms_real.append(_G_mid_real)
        G_int_terms_imag.append(_G_mid_imag)
    return G_int_terms_real, G_int_terms_imag
                          

def calculator_G_ijspecified_real(G_int_terms_real, detuning_list, duration, number_of_segments, eta, i, j):
    dur_seg = duration / number_of_segments  # duration of one segment
    G_real = []
    for k in np.arange(number_of_segments):
        _middle_real = []

        for l in np.arange(k+1):
            middle = sum([0.25 * eta[m][i] * eta[m][j] * G_int_terms_real[k][l][m] for m in
                          np.arange(len(detuning_list))])
            _middle_real.append(middle)
        
        [_middle_real.append(0) for pad in np.arange(number_of_segments - k - 1)]
        G_real.append(_middle_real)
    return G_real

def calculator_G_ijspecified_imag(G_int_terms_imag, detuning_list, duration, number_of_segments, eta, i, j):
    dur_seg = duration / number_of_segments  # duration of one segment
    G_imag = []
    for k in np.arange(number_of_segments):
        _middle_imag = []

        for l in np.arange(k+1):
            middle = sum([0.25 * eta[m][i] * eta[m][j] * G_int_terms_imag[k][l][m] for m in
                          np.arange(len(detuning_list))])
            _middle_imag.append(middle)

        [_middle_imag.append(0) for pad in np.arange(number_of_segments - k - 1)]
        G_imag.append(_middle_imag)
    return G_imag

def calculator_G(detuning_list, duration, number_of_segments, eta):
    spin_number = len(detuning_list)
    
    G_int_terms_real, G_int_terms_imag = pre_process_G_integral(detuning_list, duration, number_of_segments, eta)

    G_real = [[calculator_G_ijspecified_real(G_int_terms_real, detuning_list, duration, number_of_segments, eta, i, j) \
               for i in np.arange(spin_number)] for j in np.arange(spin_number)]
               #for [i,j] in
    G_imag = [[calculator_G_ijspecified_imag(G_int_terms_imag, detuning_list, duration, number_of_segments, eta, i, j) \
               for i in np.arange(spin_number)] for j in np.arange(spin_number)]
    return tf.complex(np.array(G_real), np.array(G_imag))


#   alpha = P_i^k*X_j^k for X and P, axis=0 is the spin index, axis=1 is the segment index
#   \theta= G_{ij}^{kl}X_i^kX_j^l*  for G, axis=0,1 is spin index, axis=2,3 is segment index
#   used to calculate \theta, and calculation of theta is not a contraction
def thetaproduct(A, B):
    thetamat = [[tf.tensordot(tf.tensordot(A[i, j], B[i], [[0], [0]]), tf.math.conj(B[j]), [[0], [0]]).numpy().imag \
                 for i in np.arange(len(A))] for j in np.arange(len(A))]
    return tf.constant(thetamat, dtype=tf.dtypes.complex128)


#   used for visualiaztion,draw the phase space diagram
def alpha_sciint(detuning, segment_amps, duration, eta, steps=100):
    seg_dur = duration / len(segment_amps)
    C1_arr = []
    S1_arr = []
    last_value = [0, 0]  # 每一段积分之后的alpha的最后一个值， real part 和 imag part
    for i in np.arange(len(segment_amps)):
        for t in np.linspace(i * seg_dur, (i + 1) * seg_dur, steps):
            C1 = 0.5 * eta * segment_amps[i] * integrate.quad(lambda x: np.cos(detuning * x), i * seg_dur, t)[0] + last_value[
                0]
            C1_arr.append(C1)
            S1 = 0.5 * eta * segment_amps[i] * integrate.quad(lambda x: np.sin(detuning * x), i * seg_dur, t)[0] + last_value[
                1]
            S1_arr.append(S1)
        last_value = np.array([C1_arr[-1], S1_arr[-1]])
    return np.array([C1_arr, S1_arr])


class AM_optimize():
    def __init__(self, detuning_list, gate_duration, segments_number, theta, eta, pulse_symmetry = True, ions_same_amps = True):
        # 计时开始
        time_start = time.time()
        self.P = calculator_P(detuning_list, gate_duration, segments_number)  # define tensor P
        self.G = calculator_G(detuning_list, gate_duration, segments_number, eta)  # define tensor G
        self.theta = tf.constant(theta, dtype=tf.dtypes.complex128)  # transfer the target interaction
        self.ions_on_index = self.theta_non_zero_index()     #   通过theta的值，看哪些离子上应该打光
        self.train_loss_results = []  # used for record cost during training
        self.detuning_list = detuning_list  # transfer detuning
        self.gate_duration = gate_duration  # transfer gate duration
        self.Jij_coupling_op_process = []
        self.segments_number = segments_number
        self.N_ions = len(self.detuning_list)       # 离子总数
        self.entangled_ions_num = len(self.ions_on_index)          # 用于约束离子链中多少个离子上打光，这个值应该比总离子数目小
        self.eta = eta
        self.pulse_symmetry = pulse_symmetry       #   设计波形是否考虑时间上对称性
        self.ions_same_amps = ions_same_amps      #   将照射离子的光强度设置成一样
        self.segments_num_even = True if self.segments_number % 2 == 0 else False     # 判断segment数量为 奇数偶数
        self.X_init = self.X_initial(symmetry = self.pulse_symmetry, ions_same_amps = self.ions_same_amps)   # P的shape 为N个离子，每个离子 N_seg段参数
        self.X_zeros = np.array( [[0.0]*self.segments_number] * self.N_ions, dtype=float )
        #   计时结束
        time_end = time.time()
        print('>>>>>>>>>> P and G calculate time used is:',time_end-time_start,'seconds')
        #print('P = ',self.P)
        #print('G = ',self.G)

    def theta_non_zero_index(self):
        theta_tri_up = np.triu( self.theta,1 )      #   将矩阵转成上三角，考虑到耦合的对称性
        non_zero_index = np.nonzero(theta_tri_up)       #   寻找非零元素，得到指引
        ions_on_index = np.unique(np.array(non_zero_index))     #   通过删除重复元素的方法，获得哪些离子上需要打光
        print('this ions should have laser shined: ', ions_on_index)
        return ions_on_index


    #   设置初始的 离子 光强 amplitude
    def X_initial(self, symmetry = False, ions_same_amps = True):
        ions_on_num = 1 if ions_same_amps == True else self.entangled_ions_num

        if symmetry == False:
            res = np.array( [[1.0]*self.segments_number] * ions_on_num )
        if symmetry == True:
            if self.segments_num_even == True:
                res = np.array( [[1.0]*int(self.segments_number/2)] * ions_on_num )
            else:
                res = np.array( [[1.0]*(int(self.segments_number/2)+1)] * ions_on_num )
        return res


    #   考虑采用对称波形的时候，优化参数减少，补充序列
    def symmetry_amps_x_extend(self, x, symmetry=False):
        res_x = []
        for sub_x in x:
            if symmetry == True:
                if self.segments_num_even == True:
                    res_x = np.append(res_x, sub_x)
                    res_x = np.append(res_x, sub_x[::-1])
                else:
                    res_x = np.append(res_x, sub_x)
                    res_x = np.append(res_x, sub_x[0:-1][::-1])
            else:
                res_x = x
        return np.ndarray.flatten( np.array(res_x) )


    #   对于优化的参数 x 进行修补，只考虑纠缠的离子上打光
    def X_append_zeors(self,x):
        x = np.reshape(x, self.X_init.shape)
        x = self.symmetry_amps_x_extend(x, symmetry=self.pulse_symmetry)    #   考虑pulse symmetry，将x的长度补齐
        if self.ions_same_amps == True:
            x = [x]*self.entangled_ions_num     #   考虑离子的光强是否相同
        x = np.reshape(x, (self.entangled_ions_num, self.segments_number))
        for index in range(len(self.ions_on_index)):        #   将x中的元素带入到出事的 [0] 列表中
            self.X_zeros[self.ions_on_index[index]] = x[index]
        return self.X_zeros


    #   定义优化的cost function，delta_theta 的贡献
    def cost_function(self, P, G, theta_target,eta):
        #   修改了 alpha的计算，将eta考虑进去
        alpha = eta * tf.tensordot(P, self.X, [[1], [1]])
        ##  cost function，考虑 alpha 的贡献
        self.cost_func_loss_value_alpha = tf.math.real(tf.norm(alpha, ord=2))
        return float(self.cost_func_loss_value_alpha )

    #   计算离子的cost function的值，只考虑纠缠的离子上打光
    def cost_function_value(self, x):
        ions_amp_x = np.array(self.X_append_zeors(x))
        ions_amp_x = np.reshape(ions_amp_x, self.P.shape)
        self.X = tf.constant(ions_amp_x, tf.dtypes.complex128)
        self.value = self.cost_function(self.P, self.G, self.theta,eta = self.eta)
        self.train_loss_results.append(self.value)  # 收集优化迭代中损失函数的值
        self.Jij_coupling_op_process.append(thetaproduct(self.G, self.X).numpy().real)  # 收集优化迭代中耦合强度的值
        return self.value

    #   对于 theta 的取值，设置约束方程
    def constrains(self, x):
        ions_amp_x = np.array(self.X_append_zeors(x))
        ions_amp_x = np.reshape(ions_amp_x, self.P.shape)
        self.X = tf.constant(ions_amp_x, tf.dtypes.complex128)
        #   计算 theta 的差异值
        theta_pulse = thetaproduct(self.G, self.X)
        # delta_theta = tf.subtract(self.theta, theta_pulse)
        delta_theta = tf.linalg.set_diag(tf.subtract(self.theta, theta_pulse), np.zeros(len(self.theta)))
        # return delta_theta.numpy().real[0][1]
        delta_theta_res = tf.norm(delta_theta, ord=1)
        return delta_theta_res.numpy().real


    # 基于 scipy.optimize.minimize 的 优化算法，优化过程
    def _optimizer_AM(self):
        # self.start = np.ndarray.flatten(self.X_init) # 优化的参数，考虑每个离子的激光参数 不一样
        #   计时开始
        time_start = time.time()
        bnd_value = 0.5
        #bnds = tuple([(-bnd_value, bnd_value) for index in range( len(self.start) )])
        
        if(self.entangled_ions_num != 2):
            print("This method applies to two-qubit entanglement only. Please try other optimizations.")
            return
        
        # 使用linalg.nullspace()函数求解null空间的基矢并进行后续的矩阵运算
        G_triang = sp.Matrix(np.imag(self.G[self.ions_on_index[0]][self.ions_on_index[1]]))
        G_symm = (G_triang + G_triang.transpose())/2
        P_mat = sp.Matrix(self.P)
        
        # 假定在两离子上打相同波形的激光
        
        Null_space = P_mat.nullspace()
        Null_space = sp.GramSchmidt(Null_space)# 正交化
        Null_vecs = []
        for nv in Null_space:
            _norm_nv=nv/nv.norm()  # 归一化
            Null_vecs.append(_norm_nv.transpose().tolist()[0]) #将Null_space中每个基矢以list形式存下来
        
        # 计算约化矩阵V，其等价于将G投影到P的null空间上，新的基矢为Null_vec
        V_reducedG = []
        for k in np.arange(len(Null_vecs)):
            _row = []
            _mid = sp.Matrix(Null_vecs[k]).transpose()*G_symm
            for l in np.arange(len(Null_vecs)):
                _row.append( (_mid * sp.Matrix(Null_vecs[l]))[0] )
            V_reducedG.append(_row)
        #print(V_reducedG)
        # 计算V的本征值并求出所需最小功率的激光波形
        V_eigenvects=sp.Matrix(V_reducedG).eigenvects()
        Eigvals = [ev[0] for ev in V_eigenvects]
        Abs_eigvals = np.abs(Eigvals)
        Eigvects = [ev[2][0].transpose().tolist()[0] for ev in V_eigenvects]
        max_eig_abs = np.max(Abs_eigvals)
        _ind = Abs_eigvals.tolist().index(max_eig_abs)
        Pulse_eigenvect = Eigvects[_ind]
        
        Pulse_shape = np.zeros(len(Null_vecs[0]))
        for k in np.arange(len(Null_vecs)):
            Pulse_shape = Pulse_shape + Pulse_eigenvect[k]*np.array(Null_vecs[k])
        Pulse_shape = np.sqrt(np.double(self.theta[self.ions_on_index[0],self.ions_on_index[1]].numpy()/max_eig_abs))*Pulse_shape
            
        self.X = np.array( [[0.0]*self.segments_number] * self.N_ions, dtype=float )
        self.X[self.ions_on_index[0]] = Pulse_shape
        self.X[self.ions_on_index[1]] = Pulse_shape
        
        #from scipy.optimize import minimize
        #   1. 测试发现，如果用Nealder-Mead的方法，bounds太小就会出现问题。
        # self.optim_results = minimize(self.cost_function_value, self.start, method='Nelder-Mead', bounds=bnds, tol=1e-18)
        # #   2. 测试发现，使用SLSQP，对于segment增加，结果依然正确
        # self.optim_results = minimize(self.cost_function_value, self.start, method='SLSQP', bounds=bnds, tol=1e-18)
        #   3. 采用SLSQP的方法，外加 constrain 对 theta 进行约束
        #self.optim_results = minimize( self.cost_function_value, self.start, method='SLSQP', bounds=bnds, constraints= {'type':'eq','fun':self.constrains}, tol=1e-18 )
        # #   4. 寻找 global minimum
        # from scipy.optimize import basinhopping
        # self.optim_results = basinhopping( self.cost_function_value, self.start )
        #   计时结束
        time_end = time.time()
        print('>>>>>>>>>> optimization time used is:',time_end-time_start,'seconds')
        return self.X  # 拿到最后优化完之后的 Amp 参数


    # 优化过程的cost function画出来，看是否收敛
    def draw_optimize_process(self):
        plt.plot(self.train_loss_results, '-o', color='black')
        plt.title('cost function value vs. iteration steps')
        plt.show()
        #   画出随着iteration过程 J_ij的变化
        legends = []
        for i in range(self.N_ions):
            for j in range(self.N_ions):
                plt.plot(np.array(self.Jij_coupling_op_process)[:, i, j])
                legends.append(str(i) + str(j))
        #   添加legends
        plt.legend(legends)
        plt.xlabel('iteration steps')
        plt.title('J_ij coupling stength, or theta_ij')
        plt.show()

    #   计算出 alpha_j^m，在所有的steps下的值
    def trajectory_alpha_calculate(self,steps):  # 画出来相空间轨迹，alpha
        self.final_result = tf.constant(np.reshape(self.X, self.P.shape), tf.dtypes.complex128)
        self.mode_number = self.N_ions
        self.alpha_modes_simu_max_trunc = []       # 获取 alpha的最大值
        self.alpha_m_j_values = []
        for m in np.arange(self.mode_number):
            temp_alpha_mode_max = []
            alpha_fix_m_j_values = []
            for j in np.arange(self.N_ions):
                segment_amps = self.final_result.numpy().real[j]
                alpha_values = alpha_sciint(self.detuning_list[m], segment_amps, self.gate_duration, self.eta[m][j], steps)
                alpha_fix_m_j_values.append(alpha_values)
                 #   计算 最大的 alpha
                temp_alpha_mode_max.append( max(np.sqrt( alpha_values[0] ** 2 + alpha_values[1] ** 2 )) )
            self.alpha_m_j_values.append(alpha_fix_m_j_values)
            self.alpha_modes_simu_max_trunc.append( int( 2*(max(temp_alpha_mode_max))**2)+1 )
        print('alpha max are:', self.alpha_modes_simu_max_trunc, ' for phonon cut-off in simulation')

    #   画出相空间的轨迹图，N*N个图，alpha_j^m
    def trajectory_plot(self, steps):  # 画出来相空间轨迹，alpha
        # 不再作trajectory_alpha_calculate的调用，因此在这一函数里补充定义
        self.mode_number=self.N_ions
        #self.final_result = tf.constant(np.reshape(self.X, self.P.shape), tf.dtypes.complex128)
        self.final_result = tf.constant(self.X)
        
        self.fig, self.axs = plt.subplots(self.mode_number, self.mode_number, constrained_layout=False)
        self.fig.set_size_inches(8, 8)
        for m in np.arange(self.mode_number):
            for j in np.arange(self.N_ions):
                segment_amps = self.final_result.numpy().real[j]
                alpha_sci = self.alpha_m_j_values[m][j]
                # 画出相空间轨迹,分段画出来
                for seg_index in range(len(segment_amps)):
                    self.axs[m, j].plot(alpha_sci[0][seg_index * steps:(seg_index + 1) * steps],
                                        alpha_sci[1][seg_index * steps:(seg_index + 1) * steps])
                self.axs[m, j].plot(alpha_sci[0][0], alpha_sci[1][0], 'g>', markersize=16)  # 画出起始点
                self.axs[m, j].plot(alpha_sci[0][-1], alpha_sci[1][-1], 'rs')  # 画出终止点
                self.axs[m, j].title.set_text('mode:' + str(m) + ' ion:' + str(j))
        #   在 subplots 之外画出legends
        labels = ['seg '+str(seg) for seg in range(self.segments_number)]
        self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.9)  ##  Need to play with this number.
        self.fig.legend(labels=labels, loc="center right", ncol=1)
        plt.show()

    def Jij_plot(self):  # 输出离子之间 耦合强度 的计算结果
        J_coupling = thetaproduct(self.G, self.X).numpy().real
        plt.matshow(J_coupling, interpolation='kaiser')
        plt.colorbar()
        plt.show()

    def save_data(self):
        folder_path = "./algebraic_results/"
        # folder_path = "new_calculation_results\\" # 考虑chi中的正弦函数进行优化的结果
        #   保存 model_pulse_op_X 到npy文件
        np.save(folder_path + "optimized_X.npy", self.X)
        #   保存 train_loss_results 到npy文件
        #np.save(folder_path + "train_loss_results.npy", self.train_loss_results)
        #   保存 Jij_coupling_op_process 到npy文件
        #np.save(folder_path + "Jij_coupling_op_process.npy", self.Jij_coupling_op_process)
        #   保存 alpha_modes_simu_max_trunc 到npy文件
        # np.save(folder_path + "alpha_modes_simu_max_trunc.npy", self.alpha_modes_simu_max_trunc)

    def print_res(self):
        theta_target = self.theta
        theta_pulse = thetaproduct(self.G, self.X)
        delta_theta = tf.subtract(theta_target, theta_pulse)
        delta_theta_amp = tf.math.real(
            tf.norm(delta_theta, ord= 2 ))  ##  cost function，考虑 alpha 和 delta_theta的综合贡献
        alpha = self.eta * tf.tensordot(self.P, self.X, [[1], [1]])

        print('---------------  print results  ---------------------------------------------------------------------------')
        # print('alpha \n', alpha)
        print('alpha_norm \n', self.cost_func_loss_value_alpha)
        print('target theta: \n', theta_target)
        print('optimized theta: \n', theta_pulse)
        # print('delta_theta: \n', delta_theta)
        # print('delta_theta_amp: \n', self.cost_func_loss_value_theta)
        print('final train_loss_results \n', self.train_loss_results[-1])
        # print(np.sqrt(sum(sum(abs(delta_theta.numpy().real)**2))))
        print('final X is \n', self.X)
        print('self.optim_results.x', self.optim_results.x)
        # print('self.start', self.start)
        print('---------------  print results end  ---------------------------------------------------------------------------')












