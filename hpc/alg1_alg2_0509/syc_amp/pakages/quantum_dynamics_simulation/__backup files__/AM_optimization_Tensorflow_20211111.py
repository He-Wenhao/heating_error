"""
This module contains functions for optimizing the pulse for entangling gate based on input pulses.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import tensorflow_probability as tfp

#   来限定只使用 CPU 进行运算。
cpus = tf.config.list_physical_devices(device_type='CPU')
tf.config.set_visible_devices(devices=cpus)


#   seperate e^{i\delta_m*t} into cos\delta_m*t+i*sin\delta_m*t and then integrate seperately
#   used for calculate the cost function
def calculator_P(detuning_list, duration, number_of_segments):
    dur_seg = duration / number_of_segments  # duration of one segment
    P_real = [[0.5 * integrate.quad(lambda x: np.cos(detuning_list[m] * x), k * dur_seg, (k + 1) * dur_seg)[0] \
               for k in np.arange(number_of_segments)] for m in np.arange(len(detuning_list))]
    P_imag = [[0.5 * integrate.quad(lambda x: np.sin(detuning_list[m] * x), k * dur_seg, (k + 1) * dur_seg)[0] \
               for k in np.arange(number_of_segments)] for m in np.arange(len(detuning_list))]
    return tf.complex(np.array(P_real), np.array(P_imag))


def calculator_G_ijspecified_real(detuning_list, duration, number_of_segments, eta, i, j):
    dur_seg = duration / number_of_segments  # duration of one segment
    G_real = []
    for k in np.arange(number_of_segments):
        _middle_real = []

        for l in np.arange(k):
            middle = sum([0.25 * eta[m][i] * eta[m][j] * integrate.dblquad(lambda t1, t2: \
                                                                               np.cos(detuning_list[m] * (t1 - t2)),
                                                                           k * dur_seg, (k + 1) * dur_seg, \
                                                                           lambda t1: l * dur_seg,
                                                                           lambda t1: (l + 1) * dur_seg)[0] for m in
                          np.arange(len(detuning_list))])
            _middle_real.append(middle)

        _middle_real.append(sum([0.25 * eta[m][i] * eta[m][j] * integrate.dblquad(lambda t1, t2: \
                                                                                      np.cos(
                                                                                          detuning_list[m] * (t1 - t2)),
                                                                                  k * dur_seg, (k + 1) * dur_seg, \
                                                                                  lambda t1: k * dur_seg,
                                                                                  lambda t1: t1)[0] for m in
                                 np.arange(len(detuning_list))]))
        [_middle_real.append(0) for pad in np.arange(number_of_segments - k - 1)]
        G_real.append(_middle_real)
    return G_real


def calculator_G_ijspecified_imag(detuning_list, duration, number_of_segments, eta, i, j):
    dur_seg = duration / number_of_segments  # duration of one segment
    G_imag = []
    for k in np.arange(number_of_segments):
        _middle_imag = []

        for l in np.arange(k):
            middle = sum([0.25 * eta[m][i] * eta[m][j] * integrate.dblquad(lambda t1, t2: \
                                                                               np.sin(detuning_list[m] * (t1 - t2)),
                                                                           k * dur_seg, (k + 1) * dur_seg, \
                                                                           lambda t1: l * dur_seg,
                                                                           lambda t1: (l + 1) * dur_seg)[0] for m in
                          np.arange(len(detuning_list))])
            _middle_imag.append(middle)

        _middle_imag.append(sum([0.25 * eta[m][i] * eta[m][j] * integrate.dblquad(lambda t1, t2: \
                                                                                      np.sin(
                                                                                          detuning_list[m] * (t1 - t2)),
                                                                                  k * dur_seg, (k + 1) * dur_seg, \
                                                                                  lambda t1: k * dur_seg,
                                                                                  lambda t1: t1)[0] for m in
                                 np.arange(len(detuning_list))]))
        [_middle_imag.append(0) for pad in np.arange(number_of_segments - k - 1)]
        G_imag.append(_middle_imag)
    return G_imag


def calculator_G(detuning_list, duration, number_of_segments, eta):
    spin_number = len(detuning_list)

    G_real = [[calculator_G_ijspecified_real(detuning_list, duration, number_of_segments, eta, i, j) \
               for i in np.arange(spin_number)] for j in np.arange(spin_number)]
    G_imag = [[calculator_G_ijspecified_imag(detuning_list, duration, number_of_segments, eta, i, j) \
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
            C1 = eta * segment_amps[i] * integrate.quad(lambda x: np.cos(detuning * x), i * seg_dur, t)[0] + last_value[
                0]
            C1_arr.append(C1)
            S1 = eta * segment_amps[i] * integrate.quad(lambda x: np.sin(detuning * x), i * seg_dur, t)[0] + last_value[
                1]
            S1_arr.append(S1)
        last_value = np.array([C1_arr[-1], S1_arr[-1]])
    return np.array([C1_arr, S1_arr])


class AM_optimize():
    def __init__(self, detuning_list, gate_duration, segments_number, theta, eta):
        # 计时开始
        import time
        time_start = time.time()

        self.P = calculator_P(detuning_list, gate_duration, segments_number)  # define tensor P
        self.G = calculator_G(detuning_list, gate_duration, segments_number, eta)  # define tensor G
        self.theta = tf.constant(theta, dtype=tf.dtypes.complex128)  # transfer the target interaction
        self.train_loss_results = []  # used for record cost during training
        self.detuning_list = detuning_list  # transfer detuning
        self.gate_duration = gate_duration  # transfer gate duration
        self.Jij_coupling_op_process = []
        self.segments_number = segments_number
        self.N_ions = len(self.detuning_list)
        self.eta = eta
        self.X = tf.Variable(1.0 + np.zeros(shape=self.P.shape), trainable=True, dtype=tf.dtypes.complex128)
        self.X_zeros = [[0]*self.segments_number]*self.N_ions

        #   计时结束
        time_end = time.time()
        print('time used is:',time_end-time_start,'seconds')

    def cost_function(self, P, G, theta_target):  # 计算 theta 和 theta_target的差异值，得到 cost function
        alpha = tf.tensordot(P, self.X, [[1], [1]])
        theta_pulse = thetaproduct(G, self.X)
        # delta_theta = tf.linalg.set_diag(tf.subtract(theta_target, theta_pulse), np.zeros(len(theta_target)))
        delta_theta = tf.subtract(theta_target, theta_pulse)
        cost_function = tf.math.real(tf.norm(alpha, ord='euclidean')) \
                        + tf.math.real(tf.norm(delta_theta, ord='euclidean'))  ##  cost function，考虑 alpha 和 delta_theta的综合贡献
        return cost_function



    # #   每个离子波形  相同
    # def cost_function_value(self, x):
    #     same_amp_x = np.array([x] * self.N_ions)
    #     same_amp_x = np.reshape(same_amp_x, self.P.shape)
    #     self.X = tf.constant(same_amp_x, tf.dtypes.complex128)
    #     self.value = self.cost_function(self.P, self.G, self.theta)
    #     self.train_loss_results.append(self.value)  # 收集优化迭代中损失函数的值
    #     self.Jij_coupling_op_process.append(thetaproduct(self.G, self.X).numpy().real)  # 收集优化迭代中耦合强度的值
    #     return self.value
    #
    # def _optimizer_AM(self):  # 基于 tensorflow 的 优化算法，优化过程
    #     self.start = tf.constant(np.ndarray.flatten((self.X.numpy().real)[0]), tf.dtypes.float64)  # 优化的参数，考虑每个离子的激光参数一样
    #     self.optim_results = tfp.optimizer.nelder_mead_minimize(self.cost_function_value, initial_vertex=self.start,
    #                                                             func_tolerance=1e-18, step_sizes=1)
    #     self.optim_results_position = [self.optim_results.position.numpy()] * self.N_ions
    #     return self.optim_results_position  # 拿到最后优化完之后的 Amp 参数





    # # 每个离子的强度  不一样
    # def cost_function_value(self, x):
    #     x = np.reshape(x, self.P.shape)
    #     self.X = tf.constant(x, tf.dtypes.complex128)
    #     self.value = self.cost_function(self.P, self.G, self.theta)
    #     self.train_loss_results.append(self.value)  # 收集优化迭代中损失函数的值
    #     self.Jij_coupling_op_process.append(thetaproduct(self.G, self.X).numpy().real)  # 收集优化迭代中耦合强度的值
    #     return self.value
    #
    # def _optimizer_AM(self):  # 基于 tensorflow 的 优化算法，优化过程
    #     self.start = tf.constant(np.ndarray.flatten(self.X.numpy().real), tf.dtypes.float64)  # 优化的参数，考虑每个离子的激光参数一样
    #     self.optim_results = tfp.optimizer.nelder_mead_minimize(self.cost_function_value, initial_vertex=self.start,
    #                                                             func_tolerance=1e-18, step_sizes=1)
    #     self.optim_results_position = self.optim_results.position.numpy()
    #     return self.optim_results_position  # 拿到最后优化完之后的 Amp 参数


    def X_append_zeors(self,x):
        self.X_zeros[0] = x[0:self.segments_number]
        self.X_zeros[1] = x[self.segments_number:2 * self.segments_number]
        return self.X_zeros

    #   只考虑  两个离子  纠缠
    def cost_function_value(self, x):
        two_amps_x = np.array(self.X_append_zeors(x))
        two_amps_x = np.reshape(two_amps_x, self.P.shape)
        self.X = tf.constant(two_amps_x, tf.dtypes.complex128)
        self.value = self.cost_function(self.P, self.G, self.theta)
        self.train_loss_results.append(self.value)  # 收集优化迭代中损失函数的值
        self.Jij_coupling_op_process.append(thetaproduct(self.G, self.X).numpy().real)  # 收集优化迭代中耦合强度的值
        return self.value

    def _optimizer_AM(self):  # 基于 tensorflow 的 优化算法，优化过程
        self.start = tf.constant(np.ndarray.flatten((self.X.numpy().real)[0:2]), tf.dtypes.float64)  # 优化的参数，考虑每个离子的激光参数一样
        self.optim_results = tfp.optimizer.nelder_mead_minimize(self.cost_function_value, initial_vertex=self.start,
                                                                func_tolerance=1e-18, step_sizes=1)
        self.optim_results_position = self.X_append_zeors( self.optim_results.position.numpy() )
        return self.optim_results_position  # 拿到最后优化完之后的 Amp 参数





    def draw_optimize_process(self):
        plt.plot(self.train_loss_results, '-o', color='black')  # 优化过程的cost function画出来，看是否收敛
        plt.title('cost function value v.s. iteration steps')
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

    def trajectory_plot(self, steps):  # 画出来相空间轨迹，alpha
        self.final_result = tf.constant(np.reshape(self.optim_results_position, self.P.shape), tf.dtypes.complex128)
        self.mode_number = self.N_ions
        self.fig, self.axs = plt.subplots(self.mode_number, self.mode_number)
        self.fig.set_size_inches(8, 8)
        for i in np.arange(self.mode_number):
            for j in np.arange(self.N_ions):
                segment_amps = self.final_result.numpy().real[j]
                alpha_sci = alpha_sciint(self.detuning_list[i], segment_amps, self.gate_duration, self.eta[j][i], steps)
                # 画出相空间轨迹,分段画出来
                for seg_index in range(len(segment_amps)):
                    self.axs[i, j].plot(alpha_sci[0][seg_index * steps:(seg_index + 1) * steps],
                                        alpha_sci[1][seg_index * steps:(seg_index + 1) * steps])
                # self.axs[i,j].plot(alpha_sci[0],alpha_sci[1])
                self.axs[i, j].plot(alpha_sci[0][0], alpha_sci[1][0], 'g>', markersize=16)  # 画出起始点
                self.axs[i, j].plot(alpha_sci[0][-1], alpha_sci[1][-1], 'rs')  # 画出终止点
                self.axs[i, j].title.set_text('mode:' + str(i) + ' ion:' + str(j))
        plt.show()

    def Jij_plot(self):  # 输出离子之间 耦合强度 的计算结果
        J_coupling = thetaproduct(self.G, self.X).numpy().real
        plt.matshow(J_coupling, interpolation='kaiser')
        plt.colorbar()
        plt.show()

    def save_data(self):
        folder_path = "C:\\Users\\s00456606\\Desktop\\trapped_ions_simulator\\2. Make an Ion simulator\\calculation results\\data\\"
        #   保存 model_pulse_op_X 到npy文件
        # np.save(folder_path + "model_pulse_op_X_"+str(self.segments_number)+"_segs.npy", self.X)
        np.save(folder_path + "optimized_X.npy", self.X)
        #   保存 train_loss_results 到npy文件
        np.save(folder_path + "train_loss_results.npy", self.train_loss_results)
        #   保存 optim_results_position 到npy文件
        np.save(folder_path + "optim_results_position.npy", self.optim_results_position)
        #   保存 Jij_coupling_op_process 到npy文件
        np.save(folder_path + "Jij_coupling_op_process.npy", self.Jij_coupling_op_process)

    def print_res(self):
        J_coupling = thetaproduct(self.G, self.X).numpy().real
        theta_target = self.theta
        theta_pulse = thetaproduct(self.G, self.X)
        delta_theta = tf.subtract(theta_target, theta_pulse)
        delta_theta_amp = tf.math.real(
            tf.norm(delta_theta, ord='euclidean'))  ##  cost function，考虑 alpha 和 delta_theta的综合贡献

        alpha = tf.tensordot(self.P, self.X, [[1], [1]])
        alpha_norm = tf.math.real(tf.norm(alpha, ord='euclidean'))
        print('---------------  print results  ---------------------------------------------------------------------------')

        # print('alpha \n', alpha)
        print('alpha_norm \n', alpha_norm)

        # print('J_ij is: \n', J_coupling)
        print('target theta: \n', theta_target)
        print('optimized theta: \n', theta_pulse)
        # print('delta_theta: \n', delta_theta)
        print('delta_theta_amp: \n', delta_theta_amp)
        # print('train_loss_results \n', self.train_loss_results)
        # print(np.sqrt(sum(sum(abs(delta_theta.numpy().real)**2))))
        print('final X is \n', self.X)
        print('---------------  print results end  ---------------------------------------------------------------------------')
