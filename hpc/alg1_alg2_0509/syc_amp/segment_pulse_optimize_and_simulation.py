#   读取 AM——modulation 的 package
import numpy as np

from pakages.quantum_dynamics_simulation.AM_optimization import *
from pakages.quantum_dynamics_simulation.segmented_simulation import *
from pakages.quantum_dynamics_simulation.visualization import *
import time

#   1. 设计的波形的参数；    2. 波形导入做时间演化；   3. 计算演化的保真度和cost function的比较；   4. 对噪声的抗性；

class pulse_optimize_and_evolution(object):
    def __init__(self):
        print('start...')


    #   给出 离子 + 阱 的参数
    def func1_parameters(self,ion_number=2,segment_num=10):
        if ion_number == 2:
            ##      针对两离子，AM 波形设计优化程序 ，收集分段过程中的Rabi强度
            self.detuning_list   = np.array([0.02, -0.02 ])
            self.N_ions = len(self.detuning_list)
            self.gate_duration = 200
            self.segments_number = segment_num
            self.mode_pattern = np.array([[1,1],
                                [-1,1]]) / np.sqrt(2)
            self.bare_eta = 0.1
            self.eta = self.bare_eta * self.mode_pattern

            self.theta = tf.constant(np.array([[0, np.pi/8],
                                           [np.pi/8, 0]]), dtype=tf.dtypes.complex128)

        if ion_number == 3:
             ##      针对三离子，AM 波形设计优化程序 ，收集分段过程中的Rabi强度
            self.detuning_list   = np.array([0.012, 0.046, 0.07])*2*np.pi
            self.N_ions = len(self.detuning_list)
            self.gate_duration   = 200
            self.segments_number = segment_num
            self.mode_pattern = np.array([[0.5774, 0.5774, 0.5774],
                                [-0.7071, 0, 0.7071],
                                [0.4082, -0.8165, 0.4082]])
            self.bare_eta = 0.1
            self.eta = self.bare_eta * self.mode_pattern

            # self.theta = tf.constant(np.array([[0, np.pi/8, 0],
            #                                [np.pi/8, 0, 0],
            #                                [0, 0, 0]]), dtype=tf.dtypes.complex128)

            # self.theta = tf.constant(np.array([[np.pi/8, np.pi/8, np.pi/8],
            #                                [np.pi/8, np.pi/8, np.pi/8],
            #                                [np.pi/8, np.pi/8, np.pi/8]]), dtype=tf.dtypes.complex128)

            self.theta = tf.constant(np.array([[0, 0, np.pi/8],
                                           [0, 0, 0],
                                           [np.pi/8, 0, 0]]), dtype=tf.dtypes.complex128)

        if ion_number == 4:
             ##      针对四离子，AM 波形设计优化程序 ，收集分段过程中的Rabi强度
             self.detuning_list = np.array([0.01, 0.053, 0.086, 0.11]) * 2 * np.pi
             self.N_ions = len(self.detuning_list)
             self.gate_duration = 150
             self.segments_number = segment_num
             self.mode_pattern = np.array([[0.5, 0.5, 0.5, 0.5],
                                           [-0.6742, -0.2132, 0.2132, 0.6742],
                                           [0.5, -0.5, -0.5, 0.5],
                                           [-0.2132, 0.6742, -0.6742, 0.2132]])
             self.bare_eta = 0.1
             self.eta = self.bare_eta * self.mode_pattern

             self.theta = tf.constant(np.array([[0, np.pi / 8, 0, 0],
                                                [np.pi / 8, 0, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0]]), dtype=tf.dtypes.complex128)

             self.theta = tf.constant(np.array([[0, 0, np.pi / 8, 0],
                                                [0, 0, 0, 0],
                                                [np.pi / 8, 0, 0, 0],
                                                [0, 0, 0, 0]]), dtype=tf.dtypes.complex128)

        if ion_number == 20:    # 随机产生一些数据用于测试
             ##      针对二十离子，AM 波形设计优化程序 ，收集分段过程中的Rabi强度
            self.detuning_list   = np.array([0.008, 0.002, -0.004,0.009]*5)*2*np.pi
            self.N_ions = len(self.detuning_list)
            self.gate_duration   = 70
            self.segments_number = segment_num
            self.mode_pattern = np.array([[0.8]*20]*20)
            self.bare_eta = 0.1
            self.eta = self.bare_eta * self.mode_pattern

            self.theta = tf.constant(np.array([[0.5]*20]*20), dtype=tf.dtypes.complex128)

    def hwh_init(self,ion_number,j_list,omega,bij,detuning,tau,segment_num,lamb_dicke):
        ##      针对两离子，AM 波形设计优化程序 ，收集分段过程中的Rabi强度
        self.j_list = np.array(j_list)
        self.detuning_list  = np.array([x-detuning for x in omega])
        self.N_ions = ion_number
        self.gate_duration = tau
        self.segments_number = segment_num
        self.mode_pattern = np.array(bij)
        self.bare_eta = lamb_dicke
        self.eta = self.bare_eta * self.mode_pattern

        self.theta = tf.constant(np.array([[0, 0],
                                        [0, 0]]), dtype=tf.dtypes.complex128)
        j0 = j_list[0]
        j1 = j_list[1]
        self.theta[j0][j1] = np.pi/8
        self.theta[j1][j0] = np.pi/8


    #   运行优化程序，得到 AM 设计波形的结果
    def func2_optimize_process_save_data(self, plotfig = False, pulse_symmetry = True, ions_same_amps = True):
        #   1. 开始optimization的过程
        op = AM_optimize(self.detuning_list, self.gate_duration, self.segments_number, self.theta, self.eta, pulse_symmetry = pulse_symmetry, ions_same_amps = ions_same_amps)
        op._optimizer_AM()

        #   计算相空间中alpha的值
        steps = 500
        op.trajectory_alpha_calculate(steps)

        if plotfig == True:
            # 2. 以下为画图
            #   以下为optimize process的图
            op.draw_optimize_process()
            #   以下为 相空间轨迹 s的图
            op.trajectory_plot(steps)
            #   以下为耦合强度的画图
            op.Jij_plot()

        #   3. 保存数据
        op.save_data()

        #   4. 打印优化之后的结果
        op.print_res()


    #   通过loading AM 的结果，作图分析
    def func3_import_saved_data(self, plotfig = False, pulse_symmetry = True, ions_same_amps = True):
        #   读取 存档的数据，       #   定义一个 公用的 folder path
        folder_path = "./calculation results/data/"
        self.train_loss_results_import = np.load(folder_path + "train_loss_results.npy")
        self.optimized_X_import = np.load(folder_path + "optimized_X.npy")
        self.Jij_coupling_op_process_import = np.load(folder_path + "Jij_coupling_op_process.npy")
        self.alpha_modes_simu_max_trunc = np.load(folder_path + "alpha_modes_simu_max_trunc.npy")

        #   将读取的数据赋值到 AM_optimize的class中
        op = AM_optimize(self.detuning_list, self.gate_duration, self.segments_number, self.theta, self.eta, pulse_symmetry = pulse_symmetry, ions_same_amps = ions_same_amps)
        op.train_loss_results = self.train_loss_results_import
        op.X = self.optimized_X_import
        op.Jij_coupling_op_process = self.Jij_coupling_op_process_import

        if plotfig == True:
            #   以下为计算segment的 Rabi强度的，作图
            time_list = np.linspace(0, self.gate_duration, self.segments_number+1)    #   注意特殊时间和segment之间的对应格式
            #   画出每个离子 操控波形的值
            for i in range(self.N_ions):
                ion_i_amp_list = np.real(self.optimized_X_import[i])
                plt.plot(time_list.repeat(2)[1:-1], ion_i_amp_list.repeat(2))
            plt.legend(['ion'+str(i) for i in range(self.N_ions)])
            plt.title('this is solved amplitude value (MHz)')
            plt.show()

            # # #   以下为optimize process的图
            # op.draw_optimize_process()
            # #   以下为 相空间轨迹 s的图
            # steps = 500
            # op.trajectory_alpha_calculate(steps)
            # op.trajectory_plot(steps)
            # #   以下为耦合强度的画图
            # op.Jij_plot()

        # #   打印优化之后的结果
        # op.print_res()


    #   通过 unitary evolution来计算量子态演化
    def func4_unitary(self ,t_step=1000, cut_off = 5, plotfig = False):
        #   计时开始
        time_start = time.time()
        #   定义 unitary evolution的一些初始参数
        mode_fre_base = 2 * 5 * np.pi
        H_para = {'spin_number': self.N_ions
            , 'mode_frequencies': self.detuning_list + mode_fre_base
            , 'motional_pattern': self.mode_pattern
            , 'Lamb_Dick_para': self.bare_eta
            , 'cut_off': cut_off}

        #   定义每段脉冲中 blue和red的参数
        detu_blue = mode_fre_base;  detu_red = -mode_fre_base;  phi_blue = 0;  phi_red = 0;

        #   将 optimized得到的值，按照格式写成 segmented pulse
        optimized_sequence = []
        for seg in range( self.segments_number):
            sub_sequence = []
            for ion in range(self.N_ions):
                amp = self.optimized_X_import[ion][seg].real
                sub_sequence.append(([amp, detu_blue, phi_blue], [amp, detu_red, phi_red]))
            optimized_sequence.append([np.array(sub_sequence),self.gate_duration/self.segments_number])
        optimized_sequence = np.array(optimized_sequence)

        # print(optimized_sequence)

        #   定义初始的量子态
        initial_state_spin = basis(2,0)
        initial_phonon_num = 0
        # initial_state_spin = [basis(2,0)]*self.N_ions
        # initial_state_phonon = [basis(2*cut_off, initial_phonon_num) for cut_off in self.alpha_modes_simu_max_trunc]
        # #   tensor 获得直积态
        # psi0 = tensor( initial_state_spin + initial_state_phonon )
        initial_state_phonon = basis(H_para['cut_off'], initial_phonon_num)
        #   tensor 获得直积态
        psi0 = tensor( [initial_state_spin]*self.N_ions + [initial_state_phonon]*self.N_ions )

        #   设置unitary 模拟过程的 时间切片
        nstep  = t_step
        #   按照 mesolve 演化进行求解
        output = segmented_run(H_para, psi0, optimized_sequence, nstep)
        #   得到计算的结果
        tlist = output[0]
        states = output[1]
        #   计时结束
        time_end = time.time()
        print('>>>>>>>>>> evolution time used is:',time_end-time_start,'seconds')

        if plotfig == True:
            #   以下为correlated population
            states_correlate_population = correlated_population(states, self.N_ions)
            #   画出 000,001，...，111 的 population岁时间演化的过程
            for correlated_index in range(len(states_correlate_population)):
                plt.plot(tlist,states_correlate_population[correlated_index])
            #   自动处理得到legends，000 或者 ijk 的编号 为离子的编号，    分别表示 第1个，第2个，第3个  离子
            plt.legend([format(i,'#0'+str(2+self.N_ions)+'b').replace('0b','') for i in range(2**self.N_ions)])
            plt.show()

            #   以下为 不同mode 的平均声字数
            modes_population_nbar = MS2_phonon_modes_nbar(states, self.N_ions, H_para['cut_off'])
            #   画出 不同 mode 上的 nbar的值，随时间变化的结果
            for mode_index in range(self.N_ions):
                plt.plot(tlist,modes_population_nbar[mode_index])
            plt.legend(['mode ' + str(i) for i in range(self.N_ions)])
            plt.show()


        #   计算纠缠态的保真度，注意不是严格的 Bell 态
        final_state_spin = states[-1].ptrace(np.arange(self.N_ions))
        print('>>>>>>>>>>>>>>>>>>>>   final_state_spin is: ')
        fidelities = []
        for a in range(2):
            for b in range(2):
                coeff = (-1)**a * (1.0j)**b
                bell_state = tensor( [tensor(basis(2), basis(2)) + coeff * tensor(basis(2, 1), basis(2, 1))] + [basis(2, initial_phonon_num)] * (self.N_ions - 2)).unit()
                fidelity_temp = fidelity(final_state_spin,bell_state)
                print('bell_state ' +str(coeff)+ ' fidelity is:', fidelity_temp)
                fidelities.append( [coeff, fidelity_temp] )
        # print(fidelities)
        self.state_fidelity = abs( max( np.array(fidelities)[:,1] ) )
        print('max fidelity is:', self.state_fidelity)

    def import_amp(self):
        folder_path = "./calculation results/data/"
        self.optimized_X_import = np.load(folder_path + "optimized_X.npy")


    def get_amp(self,t):
        delta_t = self.gate_duration/self.segments_number
        def result_func(t,j):
            ion_i_amp_list = np.real(self.optimized_X_import[j])
            ind = int(t/delta_t)
            return ion_i_amp_list[ind]
        j0 = j_list[0]
        j1 = j_list[1]
        return {j0:lambda t:result_func(t,j0) , j1:lambda t:result_func(t,j1)}




#   0. 运行程序，调用class
syc = pulse_optimize_and_evolution()

#   1. 初始化参数
syc.func1_parameters(ion_number=2, segment_num=15)

#   关闭 or 开启 画图
plotfig = True

#   2. 开始优化过程
syc.func2_optimize_process_save_data(plotfig=plotfig, pulse_symmetry = False, ions_same_amps = False)

#   3. 导入保存的数据
syc.func3_import_saved_data(plotfig=True, pulse_symmetry = False, ions_same_amps = False)

#  4. 开始演化验证
syc.func4_unitary(t_step=500,cut_off=5, plotfig=True)









class functions_backup():

    def coherent_state(self):
        nmax = 20
        nlist = np.linspace(0,nmax,nmax+1)

        from numpy import mean

        for alpha in range(0,10):
            alpha = alpha
            alpha_n = []
            for n in nlist:
                alpha_n.append( (np.exp(-alpha**2/2)*alpha**n / (np.sqrt(np.math.factorial(n))))**2 )
            print(sum(alpha_n * nlist))
            plt.plot(nlist,alpha_n,'bo-')
            plt.title('alpha is:'+str(alpha)+'; sum is:'+str(sum(alpha_n)))
            plt.show()


    def constant_X(self):
        #   4.0 假设按照理想的 MS gate，相空间的演化为一圈
        syc.gate_duration = 100     #   还可以设置成 100，最大纠缠态
        amp = 0.0358 * 2 * np.pi * np.sqrt(2)
        syc.segments_number = 1     #   还可以设置成 1
        syc.optimized_X_import = [[amp],[-amp]]   #   这里可以用来研究相空间的轨迹，很有意思
        # syc.segments_number = 2     #   还可以设置成 1
        # syc.optimized_X_import = [[amp, amp],[-amp, -amp]]



