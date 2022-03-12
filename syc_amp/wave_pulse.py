#   读取 AM——modulation 的 package
import numpy as np

from .pakages.quantum_dynamics_simulation.AM_optimization import *
from .pakages.quantum_dynamics_simulation.segmented_simulation import *
from .pakages.quantum_dynamics_simulation.visualization import *
import time

#   1. 设计的波形的参数；    2. 波形导入做时间演化；   3. 计算演化的保真度和cost function的比较；   4. 对噪声的抗性；

class syc_amp(object):



    def __init__(self,ion_number,j_list,omega,bij,detuning,tau,segment_num,lamb_dicke):
        ##      针对两离子，AM 波形设计优化程序 ，收集分段过程中的Rabi强度
        self.mu = detuning
        self.j_list = np.array(j_list)
        self.detuning_list  = np.array([x-detuning for x in omega])
        self.N_ions = ion_number
        self.gate_duration = tau
        self.segments_number = segment_num
        self.mode_pattern = np.array(bij)
        self.bare_eta = lamb_dicke
        self.eta = self.bare_eta * self.mode_pattern

        self.theta = np.array([[0.]*self.N_ions]*self.N_ions)
        j0 = j_list[0]
        j1 = j_list[1]
        self.theta[j0][j1] = np.pi/8
        self.theta[j1][j0] = np.pi/8
        self.theta = tf.constant(self.theta, dtype=tf.dtypes.complex128)

        print('detuning',self.detuning_list)
        print('tau',self.gate_duration)
        print('mode',self.mode_pattern)


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


    def import_amp(self):
        folder_path = "./calculation results/data/"
        self.optimized_X_import = np.load(folder_path + "optimized_X.npy")


    def get_amp(self):
        delta_t = self.gate_duration/self.segments_number
        def result_func(t,j):
            ion_i_amp_list = np.real(self.optimized_X_import[j])
            ind = int(t/delta_t)
            return ion_i_amp_list[ind]
        j0 = j_list[0]
        j1 = j_list[1]
        return {j0:lambda t:result_func(t,j0)*np.exp(-1.j*self.mu*t) , j1:lambda t:result_func(t,j1)*np.exp(-1.j*self.mu*t)}




#   0. 运行程序，调用class
#syc = pulse_optimize_and_evolution()

#   1. 初始化参数
#syc.func1_parameters(ion_number=3, segment_num=15)

#   关闭 or 开启 画图
#plotfig = False

#   2. 开始优化过程
#syc.func2_optimize_process_save_data(plotfig=plotfig, pulse_symmetry = False, ions_same_amps = False)

#   3. 导入保存的数据
#syc.func3_import_saved_data(plotfig=True, pulse_symmetry = False, ions_same_amps = False)

#  4. 开始演化验证
#syc.func4_unitary(t_step=500,cut_off=5, plotfig=True)









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



