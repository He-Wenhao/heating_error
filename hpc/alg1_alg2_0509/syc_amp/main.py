# -*- coding: utf-8 -*-
# Created by: PyQt5 UI code generator 5.11.3
#
"""
Created on 2021.04.16

@author: Shen Yangchao
"""


#   导入程序运行必须模块
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sys
# PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QPushButton, QStyleFactory
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QPropertyAnimation, QPoint, QParallelAnimationGroup,QThread,pyqtSignal
import time
from qutip import *
from threading import Thread

import pyqtgraph as pg


#   载入自己编写的静态函数
from pakages.Tab1_ion_pos_spec_cal import *
from pakages.Tab2_single_ion_cal import *
from pakages.Tab3_two_ions_cal import *

#   设置不以科学计数法输出，设置小数精度为3位
np.set_printoptions(suppress=True); np.set_printoptions(precision=3)



#   图形界面APP对象
class Main_TrapIons_simulator(object):

    def __init__(self):
        super(Main_TrapIons_simulator, self).__init__()
        self.ui = uic.loadUi("UI_files and backup/main_interface.ui")          # 从文件中加载UI定义，显示所有的界面




        ######   【Tab 1】：计算离子位置和能谱，简谐振动      ###################################################################################
        self.class_ion_pos_spectrum = Tab1_ion_pos_spectrum(self.ui)

        #   离子均衡位置 画图 & updata
        self.ui.pushButton_pos_cal.clicked.connect(self.class_ion_pos_spectrum.run_thread_ion_pos)
        #   离子能谱 画图 & updata
        self.ui.pushButton_spectrum_cal.clicked.connect(self.class_ion_pos_spectrum.run_thread_ion_spectrum)
        #   离子  运动轨迹 （radial direction） updata
        self.ui.pushButton_movement_cal.clicked.connect(self.class_ion_pos_spectrum.ion_movement_radial_figure_Update)
        #   离子  运动轨迹  (axial direction) updata
        self.ui.pushButton_movement_cal_axial.clicked.connect(self.class_ion_pos_spectrum.ion_movement_axial_figure_Update)




        ######   【Tab 2】：计算 单离子 + 声子        ###################################################################################
        self.class_single_ion_simulation = Tab2_single_ion_with_phonon(self.ui)

        #   当N_max变化的时候，更新initial phonon number的取值范围
        self.ui.phonon_N_max.valueChanged.connect(self.class_single_ion_simulation.N_max_update)

        #   二能级 Rabi updata
        self.ui.pushButton_two_level_cal.clicked.connect(self.class_single_ion_simulation.two_level_rabi_figure_Update)
        #   二能级 Rabi updata（with phonon）
        self.ui.pushButton_sideband_op_cal.clicked.connect(self.class_single_ion_simulation.run_thread)
        #   phonon distribution uodate (blue, red, carrier)
        self.ui.horizontalSlider_two_level_with_phonon.valueChanged.connect(self.class_single_ion_simulation.two_level_with_phonon_phonon_distribution)

        #   二能级 Rabi updata（with phonon）
        self.ui.pushButton_SDF_op.clicked.connect(self.class_single_ion_simulation.run_thread_SDF)





        ######   【Tab 3】：计算 2个离子的MS gate        ###################################################################################
        self.class_two_ions_MS_gate_simu = Tab3_MS_two_ions(self.ui)

        #   当N_max变化的时候，更新initial phonon number的取值范围
        self.ui.MS2_phonon_N_max.valueChanged.connect(self.class_two_ions_MS_gate_simu.N_max_update)

        #   2 ions MS gate simulation with parameters
        self.ui.pushButton_MS2_simu.clicked.connect(self.class_two_ions_MS_gate_simu.run_thread)




#   定义Tab 一个对象，用于计算离子位置、能谱、运动模式
class Tab1_ion_pos_spectrum(object):
    def __init__(self, ui):
        self.ui = ui
        self.Prepare_ion_pos_Canvas()  # 开始  离子 pos 准备画板
        self.Prepare_ion_spectrum_Canvas()  # 开始  离子 spectrum 准备画板

    #   更新参数：设置离子数目，ax和ra两个方向的阱频率
    def parameters_pos_spec(self):
        N_ions = self.ui.N_ions.value()
        N_ions = 0 if N_ions == '' else int(N_ions)
        omega_ax = self.ui.omega_ax.value()        ##  输入的轴向频率
        omega_ax = 0 if omega_ax == '' else 2 * np.pi * float(omega_ax) * MHz
        omega_ra = self.ui.omega_ra.text()        ##  输入的径向频率
        omega_ra = 0 if omega_ra == '' else 2 * np.pi * float(omega_ra) * MHz
        return N_ions, omega_ax, omega_ra


    #   计算离子的均衡位置
    def ion_pos_calculate(self):
        N_ions, omega_ax, omega_ra = self.parameters_pos_spec()  # 更新参数
        self.posdata = equ_pos(N_ions, omega_ax)  # 用于显示计算的均衡位置结果
        ##  设置combo——box来选择振动模式编号
        self.ui.mode_num_cb.clear()
        self.ui.mode_num_cb.addItems([str(i) for i in range(N_ions)])
        self.ui.mode_num_cb_axial.clear()
        self.ui.mode_num_cb_axial.addItems([str(i) for i in range(N_ions)])
        return self.posdata


    #   离子的 均衡位置，画板准备
    def Prepare_ion_pos_Canvas(self):  # 准备画图
        self.ion_pos_Figure = FigureCanvas()
        self.ion_pos_FigureLayout = QGridLayout(self.ui.ion_pos_displayGB)
        self.ion_pos_FigureLayout.addWidget(self.ion_pos_Figure)
    #   开启新的线程， 开始计算 离子的 均衡位置，text显示，并且画图
    def run_thread_ion_pos(self):
        self.ion_pos_Figure.ax.clear()  # 清除之前画的图像
        ion_pos_thread = Thread(target=self.ion_pos_figure_cal)   #   定义一个 子线程，指向画图的函数
        ion_pos_thread.start()    #   开始一个 子线程
        ion_pos_thread.join()     #   判断 子线程 是否终止
        #   在文本框中显示离子的位置
        self.posdata_array = np.array(np.squeeze(self.posdata, 1) * 1e6, dtype=float)
        self.ui.textBrowser_ion_config_show.setText('>>>>>> 离子的均衡位置为 (单位是um) :\n' + str(self.posdata_array))

        #   以下用于  均衡位置  画图
        self.x = self.posdata_array
        self.z = np.array([0] * len(self.x))
        size = np.array([100] * len(self.x))
        self.ion_pos_Figure.ax.scatter(self.x, self.z, s=size, c='black')
        self.ion_pos_Figure.ax.set_xlim(np.min(self.x) * 1.1, np.max(self.x) * 1.1)
        self.ion_pos_Figure.ax.set_ylim(-1, 1)
        self.ion_pos_Figure.ax.set_yticks([])  # 设置坐标轴显示
        self.ion_pos_Figure.draw()
    #   离子的 均衡位置，text显示，并且画图
    def ion_pos_figure_cal(self):  #####      计算离子的均衡位置，并且画出离子的位置分布图
        self.posdata = self.ion_pos_calculate()



    #   离子的 能谱，画板准备
    def Prepare_ion_spectrum_Canvas(self):  # 准备画图
        self.ion_spectrum_Figure = FigureCanvas()
        self.ion_spectrum_FigureLayout = QGridLayout(self.ui.ion_spectrum_displayGB)
        self.ion_spectrum_FigureLayout.addWidget(self.ion_spectrum_Figure)
    #   开启新的线程， 开始计算 离子的 均衡位置，text显示，并且画图
    def run_thread_ion_spectrum(self):
        self.ion_spectrum_Figure.ax.clear()  # 清除之前画的图像
        ion_pos_thread = Thread(target=self.ion_spectrum_figure_cal)   #   定义一个 子线程，指向画图的函数
        ion_pos_thread.start()    #   开始一个 子线程
        ion_pos_thread.join()     #   判断 子线程 是否终止

        self.posdata_array = np.array(np.squeeze(self.posdata, 1) * 1e6, dtype=float)
        self.ui.textBrowser_ion_config_show.setText('>>>>>> 离子的均衡位置为 (单位是um) :\n' + str(self.posdata_array) +
            '\n\n >>>>>> 径向的阱频 (单位是2*pi*MHz) :\n' + str(self.X_freqcal) + '\n\n >>>>>> 轴向的阱频 (单位是2*pi*MHz) :\n' + str(self.Z_freqcal)
          +'\n\n >>>>>> 径向的 b_matrix:\n' + str(self.X_modes) + '\n\n >>>>>> 轴向的 b_matrix:\n' + str(self.Z_modes))
        #   以下用于  均衡位置  画图
        self.p = [1] * len(self.X_freqcal)
        for i in range(0, len(self.X_freqcal)):
            self.ion_spectrum_Figure.ax.vlines(self.X_freqcal[i], 0, 0.9, colors="r", linestyles="solid")
            self.ion_spectrum_Figure.ax.vlines(self.Z_freqcal[i], 0, 0.9, colors="b", linestyles="solid")
        self.ion_spectrum_Figure.ax.set_xlim(0, max(self.X_freqcal) * 1.05)
        self.ion_spectrum_Figure.ax.set_ylim(0, 1)
        self.ion_spectrum_Figure.ax.set_yticks([])
        self.ion_spectrum_Figure.draw()
    #   离子的 能谱，text显示，并且画图
    def ion_spectrum_figure_cal(self):  #####      计算离子的均衡位置，并且画出离子的位置分布图

        N_ions, omega_ax, omega_ra = self.parameters_pos_spec()  # 更新参数
        self.X_freqcal, self.X_modes = radial_mode_spectrum(N_ions, omega_ax, omega_ra, self.posdata)
        self.Z_freqcal, self.Z_modes = axial_mode_spectrum(N_ions, omega_ax, omega_ra, self.posdata)


    #   定义一个函数：Rescale an arrary linearly
    def rescale_linear(self, array, new_min, new_max):
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        return (m * array + b).astype(int)
    #   离子的 movement 动态显示 并且画图 ,raidal，径向方向
    def ion_movement_radial_figure_Update(self):
        N_ions, omega_ax, omega_ra = self.parameters_pos_spec()  # 更新参数
        ###   得到画图的一些坐标位置
        center_w = int(self.ui.ion_movement_displayGB.x() + self.ui.ion_movement_displayGB.width() / 2)
        center_h = int(self.ui.ion_movement_displayGB.y() + self.ui.ion_movement_displayGB.height() / 2)

        figure_width = int(self.ui.ion_movement_displayGB.width() * 0.7)
        figure_height = int(self.ui.ion_movement_displayGB.height() * 0.9)
        ##  设置combo——box来选择振动模式编号
        modenum = np.int(self.ui.mode_num_cb.currentText())
        ###   获取N个离子的位置
        ions_vib_pos_w = self.posdata_array
        self.scaled_ions_vib_pos_w = self.rescale_linear(ions_vib_pos_w, center_w - figure_width / 2,
                                                         center_w + figure_width / 2)

        #  取N个变量的名字
        self.button_names = {};        annima_names = {};
        #####   设置动画组
        self.animation_group = QParallelAnimationGroup(self.ui)
        # self.animation_group.clear()

        #  给每个离子的动作定义好
        for i in range(0, N_ions):
            point_pos_w = self.scaled_ions_vib_pos_w[i]  ###  拿到N个离子 width 方向的位置
            ##  设置好圆形的按钮
            self.button_names[i] = QPushButton(self.ui)
            self.button_names[i].setGeometry(point_pos_w, center_h, 16, 16)
            # setting up border and radius
            self.button_names[i].setStyleSheet("border-radius : 8; border : 2px solid black; background-color: black")
            self.button_names[i].show()  ###  这一句非常重要，控制动画起来
            # 按钮 i 的动画
            annima_names[i] = QPropertyAnimation(self.button_names[i], b'pos', self.ui)
            for steps in range(0, 21):
                annima_names[i].setKeyValueAt(steps / 21, QPoint(point_pos_w, np.int(
                    center_h +60 + self.X_modes[i][modenum] * figure_height / 2 * np.sin(steps / 21 * 2 * np.pi))))   #这里有个很奇怪的平移 60
            annima_names[i].setEndValue(QPoint(30000, 30000))    ###  设置一个终止位置把小球移走
            self.animation_group.addAnimation(annima_names[i])
            annima_names[i].setDuration(2000)
        self.animation_group.start()
    #   离子的 movement 动态显示 并且画图 ,axial，轴向方向
    def ion_movement_axial_figure_Update(self):
        N_ions, omega_ax, omega_ra = self.parameters_pos_spec()  # 更新参数
        ###   得到画图的一些坐标位置
        center_w = int(self.ui.ion_movement_displayGB_axial.x() + self.ui.ion_movement_displayGB_axial.width() / 2)
        center_h = int(self.ui.ion_movement_displayGB_axial.y() + self.ui.ion_movement_displayGB_axial.height() / 2)

        figure_width = int(self.ui.ion_movement_displayGB_axial.width() * 0.7)
        figure_height = int(self.ui.ion_movement_displayGB_axial.height() * 0.9)
        ##  设置combo——box来选择振动模式编号
        modenum = np.int(self.ui.mode_num_cb_axial.currentText())
        ###   获取N个离子的位置
        ions_vib_pos_w = self.posdata_array
        self.scaled_ions_vib_pos_w = self.rescale_linear(ions_vib_pos_w, center_w - figure_width / 2,
                                                         center_w + figure_width / 2)
        ##  寻找一维长离子链中，邻近的两个离子之间的最小距离
        minimum_dis = figure_width
        for i in range(len(self.scaled_ions_vib_pos_w)-1):
            tem_dis = abs(self.scaled_ions_vib_pos_w[i+1] - self.scaled_ions_vib_pos_w[i])
            if tem_dis < minimum_dis:
                minimum_dis = tem_dis
        minimum_dis = minimum_dis / 2

        #  取N个变量的名字
        self.button_names = {};
        annima_names = {};
        #####   设置动画组
        self.animation_group = QParallelAnimationGroup(self.ui)
        # self.animation_group.clear()

        #  给每个离子的动作定义好
        for i in range(0, N_ions):
            point_pos_w = self.scaled_ions_vib_pos_w[i]  ###  拿到N个离子 width 方向的位置
            ##  设置好圆形的按钮
            self.button_names[i] = QPushButton(self.ui)
            self.button_names[i].setGeometry(point_pos_w, center_h, 16, 16)
            # setting up border and radius
            self.button_names[i].setStyleSheet(
                "border-radius : 8; border : 2px solid black; background-color: black")
            self.button_names[i].show()  ###  这一句非常重要，控制动画起来
            # 按钮 i 的动画
            annima_names[i] = QPropertyAnimation(self.button_names[i], b'pos', self.ui)
            for steps in range(0, 21):
                annima_names[i].setKeyValueAt(steps / 21, QPoint(
                    np.int(point_pos_w + self.X_modes[i][modenum] * minimum_dis * np.sin(steps / 21 * 2 * np.pi)),
                    center_h + 60))
            annima_names[i].setEndValue(QPoint(30000, 30000))  ###  设置一个终止位置把小球移走
            self.animation_group.addAnimation(annima_names[i])
            annima_names[i].setDuration(2000)
        self.animation_group.start()




#   定义Tab 一个对象，用于计算 单离子+ 声子
class Tab2_single_ion_with_phonon(object):
    def __init__(self, ui):
        self.ui = ui
        #   二能级simulation的初态选择
        self.ui.two_level_initial_state_ch.clear()
        self.ui.two_level_initial_state_ch.addItems(['↑', '↓'])
        #   设置 initial phonon num 的范围
        self.N_max = self.ui.phonon_N_max.value()
        self.N_max = 0 if self.N_max == '' else int(self.N_max)
        self.ui.initial_phonon_num_cb.clear()
        self.ui.initial_phonon_num_cb.addItems([str(i) for i in range(self.N_max)])
        self.initial_phonon_num = 0
        self.N_max = 10
        #   二能级系统的  Rabi 翻转  准备画板
        self.Prepare_two_level_Canvas()
        #   phonon的up和down states  准备画板
        self.Prepare_phonon_state_Canvas()
        self.time_slide_step = 0


    ##  设置初始的声子取值范围和初始值
    def N_max_update(self):
        #   获取当前N_max 和 initial phonon num 的值
        self.N_max = self.ui.phonon_N_max.value()
        self.N_max = 0 if self.N_max == '' else int(self.N_max)
        self.initial_phonon_num = self.ui.initial_phonon_num_cb.currentText()
        self.initial_phonon_num = 0 if self.initial_phonon_num == '' else int(self.initial_phonon_num)
        #   设置 initial phonon num 的范围
        self.ui.initial_phonon_num_cb.clear()
        self.ui.initial_phonon_num_cb.addItems([str(i) for i in range(self.N_max)])
        #   设置initial phonon num的值
        if self.initial_phonon_num >= self.N_max:
            self.ui.initial_phonon_num_cb.setCurrentIndex(0)
        else:
            self.ui.initial_phonon_num_cb.setCurrentIndex(self.initial_phonon_num)


    #   更新参数：设置离子内态初态，拉比频率，激光作用时长，激光频率detuning，相位，eta，最大声子数目，阱频，初始声子数等
    def parameters_update(self):
        # 离子和trap的参数
        initial_state_twolevel = self.ui.two_level_initial_state_ch.currentText()
        initial_state_twolevel = basis(2,0) if initial_state_twolevel == '↑' else basis(2,1)
        initial_phonon_num = self.ui.initial_phonon_num_cb.currentText()
        initial_phonon_num = 0 if initial_phonon_num == '' else int(initial_phonon_num)
        N_max = self.ui.phonon_N_max.value()
        N_max = 0 if N_max == '' else int(N_max)
        self.initial_phonon_num = initial_phonon_num
        self.N_max = N_max
        eta = self.ui.eta_num.value()
        eta = 0 if eta == '' else float(eta)
        v_trap = self.ui.v_trap_f.value()
        v_trap = 0 if v_trap == '' else 2*np.pi*float(v_trap)
        #   激光的参数
        Omega_2_level = self.ui.two_level_amp.value()
        Omega_2_level = 0 if Omega_2_level == '' else 2*np.pi*float(Omega_2_level)
        detu = self.ui.two_level_detu.value()
        detu = 0 if detu == '' else 2*np.pi*float(detu)
        phi = self.ui.two_level_phase.value()
        phi = 0 if phi == '' else float(phi)
        phi = 2 * np.pi * phi / 360.0  ###   动degree转换成 弧度
        tao = self.ui.two_level_duration.value()
        tao = 0 if tao == '' else float(tao)
        #   SDF的参数
        SDF_detu = self.ui.SDF_detu.value()
        SDF_detu = 0 if SDF_detu == '' else 2*np.pi*float(SDF_detu)
        SDF_amp = self.ui.SDF_amp.value()
        SDF_amp = 0 if SDF_amp == '' else 2*np.pi*float(SDF_amp)
        # return Omega_2_level, tao, detu, phi , initial_state_twolevel, eta, N_max, v_trap,SDF_detu,SDF_amp,initial_phonon_num
        return initial_state_twolevel, initial_phonon_num, N_max, eta, v_trap, Omega_2_level, detu, phi, tao, SDF_detu, SDF_amp


    #   二能级系统的Rabi翻转和bloch球上的演化，画板准备
    def Prepare_two_level_Canvas(self):
        self.two_level_rabi_figure = FigureCanvas()
        self.two_level_rabi_figureLayout = QGridLayout(self.ui.two_level_rabi_displayGB)
        self.two_level_rabi_figureLayout.addWidget(self.two_level_rabi_figure)
    #   phonon的up和down states  准备画板，画板准备
    def Prepare_phonon_state_Canvas(self):
        self.phonon_down_state_figure = FigureCanvas()
        self.phonon_down_state_figureLayout = QGridLayout(self.ui.phonon_down_state_displayGB)
        self.phonon_down_state_figureLayout.addWidget(self.phonon_down_state_figure)

        self.phonon_up_state_figure = FigureCanvas()
        self.phonon_up_state_figureLayout = QGridLayout(self.ui.phonon_up_state_displayGB)
        self.phonon_up_state_figureLayout.addWidget(self.phonon_up_state_figure)


    #   二能级系统的Rabi翻转和bloch球上的演化，画图
    def two_level_rabi_figure_Update(self):
        initial_state, initial_phonon_num, N_max, eta, v_trap, Omega_2_level, detu, phi, tao, SDF_detu, SDF_amp = self.parameters_update()   ###  更新参数
        #   以下用于  two_level_rabi 画图
        tlist = np.linspace(0, tao, np.int(tao/( 2 * np.pi * Omega_2_level )*20))
        output = two_level_calculate(initial_state, Omega_2_level, detu, phi, tlist)
        self.two_level_rabi_figure.ax.clear()  # 清除之前画的图像
        self.two_level_rabi_figure.ax.plot(tlist, output.expect[0],'-.', color='red')
        self.two_level_rabi_figure.ax.plot(tlist, output.expect[1],'-.', color='blue')
        self.two_level_rabi_figure.ax.legend(["P(↑)", "P(↓)"],loc='best')
        self.two_level_rabi_figure.draw()
        ##  以下为 画出   bloch  的动态过程
        output = two_level_calculate_bloch(initial_state, Omega_2_level, detu, phi, tlist)
        states_plot = (output.states)[::1]
        b = Bloch()
        b.add_states(states_plot)
        b.show()


    #   开启新的线程， 二能级 + phonon
    def run_thread(self):
        sideband_thread = Thread(target=self.two_level_with_phonon_figure_Update)   #   定义一个 子线程，指向画图的函数
        sideband_thread.start()    #   开始一个 子线程
        sideband_thread.join()     #   判断 子线程 是否终止

        ###  更新参数
        initial_state, initial_phonon_num, N_max, eta, v_trap, Omega_2_level, detu, phi, tao, SDF_detu, SDF_amp  = self.parameters_update()
        #   画出二能级的population震荡
        self.two_level_rabi_figure.ax.clear()  # 清除之前画的图像
        self.two_level_rabi_figure.ax.plot(self.tlist_two_level_with_phonon, self.population_individual[0][0],'-.', color='red')
        self.two_level_rabi_figure.ax.plot(self.tlist_two_level_with_phonon, self.population_individual[0][1],'-.', color='blue')
        self.two_level_rabi_figure.ax.legend(["P(↑)", "P(↓)"],loc='best')
        self.two_level_rabi_figure.draw()
        #   画出 barchart up state
        index=0
        self.phonon_up_state_figure.ax.clear()
        bar_up_data = self.phonon_population[index][0]
        self.phonon_up_state_figure.ax.bar(np.arange(0, len(bar_up_data), 1), bar_up_data, width=0.4, color='red')
        self.phonon_up_state_figure.ax.set_ylim(0, 1.05)
        self.phonon_up_state_figure.draw()
        #   画出 barchart down state
        self.phonon_down_state_figure.ax.clear()
        bar_down_data = self.phonon_population[index][1]
        self.phonon_down_state_figure.ax.bar(np.arange(0, len(bar_down_data), 1), bar_down_data, width=0.4, color='blue')
        self.phonon_down_state_figure.ax.set_ylim(0, 1.05)
        self.phonon_down_state_figure.draw()
        #   设置时间 slider 的最大和最小值
        self.ui.horizontalSlider_two_level_with_phonon.setMinimum(0)
        self.ui.horizontalSlider_two_level_with_phonon.setMaximum(len(self.tlist_two_level_with_phonon)-1)
        self.ui.horizontalSlider_two_level_with_phonon.setSingleStep(1)
        self.time_slide_step = np.float( (2 * np.pi * Omega_2_level) / 20  )
    #   计算二能级 + phonon，返回结果
    def two_level_with_phonon_figure_Update(self):  #####     二能级系统的  Rabi 翻转
        initial_state, initial_phonon_num, N_max, eta, v_trap, Omega_2_level, detu, phi, tao, SDF_detu, SDF_amp  = self.parameters_update()  ###  更新参数
        self.tlist_two_level_with_phonon, self.population_individual, self.phonon_population = two_level_with_phonon_calculate(
            Omega_2_level, tao, detu, phi, initial_state, eta, N_max, v_trap, initial_phonon_num )


    #   开启新的线程， SDF，
    def run_thread_SDF(self):
        SDF_thread = Thread(target=self.SDF_Update)   #   定义一个 子线程，指向画图的函数
        SDF_thread.start()    #   开始一个 子线程
        SDF_thread.join()     #   判断 子线程 是否终止

        ###  更新参数
        initial_state, initial_phonon_num, N_max, eta, v_trap, Omega_2_level, detu, phi, tao, SDF_detu, SDF_amp  = self.parameters_update()
        #   画出二能级的population震荡
        self.two_level_rabi_figure.ax.clear()  # 清除之前画的图像
        self.two_level_rabi_figure.ax.plot(self.tlist_two_level_with_phonon, self.population_individual[0][0],'-.', color='red')
        self.two_level_rabi_figure.ax.plot(self.tlist_two_level_with_phonon, self.population_individual[0][1],'-.', color='blue')
        self.two_level_rabi_figure.ax.legend(["P(↑)", "P(↓)"],loc='best')
        self.two_level_rabi_figure.draw()
        #   画出 barchart up state
        index=0
        self.phonon_up_state_figure.ax.clear()
        bar_up_data = self.phonon_population[index][0]
        self.phonon_up_state_figure.ax.bar(np.arange(0, len(bar_up_data), 1), bar_up_data, width=0.4, color='red')
        self.phonon_up_state_figure.ax.set_ylim(0, 1.05)
        self.phonon_up_state_figure.draw()
        #   画出 barchart down state
        self.phonon_down_state_figure.ax.clear()
        bar_down_data = self.phonon_population[index][1]
        self.phonon_down_state_figure.ax.bar(np.arange(0, len(bar_down_data), 1), bar_down_data, width=0.4, color='blue')
        self.phonon_down_state_figure.ax.set_ylim(0, 1.05)
        self.phonon_down_state_figure.draw()
        #   设置时间 slider 的最大和最小值
        self.ui.horizontalSlider_two_level_with_phonon.setMinimum(0)
        self.ui.horizontalSlider_two_level_with_phonon.setMaximum(len(self.tlist_two_level_with_phonon)-1)
        self.ui.horizontalSlider_two_level_with_phonon.setSingleStep(1)
        self.time_slide_step = np.float( (2 * np.pi * Omega_2_level) / 20  )
    #   计算二能级 + phonon，返回结果
    def SDF_Update(self):  #####     二能级系统的  Rabi 翻转
        initial_state, initial_phonon_num, N_max, eta, v_trap, Omega_2_level, detu, phi, tao, SDF_detu, SDF_amp  = self.parameters_update()  ###  更新参数
        self.tlist_two_level_with_phonon, self.population_individual, self.phonon_population = SDF_calculate(
            Omega_2_level, tao, detu, phi, initial_state, eta, N_max, v_trap, SDF_amp, SDF_detu,initial_phonon_num)


    #   计算二能级 + phonon，uodate barchart
    def two_level_with_phonon_phonon_distribution(self):  #####     二能级系统的  Rabi 翻转
        if self.time_slide_step !=0:
            #   用于控制当前演化时间的显示
            index = self.ui.horizontalSlider_two_level_with_phonon.value()
            current_time = round(index * self.time_slide_step,2)
            # print(current_time)
            self.ui.textBrowser.setText(str(current_time))
            #   画出 barchart up state
            self.phonon_up_state_figure.ax.clear()
            bar_up_data = self.phonon_population[index][0]
            self.phonon_up_state_figure.ax.bar(np.arange(0, len(bar_up_data), 1), bar_up_data, width=0.4, color='red')
            self.phonon_up_state_figure.ax.set_ylim(0, 1.05)
            self.phonon_up_state_figure.draw()
            #   画出 barchart down state
            self.phonon_down_state_figure.ax.clear()
            bar_down_data = self.phonon_population[index][1]
            self.phonon_down_state_figure.ax.bar(np.arange(0, len(bar_down_data), 1), bar_down_data, width=0.4, color='blue')
            self.phonon_down_state_figure.ax.set_ylim(0, 1.05)
            self.phonon_down_state_figure.draw()




#   定义Tab 一个对象，用于计算 单离子+ 声子
class Tab3_MS_two_ions(object):
    def __init__(self, ui):
        self.ui = ui
        #   2离子的二能级的初态选择
        self.ui.MS2_initial_state_ch.clear()
        self.ui.MS2_initial_state_ch.addItems(['↑↑', '↑↓','↓↑', '↓↓'])
        #   设置 initial phonon num 的范围
        self.N_max = self.ui.MS2_phonon_N_max.value()
        self.N_max = 0 if self.N_max == '' else int(self.N_max)
        self.ui.MS2_initial_phonon_num_cb.clear()
        self.ui.MS2_initial_phonon_num_cb.addItems([str(i) for i in range(self.N_max)])
        self.initial_phonon_num = 0
        self.N_max = 10
        #   准备四个画图的画板
        self.Prepare_ion_1_Canvas()
        self.Prepare_ion_2_Canvas()
        self.Prepare_ion_1_and_2_Canvas()
        self.Prepare_phonon_distribution_Canvas()



    ##  设置初始的声子取值范围和初始值
    def N_max_update(self):
        #   获取当前N_max 和 initial phonon num 的值
        self.N_max = self.ui.MS2_phonon_N_max.value()
        self.N_max = 0 if self.N_max == '' else int(self.N_max)
        self.initial_phonon_num = self.ui.MS2_initial_phonon_num_cb.currentText()
        self.initial_phonon_num = 0 if self.initial_phonon_num == '' else int(self.initial_phonon_num)
        #   设置 initial phonon num 的范围
        self.ui.MS2_initial_phonon_num_cb.clear()
        self.ui.MS2_initial_phonon_num_cb.addItems([str(i) for i in range(self.N_max)])
        #   设置initial phonon num的值
        if self.initial_phonon_num >= self.N_max:
            self.ui.MS2_initial_phonon_num_cb.setCurrentIndex(0)
        else:
            self.ui.MS2_initial_phonon_num_cb.setCurrentIndex(self.initial_phonon_num)

    #   更新参数：设置离子内态初态，拉比频率，激光作用时长，激光频率detuning，相位，eta，最大声子数目，阱频，初始声子数等
    def parameters_update(self):
        # 离子和trap的参数
            #   设置比特初态
        def state_to_basis(state):
            states = {
                '↑↑': [basis(2, 0), basis(2, 0)],
                '↑↓': [basis(2, 0), basis(2, 1)],
                '↓↑': [basis(2, 1), basis(2, 0)],
                '↓↓': [basis(2, 1), basis(2, 1)]
            }
            return states.get(state, None)
        initial_state_spin = self.ui.MS2_initial_state_ch.currentText()
        initial_state_spin = state_to_basis(initial_state_spin)
            #   设置phonon初态
        initial_phonon_num = self.ui.MS2_initial_phonon_num_cb.currentText()
        initial_phonon_num = 0 if initial_phonon_num == '' else int(initial_phonon_num)
            #   设置N_max最大值,eta,trap frequency 参数
        N_max = self.ui.MS2_phonon_N_max.value()
        N_max = 0 if N_max == '' else int(N_max)
        self.initial_phonon_num = initial_phonon_num
        self.N_max = N_max
        eta = self.ui.MS2_eta_num.value()
        eta = 0 if eta == '' else float(eta)
        v1_trap = self.ui.MS2_v1_trap_f.value()
        v1_trap = 0 if v1_trap == '' else 2*np.pi*float(v1_trap)
        v2_trap = self.ui.MS2_v2_trap_f.value()
        v2_trap = 0 if v2_trap == '' else 2*np.pi*float(v2_trap)

        #   激光的参数 强度，频率 相位
            #   离子1，blue sideband
        Omega_1b = self.ui.MS2_blue_amp_ion1.value()
        Omega_1b = 0 if Omega_1b == '' else 2*np.pi*float(Omega_1b)
        detu_1b = self.ui.MS2_blue_freq_ion1.value()
        detu_1b = 0 if detu_1b == '' else 2*np.pi*float(detu_1b)
        phi_1b = self.ui.MS2_blue_phase_ion1.value()
        phi_1b = 0 if phi_1b == '' else float(phi_1b)
        phi_1b = 2 * np.pi * phi_1b / 360.0  ###   动degree转换成 弧度
            #   离子1，red sideband
        Omega_1r = self.ui.MS2_red_amp_ion1.value()
        Omega_1r = 0 if Omega_1r == '' else 2*np.pi*float(Omega_1r)
        detu_1r = self.ui.MS2_red_freq_ion1.value()
        detu_1r = 0 if detu_1r == '' else 2*np.pi*float(detu_1r)
        phi_1r = self.ui.MS2_red_phase_ion1.value()
        phi_1r = 0 if phi_1r == '' else float(phi_1r)
        phi_1r = 2 * np.pi * phi_1r / 360.0  ###   动degree转换成 弧度
            #   离子2，blue sideband
        Omega_2b = self.ui.MS2_blue_amp_ion2.value()
        Omega_2b = 0 if Omega_2b == '' else 2*np.pi*float(Omega_2b)
        detu_2b = self.ui.MS2_blue_freq_ion2.value()
        detu_2b = 0 if detu_2b == '' else 2*np.pi*float(detu_2b)
        phi_2b = self.ui.MS2_blue_phase_ion2.value()
        phi_2b = 0 if phi_2b == '' else float(phi_2b)
        phi_2b = 2 * np.pi * phi_2b / 360.0  ###   动degree转换成 弧度
            #   离子2，red sideband
        Omega_2r = self.ui.MS2_red_amp_ion2.value()
        Omega_2r = 0 if Omega_2r == '' else 2*np.pi*float(Omega_2r)
        detu_2r = self.ui.MS2_red_freq_ion2.value()
        detu_2r = 0 if detu_2r == '' else 2*np.pi*float(detu_2r)
        phi_2r = self.ui.MS2_red_phase_ion2.value()
        phi_2r = 0 if phi_2r == '' else float(phi_2r)
        phi_2r = 2 * np.pi * phi_2r / 360.0  ###   动degree转换成 弧度

        #   设置激光作用的时长
        duration = self.ui.MS2_duration.value()
        duration = 0 if duration == '' else float(duration)

        return initial_state_spin, initial_phonon_num, N_max, eta, v1_trap, v2_trap, Omega_1b, detu_1b, phi_1b, Omega_1r, detu_1r, phi_1r, Omega_2b, detu_2b, phi_2b, Omega_2r, detu_2r, phi_2r, duration


    #     ion #1 spin popolation，画板准备
    def Prepare_ion_1_Canvas(self):
        self.ion_1_figure = FigureCanvas()
        self.ion_1_figureLayout = QGridLayout(self.ui.MS2_ion1_displayGB)
        self.ion_1_figureLayout.addWidget(self.ion_1_figure)
    #     ion #2 spin popolation，画板准备
    def Prepare_ion_2_Canvas(self):
        self.ion_2_figure = FigureCanvas()
        self.ion_2_figureLayout = QGridLayout(self.ui.MS2_ion2_displayGB)
        self.ion_2_figureLayout.addWidget(self.ion_2_figure)
    #     ion #1 & 2 spin popolation，画板准备
    def Prepare_ion_1_and_2_Canvas(self):
        self.ion_1_and_2_figure = FigureCanvas()
        self.ion_1_and_2_figureLayout = QGridLayout(self.ui.MS2_ion12_displayGB)
        self.ion_1_and_2_figureLayout.addWidget(self.ion_1_and_2_figure)
    #     ion #2 spin popolation，画板准备
    def Prepare_phonon_distribution_Canvas(self):
        self.phonon_distribution_figure = FigureCanvas()
        self.phonon_distribution_figureLayout = QGridLayout(self.ui.MS2_phonon_displayGB)
        self.phonon_distribution_figureLayout.addWidget(self.phonon_distribution_figure)




    #   开启新的线程， 开始计算 MS gate，并且收集数据画图
    def run_thread(self):
        MS2_thread = Thread(target=self.MS2_simulation_with_para)   #   定义一个 子线程，指向画图的函数
        MS2_thread.start()    #   开始一个 子线程
        MS2_thread.join()     #   判断 子线程 是否终止


        #   画出 【 ion1 】 的individual population
        self.ion_1_figure.ax.clear()  # 清除之前画的图像
        self.ion_1_figure.ax.plot(self.tlist, self.MS2_individual_population[0][0],'-.', color='red')
        self.ion_1_figure.ax.plot(self.tlist, self.MS2_individual_population[0][1],'-.', color='blue')
        self.ion_1_figure.ax.legend(["P(↑)", "P(↓)"],loc='best')
        self.ion_1_figure.draw()

        #   画出 【 ion2 】 的individual population
        self.ion_2_figure.ax.clear()  # 清除之前画的图像
        self.ion_2_figure.ax.plot(self.tlist, self.MS2_individual_population[1][0],'-.', color='red')
        self.ion_2_figure.ax.plot(self.tlist, self.MS2_individual_population[1][1],'-.', color='blue')
        self.ion_2_figure.ax.legend(["P(↑)", "P(↓)"],loc='best')
        self.ion_2_figure.draw()

        #   画出 【 ion1 和 ion2 】 的individual population
        self.ion_1_and_2_figure.ax.clear()  # 清除之前画的图像
        self.ion_1_and_2_figure.ax.plot(self.tlist, self.MS2_correlation_population[0],'-.', color='red')
        self.ion_1_and_2_figure.ax.plot(self.tlist, self.MS2_correlation_population[1],'-.', color='blue')
        self.ion_1_and_2_figure.ax.plot(self.tlist, self.MS2_correlation_population[2],'-.', color='orange')
        self.ion_1_and_2_figure.ax.plot(self.tlist, self.MS2_correlation_population[3],'-.', color='green')
        self.ion_1_and_2_figure.ax.legend(["P(↑↑)", "P(↑↓)", "P(↓↑)", "P(↓↓)"],loc='best')
        self.ion_1_and_2_figure.draw()

        #   画出 【 mode 上的 nbar 】
        self.phonon_distribution_figure.ax.clear()  # 清除之前画的图像
        self.phonon_distribution_figure.ax.plot(self.tlist, self.MS2_phonon_distribution_population[0],'-.', color='red')
        self.phonon_distribution_figure.ax.plot(self.tlist, self.MS2_phonon_distribution_population[1],'-.', color='blue')
        self.phonon_distribution_figure.ax.legend(["v1 mode", "v2 mode"],loc='best')
        self.phonon_distribution_figure.draw()


        # #   画出 【 X 和 P 的均值 】
        # plt.plot(self.MS2_phonon_distribution_alpha_x[0], self.MS2_phonon_distribution_alpha_p[0],'-.', color='red')
        # plt.plot(self.MS2_phonon_distribution_alpha_x[1], self.MS2_phonon_distribution_alpha_p[1],'-.', color='blue')
        # plt.legend(["v1 mode", "v2 mode"],loc='best')
        # plt.show()



        # #   【一种新的方法画图】
        # # self.curvel = self.ui.MS2_ion2_graphicsView.plot(self.tlist, self.MS2_phonon_distribution_population[1],pen=pg.mkPen('b'))
        # curve_MS2_mode_nbar = self.ui.MS2_ion2_graphicsView.plot()
        # curve_MS2_mode_nbar.clear()
        # curve_MS2_mode_nbar.setTitle("气温趋势",color='008080',size='12pt')
        # curve_MS2_mode_nbar.showGrid(x=True, y=True)
        # curve_MS2_mode_nbar.setYRange(min=-10,  max=50)
        # curve_MS2_mode_nbar.(pen=pg.mkPen('b'))
        # curve_MS2_mode_nbar.setData()

    #   调用函数计算MS gate的结果
    def MS2_simulation_with_para(self):
        #   获取mesolve的计算机国
        initial_state_spin, initial_phonon_num, N_max, eta, v1_trap, v2_trap, Omega_1b, detu_1b, phi_1b, Omega_1r, detu_1r, phi_1r, Omega_2b, detu_2b, phi_2b, Omega_2r, detu_2r, phi_2r, duration = self.parameters_update()
        output = MS2_calculate(initial_state_spin, initial_phonon_num, N_max, eta, v1_trap, v2_trap, Omega_1b, detu_1b, phi_1b,
                  Omega_1r, detu_1r, phi_1r, Omega_2b, detu_2b, phi_2b, Omega_2r, detu_2r, phi_2r, duration)
        self.tlist = output[0]
        self.states = output[1]
        #   在子线程中，计算population distribution等结果
        self.MS2_individual_population = individual_population(self.states, 2)
        self.MS2_correlation_population = correlated_population(self.states, 2)
        self.MS2_phonon_distribution_population = MS2_phonon_modes_nbar(self.states, 2,N_max)

        # #   计算相空间中的 X 和 P 的结果
        # self.MS2_phonon_distribution_alpha_x = MS2_phonon_alpha_x(self.states, 2,N_max)
        # self.MS2_phonon_distribution_alpha_p = MS2_phonon_alpha_p(self.states, 2,N_max)











#  准备画板的class，可以用于画多种图，line2D，plot，bar等
class FigureCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4.1, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=100)
        super(FigureCanvas, self).__init__(self.fig)
        self.ax = self.fig.add_axes([0.05, 0.15, 0.9, 0.85])
        self.ax.set_yticks([])


if __name__ == "__main__":
    from PyQt5.QtGui import QIcon
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)            # 初始化
    app.setStyle(QStyleFactory.create("fusion"))
    app.setWindowIcon(QIcon('icon.png'))  # 加载 icon
    SYC_Win = Main_TrapIons_simulator()
    SYC_Win.ui.show()      # 将窗口控件显示在屏幕上
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())

