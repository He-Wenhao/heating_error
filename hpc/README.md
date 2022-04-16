提交任务
sbatch subjob.sh

查看排队
squeue -u `whoami`
若ST 是PD(pending)则正在排队,若是R(running)正在运行

杀死排队
scancel PID

真正跑出正常结果的三个文件夹:

all_optimized

all_optimizedj1-1_modified

all_optimizedjmm_mid_mu