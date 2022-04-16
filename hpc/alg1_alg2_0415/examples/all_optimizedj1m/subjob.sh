#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=C032M0128G
#SBATCH --qos=high
#SBATCH -J simu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

PREFIX='/gpfs/share/home/1900011306'
BIN_DIR=$PREFIX/anaconda3/envs/alg1_alg2/bin
PY_COMMAND="$BIN_DIR/python3"

echo 'python script'
$PY_COMMAND -u simu_heating.py 
echo 'done'