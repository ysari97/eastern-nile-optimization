#!/bin/sh
#
#SBATCH --job-name="python_nile_opt"
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-tpm-mas

module load 2022r2
module load openmpi
module load python/3.8.12

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python3 baseline_optimization.py
