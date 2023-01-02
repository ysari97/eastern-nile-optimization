#!/bin/sh
#
#SBATCH --job-name="python_nile_opt"
#SBATCH --partition=compute
#SBATCH --time=16:20:00
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-tpm-mas
#SBATCH --output=borg_initial.out
#SBATCH --error=borg_initial.err

module load 2022r2
module load intel/oneapi-all
module load python/3.8.12

export I_MPI_PMI_LIBRARY=/cm/shared/apps/slurm/current/lib64/libpmi2.so

srun python3 optimization_borg.py
