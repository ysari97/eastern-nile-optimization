#!/bin/sh
#
#SBATCH --job-name="python_nile_opt"
#SBATCH --partition=compute
#SBATCH --time=00:01:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-tpm-mas
#SBATCH --output=borg_initial.out
#SBATCH --error=borg_initial.err

module load 2022r2
module load intel/oneapi-all
module load python/3.8.12


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export I_MPI_PMI_LIBRARY=/cm/shared/apps/slurm/current/lib64/libpmi2.so

mpirun python3 optimization_borg.py
