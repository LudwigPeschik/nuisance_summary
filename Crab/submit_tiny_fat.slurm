#!/bin/bash -l
#
# start 2 MPI processes
#SBATCH --ntasks=2
#SBATCH --nodes=1
# allocate nodes for 6 hours
#SBATCH --time=24:00:00
# job name 
#SBATCH --job-name=crab_fitting
# do not export environment variables
#SBATCH --export=NONE

# do not export environment variables
unset SLURM_EXPORT_ENV
conda env list
conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gammapy-0.19
conda info
srun --mpi=pmi2 python /home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Crab/1-Nui_Par_Fitting.py 

#sbatch.tinyfat submit_tiny_fat.slurm
#squeue.tinyfat
