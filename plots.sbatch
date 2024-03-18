#!/bin/bash

#SBATCH --job-name=plots
#SBATCH --output=slurm_out/%j-plots.out
#SBATCH --error=slurm_out/%j-plots.err
#SBATCH --account=mpcs51087
#SBATCH --time=00:40:00
#SBATCH --partition=caslake
#SBATCH --nodes=16
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=superdpsingh123@gmail.com
#SBATCH --chdir=/home/dikshant/HPC/project-1-winter-2024-dikshant293/final

module load openmpi
# module load mpich

# 1.25e-4
# echo exclusive
n=4000
dt=1.25e-4

make

mpirun --bind-to none -n 1 ./my_advection_program $n 1.0 1.0 1.414214 -1.414214 1 $dt
./pyplot.sh
mpirun --bind-to none -n 1 ./my_advection_program $n 1.0 1.0 1.414214 -1.414214 16 $dt
./pyplot.sh
mpirun --bind-to none -n 4 ./my_advection_program $n 1.0 1.0 1.414214 -1.414214 1 $dt
./pyplot.sh
mpirun --bind-to none -n 16 ./my_advection_program $n 1.0 1.0 1.414214 -1.414214 16 $dt
./pyplot.sh
