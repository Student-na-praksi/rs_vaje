#!/bin/sh
#SBATCH --job-name=gem5_simulation    
#SBATCH --output=gem5_log.txt     
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

#SBATCH --reservation=fri 

GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM5_PATH=$GEM5_ROOT/build/RISCV_ALL_RUBY

srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM5_PATH/gem5.opt --help | grep -i vlen


srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM5_PATH/gem5.opt --outdir=vector_bench cpu_benchmark.py
