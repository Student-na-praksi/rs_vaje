GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM5_PATH=$GEM5_ROOT/build/VEGA_X86/

APPTAINER_LOC=/d/hpc/projects/FRI/GEM5/gem5_workspace
APPTAINER_IMG=$APPTAINER_LOC/gcn-gpu_v24-0.sif



srun --ntasks=1 --time=00:30:00 --output=log_CU.txt \
	apptainer exec $APPTAINER_IMG \
	$GEM5_PATH/gem5.opt --outdir=CU_stats \
	$GEM5_ROOT/configs/example/apu_se.py \
	-n 3 --num-compute-units 4 --gfx-version="gfx902" \
	-c ./vec_add/bin/vec_add 
