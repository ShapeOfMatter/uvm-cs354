#!/bin/bash
# specify a partition
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=2
# Request GPUs
#SBATCH --gres=gpu:8
# Request memory 
#SBATCH --mem=16G
# Maximum runtime of 10 minutes
#SBATCH --time=59:00
# Name of this job
#SBATCH --job-name=bates_cs354_a1_0
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=%x_%j.out
echo "line one!"
source ~/.bashrc

# Move to submission directory
# Should be ~/scratch/deepgreen-keras-tutorial/src
cd ${SLURM_SUBMIT_DIR}

# your job execution follows:
source .venv/bin/activate

echo "line two!"
time python ~/uvm-cs354/assignment_1/bates_cs354_a1_train1.py
echo "line three!"


