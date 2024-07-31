#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH -t 1-23
#SBATCH -A pi_mparente_umass_edu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=370G
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -o output_%j.txt    # Standard output will be written to output_JOBID.txt
#SBATCH -e error_%j.txt     # Standard error will be written to error_JOBID.txt

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "CPUs per node: $SLURM_CPUS_ON_NODE"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Check allocated resources
echo "Checking allocated resources..."

# Check number of nodes
if [ $SLURM_JOB_NUM_NODES -eq 2 ]; then
    echo "Correct number of nodes allocated: 2"
else
    echo "Error: Incorrect number of nodes. Requested 2, got $SLURM_JOB_NUM_NODES"
fi

# Check number of GPUs per node
gpus_per_node=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
if [ $gpus_per_node -eq 4 ]; then
    echo "Correct number of GPUs per node allocated: 4"
else
    echo "Error: Incorrect number of GPUs per node. Requested 4, got $gpus_per_node"
fi

# Check total number of CPUs
if [ $SLURM_CPUS_ON_NODE -eq 32 ]; then
    echo "Correct number of CPUs per node allocated: 32"
else
    echo "Error: Incorrect number of CPUs per node. Requested 32, got $SLURM_CPUS_ON_NODE"
fi

# Check memory per node
mem_per_node=$(echo $SLURM_MEM_PER_NODE | sed 's/[^0-9]*//g')
mem_per_node_gb=$((mem_per_node / 1024))
if [ $mem_per_node_gb -eq 370 ]; then
    echo "Correct amount of memory per node allocated: 370G"
else
    echo "Error: Incorrect amount of memory per node. Requested 370G, got ${mem_per_node_gb}G"
fi

PYTHON_PATH=$(which python)
echo "Using Python interpreter: $PYTHON_PATH"


export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

# Run the Python script with the specified configuration
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


echo "Starting the main Python script..."
python3 -m main.py --config config_m3.yaml

    
# Set up environment (if needed)
# module load python/3.8.5
# source /path/to/your/virtual/environment/bin/activate


echo "Job completed"