#!/bin/bash
#SBATCH -o output_%j.txt    # Standard output will be written to output_JOBID.txt
#SBATCH -e error_%j.txt     # Standard error will be written to error_JOBID.txt

# Load Conda module and activate the jupyterlab-matlab environment

# Export PYTHONPATH to include project root
export PYTHONPATH=$PYTHONPATH:/home/kraju_umass_edu/Tokenizer/H3Tokenizer

# Run the main.py script with specified config
python main.py --config config_ca_diffusion.yaml
