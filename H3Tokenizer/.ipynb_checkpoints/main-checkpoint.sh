#!/bin/bash
#SBATCH -o output_%j.txt    # Standard output will be written to output_JOBID.txt
#SBATCH -e error_%j.txt     # Standard error will be written to error_JOBID.txt

python3 main.py --config config_ti.yaml
