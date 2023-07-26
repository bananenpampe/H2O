#!/bin/bash

#SBATCH --job-name=water_bpnn
#SBATCH --mail-user=philip.loche@epfl.ch
#SBATCH --mail-type=FAIL

#SBATCH --output=log.out
#SBATCH --ntasks-per-node=72
#SBATCH --mem=980GB
#SBATCH --time=10:00:00
#SBATCH --nodes=1
##SBATCH --qos=serial
#SBATCH --partition=bigmem

set -e

module load gcc python
source ~/h20_env/bin/activate

python3 ~/repos/H2O/example/training_BPNN_energy_force.py

