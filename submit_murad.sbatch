#!/bin/bash
#----------------------------------------------------
# Slurm job script for a Python Data Science job
# on TACC Stampede3
#----------------------------------------------------

#SBATCH -J python_job      # Job name
#SBATCH -o myjob.o%j       # Name of stdout output file
#SBATCH -e myjob.e%j       # Name of stderr error file
#SBATCH -p development     # Queue name (development is good for <2 hr jobs)
#SBATCH -N 1               # Total # of nodes (1 is sufficient for this script)
#SBATCH -n 1               # Total # of tasks (1 for a single Python script)
#SBATCH -t 00:20:00        # Run time (hh:mm:ss)
#SBATCH -A CDA24007        # Allocation name

# Change to the scratch directory where the files are
cd $SCRATCH

# Load a specific Python module
module load python/3.9.18

# Launch the Python script
python3 muradDataScience.py