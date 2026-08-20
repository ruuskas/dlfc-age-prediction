#!/bin/bash
#SBATCH --job-name=sensor_space_pipeline
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --array=0-576
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/nbe/restmeg/dlfc_age/slurm_logs/sensor_analysis_-%A_%a.out

PIPELINE_PATH=/scratch/nbe/restmeg/dlfc_age/dlfc_age_pipeline
SUBJECTS_FILE=/m/nbe/scratch/restmeg/dlfc_age/data/included_subjects.csv

ml purge
ml mamba
conda init bash >/dev/null 2>&1
source ~/.bashrc
source activate connectivity-pipeline

# Limit the minimum size of a disk-mapped array
export MNE_MEMMAP_MIN_SIZE=10M
# Set the MNE parallel cache location
export MNE_CACHE_DIR=/dev/shm
cd "$PIPELINE_PATH" || exit 1

mapfile -t subjects < <(python -c "import pandas as pd; print('\n'.join(pd.read_csv('${SUBJECTS_FILE}', index_col=0, header=0)['subject']))")

n_subjects=${#subjects[@]}
echo "Number of subjects $n_subjects."

n=$SLURM_ARRAY_TASK_ID
if [[ "$n" -ge "$n_subjects" ]]
then
  exit 0
fi

subj="${subjects[$n]}"
echo "Starting analysis for subject $subj."
srun python fc_01_compute_sensor_space_connectivity.py "$subj"
