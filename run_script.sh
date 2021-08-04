#!/usr/bin/env bash
#SBATCH  --mail-type=ALL                 # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=/itet-stor/jstuder/net_scratch/log_files/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/itet-stor/jstuder/net_scratch/log_files/error_files/%j.err  # where to store error messages
#SBATCH --gres=gpu:titan_xp:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
set -o errexit

source /itet-stor/jstuder/net_scratch/anaconda3/bin/activate ml_env
srun python /itet-stor/jstuder/net_scratch/nl-pl_moco/main.py --batch_size 4 --queue_size 512 --train_split_size 5 --validation_split_size 40
echo finished at: `date`
exit 0;

