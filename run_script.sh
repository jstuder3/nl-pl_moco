#!/usr/bin/env bash
#SBATCH  --mail-type=ALL                 # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=/itet-stor/jstuder/net_scratch/log_files/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/itet-stor/jstuder/net_scratch/log_files/error_files/%j.err  # where to store error messages
#SBATCH --gres=gpu:geforce_rtx_3090:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
set -o errexit

source /itet-stor/jstuder/net_scratch/anaconda3/bin/activate ml_env
srun python /itet-stor/jstuder/net_scratch/nl-pl_moco/main_pl_new_ds.py --augment --shuffle --num_epochs=20 --learning_rate=2e-5 --debug_data_skip_interval 10 --effective_queue_size=16384 --effective_batch_size=64 --num_gpus=4 --base_data_folder="/itet-stor/jstuder/net_scratch/nl-pl_moco/datasets/CodeSearchNet" --accelerator="ddp"
echo finished at: `date`
exit 0;

