#!/usr/bin/env bash
#SBATCH  --mail-type=ALL                 # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=/itet-stor/jstuder/net_scratch/log_files/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/itet-stor/jstuder/net_scratch/log_files/error_files/%j.err  # where to store error messages
#SBATCH --gres=gpu:geforce_rtx_3090:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
set -o errexit

source /itet-stor/jstuder/net_scratch/anaconda3/bin/activate ml

srun python /itet-stor/jstuder/net_scratch/nl-pl_moco/xMoCo_pl.py --effective_batch_size 32 --effective_queue_size=8192 --language="ruby" --debug_data_skip_interval=1 --shuffle --always_use_full_val --num_hard_negatives=8 --barlow_projector_dimension=0 --barlow_weight=7e-5 --barlow_lambda=0.005 --use_barlow

echo finished at: `date`
exit 0;

