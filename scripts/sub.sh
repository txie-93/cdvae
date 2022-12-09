#!/bin/bash
#SBATCH --time=59:00:00
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=singlegpu
#SBATCH --error=jobeval.err
#SBATCH --output=jobeval.out
. ~/.bashrc
export TMPDIR=/scratch/$SLURM_JOB_ID
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT=$PWD
export WABDB=$PWD/WABDB_T
export WABDB_DIR=$PWD/WABDB_T
export HYDRA_JOBS=$PWD/HYDRA_JOBS_T

cd /wrk/knc6/CDVAE/cdvae
conda activate cdvae
python cdvae/run.py data=supercon expname=supercon  model.predict_property=True
python scripts/evaluate.py --model_path  /wrk/knc6/CDVAE/c2db/cdvae/HYDRA_JOBS_T/singlerun/2022-12-03/supercon --tasks opt
