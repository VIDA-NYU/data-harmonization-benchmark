#!/bin/bash

module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda create -y -p ./penv_${1}_${2} python=3.9;
conda activate ./penv_${1}_${2};
conda install -y -c conda-forge openjdk;
export PATH=./penv_${1}_${2}/bin:$PATH;

if [ $2 = "gpu" ]; then
    conda install -y cuda -c nvidia;
fi


pip install -r ../requirements.txt --no-cache-dir
pip install -r ../data_harmonization_benchmark/matchers/${1}/requirements.txt --no-cache-dir