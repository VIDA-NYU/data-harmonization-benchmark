#!/bin/bash

usecases="GDC"
# usecases="ChEMBL MagellanHumanCurated OpenData TPC-DI WikidataHumanCurated"
matchers="Coma"
n_runs=3
top_k=20
use_gpu=true

rm -rf tmp/logs/*
for usecase in $usecases
do
    for matcher in $matchers
    do  
        if [[ $use_gpu = true ]]; then
            echo "[GPU] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}.out slurm_job_conda_gpu.SBATCH $matcher $usecase $n_runs $top_k
        else
            echo "[CPU] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}.out slurm_job.SBATCH $matcher $usecase $n_runs $top_k
        fi
    done
done