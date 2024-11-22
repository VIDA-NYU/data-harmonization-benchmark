#!/bin/bash

usecases="GDC_concat_columns"
# usecases="ChEMBL MagellanHumanCurated OpenData TPC-DI WikidataHumanCurated GDC"
matchers="ComaInst"
# matchers="Coma ComaInst DistributionBased JaccardDistanceMatcher SimilarityFlooding"
n_runs=3
top_k=20
use_gpu=true

rm -rf tmp/logs/*
for usecase in $usecases
do
    for matcher in $matchers
    do
        if [[ $use_gpu = true ]]; then
            sh ./penv_setup.sh $matcher gpu
            echo "[GPU] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}.out slurm_job_conda_gpu.SBATCH $matcher $usecase $n_runs $top_k
        else
            sh ./penv_setup.sh $matcher cpu
            echo "[CPU] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}.out slurm_job.SBATCH $matcher $usecase $n_runs $top_k
        fi
    done
done