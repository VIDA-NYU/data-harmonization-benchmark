#!/bin/bash

# usecases="OpenData/Joinable OpenData/Semantically-Joinable OpenData/Unionable OpenData/View-Unionable TPC-DI/Joinable TPC-DI/Semantically-Joinable TPC-DI/Unionable TPC-DI/View-Unionable"
usecases="GDC_new"
# usecases="ChEMBL MagellanHumanCurated OpenData TPC-DI WikidataHumanCurated GDC_synthetic GDC_concat_columns GDC_new"
# matchers="Coma ComaInst DistributionBased JaccardDistanceMatcher SimilarityFlooding ISResMat Unicorn Magneto"
matchers="ISResMat Unicorn"
n_runs=1
top_k=20
use_gpu=false
output="results_gdc_new_isresmat_unicorn_cpu.csv"

# rm -rf tmp/logs/*

for matcher in $matchers
do
    if [[ $use_gpu = true ]]; then
        sh ./setup_penv.sh $matcher gpu
    else
        sh ./setup_penv.sh $matcher cpu
    fi
    
    for usecase in $usecases
    do
        if [[ $use_gpu = true ]]; then
            echo "[GPU] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}.out slurm_job_gpu.SBATCH $matcher $usecase $n_runs $top_k $output
        else
            echo "[CPU] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}.out slurm_job_cpu.SBATCH $matcher $usecase $n_runs $top_k $output
        fi
    done
done