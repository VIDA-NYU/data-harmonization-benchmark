#!/bin/bash

usecase="GDC_new"

matchers="Unicorn"

samples="20 40"
n_runs=1
top_k=20
use_gpu=false

rm -rf tmp/logs/*_sample.out

for matcher in $matchers
do
    if [[ $use_gpu = true ]]; then
        sh ./penv_setup.sh $matcher gpu
    else
        sh ./penv_setup.sh $matcher cpu
    fi
    
    for sample in $samples
    do
        echo "[CPU][Target Sample] Run benchmark for ${matcher} on usecase:${usecase}"
        sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}_${sample}_sample.out slurm_job_target_sample.SBATCH $matcher $usecase $n_runs $top_k $sample
    done
done

    
