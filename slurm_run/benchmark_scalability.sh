#!/bin/bash

usecase="GDC_new/gillete"
# usecase="OpenData/Joinable/miller2_vertical_70_ec_ev"
# usecase="MagellanHumanCurated/itunes_amazon"

# matchers="Coma ComaInst JaccardDistanceMatcher DistributionBased SimilarityFlooding"
matchers="Magneto"

samples="50 100 200 300 400 500 600 700 736"
# samples="5 10 15 20 25 30 35 40 43"
# samples="1 2 3 4 5 6 7 8 9"

n_runs=1
top_k=20
use_gpu=false
output="scalability_cpu_gdc_gillete_magneto_gpt.csv"

rm -rf tmp/logs/*_sample.out

for matcher in $matchers
do
    if [[ $use_gpu = true ]]; then
        sh ./setup_penv.sh $matcher gpu
    else
        sh ./setup_penv.sh $matcher cpu
    fi
    
    for sample in $samples
    do
        if [[ $use_gpu = true ]]; then
            echo "[GPU][Target Sample] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}_${sample}_sample.out slurm_job_benchmark_scalability_gpu.SBATCH $matcher $usecase $n_runs $top_k $sample $output
        else
            echo "[CPU][Target Sample] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}_${sample}_sample.out slurm_job_benchmark_scalability_cpu.SBATCH $matcher $usecase $n_runs $top_k $sample $output
        fi
    done
done

