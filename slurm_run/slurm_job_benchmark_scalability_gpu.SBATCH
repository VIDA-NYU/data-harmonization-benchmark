#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem 64GB
#SBATCH --time 48:00:00
#SBATCH --mail-user=yfw215@nyu.edu
#SBATCH --account=pr_136_tandon_advanced

module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate ./penv_${1}_gpu;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export PATH=./penv_${1}_gpu/bin:$PATH;
export OPENAI_API_KEY=sk-; # Replace with your OpenAI API key

python3 ../data_harmonization_benchmark/runbenchmark.py -m ${1} -c ../data_harmonization_benchmark/datasets/${2} -n ${3} -k ${4} --target-sample ${5} -o ${6} --use-gpu