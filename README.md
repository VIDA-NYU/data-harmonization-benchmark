<h2 align="center">Data Harmonization Benchmark</h2>

This repository contains the code and data for the Data Harmonization Benchmark. The benchmark is a collection of datasets that are used to evaluate the performance of data harmonization methods including schema matching, value mapping.

## Code Structure
> Note: datasets can be downloaded following the instructions in the next section.

```sh
|-- data_harmonization_benchmark
    |-- datasets # Put everything downloaded from the link above here
        |-- parse_valentine_benchmark.ipynb # parse valentine data format to our format
    |-- matchers # Schema matching methods
        |-- Coma
        |-- ComaInst
        |-- DistributionBased
        |-- ISResMat # X. Du et al. - In Situ Neural Relational Schema Matcher (10.1109/ICDE60146.2024.00018)
        |-- JaccardDistance
        |-- Magneto # Magneto is introduced as a method from our team, find the source code here: https://github.com/VIDA-NYU/data-integration-eval
        |-- SimilarityFlooding
        |-- Unicorn # Tu et al. Unicorn: A unified multi-tasking model for supporting matching tasks in data integration
    |-- utils
        |-- mrr.py # Mean reciprocal rank metric
        |-- result_proc.py # Process the result of schema matching methods
    |-- config.py # Configuration file, including source, target, and running configurations
    |-- matching.py # Wrapper for different matchers
    |-- runbenchmark.py # Run benchmark tasks
|-- slurm_run # SLURM scripts for running schema matching methods on server
    |-- benchmark_batch.sh # Run all schema matching methods
    |-- benchmark_scalabilty.sh # Run scalability benchmarks on various target samples
    |-- setup_penv.sh # Setup python environment with conda
    |-- slurm_job_cpu.SBATCH # SLURM job script for CPU
    |-- slurm_job_gpu.SBATCH # SLURM job script for GPU
```

## 0. Dataset Accessability
The datasets used in this benchmark are available for download via the following links:
- [Dropbox Link](https://nyu.box.com/s/k115e9tcdg33rj13ssfyruwkuyzc88uo)
- [Google Drive Link](https://drive.google.com/file/d/1MUI1BlQt-u6sxdGZOV2VtGGh6ayL5Ipl/view?usp=drive_link)

After downloading the datasets, unzip the subfolders under the `datasets` directory. The directory structure should look like this:
```sh
|-- data_harmonization_benchmark
    |-- datasets
        |-- datasets
            |-- GDC
            |-- OpenData
            |-- TPC-DI
            |-- ...
```

## 1. Schema Matching
Schema matching is the process of identifying correspondences between attributes from two database schemas. Typically, schema matching methods employ one or more functions to establish a similarity value between pairs of elements from the schemas, referred to as _matching candidates_. These functions, known as _matchers_, take two elements as input and estimate a similarity value between 0 and 1, where a higher value indicates greater similarity. Matchers can utilize a variety of strategies to estimate similarities, such as comparing schema element names, assessing their semantic similarity using a thesaurus, analyzing data types and cardinality, or even examining data values when available.

### 1.1. Supported Matchers
We support the following schema matching methods, all of them can be run on-server with SLURM or locally.

#### 1.1.1 Coma
https://github.com/delftdata/valentine/tree/master/valentine/algorithms/coma

#### 1.1.2 Coma++
https://github.com/delftdata/valentine/tree/master/valentine/algorithms/coma

#### 1.1.3 Distribution-based
https://github.com/delftdata/valentine/tree/master/valentine/algorithms/distribution_based

#### 1.1.4 Jaccard Distance
https://github.com/delftdata/valentine/tree/master/valentine/algorithms/jaccard_distance

#### 1.1.5 Similarity Flooding
https://github.com/delftdata/valentine/tree/master/valentine/algorithms/similarity_flooding

#### 1.1.6 Unicorn
https://github.com/ruc-datalab/Unicorn

#### 1.1.7 ISResMat
https://github.com/duxyad/ISResMat

#### 1.1.8 Magneto
https://github.com/VIDA-NYU/data-integration-eval