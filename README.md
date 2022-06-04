# Integration Data Lake Tables
Supplementary materials and Implementation codes of Integrating Data Lake Tables (ALITE)

## Abstract
Over the last decade, we have made tremendous strides in providing tools for data scientists to discover new tables useful for their tasks. But despite these advances, the proper integration of discovered tables has been under-explored. An interesting semantics for integration, called Full Disjunction, was proposed in the 1980’s, but there has been little advancement in using Full Disjunction for data science to integrate tables culled from data lakes. We provide ALITE, the first proposal for scalable integration of tables that may have been discovered using join, union or related table search. We show that ALITE can outperform (both theoretically and empirically) previous algorithms for computing the Full Disjunction. ALITE relaxes previous assumptions that tables share a common attribute names (which completely determine the join columns), are complete (without null values), and have acyclic join patterns. To evaluate our work, we develop and share three new benchmarks for integration that use real data lake tables.

## Repository Organization

- **codes** folder contains ALITE and baseline source codes.
- **statistics** folder contains the statistics of benchmarks and the time taken to integrate tables on each benchmark using different techniques.
- **README.md** file explains the repository.
- **requirements.txt** file contains necessary packages to run the project.

## Benchmark

Please visit [this link](https://drive.google.com/drive/folders/1yUgL8TjQievzp8zvmHLpa_ClNzc5mTmD?usp=sharing) to download Align Benchmark, Real Benchmark, Join Benchmark and the samples of IMDB Benchmark used in the experiments. The original IMDB benchmark is available at [https://datasets.imdbws.com/](https://datasets.imdbws.com/).

## Setup

1. Clone the repo
2. CD to the repo directory. Create and activate a virtual environment for this project  
  * On macOS or Linux:
      ```
      python3 -m venv env
      source env/bin/activate
      which python
      ```
  * On windows:
      ```
      python -m venv env
      .\env\Scripts\activate.bat
      where.exe python
      ```

3. Install necessary packages. We recommend using python version 3.7 or higher.
   ```
   pip install -r requirements.txt
   ```

## Reproducibility

1. Download benchmarks from [this link](https://drive.google.com/drive/folders/1yUgL8TjQievzp8zvmHLpa_ClNzc5mTmD?usp=sharing) and upload them to the root folder.

2. Run [align_integration_ids.py](codes/align_integration_ids.py) to run the clustering algorithm that assigns the integration ids.

3. Run [align_fd.py](codes/align_fd.py) to compute full disjunction using ALITE.

4. Run [pdelay.py](codes/pdelay.py) to compute full disjunction using BICOMNLOJ.

5. Run [parafd.py](codes/parafd.py) to compute full disjunction using ParaFD. Note that this algorithm can be used only for the tables having functional relationship.