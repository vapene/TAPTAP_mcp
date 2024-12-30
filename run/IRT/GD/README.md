
# 기출탭탭 - Instruction Guide

This document provides step-by-step instructions for running the 기출탭탭 system, from data preprocessing to model evaluation and reliability analysis.

## Getting Started

### 0.Prerequisites
Ensure you are in the correct directory before beginning the process:

```bash
conda activate lanchain
cd EduCDM/examples/IRT/GD    -> cd jw/TAPTAP_IRT/run/IRT/GD
```

### 0-1.Data Preparation
Data files are located at `EduCDM/data/full_data` and include:
- `student_response.csv`: Contains student responses.
- `cumul_csv`: Cumulative frequency data for 수능 빈출 for each subject.    -> FROM (콘텐츠 통계요청) 문제 유형에 따른 누적 문항 수_{subject} 파일
- `{문학}_middle_freq_list.json`: Specifies the number of required questions per subject and category. -> FROM 기출탭탭_AI진단문제 문제유형별 구성_{subject} 파일

### 1.Preprocessing

Run the `preprocessing_run_all.sh` script to clean and select data:
```bash
sh preprocessing_run_all.sh
```
This script performs the following actions:
- Replaces "." with "?" in data files.
- 윤리와 사상을 따로 돌리는 이유는 {middle_cat: 서양의 여러 가지 윤리 사상} 아래에 small_cat가 서양 "사상가들의 입장 비교" 하나 밖에 없기 때문이다. combination을 찾을 때, 이것을 special_cat로 놓는다. 
- Selects 10 problems for IRT.
- Outputs two main folders:
  - `detailed`: Contains `nan_ratio.csv` (indicating how many students solved the 10 problems), `p_theta_problem.csv` (probability of correct answers per problem given. This is a fixed value nation wide.), `p_theta_response.csv` (probability of correct answers per problem calculated by response data), and `question_detailed.csv` (details about the 10 problems including answer ratio and source).
  - `processed`: Contains `member_idx_mapping_dict.json` (maps reindexed member IDs to unique LMS member IDs), `after_NA.csv` (used as the training dataset), and `problem_seq_mapping_dict.json` (maps reindexed problem IDs to unique LMS problem sequences).

## Model Execution

### 2.Running IRT Models
Execute the following scripts to run the IRT models and perform hyperparameter tuning:
```bash
bash IRT_mine_NA1.sh
# bash IRT_mine_NA2.sh
```
These scripts generate:
- `results_NA1/` and `results_NA2/`: Folders containing the outcomes of the hyperparameter search and the best parameter sets saved in `best_params.txt`.

### 3.Generating Results
Run the `results.py` script to generate results files:
```bash
python results.py
```
- BEFORE running, **delete** old best_params.txt file at `results` folder. Cannot take in multiple best_params.txt files. 

This script creates:
- `NA_best_results.csv` in `results_NA1` and `results_NA2`. Shows best performances per subject.
- `student_counts.csv` in the current directory.
- `configs.yaml` to configure the model with the best hyperparameters.

### 4.Saving Model Parameters
To save the learned model parameters:
```bash
bash IRT_mine_NA1_saveIRT.sh
bash IRT_mine_NA2_saveIRT.sh
```
These scripts gives --evalution argument and save `irt.params` in each `results_NA/subject` folder.


### 5. Reliability Analysis
Execute `reliability.py` to analyze model reliability and generate plots:
```bash
python reliability.py
```
Outputs at results_NA/subject include:
- `info_comparison.csv`: Information for each p_theta and differences among themselves.
- `NA_info_plot.png`: Information plot for the subject in total.
- `NA1_information_plots.png`: Information for each question in the subject.
- `ICC_plots.png`: ICC curves for each problem in the subject.

### 6. Additional Testing
To test the model against non-augmented data:
```bash
bash IRT_mine_NA1_useOriginal.sh
bash IRT_mine_NA2_useOriginal.sh
```
This generates `useOriginal.txt` to show performance results and tests for overfitting to the augmentation method.

### 7. Create 메인.csv file   OR    Delete files in all results folders
To create a csv used for 메인 in Excel:
```bash
delete_results.ipynb
```

### 8. 기타파일
useOriginal.py: test 데이터에 증강한 데이터를 포함하지 않음. 
  

## Conclusion
Follow these steps carefully to ensure the successful execution of the 기출탭탭 processes. Each step has been designed to provide clear outputs and results for effective analysis and evaluation.
