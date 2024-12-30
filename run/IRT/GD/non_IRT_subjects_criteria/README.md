# README

## Overview

This directory is designated for evaluating performance levels—상 (high), 중 (medium), and 하 (low)—for subjects where Item Response Theory (IRT) cannot be applied.

## Requirements
- **Subject List**: A comprehensive list of subjects is necessary.
- **Problem Sets**: Each subject requires 10 associated problems, generated through a `preprocesing_run_all.sh`

## Preparation
### Step 1: Generate Problem Sets
- Make sure you have ran the following script to generate problem sets for each subject:
  ```bash
  ./preprocessing_run_all.sh
  ```
- returns `../detailed/10_{subject_kr}_answer_count_taptap.csv`

### Step 2: Retrieve and Organize Problem Data
## Execution
- Open and execute the `non_IRT.py` to begin the analysis based on the provided problem data.

## Results
- The analysis results will be outputted to the `criteria_output` directory in JSON format, as illustrated below:
```json
{
  "subject1": [float1, float2],
  "subject2": [float1, float2]
}

float1 represents the threshold score for transitioning from low to medium performance.
float2 indicates the threshold score for moving from medium to high performance.

The scoring thresholds (float1 and float2) determine the performance tier of students taking the exam. Students will be classified into 상, 중, or 하 based on their scores relative to these thresholds.