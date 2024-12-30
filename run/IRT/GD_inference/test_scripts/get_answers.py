import pandas as pd
import numpy as np
import random
import torch
import argparse
from scipy.stats import mode
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import yaml
import json
import warnings
import itertools
import os
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')
from types import SimpleNamespace


def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  


set_seed(42)

# Define all subjects with potential sub-subjects separated by commas
# all_subjects = [
#     '수학Ⅰ,수학Ⅱ', '독해,듣기', '독서,문학', '물리학Ⅰ', '화학Ⅰ',
#     '생명과학Ⅰ', '지구과학Ⅰ', '지구과학Ⅱ', '생명과학Ⅱ', '화학Ⅱ',
#     '물리학Ⅱ', '한국사', '생활과 윤리', '정치와 법', '한국지리',
#     '세계지리', '사회?문화', '동아시아사', '세계사', '윤리와 사상', '경제']
all_subjects = ['사회?문화']


###### import original datasets
category = pd.read_csv(f'/root/jw/TAPTAP_IRT/data/full_data/perfectexam_category_202404031323.csv', encoding='cp949')
problem = pd.read_csv(f'/root/jw/TAPTAP_IRT/data/full_data/perfectexam_problem_202404031323.csv', encoding='cp949')
member_exam_answer = pd.read_csv(f'/root/jw/TAPTAP_IRT/data/full_data/perfectexam_member_exam_answer_202404031323.csv', encoding='cp949')
member_exam = pd.read_csv(f'/root/jw/TAPTAP_IRT/data/full_data/perfectexam_member_exam_202404031323.csv', encoding='cp949')
##### 여기서 시험 범위 제한 가능 ######
exam_in_range = pd.read_csv(f'/root/jw/TAPTAP_IRT/data/full_data/perfectexam_exam_202404031323.csv', encoding='cp949')

category_simple = category[['code', 'code_name']]
problem_simple = problem[['seq', 'subject', 'big_category', 'middle_category','small_category','ref_seq','point','answer_rate',"set_type", "type1", "type2", "num1", "ref_num", "represent_yn", "answer"]]
member_exam_answer_simple = member_exam_answer[['member_idx', 'member_exam_seq', 'problem_seq', 'answer_dttm','correct_yn']]
member_exam_simple = member_exam[['seq', 'exam_seq', 'title']]
first_merge = member_exam_answer_simple.merge(member_exam_simple, left_on='member_exam_seq', right_on='seq', how='left').drop(columns='seq')
second_merge = first_merge[first_merge['exam_seq'].isin(exam_in_range['seq'])]
third_merge = second_merge.merge(exam_in_range[['seq', 'title']], left_on='exam_seq', right_on='seq', how='left').drop(columns=['seq', 'exam_seq', 'title_x'])
member_exam_answer_in_range = third_merge.merge(problem_simple, left_on='problem_seq', right_on='seq', how='left').drop(columns='seq')
category_dict = pd.Series(category_simple['code_name'].values, index=category_simple['code']).to_dict()
for column in ['subject', 'big_category', 'middle_category','small_category']:
    member_exam_answer_in_range[column] = member_exam_answer_in_range[column].map(category_dict)


###### get first attemps
filtered_data_sorted = member_exam_answer_in_range.sort_values(by=['member_idx', 'problem_seq', 'answer_dttm'])

################### get last attempts
first_attempts = filtered_data_sorted.groupby(['member_idx', 'problem_seq']).last().reset_index()
first_attempts.head(2)

import pandas as pd
import json
import os
from tqdm import tqdm

# Define directories
filtered_csv_dir = "/root/jw/EduCDM/examples/IRT/GD_inference/integration_test/filtered_csv_outputs"

os.makedirs(filtered_csv_dir, exist_ok=True)
for subject_ in all_subjects:
    with open(f"/root/jw/EduCDM/examples/IRT/GD/processed/10_{subject_}_problem_seq_mapping_dict.json", 'r', encoding='utf-8') as file:
        problem_seq_mapping = json.load(file)
    
    # Load subject problems CSV (assuming 'first_attempts' is the DataFrame you're working with)
    subject_problems = first_attempts

    # Get the problem sequence from the mapping (convert to integers)
    problem_seq = list(map(int, problem_seq_mapping.keys()))

    # Convert problem_seq_mapping to a DataFrame to merge with subject_problems
    mapping_df = pd.DataFrame({
        'problem_seq': list(map(int, problem_seq_mapping.keys())),
        'mapping_value': list(problem_seq_mapping.values())  # The values from the mapping
    })

    # Filter the DataFrame by problem sequence and drop duplicates by 'problem_seq'
    filtered_df = subject_problems[subject_problems['problem_seq'].isin(problem_seq)].drop_duplicates(subset='problem_seq')

    # Merge the filtered DataFrame with the mapping DataFrame to add the new column
    merged_df = pd.merge(filtered_df, mapping_df, on='problem_seq')

    # Sort the merged DataFrame by the 'mapping_value' column
    merged_df = merged_df[['mapping_value','problem_seq','subject','answer']].sort_values(by='mapping_value')

    # Save the sorted DataFrame to a new CSV file
    merged_df.to_csv(f"{filtered_csv_dir}/{subject_}.csv", index=False)
    merged_df

