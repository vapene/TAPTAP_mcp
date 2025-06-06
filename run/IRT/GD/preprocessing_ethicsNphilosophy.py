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
##### subject list #####
# 한국사: 한국사
# 사회 탐구: 생활과 윤리, 정치와 법, 한국지리, 세계지리, 사회?문화, 동아시아사, 통합사회, 세계사, 윤리와 사상, 경제
# 수학: 수학Ⅰ, 수학Ⅱ     ### 사용안하는 선택 과목들:  확률과 통계, 기하, 중학, 미적분, 고등수학, 
# 영어: 독해     ###  사용안하는 선택 과목들: 듣기
# 과학 탐구: 화학Ⅰ, 통합과학, 물리학Ⅰ, 지구과학 Ⅰ, 생명과학 Ⅰ, 생명과학Ⅱ, 지구과학Ⅱ, 물리학Ⅱ, 화학Ⅱ
# 국어: 독서, 문학            ###  사용안하는 선택 과목들: 언어와 매체, 화법과 작문

parser = argparse.ArgumentParser(description='Train a model on user data')
parser.add_argument('--subjects', type=str, default='윤리와 사상', help='Comma-separated list of subjects to include') # '수학Ⅰ,수학Ⅱ'
parser.add_argument('--set_type', type=lambda s: s.split(','), default='EC040003,EC040014', help='국어 지문 유형')
parser.add_argument('--num_problems', type=int, default=10, help='몇 문제 뽑을 건가?')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--original_data_path', type=str, default='../../../data/full_data', help='data path for original datas')
parser.add_argument('--same_standard', type=str, default='middle_category', help='같은 문제로 볼 기준(new_seq)')
parser.add_argument('--IQR_range', type=float, default=1.5, help='이상치 기준. 1.5*IQR')
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

set_seed(args.seed)

###### import original datasets
category = pd.read_csv(f'{args.original_data_path}/perfectexam_category_202404031323.csv', encoding='cp949')
problem = pd.read_csv(f'{args.original_data_path}/perfectexam_problem_202404031323.csv', encoding='cp949')
member_exam_answer = pd.read_csv(f'{args.original_data_path}/perfectexam_member_exam_answer_202404031323.csv', encoding='cp949')
member_exam = pd.read_csv(f'{args.original_data_path}/perfectexam_member_exam_202404031323.csv', encoding='cp949')
##### 기탭에서 답변에 따라 ######
exam_in_range = pd.read_csv(f'{args.original_data_path}/perfectexam_exam_202404031323.csv', encoding='cp949')

category_simple = category[['code', 'code_name']]
problem_simple = problem[['seq', 'subject', 'big_category', 'middle_category','small_category','ref_seq','point','answer_rate',"set_type", "type1", "type2", "num1", "ref_num", "represent_yn"]]
member_exam_answer_simple = member_exam_answer[['member_idx', 'member_exam_seq', 'problem_seq', 'answer_dttm','correct_yn']]
member_exam_simple = member_exam[['seq', 'exam_seq', 'title']]
first_merge = member_exam_answer_simple.merge(member_exam_simple, left_on='member_exam_seq', right_on='seq', how='left').drop(columns='seq')
second_merge = first_merge[first_merge['exam_seq'].isin(exam_in_range['seq'])]
third_merge = second_merge.merge(exam_in_range[['seq', 'title']], left_on='exam_seq', right_on='seq', how='left').drop(columns=['seq', 'exam_seq', 'title_x'])
member_exam_answer_in_range = third_merge.merge(problem_simple, left_on='problem_seq', right_on='seq', how='left').drop(columns='seq')
category_dict = pd.Series(category_simple['code_name'].values, index=category_simple['code']).to_dict()
for column in ['subject', 'big_category', 'middle_category','small_category']:
    member_exam_answer_in_range[column] = member_exam_answer_in_range[column].map(category_dict)

###### filter by subjects
subjects_list = args.subjects.split(',')
filtered_data = member_exam_answer_in_range[member_exam_answer_in_range['subject'].isin(subjects_list)]
###### get first attemps
filtered_data_sorted = filtered_data.sort_values(by=['member_idx', 'problem_seq', 'answer_dttm'])
first_attempts = filtered_data_sorted.groupby(['member_idx', 'problem_seq']).last().reset_index()

###### get new_seq (같은 문제로 볼 기준 정하기 -> standardized_problem_seq 정해주기)
def generate_new_seq(row, standard):
    standards = {
        'big_category': ['subject', 'big_category', 'point'],
        'middle_category': ['subject', 'big_category', 'middle_category', 'point'],
        'small_category': ['subject', 'big_category', 'middle_category', 'small_category', 'point']}
    if standard in standards:
        return '-'.join(str(row[col]) for col in standards[standard])
    else:
        raise ValueError(f"Unsupported standard: {standard}")

first_attempts['new_seq_middle'] = first_attempts.apply(generate_new_seq, axis=1, standard=args.same_standard)
representative_problem_seq = (first_attempts.groupby('new_seq_middle')['problem_seq'].apply(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]).reset_index())
problem_seq_map = dict(zip(representative_problem_seq['new_seq_middle'], representative_problem_seq['problem_seq']))
first_attempts['standardized_problem_seq'] = first_attempts['new_seq_middle'].map(problem_seq_map)

##### new_seq별 너무 accuracy가 다른 response 제거   
def remove_outliers(df):
    Q1 = df['answer_rate'].quantile(0.25)
    Q3 = df['answer_rate'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - args.IQR_range * IQR
    upper_bound = Q3 + args.IQR_range * IQR
    return df[(df['answer_rate'] >= lower_bound) & (df['answer_rate'] <= upper_bound)]

first_attempts_after = first_attempts.groupby('new_seq_middle').apply(remove_outliers).reset_index(drop=True)
columns_to_drop_candidate = ['answer_dttm', 'answer_time(문제풀이 시간)', 'ref_seq','member_exam_seq']
columns_to_drop = [col for col in columns_to_drop_candidate if col in first_attempts_after.columns]
preprocessing_done = first_attempts_after.drop(columns=columns_to_drop)
preprocessing_done['correct_yn'] = preprocessing_done['correct_yn'].map({'Y': 1, 'N': 0})
preprocessing_done = preprocessing_done.applymap(lambda x: x.replace('・', '?').replace('∙', '?') if isinstance(x, str) else x) 

##### 질문지 제한 사항 (point, small_category 랭킹)
cumul_data = {}
for subject in subjects_list:
    file_path = f'{args.original_data_path}/cumul_{subject}.csv'
    try: 
        cumul_data[subject] = pd.read_csv(file_path, encoding='cp949')
    except:
        cumul_data[subject] = pd.read_csv(file_path, encoding='UTF-8')
##### small_category 뽑기 #####
category_structures = {}
for subject in subjects_list:
    with open(f'{args.original_data_path}/{subject}_middle_freq_list.json', 'r', encoding='UTF-8') as file:
        category_structures[subject] = json.load(file)
def select_frequent_small_category(data_df, category_structure):
    selected_small_category_list = []
    for big_category, middle_categories in category_structure.items():
        for middle_category, freq in middle_categories.items():
            filtered_data = data_df[(data_df['big_category'] == big_category) & 
                                    (data_df['middle_category'] == middle_category)]
            value_counts = filtered_data['small_category'].value_counts()
            categories = value_counts.index.tolist()
            selections = {category: 0 for category in categories}
            most_frequent_small_categories = []
            for _ in range(freq):
                max_category = max(categories, key=lambda x: value_counts[x] / (selections[x] + 1))
                selections[max_category] += 1
                most_frequent_small_categories.append(max_category)

            selected_small_category_list.extend(most_frequent_small_categories)
    return selected_small_category_list

##### 문제 뽑기 ######
all_selected_categories = []
for subject, df in cumul_data.items():
    df = df.replace(to_replace=["・", "∙"], value="?", regex=True)
    category_structure = category_structures[subject]
    
    selected_categories = select_frequent_small_category(df, category_structure)
    all_selected_categories.extend(selected_categories)


category_counts = {category: all_selected_categories.count(category) for category in set(all_selected_categories)}
# Initialize dictionaries and counters

def solved_all_problems(group, required_problems):
    solved_problems = set(group['standardized_problem_seq'])
    return solved_problems == set(required_problems)

def solved_all_problems_korean(group, required_problems):
    solved_problems = set(group['problem_seq'])
    return solved_problems == set(required_problems)

# Loop through each category
set_problems_dict = {}
total_problems = args.num_problems

for category, needed_top_n in category_counts.items():
    problems_in_category = preprocessing_done[preprocessing_done['small_category'] == category]
    top_problems = problems_in_category.groupby('standardized_problem_seq').size().nlargest(needed_top_n * 5).index.tolist()
    set_problems_dict[category] = top_problems

special_categories = {cat: count for cat, count in category_counts.items() if count > 1}
normal_categories = {cat: probs for cat, probs in set_problems_dict.items() if cat not in special_categories}

def generate_balanced_combinations(items, count):
    if len(items) >= count:
        return list(itertools.combinations(items, count))
    else:
        base_combinations = list(itertools.combinations_with_replacement(items, count))
        valid_combinations = []
        for combo in base_combinations:
            if all(combo.count(item) <= 2 for item in items):
                valid_combinations.append(combo)
        return valid_combinations

all_combinations = []
normal_combinations = list(itertools.product(*[normal_categories[cat] for cat in normal_categories]))
for normal_combo in normal_combinations:
    special_combos = [normal_combo]  
    for special_cat, special_count in special_categories.items():
        new_combos = []
        for base_combo in special_combos:
            for special_combo in generate_balanced_combinations(set_problems_dict[special_cat], special_count):
                new_combos.append(base_combo + special_combo) 
        special_combos = new_combos 
    all_combinations.extend(special_combos) 

def process_combination(combination):
    
    problem_combination = list(combination)
    filtered_df = preprocessing_done[preprocessing_done['standardized_problem_seq'].isin(problem_combination)]
    solved_all = filtered_df.groupby('member_idx').filter(lambda x: solved_all_problems(x, problem_combination))
    return len(solved_all['member_idx'].unique()), problem_combination
max_member_count = 0
all_top_standardized = []
with ProcessPoolExecutor(max_workers=32) as executor:
    results = executor.map(process_combination, all_combinations)
    for unique_member_count, problem_combination in results:
        if unique_member_count > max_member_count:
            max_member_count = unique_member_count
            all_top_standardized = problem_combination
            print('\n Current Best:', all_top_standardized, max_member_count)

##### 뽑힌 문제 확인 csv 저장 #####
results = all_top_standardized 

selected_df = pd.DataFrame(results, columns=['Problem Seq'])

detailed_df = selected_df.merge(preprocessing_done, left_on='Problem Seq', right_on='standardized_problem_seq', how='left')
detailed_df = detailed_df[['Problem Seq', 'subject', 'big_category', 'middle_category', 'small_category', 'point', 'new_seq_middle','title_y']]
detailed_df = detailed_df.drop_duplicates()
detailed_df.sort_values(['subject', 'big_category', 'middle_category', 'small_category'], inplace=True)
response_counts = preprocessing_done.groupby('new_seq_middle').size().rename('response_count')
correct_yn_counts = preprocessing_done.groupby('new_seq_middle')['correct_yn'].value_counts().unstack(fill_value=0)
correct_yn_counts = correct_yn_counts.rename(columns={0: 'incorrect_count', 1: 'correct_count'})
correct_yn_counts['response_summary'] = correct_yn_counts.apply(lambda x: f"0:{x['incorrect_count']}, 1:{x['correct_count']}", axis=1)
detailed_df = detailed_df.merge(response_counts, left_on='new_seq_middle', right_index=True, how='left')
detailed_df = detailed_df.merge(correct_yn_counts[['response_summary']], left_on='new_seq_middle', right_index=True, how='left')
detailed_df = detailed_df[['Problem Seq', 'subject', 'title_y', 'big_category', 'middle_category', 'small_category', 'point', 'response_count', 'response_summary']]

def replace_titles(group):
    try:
        filtered_titles = group[group['title_y'].str.contains('학년도', na=False)]['title_y']
        if not filtered_titles.empty:
            academic_year_title = filtered_titles.iloc[0]
            group['title_y'] = academic_year_title
        else:
            raise IndexError  # Force jump to except block if no titles match
    except IndexError:
        if not group['title_y'].empty:
            mode_result = group['title_y'].mode()
            if not mode_result.empty:
                most_frequent_title = mode_result[0]
            else:
                most_frequent_title = "Default Title"
        else:
            most_frequent_title = "Default Title"
        group['title_y'] = most_frequent_title
    return group

detailed_df_2= detailed_df.groupby('Problem Seq').apply(replace_titles)
detailed_df_2 = detailed_df_2.drop_duplicates(subset='Problem Seq', keep='first')
detailed_df_2 = detailed_df_2.reset_index(drop=True)
def calculate_response_ratio(summary):
    pairs = summary.split(', ')
    nums = [int(pair.split(':')[1]) for pair in pairs]
    total = sum(nums)
    ratios = [round(num / total, 2) for num in nums]
    return ratios
detailed_df_2['response_ratio'] = detailed_df_2['response_summary'].apply(calculate_response_ratio)

target_counts = Counter(all_top_standardized)
rows_to_add = []
for problem, required_count in target_counts.items():
    mask = detailed_df_2['Problem Seq'] == problem
    current_count = mask.sum()  
    additional_needed = required_count - current_count 
    if additional_needed > 0:
        rows_to_duplicate = detailed_df_2[mask].copy() 
        for _ in range(additional_needed):
            top_sequences = preprocessing_done[preprocessing_done['standardized_problem_seq'] == problem]['problem_seq'].value_counts().nlargest(5)
            selected_seq = None
            for seq, _ in top_sequences.items():
                if seq not in detailed_df_2['Problem Seq'].values:
                    selected_seq = seq
                    break
            if selected_seq is not None:
                rows_to_duplicate['Problem Seq'] = selected_seq  
            else:
                rows_to_duplicate['Problem Seq'] = rows_to_duplicate['Problem Seq'].iloc[0]  
            rows_to_add.append(rows_to_duplicate)
if rows_to_add:
    detailed_df_2 = pd.concat([detailed_df_2] + rows_to_add, ignore_index=True)
detailed_df_2.to_csv(f'./detailed/{args.num_problems}_{args.subjects}_question_detailed.csv', encoding='utf-8-sig', index=False)

detailed_df_2[['Problem Seq','response_summary']].to_csv(f'./detailed/{args.num_problems}_{args.subjects}_answer_count_taptap.csv', encoding='utf-8-sig', index=False)

######  Create problem point mapping dict  ########
problem_point_mapping_dict = detailed_df_2.set_index("Problem Seq")["point"].to_dict()

with open(f'./processed/{args.num_problems}_{args.subjects}_problem_point_mapping_dict.json', 'w') as f:
    json.dump(problem_point_mapping_dict, f)


##### Create full_data (response matrix with no re-index)
def get_frequent_value(series):
    value_counts = series.value_counts(dropna=True)
    if value_counts.empty:
        return None  
    if len(value_counts) > 1 and value_counts.iloc[0] == value_counts.iloc[1]:
        return series.dropna().iloc[0] if not series.dropna().empty else None
    return value_counts.idxmax()  

def map_first_occurrence(problem_seq):
    return problem_seq_mapping_dict.get(problem_seq, [None])[0]

problem_seq_mapping_dict = defaultdict(list)
for idx, problem in enumerate(all_top_standardized):
    problem_seq_mapping_dict[problem].append(idx + 1)
    

mask = preprocessing_done['standardized_problem_seq'].isin(all_top_standardized)
selected_data = preprocessing_done[mask].copy()
selected_data['standardized_problem_seq_mapped'] = selected_data['standardized_problem_seq'].apply(map_first_occurrence)
aggregated_data = selected_data.groupby(['member_idx', 'standardized_problem_seq_mapped'])['correct_yn'].agg(get_frequent_value).reset_index()

standardized_problem_seq_list = np.arange(1, len(all_top_standardized)+1)
full_index = pd.MultiIndex.from_product(
    [selected_data['member_idx'].unique(), standardized_problem_seq_list],
    names=['member_idx', 'standardized_problem_seq_mapped'])
full_data = pd.DataFrame(index=full_index).reset_index()
full_data = full_data.merge(aggregated_data, on=['member_idx', 'standardized_problem_seq_mapped'], how='left')

for problem, indices in problem_seq_mapping_dict.items():
    if len(indices) > 1: 
        source_index = indices[0]
        source_data = full_data[full_data['standardized_problem_seq_mapped'] == source_index][['member_idx', 'correct_yn']].set_index('member_idx')
        

        for target_index in indices[1:]:
            condition = full_data['standardized_problem_seq_mapped'] == target_index
            full_data.loc[condition, 'correct_yn'] = full_data.loc[condition, 'member_idx'].map(source_data['correct_yn'])
full_data = full_data.sort_values(by=['member_idx', 'standardized_problem_seq_mapped'])

full_data['original'] = np.where(full_data['correct_yn'].isna(), 'N', 'Y')
################


######  Create problem seq mapping dict  ########
problem_seq_mapping_dict_save = {key: i+1 for i, key in enumerate(problem_point_mapping_dict.keys())}

with open(f'./processed/{args.num_problems}_{args.subjects}_problem_seq_mapping_dict.json', 'w') as file:
        json.dump(problem_seq_mapping_dict_save, file)
        
##### 응답률에 따른 학생 수 보기
def analyze_data_quality(full_data):
    unique_problems = len({index for sublist in problem_seq_mapping_dict.values() for index in sublist})
    results_df = pd.DataFrame(columns=["Threshold", "Remaining Member Count", "Filtered Out Member Count"])
    for threshold_percent in range(0, 100, 10):
        threshold = unique_problems * (threshold_percent / 100.0)
        nan_counts = full_data['correct_yn'].isna().groupby(full_data['member_idx']).sum()
        members_to_remove = nan_counts[nan_counts > threshold].index
        filtered_data = full_data[~full_data['member_idx'].isin(members_to_remove)]
        member_idx_mapping = {idx: i for i, idx in enumerate(filtered_data['member_idx'].unique())}
        filtered_data['member_idx'] = filtered_data['member_idx'].map(member_idx_mapping)
        remaining_member_count = filtered_data['member_idx'].nunique()
        filtered_out_member_count = full_data['member_idx'].nunique() - remaining_member_count
        new_row = pd.DataFrame([{
            "Threshold": f"{threshold_percent}%",
            "Remaining Member Count": remaining_member_count,
            "Filtered Out Member Count": filtered_out_member_count
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)   
    return results_df
##### 몇% 이상 푼 학생들 filter

def data_filtering(full_data, threshold, num_problems):
    score_column = 'correct_yn'
    threshold_percent = threshold / 100.0  
    threshold = num_problems * threshold_percent 
    nan_counts = full_data[score_column].isna().groupby(full_data['member_idx']).sum()
    members_to_remove = nan_counts[nan_counts > threshold].index
    filtered_data = full_data[~full_data['member_idx'].isin(members_to_remove)]
    unique_members = pd.unique(filtered_data['member_idx'])
    unique_members.sort()  
    member_idx_mapping_dict = {member: idx + 1 for idx, member in enumerate(unique_members)}
    filtered_data['member_idx_mapped'] = filtered_data['member_idx'].map(member_idx_mapping_dict)
    filtered_data.sort_values(by=['member_idx_mapped', 'standardized_problem_seq_mapped'], inplace=True)
    return filtered_data, member_idx_mapping_dict


def process_member_NA1(member, full_data_merged, preprocessing_done_reduced):
    updates = {}
    preprocessing_done_reduced = preprocessing_done_reduced[preprocessing_done_reduced['member_idx'] == member]
    member_data = full_data_merged[full_data_merged['member_idx'] == member]  
    for index, row in member_data.iterrows():
        if pd.isna(row['correct_yn']):
            mask = (
                (preprocessing_done_reduced['subject'] == row['subject']) &
                (preprocessing_done_reduced['big_category'] == row['big_category']) &
                (preprocessing_done_reduced['middle_category'] == row['middle_category'])
            )
            filtered_attempts = preprocessing_done_reduced[mask]
            higher_points_correct = filtered_attempts[(filtered_attempts['point'] > row['point']) & (filtered_attempts['most_frequent'] == 1)]
            lower_points_incorrect = filtered_attempts[(filtered_attempts['point'] < row['point']) & (filtered_attempts['most_frequent'] == 0)]
            higher_count = len(higher_points_correct)
            lower_count = len(lower_points_incorrect)
            if higher_count > 0 or lower_count > 0:
                result = 1 if higher_count > lower_count else 0
                updates[index] = result
    return updates

def fill_nan_correct_yn(full_data, preprocessing_done, problem_seq_mapping_dict):
    inverse_mapping = {i: problem_seq for problem_seq, indices in problem_seq_mapping_dict.items() for i in indices}
    full_data['standardized_problem_seq'] = full_data['standardized_problem_seq_mapped'].map(inverse_mapping)
    valid_data = preprocessing_done.dropna(subset=['correct_yn'])
    mode_df = valid_data.groupby(['member_idx', 'standardized_problem_seq'])['correct_yn'].agg(lambda x: x.value_counts().idxmax()).reset_index()
    mode_df.columns = ['member_idx', 'standardized_problem_seq', 'most_frequent']
    preprocessing_done_reduced = preprocessing_done.drop_duplicates(subset=['member_idx', 'problem_seq'])[['member_idx', 'problem_seq', 'correct_yn', 'subject',
       'big_category', 'middle_category', 'point','standardized_problem_seq']]
    preprocessing_done_reduced = preprocessing_done_reduced.merge(mode_df, left_on=['member_idx', 'standardized_problem_seq'], right_on=['member_idx', 'standardized_problem_seq'], how='left')
    full_data_merged = full_data.merge(
        preprocessing_done_reduced.drop_duplicates(subset=['standardized_problem_seq'], keep='first'), left_on=['standardized_problem_seq'], right_on=['standardized_problem_seq'], how='left', suffixes=('', '_drop'))
    full_data_merged.drop([col for col in full_data_merged.columns if 'drop' in col], axis=1, inplace=True)
    members = full_data_merged['member_idx'].unique()
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_member_NA1, member, full_data_merged, preprocessing_done_reduced): member for member in members}
        for future in futures:
            member_updates = future.result()
            for index, value in member_updates.items():
                if pd.isna(full_data_merged.at[index, 'correct_yn']):
                    full_data_merged.at[index, 'correct_yn'] = value
    return full_data_merged[['member_idx', 'standardized_problem_seq_mapped', 'correct_yn','original']]

##### Get p_thetas for information function. 
if set(subjects_list) & set(['독서','문학']): 
    standard = 'problem_seq'
else:
    standard = 'standardized_problem_seq'
    
preprocessing_done_answer_rate = preprocessing_done[
    preprocessing_done[standard].isin(problem_seq_mapping_dict.keys())]
answer_rate_per_standard_problem_seq = preprocessing_done_answer_rate.groupby(standard)['answer_rate'].mean()/100
p_theta_problem = defaultdict(list)
for problem_seq in problem_seq_mapping_dict:
    answer_rate = answer_rate_per_standard_problem_seq.get(problem_seq, np.nan)
    for index in problem_seq_mapping_dict[problem_seq]:
        p_theta_problem[index].append(answer_rate)

for key in list(p_theta_problem):
    p_theta_problem[key] = np.nanmean(p_theta_problem[key])  # Compute mean if multiple values exist, handling NaNs

p_theta_problem=dict(p_theta_problem)
with open(f"./detailed/10_{args.subjects}_p_theta_problem.json", 'w') as json_file:
    json.dump(p_theta_problem, json_file, indent=4)  


# p theta response
filtered_data_0, _= data_filtering(full_data,threshold=30, num_problems=len(all_top_standardized)) 
prob_df = filtered_data_0[filtered_data_0['correct_yn'].isin([0, 1])]
group_counts = prob_df.groupby('standardized_problem_seq_mapped').size()
prob_df = prob_df.groupby('standardized_problem_seq_mapped')['correct_yn'].agg(
    count='count',
    count_1=lambda x: (x == 1).sum(),  # Count of 1s
    count_0=lambda x: (x == 0).sum(),  # Count of 0s
    probability=lambda x: (x == 1).sum() / x.count()
).reset_index()
p_theta_response = prob_df.set_index('standardized_problem_seq_mapped')['probability'].to_dict()
with open(f"./detailed/10_{args.subjects}_p_theta_response.json", 'w') as json_file:
    json.dump(p_theta_response, json_file, indent=4)  

### p_theta_response_answer_ratio_counts
p_theta_response_answer_ratio_counts = prob_df.set_index('standardized_problem_seq_mapped')[['count_0', 'count_1']].apply(lambda x: [x['count_0'], x['count_1']], axis=1).to_dict()
p_theta_response_answer_ratio_counts = {int(k): [int(v[0]), int(v[1])] for k, v in p_theta_response_answer_ratio_counts.items()}
with open(f"./detailed/10_{args.subjects}_p_theta_response_ratio_counts.json", 'w') as json_file:
    json.dump(p_theta_response_answer_ratio_counts, json_file, indent=4)  
###



num_students_by_NaN_ratio = analyze_data_quality(full_data)
num_students_by_NaN_ratio.to_csv(f'./detailed/{args.num_problems}_{args.subjects}_NaN_ratio_analyze_original.csv', encoding='utf-8-sig', index=False)
######
filled_data1 = fill_nan_correct_yn(full_data, preprocessing_done, problem_seq_mapping_dict)
analyze_data=analyze_data_quality(filled_data1)
analyze_data.to_csv(f'./detailed/{args.num_problems}_{args.subjects}_NaN_ratio_analyze_afterNA1.csv', encoding='utf-8-sig', index=False)
filtered_data_1,  member_idx_mapping_dict_1= data_filtering(filled_data1,threshold=30, num_problems=len(all_top_standardized)) # NaN 비율이 30%가 넘어가면

filtered_data_1[['member_idx_mapped','standardized_problem_seq_mapped','correct_yn','original']].to_csv(f'./processed/{args.num_problems}_{args.subjects}_30_ratio_afterNA1.csv', encoding='utf-8-sig', index=False)
member_idx_mapping_dict_1 = {int(k): v for k, v in member_idx_mapping_dict_1.items()}
with open(f'./processed/{args.num_problems}_{args.subjects}_30_ratio_afterNA1_member_idx_mapping_dict_1.json', 'w') as file:
        json.dump(member_idx_mapping_dict_1, file)
        
##### Fill NA 2. rule3def calculate_similarity(series1, series2):
def calculate_similarity(series1, series2):
    where_are_nans = np.logical_or(np.isnan(series1), np.isnan(series2))
    series1 = series1[~where_are_nans]
    series2 = series2[~where_are_nans]
    return (series1 == series2).sum()


def process_member_NA2(member, filled_data, non_nan_members):
    member_data = filled_data[filled_data['member_idx'] == member]
    nan_problems = member_data[member_data['correct_yn'].isna()]['standardized_problem_seq_mapped'].unique()
    results = []
    for problem in nan_problems:
        member_responses = member_data[member_data['standardized_problem_seq_mapped'] != problem]['correct_yn']
        similarity_scores = {}
        for other_member in non_nan_members:
            other_member_data = filled_data[filled_data['member_idx'] == other_member]
            other_member_responses = other_member_data[other_member_data['standardized_problem_seq_mapped'] != problem]['correct_yn']

            if len(member_responses) == len(other_member_responses):
                similarity = calculate_similarity(np.array(member_responses), np.array(other_member_responses))
                similarity_scores[other_member] = similarity

        if similarity_scores:
            most_similar_members = [member for member, score in similarity_scores.items() if score == max(similarity_scores.values())]
            most_similar_responses = filled_data[(filled_data['member_idx'].isin(most_similar_members)) & 
                                                  (filled_data['standardized_problem_seq_mapped'] == problem)]['correct_yn']
            try:
                mode_response = mode(most_similar_responses, nan_policy='omit')
                mode_value = mode_response.mode 
            except IndexError:
                mode_value = mode_response.mode[0] if mode_response.count[0] > 0 else np.nan
            results.append((member, problem, mode_value))

    return results
def identify_non_nan_members(filled_data):
    nan_counts = filled_data.groupby('member_idx')['correct_yn'].apply(lambda x: x.isna().sum())
    non_nan_members = nan_counts[nan_counts <= len(all_top_standardized)*0.4].index
    return non_nan_members

def fill_remaining_nans(filled_data):
    filled_data.sort_values(by=['member_idx', 'standardized_problem_seq_mapped'], inplace=True)
    nan_members = filled_data[filled_data['correct_yn'].isna()]['member_idx'].unique()
    candidate_members = identify_non_nan_members(filled_data)
    non_nan_members = np.setdiff1d(candidate_members, nan_members)
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {}
        for member in tqdm(nan_members, desc="Processing members"):
            future = executor.submit(process_member_NA2, member, filled_data, non_nan_members)
            futures[future] = member
        for future in tqdm(futures, desc="Completing tasks"):
            results = future.result()
            for member, problem, value in results:
                condition = ((filled_data['member_idx'] == member) & 
                             (filled_data['standardized_problem_seq_mapped'] == problem))
                filled_data.loc[condition & filled_data['correct_yn'].isna(), 'correct_yn'] = value

    return filled_data

filled_data2 = fill_remaining_nans(filled_data1)
analyze_data=analyze_data_quality(filled_data2)
analyze_data.to_csv(f'./detailed/{args.num_problems}_{args.subjects}_NaN_ratio_analyze_afterNA2.csv', encoding='utf-8-sig', index=False)
filtered_data_2, member_idx_mapping_dict_2 = data_filtering(filled_data2,threshold=30,num_problems=len(all_top_standardized)) # NaN 비율이 30%가 넘어가면
filtered_data_2[['member_idx_mapped','standardized_problem_seq_mapped','correct_yn','original']].to_csv(f'./processed/{args.num_problems}_{args.subjects}_30_ratio_afterNA2.csv', encoding='utf-8-sig', index=False)
member_idx_mapping_dict_2 = {int(k): v for k, v in member_idx_mapping_dict_2.items()}
with open(f'./processed/{args.num_problems}_{args.subjects}_30_ratio_afterNA1_member_idx_mapping_dict_2.json', 'w') as file:
        json.dump(member_idx_mapping_dict_2, file)


