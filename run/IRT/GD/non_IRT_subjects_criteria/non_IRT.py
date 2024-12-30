import pandas as pd
import numpy as np
import random
import torch
import json
import warnings
import matplotlib.pyplot as plt
import argparse
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--num_problems', type=int, default=10, help='Number of problems')
parser.add_argument('--set_type', nargs='+', default=['EC040003','EC040014'], help='Set types')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--original_data_path', type=str, default='/root/jw/TAPTAP_IRT/data/full_data', help='Path to original data')
parser.add_argument('--same_standard', type=str, default='middle_category', help='Standard to be used')
parser.add_argument('--IQR_range', type=float, default=1.5, help='IQR range')
parser.add_argument('--num_students', type=int, default=10000, help='number of students to simulate')
args = parser.parse_args()

num_students = args.num_students
set_seed(args.seed)

subject_mapping_dict = {
    "통합과학": "Science",
    "통합사회": "Humanities",
    "수학Ⅰ,수학Ⅱ": 'Math',
    "독해": 'English',
    "독서,문학": "Korean",
    "한국사": "Korean History",
    "생활과 윤리": "Life N Ethics",
    "정치와 법": "Politics N Law",
    "한국지리": "Korean Geo.",
    "세계지리": "World Geo.",
    "사회?문화": "Life Culture",
    "동아시아사": "East Asian History",
    "세계사": "World History",
    "윤리와 사상": "Ethical Philosopy",
    "경제": "Economics",
    "물리학Ⅰ": "PhysicsⅠ",
    "화학Ⅰ": "ChemistryⅠ",
    "생명과학Ⅰ": "BiologyⅠ",
    "지구과학Ⅰ": "Earth ScienceⅠ",
    "지구과학Ⅱ": "Earth ScienceⅡ",
    "생명과학Ⅱ": "BiologyⅡ",
    "화학Ⅱ": "ChemistryⅡ",
    "물리학Ⅱ": "PhysicsⅡ"
}

# Import original datasets
category = pd.read_csv(f'{args.original_data_path}/perfectexam_category_202404031323.csv', encoding='cp949')
problem = pd.read_csv(f'{args.original_data_path}/perfectexam_problem_202404031323.csv', encoding='cp949')
member_exam_answer = pd.read_csv(f'{args.original_data_path}/perfectexam_member_exam_answer_202404031323.csv', encoding='cp949')
member_exam = pd.read_csv(f'{args.original_data_path}/perfectexam_member_exam_202404031323.csv', encoding='cp949')
exam_in_range = pd.read_csv(f'{args.original_data_path}/perfectexam_exam_202404031323.csv', encoding='cp949')

category_simple = category[['code', 'code_name']]
problem_simple = problem[['seq', 'subject', 'big_category', 'middle_category','small_category','point','answer_rate',"set_type", "ref_num"]]
member_exam_answer_simple = member_exam_answer[['member_idx', 'member_exam_seq', 'problem_seq', 'answer_dttm','correct_yn']]
member_exam_simple = member_exam[['seq', 'exam_seq', 'title']]
first_merge = member_exam_answer_simple.merge(member_exam_simple, left_on='member_exam_seq', right_on='seq', how='left').drop(columns='seq')
second_merge = first_merge[first_merge['exam_seq'].isin(exam_in_range['seq'])]
third_merge = second_merge.merge(exam_in_range[['seq', 'title']], left_on='exam_seq', right_on='seq', how='left').drop(columns=['seq', 'exam_seq', 'title_x'])
member_exam_answer_in_range = third_merge.merge(problem_simple, left_on='problem_seq', right_on='seq', how='left').drop(columns='seq')
category_dict = pd.Series(category_simple['code_name'].values, index=category_simple['code']).to_dict()

for column in ['subject', 'big_category', 'middle_category','small_category']:
    member_exam_answer_in_range[column] = member_exam_answer_in_range[column].map(category_dict)

# Process each subject
for subject_kr, subject_en in subject_mapping_dict.items():
    subjects_list = subject_kr.split(',')

    filtered_data = member_exam_answer_in_range[member_exam_answer_in_range['subject'].isin(subjects_list)]
    # filtered_data['correct_yn'] = filtered_data['correct_yn'].map({'Y': 1, 'N': 0})
    

    response_count= pd.read_csv(f'../detailed/10_{subject_kr}_answer_count_taptap.csv', encoding='utf-8')
    def calculate_probability(summary):
        parts = summary.split(', ')
        count_0 = int(parts[0].split(':')[1])
        count_1 = int(parts[1].split(':')[1])
        total = count_0 + count_1
        probability = round((count_1 / total)*100,1)
        return probability

    response_count['probability'] = response_count['response_summary'].apply(calculate_probability)

    # point 구하기
    problem_seq_keys = list(map(int, response_count['problem_seq']))
    filtered_problems = problem[problem['seq'].isin(problem_seq_keys)]

    max_points = filtered_problems.groupby('seq')['point'].max()

    if 'max_point' in response_count.columns:
        response_count = response_count.rename(columns={'max_point': 'existing_max_point'})
    # Merge the max_points DataFrame with the response_count DataFrame
    response_count = response_count.merge(max_points, left_on='problem_seq', right_on='seq', how='left')
    # Drop the unnecessary columns
    columns_to_drop = ['seq_x', 'max_point_x', 'seq_y', 'max_point_y', 'existing_max_point']
    response_count_taptap_df = response_count.drop(columns=[col for col in columns_to_drop if col in response_count.columns])


    # Number of students to simulate
    students_scores = pd.DataFrame(index=range(num_students), columns=['score'])
    correct_rate_taptap = response_count_taptap_df['probability']
    points = response_count_taptap_df['point']

    score_list_taptapp = []
    # Run the simulation
    for i in range(num_students):
        # Simulate student answers (1 for correct, 0 for incorrect) using Bernoulli distribution
        correct_answers = np.random.binomial(1, response_count_taptap_df['probability'] / 100)
        target_score = sum(r * d for r, d in zip(correct_answers, points))
        # Initialize inconsistency count
        inconsistency_count = 0
        total_possible_inconsistencies = 0
        # Iterate over all pairs of questions
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if points[i] < points[j]:
                    total_possible_inconsistencies += 1
                    if correct_answers[i] == 0 and correct_answers[j] == 1:
                        inconsistency_count += 1
                elif points[i] > points[j]:
                    total_possible_inconsistencies += 1
                    if correct_answers[i] == 1 and correct_answers[j] == 0:
                        inconsistency_count += 1

        # Calculate inconsistency rate
        if total_possible_inconsistencies > 0:
            inconsistency_rate = inconsistency_count / total_possible_inconsistencies
        else:
            inconsistency_rate = 0

        # Parameters for adjusted penalty
        penalty_strength = 0.3    # Maximum penalty reduction
        penalty_threshold = 0.2   # Threshold above which penalty starts
        alpha = 2                 # Controls sensitivity to inconsistency increases

        # Apply adjusted penalty
        if inconsistency_rate <= penalty_threshold:
            penalty_factor = 1  
        else:
            normalized_inconsistency = (inconsistency_rate - penalty_threshold) / (1 - penalty_threshold)
            penalty_factor = 1 - penalty_strength * (normalized_inconsistency) ** alpha
            penalty_factor = max(penalty_factor, 1 - penalty_strength)

        target_score *= penalty_factor
        ################
        score_list_taptapp.append(target_score)

    score_series = pd.Series(score_list_taptapp)
    # Calculate the score thresholds for the top 30% and 60%
    threshold_33 = score_series.quantile(0.33)  # Bottom 33%
    threshold_66 = score_series.quantile(0.66)  # Top 66%

    # Save results to a JSON file
    results = {
        "subject": subject_en,
        "top_30_percent_threshold": threshold_33,
        "top_60_percent_threshold": threshold_66
    }
    with open(f"./criteria_output/{subject_kr}_results_TAPTAP.json", "w") as outfile:
        json.dump(results, outfile)
    # Plot and save the histogram
    plt.figure()
    score_series.hist(bins=20)
    plt.xlabel('Score')
    plt.ylabel('Number of Students')
    plt.title(f'Histogram of Student Scores for {subject_en} using TAPTAP prob.')
    plt.savefig(f"./figures/{subject_en}_histogram_TAPTAP.png")
    plt.close()

    ####################### 
    # using nationwide data #
    ######################
    answer_rate_dict = {str(row['seq']): [row['answer_rate'], row['point']] for _, row in filtered_problems.iterrows()}
    students_scores = pd.DataFrame(index=range(num_students), columns=['score'])
    correct_rate_nation = [value[0] for value in answer_rate_dict.values()]
    points = [value[1] for value in answer_rate_dict.values()]

    score_list_nation = []
    # Run the simulation
    for i in range(num_students):
        # Simulate student answers (1 for correct, 0 for incorrect) using Bernoulli distribution
        correct_answers = [np.random.binomial(1, rate / 100) for rate in correct_rate_nation]
        target_score = sum(r * d for r, d in zip(correct_answers, points))
        # Initialize inconsistency count
        inconsistency_count = 0
        total_possible_inconsistencies = 0
        # Iterate over all pairs of questions
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if points[i] < points[j]:
                    total_possible_inconsistencies += 1
                    if correct_answers[i] == 0 and correct_answers[j] == 1:
                        inconsistency_count += 1
                elif points[i] > points[j]:
                    total_possible_inconsistencies += 1
                    if correct_answers[i] == 1 and correct_answers[j] == 0:
                        inconsistency_count += 1
        # Calculate inconsistency rate
        if total_possible_inconsistencies > 0:
            inconsistency_rate = inconsistency_count / total_possible_inconsistencies
        else:
            inconsistency_rate = 0
        # Parameters for adjusted penalty
        penalty_strength = 0.3    # Maximum penalty reduction
        penalty_threshold = 0.2   # Threshold above which penalty starts
        alpha = 2                 # Controls sensitivity to inconsistency increases
        # Apply adjusted penalty
        if inconsistency_rate <= penalty_threshold:
            penalty_factor = 1  
        else:
            normalized_inconsistency = (inconsistency_rate - penalty_threshold) / (1 - penalty_threshold)
            penalty_factor = 1 - penalty_strength * (normalized_inconsistency) ** alpha
            penalty_factor = max(penalty_factor, 1 - penalty_strength)

        target_score *= penalty_factor
        ################
        score_list_nation.append(target_score)

    score_series = pd.Series(score_list_nation)
    # Calculate the score thresholds for the top 30% and 60%
    threshold_33 = score_series.quantile(0.33)  # Bottom 33%
    threshold_66 = score_series.quantile(0.66)  # Top 66%

    results = {
        "subject": subject_en,
        "top_30_percent_threshold": threshold_33,
        "top_60_percent_threshold": threshold_66
    }
    with open(f"./criteria_output/{subject_kr}_results.json", "w") as outfile:
        json.dump(results, outfile)

    # Plot and save the histogram
    plt.figure()
    score_series.hist(bins=20)
    plt.xlabel('Score')
    plt.ylabel('Number of Students')
    plt.title(f'Histogram of Student Scores for {subject_en}')
    plt.savefig(f"./figures/{subject_en}_histogram.png")
    plt.close()

    ################### save CSV
    print(f'done saving criteria_output/{subject_kr}_results.json')


