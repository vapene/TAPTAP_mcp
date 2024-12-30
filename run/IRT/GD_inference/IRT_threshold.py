import json
import pandas as pd
import numpy as np

irt_subjects = ['독서,문학', '독해','사회?문화','생명과학Ⅰ', '생활과 윤리', '수학Ⅰ,수학Ⅱ', '지구과학Ⅰ']
all_criteria_dict = {}

for subject in irt_subjects:
    answer_rate_dict = json.load(open(f"../GD/detailed/10_{subject}_p_theta_response.json"))

    answer_list_df = pd.read_csv("./integration_test/compiled_data.csv")

    num_students = 3000

    total_list = []

    for _ in range(num_students):
        solving_list = [np.random.binomial(1, answer_rate_dict[str(i)]) for i in range(1, len(answer_rate_dict) + 1)]
        score = answer_list_df.loc[(answer_list_df['Solving History'] == str(solving_list)) & (answer_list_df['Subject']==subject), 'Target Score']
        if not score.empty:
            total_list.append(score.values[0])
        else:
            print(f"{subject} Solving list {solving_list} not found in DataFrame")

    if total_list: 
        total_list = sorted(total_list, reverse=True)
        top_70_percent_threshold = total_list[int(num_students * 0.7) - 1]
        top_40_percent_threshold = total_list[int(num_students * 0.4) - 1]

        all_criteria_dict[subject] = [top_70_percent_threshold,top_40_percent_threshold]
    else:
        print(f"{subject} No scores available to calculate thresholds.")
    print(f'{subject} done')
print(json.dumps(all_criteria_dict, ensure_ascii=False, indent=4))