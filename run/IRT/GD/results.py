import os
import pandas as pd
import numpy as np

# 1.현재 폴더에 student_counts.csv
# 2.results 폴더에 best_results.csv
# 3.현재 폴더에 configs_NA1.yml, configs_NA2.yml 만들기

#####################################
####### Get Student count ###########
#####################################

# Correct the subject list
subject_list = ['통합사회','통합과학','경제', '사회?문화', '세계사', '윤리와 사상', '세계지리', '생활과 윤리', '동아시아사', '한국지리', '물리학Ⅱ', '화학Ⅰ', '수학Ⅰ,수학Ⅱ', '독서,문학', '생명과학Ⅱ', '지구과학Ⅰ', '지구과학Ⅱ', '생명과학Ⅰ', '한국사', '독해', '화학Ⅱ', '정치와 법', '물리학Ⅰ']
# subject_list = ['독해,듣기']
base_dir = './detailed'
data = []
for subject in subject_list:
    try:
        path_original = os.path.join(base_dir, f'10_{subject}_NaN_ratio_analyze_original.csv')
        path_afterNA1 = os.path.join(base_dir, f'10_{subject}_NaN_ratio_analyze_afterNA1.csv')
        path_afterNA2 = os.path.join(base_dir, f'10_{subject}_NaN_ratio_analyze_afterNA2.csv')
        df_original = pd.read_csv(path_original)
        df_afterNA1 = pd.read_csv(path_afterNA1)
        df_afterNA2 = pd.read_csv(path_afterNA2)
        original_count = df_original[df_original['Threshold'] == '30%']['Remaining Member Count'].values[0]
        na1_count = df_afterNA1[df_afterNA1['Threshold'] == '30%']['Remaining Member Count'].values[0]
        na2_count = df_afterNA2[df_afterNA2['Threshold'] == '30%']['Remaining Member Count'].values[0]
        data.append({'Subject': subject, 'Original_count': original_count, 'NA1_count': na1_count, 'NA2_count': na2_count})
    except Exception as e:
        print(f"Error processing subject {subject}: {e}")
result_df = pd.DataFrame(data)
output_filepath = './students_counts.csv'
result_df.to_csv(output_filepath, encoding='utf-8-sig', index=False)



####################################
###### Get best resulsts ###########
####################################
NA_list = ['NA1', 'NA2']
for NA in NA_list:
    base_dir = f'./results_{NA}' #'results_NA2' #'results_NA1'
    results_df = pd.DataFrame(columns=['auc_score', 'acc_score', 'fpr_score', 'precision_score'])
    for subdir, dirs, files in os.walk(base_dir):
        results_found = False  # Flag to check if valid results have been found
        for file in files:
            if file.startswith('best_params'):
                full_file_path = os.path.join(subdir, file)
                with open(full_file_path, 'r') as f:
                    lines = f.readlines()
                    auc_scores = np.array([float(x.strip()) for x in lines[2].strip().split(': ')[1].replace('[', '').replace(']', '').split(',')])*100
                    acc_scores = np.array([float(x.strip()) for x in lines[3].strip().split(': ')[1].replace('[', '').replace(']', '').split(',')])*100
                    fpr_scores = np.array([float(x.strip()) for x in lines[4].strip().split(': ')[1].replace('[', '').replace(']', '').split(',')])*100
                    precision_scores = np.array([float(x.strip()) for x in lines[5].strip().split(': ')[1].replace('[', '').replace(']', '').split(',')])*100
                    auc_mean = np.mean(auc_scores)
                    auc_std = np.std(auc_scores)
                    acc_mean = np.mean(acc_scores)
                    acc_std = np.std(acc_scores)
                    fpr_mean = np.mean(fpr_scores)
                    fpr_std = np.std(fpr_scores)
                    precision_mean = np.mean(precision_scores)
                    precision_std = np.std(precision_scores)

                    auc_result = f"{auc_mean:.2f} ± {auc_std:.2f}"
                    acc_result = f"{acc_mean:.2f} ± {acc_std:.2f}"
                    fpr_result = f"{fpr_mean:.2f} ± {fpr_std:.2f}"
                    precision_result = f"{precision_mean:.2f} ± {precision_std:.2f}"
                    results_found = True
        folder_name = os.path.basename(subdir)
        if results_found:
            results_df.loc[folder_name] = [auc_result, acc_result, fpr_result, precision_result]
        else:
            results_df.loc[folder_name] = ["NA","NA","NA","NA"]
            
    print(results_df)
    results_df.to_csv(f'{base_dir}/{NA}_best_results.csv', encoding='utf-8-sig', index=True)


##############################
###### Create config #########
##############################


# Define the base directories containing your results
base_dirs = [
    './results_NA1',
    './results_NA2'
]

# Default parameters
default_params = {
    'alpha': '0.7',
    'gamma': '1',
    'batch_size': '128'
}

def get_combined_mean_score(data):
    # Extract the combined mean score from the file data
    for line in data.splitlines():
        if 'Combined Mean Score:' in line:
            return float(line.split(':')[1].strip())
    return None

for base_dir in base_dirs:
    summary_file_path = os.path.join('./', f'configs_{base_dir[-3:]}.yml')
    # Track subjects with missing data and subjects using default values
    subjects_with_missing_data = []
    subjects_with_default_values = []

    # Open the summary file for writing
    with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
        # List actual folders in the base directory
        actual_folders = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

        for subject in actual_folders:
            subject_path = os.path.join(base_dir, subject)
            best_file = None
            highest_score = float('-inf')

            # Evaluate each file to find the best one
            for file in os.listdir(subject_path):
                if file.startswith('best_params_') and file.endswith('.txt'):
                    file_path = os.path.join(subject_path, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                        score = get_combined_mean_score(data)
                        if score is not None and score > highest_score:
                            best_file = file
                            highest_score = score

            # Delete all other files and keep only the best one
            for file in os.listdir(subject_path):
                if file.startswith('best_params_') and file.endswith('.txt') and file != best_file:
                    os.remove(os.path.join(subject_path, file))

            # Process the best file or apply defaults
            if best_file:
                file_path = os.path.join(subject_path, best_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
                    # Initialize parameters
                    alpha = gamma = batch_size = None

                    # Parsing each line to find the parameters
                    for line in data.splitlines():
                        if 'Alpha:' in line:
                            alpha = line.split('Alpha:')[1].split(',')[0].strip()
                        if 'Gamma:' in line:
                            gamma = line.split('Gamma:')[1].split(',')[0].strip()
                        if 'Batch Size:' in line:
                            batch_size = line.split('Batch Size:')[1].split(',')[0].strip()
            else:
                # Apply default values if no file is found
                alpha = default_params['alpha']
                gamma = default_params['gamma']
                batch_size = default_params['batch_size']
                subjects_with_default_values.append(subject)

            # Write the formatted output to the summary file
            summary_file.write(f'{subject}:\n')
            summary_file.write(f'  alpha: {alpha if alpha else "missing"}\n')
            summary_file.write(f'  gamma: {gamma if gamma else "missing"}\n')
            summary_file.write(f'  batch_size: {batch_size}\n')

            # Check for missing data
            if not alpha or not gamma:
                subjects_with_missing_data.append(subject)

    # Reporting on completion
    if subjects_with_missing_data:
        print(f"Subjects with missing alpha or gamma parameters in {os.path.basename(base_dir)}:", subjects_with_missing_data)

    if subjects_with_default_values:
        print(f"Subjects that used default values due to missing files in {os.path.basename(base_dir)}:", subjects_with_default_values)
    else:
        print(f"No subjects needed to use default values in {os.path.basename(base_dir)}.")

    print(f"Summary of parameters has been saved to {summary_file_path}")