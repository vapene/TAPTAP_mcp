import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

#################################
subject_mapping_dict = {"통합과학":"Science","통합사회":"Humanity","수학Ⅰ,수학Ⅱ":'Math',"독해":'English',"독서,문학":"Korean","한국사":"Korean History", "생활과 윤리":"Life N Ethics", "정치와 법":"Politics N Law", "한국지리":"Korean Geo.", "세계지리":"World Geo.", "사회?문화":"Life Culture", "동아시아사":"East Asian History", "세계사":"World History", "윤리와 사상":"Ethical Philosopy", "경제":"Economics",
 "물리학Ⅰ":"PhysicsⅠ", "화학Ⅰ":"ChemistryⅠ", "생명과학Ⅰ":"BiologyⅠ", "지구과학Ⅰ":"Earth ScienceⅠ", "지구과학Ⅱ":"Earth ScienceⅡ", "생명과학Ⅱ":"BiologyⅡ", "화학Ⅱ":"ChemistryⅡ", "물리학Ⅱ":"PhysicsⅡ"}
def irt3pl_tensor(theta, a, b, c, D=torch.tensor(1.702), *, F=torch):
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))
def irt2pl(theta, a, b, D=1.702, *, F=np):
    return F.exp(D * a * (theta - b)) / (1 + F.exp(D * a * (theta - b)))
def irt1pl(theta, b, D=1.702, *, F=np):
    return F.exp(theta - b) / (1 + F.exp(theta - b))
#################################
def plot_icc_curve(subject, NA, item_ids, a, b, c, theta, data_store):
    ICC_plot = torch.linspace(-5, 5, 100) 
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  
    axes = axes.flatten() 

    for index, ax in enumerate(axes):
        if index < len(item_ids):  
            item_id = next((k for k, v in item_ids.items() if v == index + 1), None)
            a_i = a[index].squeeze()
            b_i = b[index].squeeze()
            c_i = c[index].squeeze()
            probabilities = irt3pl_tensor(ICC_plot, a_i, b_i, c_i).numpy()
            ax.plot(ICC_plot.numpy(), probabilities, label=f'Item {item_id}')

            for theta_value in theta:
                prob_point = irt3pl_tensor(torch.tensor([theta_value]), a_i, b_i, c_i).item()
                ax.plot(theta_value.item(), prob_point, 'ro', markersize=5)

            ax.set_title(f'Item {item_id}')
            ax.set_xlabel('Theta')
            ax.set_ylabel('Probability')
            ax.grid(True)
    fig.suptitle(f"{subject_mapping_dict[subject]} {NA} - Item Response Curves", fontsize=16)        
    plt.tight_layout()
    fig_path = f"./results_{NA}/{subject}/{subject}_{NA}_ICC_plots.png"
    fig.savefig(fig_path, dpi=300)
    print(f"Plot saved to {fig_path}")
#######

def P(t, a, b, c):
    return c + ((1.0 - c) / (1.0 + np.exp(-1.7 * a * (t - b))))
def Q(t, a, b, c):
    return 1.0 - P(t, a, b, c)

def safe_exp(x):
    x = np.clip(x, -np.inf, 50) 
    return np.exp(x).astype(np.float64)  

def d_P(t, a, b, c):
    exponent = -1.7 * a * (t - b)
    v = safe_exp(exponent)
    v = np.clip(v, 0, 1e18)  
    numerator = (1.0 - c) * 1.7 * a * v
    denominator = (1.0 + v) ** 2
    if np.any(denominator > 1e300):
        return 0
    else:
        return np.round(numerator / denominator, 4)

def I_item(t, a, b, c, i):
    epsilon = 1e-8
    d_p_value = d_P(t, a[i], b[i], c[i])
    p_value = P(t, a[i], b[i], c[i])
    q_value = Q(t, a[i], b[i], c[i])
    denominator = p_value * q_value + epsilon
    if isinstance(d_p_value, np.ndarray) and d_p_value.size > 1:
        return np.mean((d_p_value ** 2.0) / denominator)
    else:
        return (d_p_value ** 2.0) / denominator

def I_item_with_p_theta(p, a, b, c, theta, i):
    Q = 1 - p
    exponent = -1.7 * a[i] * (theta - b[i])
    v = safe_exp(exponent) 
    d_P = (1.0 - c[i]) * 1.7 * a[i] * v / ((1.0 + v) ** 2.0)
    return (d_P ** 2.0) / (p * Q + 1e-8)

def csv_information_function(subject, data_store):
    def round_values_up(data, decimals=4):
        if isinstance(data, dict):
            return {key: round_values_up(value, decimals) for key, value in data.items()}
        elif isinstance(data, list):
            return [round_values_up(item, decimals) for item in data]
        elif isinstance(data, (float, np.float32, np.float64)):
            return round(data, decimals)
        return data

    # Applying rounding to individual parameters
    a_1 = np.round(data_store[subject]['NA1']['a'], 4)
    b_1 = np.round(data_store[subject]['NA1']['b'], 4)
    c_1 = np.round(data_store[subject]['NA1']['c'], 4)
    a_2 = np.round(data_store[subject]['NA2']['a'], 4)
    b_2 = np.round(data_store[subject]['NA2']['b'], 4)
    c_2 = np.round(data_store[subject]['NA2']['c'], 4)

    # Rounding dictionary values
    p_theta_response = round_values_up(data_store[subject]['NA1'].get('p_theta_response', {}))
    p_theta_problem = round_values_up(data_store[subject]['NA1'].get('p_theta_problem', {}))

    # Rounding array values
    theta_NA1 = np.round(data_store[subject]['NA1']['theta'], 4)
    theta_NA2 = np.round(data_store[subject]['NA2']['theta'], 4)
    item_ids = data_store[subject]['NA1']['item_ids'] # item_ids {'8094': 1, '29843': 2}
    print('p_theta_response',p_theta_response)
    print('theta',theta_NA1)
    print('item_ids',item_ids)
    M = len(a_1) # 문항 수
    ####### Draw plot Plot x,y-axis.
    x_range = np.arange(-3, 3.01, 0.01)
    y_1 = [sum(I_item(x, a_1, b_1, c_1, i) for i in range(M)) for x in x_range] # 그림 만들기
    y_2 = [sum(I_item(x, a_2, b_2, c_2, i) for i in range(M)) for x in x_range]
    theta_info_1 = [sum(I_item(theta, a_1, b_1, c_1, i) for i in range(M)) for theta in theta_NA1] # 실제 빨간 점. (10문항 total)
    theta_info_2 = [sum(I_item(theta, a_2, b_2, c_2, i) for i in range(M)) for theta in theta_NA2]

    plt.figure()
    plt.plot(x_range, y_1, 'k-', label='NA1 Total Information')
    plt.scatter(theta_NA1, theta_info_1, color='red')  # Red dots for each theta value
    plt.title(f"{subject_mapping_dict[subject]}_NA1 Information Function")
    plt.xlabel('Theta (proficiency levels)')
    plt.ylabel('Information')
    plt.legend()
    plt.savefig(f"./results_NA1/{subject}/NA1_info_plot.png")
    plt.clf()  
    
    plt.plot(x_range, y_2, 'k-', label='Total Information')
    plt.scatter(theta_NA2, theta_info_2, color='red')  # Red dots for each theta value
    plt.title(f"{subject_mapping_dict[subject]}_NA2 Information Function")
    plt.xlabel('Theta (proficiency levels)')
    plt.ylabel('Information')
    plt.legend()
    plt.savefig(f"./results_NA1/{subject}/NA2_info_plot.png")
    plt.clf()  
    
    ####### Calculate info for NA1 #################
    p_theta_problem_Na1_info = [np.mean(I_item_with_p_theta(p_theta_problem[str(i+1)], a_1, b_1, c_1, theta_NA1, i))for i in range(len(a_1))] # 10
    p_theta_response_Na1_info = [np.mean(I_item_with_p_theta(p_theta_response[str(i+1)], a_1, b_1, c_1, theta_NA1, i))for i in range(len(a_1))] # 10
    p_theta_na_1_info = [np.mean(I_item(theta_NA1, a_1, b_1, c_1, i))for i in range(len(a_1))] # 10
    p_theta_na_1_info = np.array(p_theta_na_1_info)
    p_theta_response_Na1_info = np.array(p_theta_response_Na1_info)
    p_theta_problem_Na1_info = np.array(p_theta_problem_Na1_info)
    ####### Calculate info for NA2 #################
    p_theta_problem_Na2_info = [np.mean(I_item_with_p_theta(p_theta_problem[str(i+1)], a_2, b_2, c_2, theta_NA2, i))for i in range(len(a_1))] # 10
    p_theta_response_Na2_info = [np.mean(I_item_with_p_theta(p_theta_response[str(i+1)], a_2, b_2, c_2, theta_NA2, i))for i in range(len(a_1))] # 10
    p_theta_na_2_info = [np.mean(I_item(theta_NA2, a_2, b_2, c_2, i))for i in range(len(a_2))] # 10
    p_theta_na_2_info = np.array(p_theta_na_2_info)
    p_theta_response_Na2_info = np.array(p_theta_response_Na2_info)
    p_theta_problem_Na2_info = np.array(p_theta_problem_Na2_info)
    ### Are NA1 NA2 different? ###
    t_stat, p_value = ttest_ind(p_theta_na_1_info, p_theta_na_2_info)
    different_NA1_NA2 = p_value < 0.05  
    #############  Cohen's d ######################
    def cohen_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
        return (np.mean(x) - np.mean(y)) / pooled_std

    d_NA1_NA2 = cohen_d(p_theta_na_1_info, p_theta_na_2_info)
    d_NA1_Response = cohen_d(p_theta_na_1_info, p_theta_response_Na1_info)
    d_NA1_Problem = cohen_d(p_theta_na_1_info, p_theta_problem_Na1_info)
    d_NA1_Response_Problem = cohen_d(p_theta_response_Na1_info, p_theta_problem_Na1_info)
    ####### Dataframe for  NA1 #################
    results_NA1 = {
    'Item ID': list(item_ids.keys()),
    "NA_1_info": list(p_theta_na_1_info),
    "Are NA1 NA2 Different?": [different_NA1_NA2] + [''] * (M - 1),
    "Cohen's d (NA1 vs NA2)": [d_NA1_NA2] + [''] * (M - 1),
    "Cohen's d (NA1 vs Response)": [d_NA1_Response] + [''] * (M - 1),
    "Cohen's d (NA1 vs Problem)": [d_NA1_Problem] + [''] * (M - 1),
    "Cohen's d (Response vs Problem)": [d_NA1_Response_Problem] + [''] * (M - 1),
    'info_response': list(p_theta_response_Na1_info),
    'info_problem': list(p_theta_problem_Na1_info),
    'diff_response_NA1': list(np.array(p_theta_response_Na1_info) - np.array(p_theta_na_1_info)),
    'diff_problem_NA1': list(np.array(p_theta_problem_Na1_info) - np.array(p_theta_na_1_info)),
    'diff_response_problem': list(np.array(p_theta_response_Na1_info) - np.array(p_theta_problem_Na1_info))}
    df_results = pd.DataFrame(results_NA1)
    data = np.concatenate([p_theta_na_1_info, p_theta_response_Na1_info, p_theta_problem_Na1_info])
    groups = ['NA1'] * len(p_theta_na_1_info) + ['Response'] * len(p_theta_response_Na1_info) + ['Problem'] * len(p_theta_problem_Na1_info)
    f_stat, p_value = f_oneway(p_theta_na_1_info, p_theta_response_Na1_info, p_theta_problem_Na1_info)
    num_rows = len(df_results)
    df_results['ANOVA_F-stat'] = [f_stat] + [''] * (num_rows - 1)
    df_results['ANOVA_P-value'] = [p_value] + [''] * (num_rows - 1)
    if p_value < 0.05:
        tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
        tukey_results = str(tukey.summary())
    else:
        tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
        tukey_results = "No significant differences found by ANOVA.\n" + str(tukey.summary())
    df_results['Tukey_HSD'] = [tukey_results] + [''] * (num_rows - 1)
    csv_path = f'./results_NA1/{subject}/{subject}_NA1_info_comparison.csv'
    df_results.to_csv(csv_path, index=False)

    ####### Dataframe for  NA2 #################
    d_NA2_Response = cohen_d(p_theta_na_2_info, p_theta_response_Na2_info)
    d_NA2_Problem = cohen_d(p_theta_na_2_info, p_theta_problem_Na2_info)
    d_NA2_Response_Problem = cohen_d(p_theta_response_Na2_info, p_theta_problem_Na2_info)
    results_NA2 = {
    'Item ID': list(item_ids.keys()),
    "NA_2_info": list(p_theta_na_2_info),
    "Are NA1 NA2 Different?": [different_NA1_NA2] + [''] * (M - 1),
    "Cohen's d (NA1 vs NA2)": [d_NA1_NA2] + [''] * (M - 1),
    "Cohen's d (NA2 vs Response)": [d_NA2_Response] + [''] * (M - 1),
    "Cohen's d (NA2 vs Problem)": [d_NA2_Problem] + [''] * (M - 1),
    "Cohen's d (Response vs Problem)": [d_NA2_Response_Problem] + [''] * (M - 1),
    'info_response': list(p_theta_response_Na2_info),
    'info_problem': list(p_theta_problem_Na2_info),
    'diff_response_NA2': list(np.array(p_theta_response_Na2_info) - np.array(p_theta_na_2_info)),
    'diff_problem_NA2': list(np.array(p_theta_problem_Na2_info) - np.array(p_theta_na_2_info)),
    'diff_response_problem': list(np.array(p_theta_response_Na2_info) - np.array(p_theta_problem_Na2_info))}
    df_results = pd.DataFrame(results_NA2)

    data = np.concatenate([p_theta_na_2_info, p_theta_response_Na2_info, p_theta_problem_Na2_info])
    groups = ['NA2'] * len(p_theta_na_2_info) + ['Response'] * len(p_theta_response_Na2_info) + ['Problem'] * len(p_theta_problem_Na2_info)
    f_stat, p_value = f_oneway(p_theta_na_2_info, p_theta_response_Na2_info, p_theta_problem_Na2_info)
    df_results['ANOVA_F-stat'] = [f_stat] + [''] * (num_rows - 1)
    df_results['ANOVA_P-value'] = [p_value] + [''] * (num_rows - 1)
    if p_value < 0.05:
        tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
        tukey_results = str(tukey.summary())
    else:
        tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
        tukey_results = "No significant differences found by ANOVA.\n" + str(tukey.summary())
    df_results['Tukey_HSD'] = [tukey_results] + [''] * (num_rows - 1)
    csv_path = f'./results_NA2/{subject}/{subject}_NA2_info_comparison.csv'
    df_results.to_csv(csv_path, index=False)
    
    
def plot_information_function(subject, NA, item_ids, a, b, c, theta, data_store, p_theta_response):
    
    def I_item_plot(t, a, b, c, epsilon=1e-8):
        P_theta = 1 / (1 + np.exp(-1.7 * a * (t - b)))
        Q_theta = 1.0 - P_theta
        d_P = (1.0 - c) * 1.7 * a * P_theta * Q_theta
        info = (d_P ** 2.0) / (P_theta * Q_theta + epsilon)
        return info

    info_plot = np.linspace(-3, 3, 100)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    item_ids_dict = item_ids.to_dict()
    for index, ax in enumerate(axes):
        try:
            item_id = next((v for k, v in item_ids_dict.items() if k == index), None)
            if item_id is None:
                continue
            a_i = a[index]
            b_i = b[index]
            c_i = c[index]

            info_values_plot = [I_item_plot(t, a_i, b_i, c_i) for t in info_plot]
            ax.plot(info_plot, info_values_plot, 'b-', label='Avg Info (Theta -3 to 3)')
            info_values_real = [I_item_plot(t, a_i, b_i, c_i) for t in theta]
            ax.plot(theta, info_values_real, 'ro', label='Avg Info (Theta from NA)')
            ax.set_title(f'Item ID: {item_id}', fontsize=12)
            ax.set_xlabel('Theta')
            ax.set_ylabel('Information')
            ax.grid(True)
        except Exception as e:
            print(f"Error processing item {index + 1}: {e}")

    # Clean up plot layout
    for j in range(len(item_ids), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{subject_mapping_dict[subject]}_{NA} Information Function", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = f"./results_{NA}/{subject}/{subject}_{NA}_information_plots.png"
    fig.savefig(fig_path, dpi=300)
    print(f"Info Plot saved to {fig_path}")
######
## Loading the model parameters ##
subject_list = ['수학Ⅰ,수학Ⅱ', '독해', '독서,문학', '한국사', '생활과 윤리', '정치와 법', '한국지리', '세계지리', '사회?문화', '동아시아사', '세계사', '윤리와 사상', '경제', '물리학Ⅰ', '화학Ⅰ', '생명과학Ⅰ', '지구과학Ⅰ', '지구과학Ⅱ', '생명과학Ⅱ', '화학Ⅱ', '물리학Ⅱ']
subject_list = ['동아시아사']
NA_list = ["NA1", "NA2"]
# NA_list = ["NA1"]

# Import datas
data_store = {}
for subject in subject_list:
    data_store[subject] = {}
    for NA in NA_list:
        problem_seq_mapping_path = os.path.join('./processed', f'10_{subject}_problem_seq_mapping_dict.json')
        model_parameters_path = os.path.join(f"./results_{NA}/{subject}/irt.params")
        p_theta_response_path = os.path.join("./detailed", f'10_{subject}_p_theta_response.json')
        p_theta_problem_path = os.path.join("./detailed", f'10_{subject}_p_theta_problem.json')

        # item id from question_detailed.csv
        file_path = f'./detailed/10_{subject}_question_detailed.csv'
        item_id_temp = pd.read_csv(file_path, encoding='utf-8')
        try:
            data_store[subject][NA] = {'item_ids': item_id_temp['Problem Seq']}
        except:
            data_store[subject][NA] = {'item_ids': item_id_temp['problem_seq']}
        try:
            model_parameters = torch.load(model_parameters_path)
            a = torch.squeeze(model_parameters['a.weight'], dim=-1).detach().cpu().numpy()
            # a = torch.nn.functional.softplus(torch.tensor(a)).numpy()
            b = torch.squeeze(model_parameters['b.weight'], dim=-1).detach().cpu().numpy()
            c = torch.squeeze(model_parameters['c.weight'], dim=-1).detach().cpu().numpy()
            # c = torch.sigmoid(torch.tensor(c)).numpy()
            theta = torch.squeeze(model_parameters['theta.weight'], dim=-1).detach().cpu().numpy()
            data_store[subject][NA].update({
                'a': torch.nn.functional.softplus(torch.tensor(a)).numpy(),
                'b': b,
                'c': torch.sigmoid(torch.tensor(c)).numpy(),
                'theta': theta
            })

        except Exception as e:
            print(f"Error loading model parameters: {e}")
            continue

        try:
            with open(p_theta_response_path, 'r', encoding='utf-8') as f:
                p_theta_response = json.load(f)
            with open(p_theta_problem_path, 'r', encoding='utf-8') as f:
                p_theta_problem = json.load(f)
            data_store[subject][NA].update({
                'p_theta_response': p_theta_response,
                'p_theta_problem': p_theta_problem
            })
        except FileNotFoundError as e:
            print(f"Error loading P theta data: {e}")
            continue

for subject in subject_list:
    for NA in NA_list:
        if 'a' in data_store[subject][NA]:
            plot_information_function(subject, NA, data_store[subject][NA]['item_ids'], data_store[subject][NA]['a'], data_store[subject][NA]['b'], data_store[subject][NA]['c'], data_store[subject][NA]['theta'], data_store, data_store[subject][NA].get('p_theta_response', {}))
            plot_icc_curve(subject, NA, data_store[subject][NA]['item_ids'], data_store[subject][NA]['a'], data_store[subject][NA]['b'], data_store[subject][NA]['c'], data_store[subject][NA]['theta'], data_store)
    if 'a' in data_store[subject][NA]:
        csv_information_function(subject,data_store)
            
            

        