import os
import random
import json
import torch
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool, cpu_count
from IRT_freeze_abc import GDIRT_inference
from torch.utils.data import TensorDataset, DataLoader
import yaml
import itertools


def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

# CSV 파일을 TensorDataset으로 변환
def transform(user_id, problem_id, correct_yn, batch_size, device):
    dataset = TensorDataset(
        torch.tensor(user_id, dtype=torch.int64).to(device) - 1,
        torch.tensor(problem_id, dtype=torch.int64).to(device) - 1,
        torch.tensor(correct_yn, dtype=torch.float32).to(device)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_best_configs(args, path):
        with open(path, "r") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = configs.get(args.subjects, {})
        for k, v in configs.items():
            if hasattr(args, k):
                if k == "batch_size":
                    setattr(args, k, int(v))
                elif k in ["alpha", "gamma"]:
                    setattr(args, k, float(v))
        return args
    
def process_subject(args): 
    subject_ = args['subject'] 
    irt_subjects = args['irt_subjects']
    device = args['device']
    class Args:
        def __init__(self):
            self.subjects = subject_  
            self.device = -1            
            self.seed = 42              
            self.alpha = 0.70           
            self.gamma = 1              
            self.batch_size = 256      
    args_model = Args()
    args_model = load_best_configs(args_model, "../GD/configs_NA1.yml")
    set_seed(args_model.seed)
    original_weights = torch.load(f"../GD/results_NA1/{args_model.subjects}/irt.params", map_location=device)
    if subject_ in irt_subjects:
        apply_IRT = True        
    else:
        apply_IRT = False
    
    n_user = original_weights['theta.weight'].shape[0] + 1   
    n_item = original_weights['a.weight'].shape[0]    
    
    with open("./middle_high_rank.json", 'r', encoding='utf-8') as f:
        irt_criteria = json.load(f)
    with open(f"../GD/processed/10_{args_model.subjects}_problem_point_mapping_dict.json", 'r', encoding='utf-8') as f:
        baejeom = json.load(f)

    len_random_list = len(baejeom) # 국어는 8문항
    
    subject_random_lists = {}
    all_combinations = list(itertools.product([0, 1], repeat=len_random_list))
    random.shuffle(all_combinations)
    subject_random_lists = {idx: list(comb) for idx, comb in enumerate(all_combinations)} # {0: [1, 0, 1, 1, 1],

    final_json = {"subject": args_model.subjects, "rank": [], "solving_history": [], "target_score": []}

    rank_counts = {'low': 0, 'middle': 0, 'high': 0}  # Initialize rank counts
    
    for idx_random_list, random_list_ in subject_random_lists.items():
        
        set_seed(args_model.seed)
        user_id = [n_user] * n_item
        item_ids = np.arange(1, n_item + 1, dtype='int64')
        train_th_inference = transform(np.array(user_id, dtype='int64'),
        item_ids,
        np.array(random_list_, dtype='int64'),
        # np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype='int64'),
        args_model.batch_size, device)
        
        if apply_IRT:
            cdm_inference = GDIRT_inference(n_user, n_item, alpha=args_model.alpha, gamma=args_model.gamma, device=device)
            cdm_inference.load_param(original_weights)
            cdm_inference.train(train_data=train_th_inference)
            rank, target_score = cdm_inference.get_rank(irt_criteria[args_model.subjects])
            # print('\n result', rank, 'target_score', target_score, 'random_list_', random_list_)
        else:
            difficulties = list(baejeom.values())
            responses = random_list_

            # Calculate total count and target score
            target_score = sum(r * d for r, d in zip(responses, difficulties))

            # Initialize inconsistency count
            inconsistency_count = 0
            total_possible_inconsistencies = 0

            # Iterate over all pairs of questions
            for i in range(len(difficulties)):
                for j in range(i + 1, len(difficulties)):
                    if difficulties[i] < difficulties[j]:
                        total_possible_inconsistencies += 1
                        if responses[i] == 0 and responses[j] == 1:
                            inconsistency_count += 1
                    elif difficulties[i] > difficulties[j]:
                        total_possible_inconsistencies += 1
                        if responses[i] == 1 and responses[j] == 0:
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

            criteria = irt_criteria[args_model.subjects]
            rank = 'low' if target_score < criteria[0] else 'middle' if target_score < criteria[1] else 'high'
             
            # print('\n result', rank, 'target_score', target_score, 'random_list_', random_list_)f

        final_json["rank"].append(rank)
        final_json["target_score"].append(target_score)
        final_json["solving_history"].append(random_list_)
        rank_counts[rank] += 1  

        if idx_random_list == len(subject_random_lists) - 1:
            folder_path = 'integration_test'
            os.makedirs(folder_path, exist_ok=True)  

            file_name = f"{args_model.subjects}.json"
            file_path = os.path.join(folder_path, file_name)

            def convert_to_native(data):
                if isinstance(data, dict):
                    return {k: convert_to_native(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [convert_to_native(v) for v in data]
                elif isinstance(data, np.generic):
                    return data.item()
                else:
                    return data

            final_json_native = convert_to_native(final_json)

            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(final_json_native, json_file, ensure_ascii=False, indent=4)

            print(f"Saved {file_name} successfully!")
            break

if __name__ == '__main__':
    all_subjects = [ '통합사회', '통합과학', '수학Ⅰ,수학Ⅱ', '독해', '독서,문학', '물리학Ⅰ', '화학Ⅰ', '생명과학Ⅰ', '지구과학Ⅰ', '지구과학Ⅱ', '생명과학Ⅱ', '화학Ⅱ', '물리학Ⅱ', '한국사', '생활과 윤리', '정치와 법', '한국지리', '세계지리', '사회?문화', '동아시아사', '세계사', '윤리와 사상', '경제' ]
    # all_subjects = ["동아시아사"]
    irt_subjects = ['독서,문학', '독해','사회?문화','생명과학Ⅰ', '생활과 윤리', '수학Ⅰ,수학Ⅱ', '지구과학Ⅰ']
    # irt_subjects = []
    non_irt_subjects = [subject for subject in all_subjects if subject not in irt_subjects]
        
    device = torch.device("cpu") 
    args_list = [{'subject': subject_, 'irt_subjects': irt_subjects, 'device': device} for subject_ in all_subjects]
    # for args in args_list:
    #     process_subject(args)
    with Pool(16) as pool:
        pool.map(process_subject, args_list)

    csv_data = []
    folder_path = "./integration_test"
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                subjects = data.get('subject', 'Unknown')
                for rank, history, score in zip(data['rank'], data['solving_history'], data['target_score']):
                    csv_data.append({
                        'Subject': subjects,
                        'Rank': rank,
                        'Solving History': history,
                        'Target Score': score
                    })

    csv_df = pd.DataFrame(csv_data)
    csv_file_path = os.path.join(folder_path, 'compiled_data.csv')
    csv_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

    print(f"CSV file created successfully at {csv_file_path}!")
    
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

    print("Temporary JSON files deleted successfully!")