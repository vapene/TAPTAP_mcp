# coding: utf-8
# 2021/3/23 @ tongshiwei
from IRT_freeze_abc import GDIRT_inference
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import random 
import argparse
import torch
import yaml
import numpy as np
import json

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

# CSV 파일을 TensorDataset으로 변환
def transform(user, item, score, batch_size, device):
    data_set = TensorDataset(
        torch.tensor(user.to_numpy(), dtype=torch.int64).to(device) - 1,
        torch.tensor(item.to_numpy(), dtype=torch.int64).to(device) - 1, 
        torch.tensor(score.to_numpy(), dtype=torch.float32).to(device))
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

# argument 받기. 
# subjects = {"수학Ⅰ,수학Ⅱ":'수학',"독해":'영어',"독서,문학":"국어"}
# alpha, gamma, batch_size는 best_configs.yml 파일에서 가져옴.
parser = argparse.ArgumentParser(description='Train a model on user data')
parser.add_argument('--subjects', type=str, default='독해', help='Comma-separated list of subjects to include') # "수학Ⅰ,수학Ⅱ", "생명과학Ⅰ", "독서,문학", "사회?문화", "생활과 윤리", "지구과학Ⅰ", "독해,듣기"
parser.add_argument('--device', type=int, default=-1, help='device')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--alpha', type=float, default=0.70, help='alpha 값이 <0.5 면 1에 more weight, >0.5면 0에 more weight. ')
parser.add_argument('--gamma', type=float, default=1, help='gamma 값이 높으면 prediction error 가 높은 hard saple에 more weight')
parser.add_argument('--batch_size', type=int, default=256, help='seed')
args = parser.parse_args()

# best_configs.yml 파일에서 과목 당 최적의 alpha, gamma, batch_size 가져오기.
def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)
    configs = configs[args.subjects]
    for k, v in configs.items():
        if hasattr(args, k):
            if k in ["batch_size"]:
                setattr(args, k, int(v))
            elif k in ["alpha", "gamma"]:
                setattr(args, k, float(v))
    return args
args = load_best_configs(args, "../GD/configs_NA1.yml")
set_seed(args.seed)
# cpu만 사용하도록 설정
device = "cpu"

#################### load weights #############################
original_weights = torch.load(f"../GD/results_NA1/{args.subjects}/irt.params")
n_user = original_weights['theta.weight'].shape[0]+1   # 518
n_item = original_weights['a.weight'].shape[0]     # 10
#################################################


#################### load dataset ############################
train_data = pd.read_csv(f"../GD/processed/10_{args.subjects}_30_ratio_afterNA1.csv")
train_data.rename(columns={
    'member_idx_mapped': 'user_id',
    'standardized_problem_seq_mapped': 'item_id',
    'correct_yn': 'score'
}, inplace=True)
columns_to_keep = ['user_id', 'item_id', 'score']
train_data = train_data[columns_to_keep]
train_data.dropna(inplace=True)
train_data['score'] = train_data['score'].astype(int)
# 과목별 상, 중, 하 threshold 기준 load
with open("middle_high_rank.json", "r") as file:
    criteria = json.load(file)

# 여기는 확인용 나중에 없에주세요. 
train_data = train_data.sample(frac=1).reset_index(drop=True)
# which student to inference? 
infer_student = 11
train_data = train_data[train_data['user_id'] == infer_student]
train_data['user_id'] = train_data['user_id'].replace(infer_student, n_user-1) # 107을 518로 바꿔줌.

# transform dataset to tensor
train_th_inference, _, _ = [
    transform(data["user_id"], data["item_id"], data["score"], args.batch_size, device)
    for data in [train_data, train_data, train_data]]
#################################################

cdm_inference = GDIRT_inference(n_user, n_item, alpha=args.alpha, gamma=args.gamma, device=device)
cdm_inference.load_param(original_weights)
cdm_inference.train(train_data=train_th_inference)
print('args.subjects', criteria[args.subjects], args.subjects)
rank,target_score = cdm_inference.get_rank(criteria[args.subjects])
# rank: 등수, tag: 상,중,하 등급, target_score: theta 학생의 과목 능력치
print('result', rank,'rank', target_score,'능력치')