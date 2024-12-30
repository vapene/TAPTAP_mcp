# coding: utf-8
# 2021/3/23 @ tongshiwei
import sys
sys.path.append('../../..') 
from EduCDM import GDIRT
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import random 
import argparse
import torch
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

def transform(user, item, score, batch_size, device):
    data_set = TensorDataset(
        torch.tensor(user.to_numpy(), dtype=torch.int64).to(device) - 1,  # Convert user Series to numpy array
        torch.tensor(item.to_numpy(), dtype=torch.int64).to(device) - 1,  # Convert item Series to numpy array
        torch.tensor(score.to_numpy(), dtype=torch.float32).to(device)    # Convert score Series to numpy array
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


parser = argparse.ArgumentParser(description='Train a model on user data')
parser.add_argument('--subjects', type=str, default='독해,듣기', help='수학Ⅰ,수학Ⅱ')
parser.add_argument('--num_problems', type=int, default=10, help='device')

parser.add_argument('--device', type=int, default=-1, help='device')
parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train the model')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
parser.add_argument('--timestamp', type=str, default='20240626_091911', help='timestamp for saving files')
parser.add_argument('--trial', type=int, default=5, help='numer of trials')
parser.add_argument('--evaluation', action='store_true', default=False, help='Use to indicate saving IRT evaluation results.')

parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--alpha', type=float, default=0.70, help='alpha 값이 <0.5 면 1에 more weight, >0.5면 0에 more weight. ')
parser.add_argument('--gamma', type=float, default=1, help='gamma 값이 높으면 prediction error 가 높은 hard saple에 more weight')
parser.add_argument('--batch_size', type=int, default=256, help='seed')
parser.add_argument('--param_type', type=int, default=3, help='3 parameter model')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of data to use for training (remainder split between validation and test)')
args = parser.parse_args()

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)
    configs = configs[args.subjects]
    print(f"Loaded configs: {configs}")
    for k, v in configs.items():
        # if hasattr(args, k):
            # if k == "learning_rate":
            #     print('yes learning rate')
            #     setattr(args, k, float(v))
        if k == "batch_size":
            print('yes batch size')
            setattr(args, k, int(v))
        elif k in ["alpha", "gamma"]:
            setattr(args, k, float(v))
        
    print("------ Use best configs ------")
    return args

if args.evaluation == True:
    args.trial = 1
    args = load_best_configs(args, "./configs_NA1.yml")


set_seed(args.seed)
if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
try:
    train_data = pd.read_csv(f"./processed/{args.num_problems}_{args.subjects}_30_ratio_afterNA1.csv")  # Assume file path is correct
except FileNotFoundError:
    print('arguments',args)
    exit()
# ################################################
train_data.rename(columns={
    'member_idx_mapped': 'user_id',
    'standardized_problem_seq_mapped': 'item_id',
    'correct_yn': 'score' 
}, inplace=True)
columns_to_keep = ['user_id', 'item_id', 'score']
train_data = train_data[columns_to_keep]

train_data.dropna(inplace=True)
train_data['score'] = train_data['score'].astype(int) 


auc_score_list=[]
acc_score_list=[]
fpr_score_list=[]
recall_score_list=[]
mean_score_list=[]
try:
    for i in range(args.trial):
    # for i in range(10):
        train, temp = train_test_split(train_data, test_size=(1-args.train_ratio), random_state=i)
        # 나머지 데이터를 검증 및 테스트로 분할 (각각 50%씩 -> 전체 데이터의 20%)
        valid, test = train_test_split(temp, test_size=0.5, random_state=i)

        #####################################
        n_user = np.max(train_data['user_id']) # 4128
        n_item = np.max(train_data['item_id'])

        train_th, valid_th, test_th = [
            transform(data["user_id"], data["item_id"], data["score"], args.batch_size, device)
            for data in [train, valid, test]]

        if len(train)==len(valid):
            valid=None

        cdm = GDIRT(n_user, n_item, alpha=args.alpha, gamma=args.gamma, type_=args.param_type, device=device)

        cdm.train(train_data=train_th, valid_data=valid_th, test_data=test_th, epoch=args.epochs, lr=args.learning_rate, patience=args.patience)
        if args.evaluation == True:
            cdm.save(f"./results_NA1/{args.subjects}/irt.params")
            cdm.load(f"./results_NA1/{args.subjects}/irt.params")
        auc_score, acc_score, fpr, recall = cdm.eval(test_th)

        print("auc: %.6f, accuracy: %.6f, fpr: %.6f, recall: %.6f" % (auc_score, acc_score, fpr, recall))
        print('args.alpha',args.alpha)
        # FPR은 낮은 수록 좋음. 나머지는 높을수록 좋음.
        # FPR(False Positive Rate): 실제 False인 data 중에서 모델이 True라고 예측한 비율입니다. 
        # Recall: 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율입니다. 
        auc_score_list.append(auc_score)
        acc_score_list.append(acc_score)
        fpr_score_list.append(fpr)
        recall_score_list.append(recall)
        mean_score = auc_score+acc_score+(100-fpr)
        print('mean',mean_score)
        mean_score_list.append(mean_score)
except Exception as e:
    print(f"Encountered an error, stopping further trials: {e}")
    print(f'mean auc score is {0}, {0}')
    print(f'mean acc score is {0}, {0}' )
    print(f'mean fpr score is {100}, {100}')
    print(f'mean recall score is {0}, {0}')
    print(f"Combined Mean Score:{0}, {0}")
    
print(f'mean auc score is {np.mean(auc_score_list)}, {auc_score_list}')
print(f'mean acc score is {np.mean(acc_score_list)}, {acc_score_list}' )
print(f'mean fpr score is {np.mean(fpr_score_list)}, {fpr_score_list}')
print(f'mean recall score is {np.mean(recall_score_list)}, {recall_score_list}')
print(f"Combined Mean Score:{np.mean(mean_score_list)}, {mean_score_list}")


# #################    save parameters   ###########################
# if args.evaluation==True:
#     a = torch.squeeze(cdm.irt_net.a.weight, dim=-1).detach().cpu().numpy()
#     a = torch.nn.functional.softplus(torch.tensor(a)).numpy()
#     b = torch.squeeze(cdm.irt_net.b.weight, dim=-1).detach().cpu().numpy()
#     c = torch.squeeze(cdm.irt_net.c.weight, dim=-1).detach().cpu().numpy()
#     c = torch.sigmoid(torch.tensor(c)).numpy()
#     theta = torch.squeeze(cdm.irt_net.theta.weight, dim=-1).detach().cpu().numpy()

#     abc_df = pd.DataFrame({
#         'a': a,
#         'b': b,
#         'c': c
#     })

#     theta_df = pd.DataFrame({
#         'theta': theta
#     })

#     abc_df.to_csv(f'./draw_figure/abc_{args.param_type}.csv', index=False)
#     theta_df.to_csv(f'./draw_figure/theta_{args.param_type}.csv', index=False)
