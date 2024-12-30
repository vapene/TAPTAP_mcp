# 2024/7/23 @ vapene208
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR    

class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, irf_kwargs=None):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.b.weight)
        nn.init.xavier_uniform_(self.c.weight)
        nn.init.xavier_uniform_(self.theta.weight)

    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        c = torch.sigmoid(c)
        a = F.softplus(a)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):
            if torch.isnan(a).any():
                print("Warning: A NaN values found a parameters!")
            elif torch.isnan(b).any():
                print("Warning: B NaN values found b parameters!")
            elif torch.isnan(c).any():
                print("Warning: C NaN values found c parameters!")
            elif torch.isnan(theta).any():
                print("Warning: Theta NaN values found theta parameters!")
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')
        return c + (1 - c) / (1 + torch.exp(-1.73 * a * (theta - b)+ 1e-8))

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.75, gamma=1, device='cpu'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        try:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        except:
            inputs = inputs.unsqueeze(0)
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
class GDIRT_inference(object):
    def __init__(self, user_num, item_num, alpha, gamma, device="cpu"):
        super(GDIRT_inference, self).__init__()
        self.irt_net = IRTNet(user_num, item_num).to(device)
        self.loss_function = WeightedFocalLoss(alpha, gamma, device=device)

    def train(self, train_data) -> ...:
        
        for param in self.irt_net.parameters():
            param.requires_grad = False  # Freeze all parameters initially

        for param in self.irt_net.theta.parameters():
            param.requires_grad = True

        trainer = torch.optim.Adam(self.irt_net.parameters(), lr=0.01)
        scheduler = StepLR(trainer, step_size=100, gamma=0.1)

        for _ in range(400): 
            self.irt_net.train()
            for batch_data in train_data:
                user_id, item_id, response = batch_data
                user_id: torch.Tensor = user_id
                item_id: torch.Tensor = item_id
                predicted_response: torch.Tensor = self.irt_net(user_id, item_id)
                response: torch.Tensor = response
                loss = self.loss_function(predicted_response.squeeze(), response.float())
                trainer.zero_grad()
                loss.backward()
                trainer.step()
                scheduler.step()

    def load_param(self, weights_dict):
        with torch.no_grad():  
            self.irt_net.a.weight.data.copy_(weights_dict['a.weight'].clone().detach())
            self.irt_net.b.weight.data.copy_(weights_dict['b.weight'].clone().detach())
            self.irt_net.c.weight.data.copy_(weights_dict['c.weight'].clone().detach())
            self.irt_net.theta.weight.data[:-1]= weights_dict['theta.weight'].clone().detach()

    def get_rank(self, criteria):
        theta_weights = self.irt_net.theta.weight.data.clone().detach().numpy()
        target_score = theta_weights[-1]
        rate = 'low' if target_score[0] < criteria[0] else 'middle' if target_score[0] < criteria[1] else 'high'

        return rate, target_score[0]

    def get_rank_threshold(self, criteria):
        theta_weights = self.irt_net.theta.weight.data.clone().detach().numpy()
        ###
        # criteria = np.percentile(theta_weights, [30, 60])
        # criteria_dict = {
        #     '30_percentile': float(criteria[0]),
        #     '60_percentile': float(criteria[1])
        # }
        # file_path = f"/root/jw/EduCDM/examples/IRT/GD_inference/integration_test/{subject}_threshold.json"
        # import json
        # with open(file_path, 'w') as json_file:
        #     json.dump(criteria_dict, json_file, indent=4)
        ##
        target_score = theta_weights[-1]
        rate = 'low' if target_score[0] < criteria[0] else 'middle' if target_score[0] < criteria[1] else 'high'

        return rate, target_score[0]