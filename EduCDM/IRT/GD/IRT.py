# coding: utf-8
# 2021/4/23 @ tongshiwei

import logging
import numpy as np
import torch

from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from ..irt import irt3pl, irt2pl, irt1pl
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, value_range, a_range, type_, irf_kwargs=None):
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
        self.value_range = value_range
        self.a_range = a_range
        self.type_ = type_
        
    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        c = torch.sigmoid(c)
        a = F.softplus(a)
        # print('soft a',a)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            if torch.isnan(a).any():
                print("Warning: A NaN values found a parameters!")
            elif torch.isnan(b).any():
                print("Warning: B NaN values found b parameters!")
            elif torch.isnan(c).any():
                print("Warning: C NaN values found c parameters!")
            elif torch.isnan(theta).any():
                print("Warning: Theta NaN values found theta parameters!")
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')

        return self.irf(theta, a, b, c, self.type_, **self.irf_kwargs)
    
    # self.irt = return c + (1 - c) / (1 + np.exp(-1.73 * a * (theta - b)))
    

    @classmethod
    def irf(cls, theta, a, b, c, type_, **kwargs):
        if type_==3:
            return irt3pl(theta, a, b, c, **kwargs)
        elif type_==2:
            return irt2pl(theta, a, b, c, **kwargs)
        elif type_==1:
            return irt1pl(theta, a, b, c, **kwargs)
        else:
            print('check IRT type')


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.75, gamma=1, device='cpu'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        # print('\n inputs',inputs, inputs.shape)
        # print('\n targets',targets, targets.shape)
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
    
class IRT(CDM):
    def __init__(self, user_num, item_num, alpha, gamma, value_range=None, a_range=None, type_=3, device="cpu"):
        super(IRT, self).__init__()
        self.irt_net = IRTNet(user_num, item_num, value_range, a_range, type_=type_).to(device)
        self.loss_function = WeightedFocalLoss(alpha, gamma, device=device)
        # self.loss_function = nn.BCELoss()
    def train(self, train_data, valid_data=None, test_data=None, *, epoch: int, lr=0.001, patience= 50) -> ...:
    
        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        if valid_data:
            best_score = 0  # Initialize the best AUC
            best_model_params = None  # Store the best model parameters
            no_improve_epoch = 0  # Counter for epochs without improvement

            for e in range(epoch):
                self.irt_net.train()
                for batch_data in tqdm(train_data, desc=f"Training Epoch {e}"):
                    user_id, item_id, response = batch_data
                    predicted_response = self.irt_net(user_id, item_id)
                    loss = self.loss_function(predicted_response.squeeze(), response.float())
                    # loss = self.loss_function(predicted_response, response)
                    trainer.zero_grad()
                    loss.backward()
                    trainer.step()

                # Validation loop
                self.irt_net.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for valid_batch_data in tqdm(valid_data, desc=f"Validation Epoch {e}"):
                        user_id, item_id, response = valid_batch_data
                        user_id = user_id
                        item_id = item_id
                        response = response

                        predicted_response = self.irt_net(user_id, item_id)
                        all_preds.extend(predicted_response.squeeze().cpu().numpy())
                        all_labels.extend(response.cpu().numpy())

                # Compute AUC
                if len(np.unique(all_labels)) > 1:
                    acc_score = accuracy_score(all_labels, np.array(all_preds) >= 0.5)
                    
                    auc_score = roc_auc_score(all_labels, all_preds)
                    mean_score = (auc_score+acc_score)/2
                    # print(f"[Epoch {e}] Validation AUC: {auc_score:.6f} Validation ACC: {acc_score:.6f}")

                    if mean_score > best_score:
                        # print(f"AUC improved to {auc_score:.6f}  ACC improved to {acc_score:.6f}")
                        best_score = mean_score
                        no_improve_epoch = 0
                        # Save the best model parameters
                        best_model_params = self.irt_net.state_dict()
                    else:
                        no_improve_epoch += 1
                        # print(f"No improvement in AUC for {no_improve_epoch} epochs")
                        if no_improve_epoch >= patience:
                            print("Early stopping triggered")
                            # Load the best model parameters
                            self.irt_net.load_state_dict(best_model_params)
                            break
                else:
                    print("Not enough class labels to calculate AUC")
        else:
         
            for e in range(epoch):
                self.irt_net.train()

                losses = []
                for batch_data in tqdm(train_data, "Epoch %s" % e):
                    user_id, item_id, response = batch_data
                    user_id: torch.Tensor = user_id
                    item_id: torch.Tensor = item_id
                    predicted_response: torch.Tensor = self.irt_net(user_id, item_id)
                    response: torch.Tensor = response
                    loss = self.loss_function(predicted_response.squeeze(), response.float())
                    # back propagation
                    trainer.zero_grad()
                    loss.backward()
                    trainer.step()

                #     losses.append(loss.mean().item())
                # print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

        # if test_data is not None:
        #     auc_score, acc_score, fpr, recall = self.eval(test_data)
        #     print("[Epoch %d] auc: %.6f, accuracy: %.6f, fpr: %.6f, recall: %.6f" % (e, auc_score, acc_score, fpr, recall))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():  # Ensures that operations do not track history
            for batch_data in tqdm(test_data, "evaluating"):
                user_id, item_id, response = batch_data
                user_id: torch.Tensor = user_id
                item_id: torch.Tensor = item_id
                pred: torch.Tensor = self.irt_net(user_id, item_id)
                y_pred.extend(pred.squeeze().detach().cpu().numpy())  # Use .detach() here
                y_true.extend(response.cpu().numpy())

        binary_predictions = np.array(y_pred) >= 0.5
        auc_score = roc_auc_score(y_true, y_pred)
        acc_score = accuracy_score(y_true, binary_predictions)
        tn, fp, fn, tp = confusion_matrix(y_true, binary_predictions).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return auc_score, acc_score, fpr, recall

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        # logging.info("save parameters to %s" % filepath)

    def load_leaveoneout(self, weights_dict, num_users, num_items, leaveoneout_index=10):
        with torch.no_grad():  
            # replace all a,b,c
            self.irt_net.a.weight.data.copy_(weights_dict['a.weight'].clone().detach())
            self.irt_net.b.weight.data.copy_(weights_dict['b.weight'].clone().detach())
            self.irt_net.c.weight.data.copy_(weights_dict['c.weight'].clone().detach())
            all_indices = list(range(num_users))
            indices_to_update = all_indices[:leaveoneout_index] + all_indices[leaveoneout_index+1:]
            mask = torch.tensor(indices_to_update)
            # print("weights_dict['theta.weight']",weights_dict['theta.weight'][:5])
            self.irt_net.theta.weight.data[:-1] = weights_dict['theta.weight'].clone().detach()[mask]
            # print('self.irt_net_theta_weight',self.irt_net.theta.weight.data[:5])

            
    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))
        # logging.info("load parameters from %s" % filepath)
