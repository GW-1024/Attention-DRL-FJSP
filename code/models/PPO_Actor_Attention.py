# 这是 PPO_Actor_Attention.py 文件的完整内容
import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic,MLP
import torch.nn.functional as F
# --- (修改 1) 导入 AttentionEncoder 而不是 GraphCNN ---
from models.attentionEncoder import GraphAttentionEncoder
# ---
from torch.distributions.categorical import Categorical
import torch
from Params import configs
from agent_utils import select_action1,greedy_select_action,select_action2

INIT = configs.Init
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        encoder_transform = self.W1(encoder_outputs)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        return u_i

# --- (修改 2) 删除了 GNN 的 Encoder class ---

class Job_Actor(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 num_layers,
                 learn_eps, # <--- 确保 learn_eps 在参数里
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device
                 ):
        super(Job_Actor, self).__init__()
        self.n_j = n_j
        self.device=device
        self.bn = torch.nn.BatchNorm1d(input_dim).to(device)
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device

        # --- (修改 3) 替换 GNN Encoder 为 Attention Encoder ---
        self.n_heads = 8  # Transformer 的头数，您可以稍后将其添加到 Params.py
        self.encoder = GraphAttentionEncoder(
                            n_heads=self.n_heads,
                            embed_dim=hidden_dim, 
                            n_layers=num_layers,
                            node_dim=input_dim, # <--- 确保 node_dim 是 input_dim
                            normalization='batch',
                            feed_forward_hidden=hidden_dim * 4 
                        ).to(device)
        # ---

        self._input = nn.Parameter(torch.Tensor(hidden_dim))
        self._input.data.uniform_(-1, 1).to(device)
        self.actor1 = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self,
                x,
                graph_pool, # (Attention 模型不使用)
                padded_nei, # (Attention 模型不使用)
                adj,        # (Attention 模型不使用)
                candidate,
                mask,
                mask_mch,
                dur,
                a_index,
                old_action,
                mch_pool,
                old_policy=True,
                T=1,
                greedy=False
                ):

        # --- (修改 4) 修改 Encoder 的调用方式 ---
        batch_size = candidate.size(0)
        n_nodes = self.n_j * self.n_m

        # 确保 x 是正确的形状 (batch_size, n_nodes, input_dim)
        x_reshaped = x.view(batch_size, n_nodes, configs.input_dim) 

        # 传入 GraphAttentionEncoder (它返回 h_nodes_batch, h_pooled)
        h_nodes_batch, h_pooled = self.encoder(x=x_reshaped, mask=None)

        # Reshape h_nodes 回 (batch_size * n_nodes, hidden_dim) 以便下游代码使用
        h_nodes = h_nodes_batch.view(batch_size * n_nodes, -1)
        # ---

        # --- (下游代码与 PPO_Actor1.py 保持一致) ---
        if old_policy:
            dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)

            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(candidate_feature)
            if mch_pool==None:
                mch_pooled_repeated = self._input[None,None, :].expand_as(candidate_feature).to(self.device)
            else:
                mch_pooled_repeated = mch_pool.unsqueeze(-2).expand_as(candidate_feature).to(self.device)
            concateFea = torch.cat((candidate_feature, h_pooled_repeated,mch_pooled_repeated), dim=-1)
            candidate_scores = self.actor1(concateFea)

            candidate_scores = candidate_scores * 10
            mask_reshape = mask.reshape(candidate_scores.size())
            candidate_scores[mask_reshape] = float('-inf')

            pi = F.softmax(candidate_scores, dim=1)
            if greedy:
                action = greedy_select_action(pi,candidate)
                log_a = 0
                index = 0
            else:
                action, index, log_a = select_action1(pi, candidate)
            action1 = action.type(torch.long).to(self.device)
            batch_x = dur.reshape(dummy.size(0), self.n_j * self.n_m, -1).to(self.device)
            mask_mch = mask_mch.reshape(dummy.size(0), -1, self.n_m)
            mask_mch_action = torch.gather(mask_mch, 1,
                                           action1.unsqueeze(-1).unsqueeze(-1).expand(mask_mch.size(0), -1,
                                                                                      mask_mch.size(2)))
            action_feature = torch.gather(batch_node, 1,
                                          action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1,
                                                                                     batch_node.size(2))).squeeze(1)
            action_node = torch.gather(batch_x, 1,
                                       action1.unsqueeze(-1).unsqueeze(-1).expand(batch_x.size(0), -1,
                                                                                  batch_x.size(2))).squeeze(1)

            return action,index, log_a, action_node.detach(), action_feature.detach(), mask_mch_action.detach(), h_pooled.detach()

        else:
            dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)

            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(candidate_feature)
            if mch_pool == None:
                mch_pooled_repeated = self._input[None, None, :].expand_as(candidate_feature).to(self.device)
            else:
                mch_pooled_repeated = mch_pool.unsqueeze(-2).expand_as(candidate_feature).to(self.device)
            concateFea = torch.cat((candidate_feature, h_pooled_repeated, mch_pooled_repeated), dim=-1)
            candidate_scores = self.actor1(concateFea)

            candidate_scores = candidate_scores.squeeze(-1) * 10
            mask_reshape = mask.reshape(candidate_scores.size())
            candidate_scores[mask_reshape] = float('-inf')

            pi = F.softmax(candidate_scores, dim=1)
            dist = Categorical(pi)

            log_a = dist.log_prob(a_index.to(self.device))
            entropy = dist.entropy()
            action1 = old_action.type(torch.long).cuda()
            batch_x = dur.reshape(dummy.size(0), self.n_j*self.n_m, -1).to(self.device)
            mask_mch = mask_mch.reshape(dummy.size(0), -1, self.n_m)
            mask_mch_action = torch.gather(mask_mch, 1,
                                           action1.unsqueeze(-1).unsqueeze(-1).expand(mask_mch.size(0), -1,
                                                                                      mask_mch.size(2)))
            action_feature = torch.gather(batch_node, 1,
                                          action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1,
                                                                                     batch_node.size(2))).squeeze(1)
            action_node = torch.gather(batch_x, 1,
                                       action1.unsqueeze(-1).unsqueeze(-1).expand(batch_x.size(0), -1,
                                                                                  batch_x.size(2))).squeeze(1)
            v = self.critic(h_pooled)

            return entropy, v, log_a, action_node.detach(), action_feature.detach(), mask_mch_action.detach(), h_pooled.detach()


# ( Mch_Actor 保持不变, 所以我们把它也复制过来 )
class Mch_Actor(nn.Module):
    def __init__(self,n_j,
                 n_m,
                 num_layers,
                 learn_eps, # <--- 确保 learn_eps 在参数里
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 device):
        super(Mch_Actor,self).__init__()
        self.n_j = n_j
        self.bn = torch.nn.BatchNorm1d(hidden_dim).to(device)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim).to(device)
        self.n_m = n_m
        self.hidden_size=hidden_dim
        self.n_ops_perjob = n_m
        self.device = device

        self.fc2 = nn.Linear(2, hidden_dim, bias=False).to(device)
        self.actor = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self,action_node,hx,mask_mch_action,mch_time,mch_a=None,last_hh=None,policy=False):
        mch_time = mch_time/configs.et_normalize_coef
        action_node = action_node/configs.et_normalize_coef

        feature = torch.cat([mch_time.unsqueeze(-1), action_node.unsqueeze(-1)], -1)
        action_node = self.bn(self.fc2(feature).reshape(-1, self.hidden_size)).reshape(-1,self.n_m,self.hidden_size)
        pool = action_node.mean(dim=1)
        h_pooled_repeated = pool.unsqueeze(1).expand_as(action_node)
        pooled_repeated = hx.unsqueeze(1).expand_as(action_node)
        concateFea = torch.cat((action_node, h_pooled_repeated,pooled_repeated), dim=-1)
        mch_scores = self.actor(concateFea)

        mch_scores = mch_scores.squeeze(-1) * 10

        mch_scores = mch_scores.masked_fill(mask_mch_action.squeeze(1).bool(), float("-inf"))
        pi_mch = F.softmax(mch_scores, dim=1)

        return pi_mch,pool

if __name__ == '__main__':
    print('This is the new Attention-based Actor model file.')