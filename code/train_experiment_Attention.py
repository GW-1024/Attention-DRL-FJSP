import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 依然只用一张卡
import time
from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action, select_action2
from models.PPO_Actor_Attention import Job_Actor, Mch_Actor
from copy import deepcopy
import torch
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from Params import configs
from epsGreedyForMch import PredictMch
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import gc 
import traceback 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[Info] Global seed set to: {seed}")

class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.job_logprobs = []
        self.mch_logprobs = []
        self.mask_mch = []
        self.first_task = []
        self.pre_task = []
        self.action = []
        self.mch = []
        self.dur = []
        self.mch_time = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.job_logprobs[:]
        del self.mch_logprobs[:]
        del self.mask_mch[:]
        del self.first_task[:]
        del self.pre_task[:]
        del self.action[:]
        del self.mch[:]
        del self.dur[:]
        del self.mch_time[:]

def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

class PPO:
    def __init__(self, lr, gamma, k_epochs, eps_clip, n_j, n_m, num_layers, neighbor_pooling_type, input_dim, hidden_dim, num_mlp_layers_feature_extract, num_mlp_layers_actor, hidden_dim_actor, num_mlp_layers_critic, hidden_dim_critic):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.policy_job = Job_Actor(n_j=configs.n_j, n_m=configs.n_m, num_layers=configs.num_layers, learn_eps=False, neighbor_pooling_type=configs.neighbor_pooling_type, input_dim=configs.input_dim, hidden_dim=configs.hidden_dim, num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract, num_mlp_layers_critic=num_mlp_layers_critic, hidden_dim_critic=hidden_dim_critic, device=device)
        self.policy_mch = Mch_Actor(n_j=configs.n_j, n_m=configs.n_m, num_layers=configs.num_layers, learn_eps=False, neighbor_pooling_type=configs.neighbor_pooling_type, input_dim=configs.input_dim, hidden_dim=configs.hidden_dim, num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract, device=device)
        self.policy_old_job = deepcopy(self.policy_job)
        self.policy_old_mch = deepcopy(self.policy_mch)
        self.policy_old_job.load_state_dict(self.policy_job.state_dict())
        self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())
        self.job_optimizer = torch.optim.Adam(self.policy_job.parameters(), lr=lr)
        self.mch_optimizer = torch.optim.Adam(self.policy_mch.parameters(), lr=lr)
        if hasattr(configs, 'decayflag') and configs.decayflag:
             self.job_scheduler = torch.optim.lr_scheduler.StepLR(self.job_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)
             self.mch_scheduler = torch.optim.lr_scheduler.StepLR(self.mch_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)
        else:
             self.job_scheduler = None
             self.mch_scheduler = None
        self.MSE = nn.MSELoss()

    def update(self, memories, epoch):
        torch.cuda.empty_cache()
        rewards_all_env = []
        for i in range(configs.batch_size):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed((memories.r_mb[0][i]).tolist()), reversed(memories.done_mb[0][i].tolist())):
                if is_terminal: discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)
        rewards_all_env = torch.stack(rewards_all_env, 0)
        
        for _ in range(configs.k_epochs):
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type, batch_size=torch.Size([configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]), n_nodes=configs.n_j * configs.n_m, device=device)
            job_log_prob, mch_log_prob, val = [], [], []
            job_entropy, mch_entropies = [], []
            job_log_old_prob = memories.job_logprobs[0]
            mch_log_old_prob = memories.mch_logprobs[0]
            env_mask_mch = memories.mask_mch[0]
            env_dur = memories.dur[0]
            pool = None
            for i in range(len(memories.fea_mb)):
                env_fea = memories.fea_mb[i].to(device)
                env_adj = memories.adj_mb[i].to(device)
                env_candidate = memories.candidate_mb[i].to(device)
                env_mask = memories.mask_mb[i].to(device)
                a_index = memories.a_mb[i]
                env_mch_time = memories.mch_time[i].to(device)
                old_action = memories.action[i].to(device)
                old_mch = memories.mch[i].to(device)
                
                a_entropy, v, log_a, action_node, _, mask_mch_action, hx = self.policy_job(x=env_fea, graph_pool=g_pool_step, padded_nei=None, adj=env_adj, candidate=env_candidate, mask=env_mask, mask_mch=env_mask_mch, dur=env_dur, a_index=a_index, old_action=old_action, mch_pool=pool, old_policy=False)
                pi_mch, pool = self.policy_mch(action_node, hx, mask_mch_action, env_mch_time, mch_a=None, last_hh=None, policy=True)
                val.append(v)
                dist = Categorical(pi_mch)
                log_mch = dist.log_prob(old_mch)
                mch_entropy = dist.entropy()
                job_entropy.append(a_entropy)
                mch_entropies.append(mch_entropy)
                job_log_prob.append(log_a)
                mch_log_prob.append(log_mch)
            job_log_prob, job_log_old_prob = torch.stack(job_log_prob, 0).permute(1, 0), torch.stack(job_log_old_prob, 0).permute(1, 0).to(device)
            mch_log_prob, mch_log_old_prob = torch.stack(mch_log_prob, 0).permute(1, 0), torch.stack(mch_log_old_prob, 0).permute(1, 0).to(device)
            val = torch.stack(val, 0).squeeze(-1).permute(1, 0)
            job_entropy = torch.stack(job_entropy, 0).permute(1, 0)
            mch_entropies = torch.stack(mch_entropies, 0).permute(1, 0)
            job_loss_sum, mch_loss_sum = 0, 0
            for j in range(configs.batch_size):
                job_ratios = torch.exp(job_log_prob[j] - job_log_old_prob[j].detach())
                mch_ratios = torch.exp(mch_log_prob[j] - mch_log_old_prob[j].detach())
                advantages = rewards_all_env[j] - val[j].detach()
                advantages = adv_normalize(advantages)
                job_surr1 = job_ratios * advantages
                job_surr2 = torch.clamp(job_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                job_v_loss = self.MSE(val[j], rewards_all_env[j])
                job_loss = -1 * torch.min(job_surr1, job_surr2) + 0.5 * job_v_loss - 0.01 * job_entropy[j]
                job_loss_sum += job_loss
                mch_surr1 = mch_ratios * advantages
                mch_surr2 = torch.clamp(mch_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                mch_loss = -1 * torch.min(mch_surr1, mch_surr2) - 0.01 * mch_entropies[j]
                mch_loss_sum += mch_loss
            self.job_optimizer.zero_grad()
            job_loss_sum.mean().backward(retain_graph=True)
            self.policy_old_job.load_state_dict(self.policy_job.state_dict())
            self.mch_optimizer.zero_grad()
            mch_loss_sum.mean().backward(retain_graph=True)
            self.job_optimizer.step()
            self.mch_optimizer.step()
            self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())
            if self.job_scheduler is not None:
                self.job_scheduler.step()
                self.mch_scheduler.step()
        return job_loss_sum.mean().item(), mch_loss_sum.mean().item()

def validate_placeholder(valid_loader, batch_size, policy_job, policy_mch):
    return torch.tensor([0.0])
try:
    from validation import validate
except ImportError:
    validate = validate_placeholder

def main(epochs):
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        from uniform_instance import FJSPDataset
        from FJSP_Env import FJSP

        if not hasattr(configs, 'log_dir'): configs.log_dir = './logs'
        if not os.path.exists(configs.log_dir): os.makedirs(configs.log_dir)
        
        #log_csv_path = os.path.abspath(os.path.join(configs.log_dir, f"log_{configs.n_j}_{configs.n_m}_seed{configs.seed}.csv"))
        # 1. 获取当前时间 (例如: 20260125_183000)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 2. 拼接文件名：除了种子号，还加上了时间戳，保证文件名唯一
        # 格式变成: log_15_15_seed200_20260125_183000.csv
        log_csv_path = os.path.abspath(os.path.join(configs.log_dir, f"log_{configs.n_j}_{configs.n_m}_seed{configs.seed}_{timestamp}.csv"))
    
        with open(log_csv_path, 'w') as f:
            f.write("Batch_Idx,Avg_Reward,Loss_Job,Loss_Mch,Val_Makespan,Time\n")
            f.flush()
        print(f"[Info] CSV 日志将保存至: {log_csv_path}")

        g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type, batch_size=torch.Size([configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]), n_nodes=configs.n_j * configs.n_m, device=device)

        ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip, n_j=configs.n_j, n_m=configs.n_m, num_layers=configs.num_layers, neighbor_pooling_type=configs.neighbor_pooling_type, input_dim=configs.input_dim, hidden_dim=configs.hidden_dim, num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract, num_mlp_layers_actor=configs.num_mlp_layers_actor, hidden_dim_actor=configs.hidden_dim_actor, num_mlp_layers_critic=configs.num_mlp_layers_critic, hidden_dim_critic=configs.hidden_dim_critic)
        
        train_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, configs.num_ins, 200)
        validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 128, 200)
        data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
        valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

        record = 1000000
        
        for epoch in range(epochs):
            memory = Memory()
            ppo.policy_old_job.train()
            ppo.policy_old_mch.train()
            start = time.time()
            
            times, costs, losses, critic_loss = [], [], [], []
            
            for batch_idx, batch in enumerate(data_loader):
                print(f"Debug: Seed {configs.seed} Processing Batch {batch_idx} (Device: {device})...") 
                
                env = FJSP(configs.n_j, configs.n_m)
                data = batch.numpy()
                adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)
                
                job_log_prob, mch_log_prob = [], []
                r_mb, done_mb = [], []
                first_task, pretask = [], []
                j, mch_a, pool = 0, None, None
                ep_rewards = - env.initQuality
                env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
                env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
                
                while True:
                    cpu_sparse_adj = deepcopy(adj).to_sparse()
                    env_adj_cpu = aggr_obs(cpu_sparse_adj, configs.n_j * configs.n_m)
                    env_fea_cpu = torch.from_numpy(np.copy(fea)).float()
                    env_fea_cpu = deepcopy(env_fea_cpu).reshape(-1, env_fea_cpu.size(-1))
                    env_candidate_cpu = torch.from_numpy(np.copy(candidate)).long()
                    env_mask_cpu = torch.from_numpy(np.copy(mask))
                    env_mch_time_cpu = torch.from_numpy(np.copy(mch_time)).float()

                    env_adj_gpu = env_adj_cpu.to(device)
                    env_fea_gpu = env_fea_cpu.to(device)
                    env_candidate_gpu = env_candidate_cpu.to(device)
                    env_mask_gpu = env_mask_cpu.to(device)
                    env_mch_time_gpu = env_mch_time_cpu.to(device)

                    # =======================================================
                    # 【Debug 关键点】打印输入 tensor 的形状，看看是不是这里爆了
                    if batch_idx == 0 and j == 0:
                        print(f"DEBUG: env_fea_gpu shape: {env_fea_gpu.shape}")
                        print(f"DEBUG: env_adj_gpu shape: {env_adj_gpu.shape}")
                    # =======================================================

                    action, a_idx, log_a, action_node, _, mask_mch_action, hx = ppo.policy_old_job(x=env_fea_gpu, graph_pool=g_pool_step, padded_nei=None, adj=env_adj_gpu, candidate=env_candidate_gpu, mask=env_mask_gpu, mask_mch=env_mask_mch, dur=env_dur, a_index=0, old_action=0, mch_pool=pool, old_policy=True)
                    pi_mch, pool = ppo.policy_old_mch(action_node, hx, mask_mch_action, env_mch_time_gpu, mch_a, None, policy=True)
                    mch_a, log_mch = select_action2(pi_mch)
                    
                    job_log_prob.append(log_a.cpu())
                    mch_log_prob.append(log_mch.cpu())
                    memory.mch.append(mch_a.cpu())
                    memory.pre_task.append(pretask)
                    memory.adj_mb.append(env_adj_cpu)
                    memory.fea_mb.append(env_fea_cpu)
                    memory.candidate_mb.append(env_candidate_cpu)
                    memory.action.append(deepcopy(action.cpu()))
                    memory.mask_mb.append(env_mask_cpu)
                    memory.mch_time.append(env_mch_time_cpu)
                    memory.a_mb.append(a_idx)

                    adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time = env.step(action.cpu().numpy(), mch_a)
                    ep_rewards += reward
                    r_mb.append(deepcopy(reward))
                    done_mb.append(deepcopy(done))

                    j += 1
                    if env.done(): break
                
                memory.dur.append(env_dur)
                memory.mask_mch.append(env_mask_mch)
                memory.first_task.append(first_task)
                memory.job_logprobs.append(job_log_prob)
                memory.mch_logprobs.append(mch_log_prob)
                memory.r_mb.append(torch.tensor(r_mb).float().permute(1, 0))
                memory.done_mb.append(torch.tensor(done_mb).float().permute(1, 0))
                
                ep_rewards -= env.posRewards
                loss, v_loss = ppo.update(memory, batch_idx)
                memory.clear_memory()
                
                losses.append(loss)
                critic_loss.append(v_loss)
                cost = env.mchsEndTimes.max(-1).max(-1)
                costs.append(cost.mean())
                
                if batch_idx % 10 == 0: torch.cuda.empty_cache()

                step = 1
                filepath = 'my_experiment_Attention'
                
                if (batch_idx + 1) % step == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end
                    mean_loss = np.mean(losses[-step:]) if len(losses) >= step else losses[-1]
                    mean_reward_val = np.mean(costs[-step:]) 
                    mean_ep_reward = np.mean(ep_rewards)
                    critic_losss = np.mean(critic_loss[-step:]) if len(critic_loss) >= step else critic_loss[-1]

                    print(' Seed %d | Batch %d | Reward: %.3f | Loss: %.4f' % (configs.seed, batch_idx, mean_reward_val, mean_loss))

                    if (batch_idx + 1) % 20 == 0:
                        validation_log = validate(valid_loader, configs.batch_size, ppo.policy_job, ppo.policy_mch).mean()
                    else:
                        validation_log = 0.0

                    try:
                        with open(log_csv_path, 'a') as f:
                            f.write(f"{batch_idx},{mean_ep_reward:.4f},{mean_loss:.4f},{critic_losss:.4f},{validation_log:.4f},{times[-1]:.2f}\n")
                            f.flush()
                    except Exception as e:
                        print(f"写入日志失败: {e}")

                    if (batch_idx + 1) % 100 == 0:
                        filename = 'FJSP_{}'.format('J'+str(configs.n_j)+'M'+str(configs.n_m))
                        filepath_full = os.path.join(filepath, filename)
                        epoch_dir = os.path.join(filepath_full, '%s_%s' % (100, batch_idx))
                        if not os.path.exists(epoch_dir): os.makedirs(epoch_dir)
                        torch.save(ppo.policy_job.state_dict(), os.path.join(epoch_dir, 'policy_job.pth'))
                        torch.save(ppo.policy_mch.state_dict(), os.path.join(epoch_dir, 'policy_mch.pth'))

                        if validation_log < record and validation_log > 0:
                            best_dir = os.path.join(filepath_full, 'best_value100')
                            if not os.path.exists(best_dir): os.makedirs(best_dir)
                            torch.save(ppo.policy_job.state_dict(), os.path.join(best_dir, 'policy_job.pth'))
                            torch.save(ppo.policy_mch.state_dict(), os.path.join(best_dir, 'policy_mch.pth'))
                            record = validation_log
    
    except Exception as e:
        error_msg = f"Crash Report:\n{traceback.format_exc()}"
        print(error_msg)
        with open("ERROR_LOG.txt", "w") as f:
            f.write(error_msg)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    if configs.seed is not None:
        setup_seed(configs.seed)
    else:
        setup_seed(200)
    main(1)