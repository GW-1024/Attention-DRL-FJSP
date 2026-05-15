# --- 完整复制并覆盖您原来的 validation.py 文件 ---

from epsGreedyForMch import PredictMch
from mb_agg import *
from Params import configs
from copy import deepcopy
from FJSP_Env import FJSP,DFJSP_GANTT_CHART
from mb_agg import g_pool_cal
import copy
from agent_utils import sample_select_action
from agent_utils import greedy_select_action
import numpy as np
import torch
import matplotlib.pyplot as plt
from Params import configs
def validate(vali_set,batch_size, policy_jo,policy_mc):
    policy_job = copy.deepcopy(policy_jo)
    policy_mch = copy.deepcopy(policy_mc)
    policy_job.eval()
    policy_mch.eval()
    def eval_model_bat(bat,i):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()

            env = FJSP(n_j=configs.n_j, n_m=configs.n_m)
            
            # Bug 1 修复：在验证时不创建甘特图对象，防止 plt.figure 内存泄漏
            # gantt_chart = DFJSP_GANTT_CHART( configs.n_j, configs.n_m) 
            
            
            
            
            
            device = torch.device(configs.device)
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                     n_nodes=configs.n_j * configs.n_m,
                                     device=device)

            adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)

            j = 0

            ep_rewards = - env.initQuality
            rewards = []
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            pool=None
            while True:
                
                # --- Bug 2 修复：应用 CPU 内存策略，防止 GPU 显存溢出 ---
                # 1. 在 CPU 上创建所有张量
                cpu_sparse_adj = deepcopy(adj).to_sparse()
                env_adj_cpu = aggr_obs(cpu_sparse_adj, configs.n_j * configs.n_m)
                env_fea_cpu = torch.from_numpy(np.copy(fea)).float()
                env_fea_cpu = deepcopy(env_fea_cpu).reshape(-1, env_fea_cpu.size(-1))
                env_candidate_cpu = torch.from_numpy(np.copy(candidate)).long()
                env_mask_cpu = torch.from_numpy(np.copy(mask))
                env_mch_time_cpu = torch.from_numpy(np.copy(mch_time)).float()

                # 2. 仅在调用模型前，才把数据发送到 GPU
                env_adj_gpu = env_adj_cpu.to(device)
                env_fea_gpu = env_fea_cpu.to(device)
                env_candidate_gpu = env_candidate_cpu.to(device)
                env_mask_gpu = env_mask_cpu.to(device)
                env_mch_time_gpu = env_mch_time_cpu.to(device)
                
                # 3. 使用 GPU 上的张量 (_gpu) 来调用模型
                action, a_idx, log_a, action_node, _, mask_mch_action, hx = policy_job(x=env_fea_gpu,
                                                                                       graph_pool=g_pool_step,
                                                                                       padded_nei=None,
                                                                                       adj=env_adj_gpu,
                                                                                       candidate=env_candidate_gpu
                                                                                       , mask=env_mask_gpu
                                                                                       , mask_mch=env_mask_mch
                                                                                       , dur=env_dur
                                                                                       , a_index=0
                                                                                       , old_action=0
                                                                                       , mch_pool=pool
                                                                                       ,old_policy=True,
                                                                                       T=1
                                                                                       ,greedy=True
                                                                                       )

                pi_mch,pool = policy_mch(action_node, hx, mask_mch_action, env_mch_time_gpu)

                _, mch_a = pi_mch.squeeze(-1).max(1)

                # 4. Bug 1 修复：移除 gantt_chart 参数，停止绘图
                adj, fea, reward, done, candidate, mask,job,_,mch_time,job_time = env.step(action.cpu().numpy(), mch_a)
                # --------------------- 修复结束 ---------------------

                j += 1
                if env.done():
                    break
            cost = env.mchsEndTimes.max(-1).max(-1)
            C_max.append(cost)
        return torch.tensor(cost)
    
    totall_cost = torch.cat([eval_model_bat(bat,i) for i,bat in enumerate(vali_set)], 0)

    return totall_cost


# --- (新添加的代码) --- 
# --- 让这个脚本可以被 python 命令直接运行 ---
if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    from uniform_instance import FJSPDataset
    from models.PPO_Actor_Attention import Job_Actor, Mch_Actor
     # 确保导入与训练时相同的模型结构
    from PPOwithValue import PPO # 导入 PPO 只是为了加载模型结构, 也可以像下面那样手动创建

    # 1. --- (!! 关键 !!) ---
    #    --- 请在这里设置您要验证的模型路径 ---
    
    # 示例1: 您在 J10M10 下的 'my_trained_models' 文件夹

    model_save_path = 'my_experiment_Attention/FJSP_J10M10/best_value100/'
    # 示例2: 作者的 J30M20 预训练模型
    # model_save_path = 'saved_network/FJSP_J30M20/best_value0/'
    
    # --------------------------------------------------

    job_path = os.path.join(model_save_path, 'policy_job.pth')
    mch_path = os.path.join(model_save_path, 'policy_mch.pth')

    if not (os.path.exists(job_path) and os.path.exists(mch_path)):
        print(f"错误：在 {model_save_path} 中找不到模型文件。")
        print("请检查：")
        print("1. 'model_save_path' 路径是否正确？")
        print("2. 'Params.py' 中的 n_j 和 n_m 是否与文件夹名称 (例如 FJSP_J10M10) 匹配？")
        exit()

    print(f"--- 正在加载模型用于独立验证 ---")
    print(f"模型路径: {model_save_path}")
    print(f"问题规模 (来自 Params.py): {configs.n_j}x{configs.n_m}")
    
    # 2. 设置设备
    device = torch.device(configs.device)

    # 3. 重新创建模型架构 (必须与训练时完全一致)
    #    它会自动从 import 的 configs 中读取 n_j, n_m, hidden_dim 等参数
    policy_job = Job_Actor(n_j=configs.n_j,
                           n_m=configs.n_m,
                           num_layers=configs.num_layers,
                           learn_eps=False,
                           neighbor_pooling_type=configs.neighbor_pooling_type,
                           input_dim=configs.input_dim,
                           hidden_dim=configs.hidden_dim,
                           num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                           num_mlp_layers_critic=configs.num_mlp_layers_critic,
                           hidden_dim_critic=configs.hidden_dim_critic,
                           device=device).to(device)
    
    policy_mch = Mch_Actor(n_j=configs.n_j,
                           n_m=configs.n_m,
                           num_layers=configs.num_layers,
                           learn_eps=False,
                           neighbor_pooling_type=configs.neighbor_pooling_type,
                           input_dim=configs.input_dim,
                           hidden_dim=configs.hidden_dim,
                           num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                           device=device).to(device)

    # 4. 加载您训练好的权重
    policy_job.load_state_dict(torch.load(job_path, map_location=device))
    policy_mch.load_state_dict(torch.load(mch_path, map_location=device))

    # 5. 加载验证数据集 (同样从 configs 读取设置)
    #   
    validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 128, 200) 
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size) 

    # 6. 开始验证
    print("正在验证...")
    validation_makespan = validate(valid_loader, configs.batch_size, policy_job, policy_mch).mean()
    
    # 7. 打印最终结果
    print("---------------------------------------------------------")
    print(f"您的模型 ({model_save_path})")
    print(f"在 {configs.n_j}x{configs.n_m} 验证集上的平均完工时间 (Makespan) 为: {validation_makespan}")
    print("---------------------------------------------------------")