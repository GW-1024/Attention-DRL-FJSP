import matplotlib
# 【关键】必须在导入 pyplot 之前设置后端为 'Agg'，否则在服务器上会报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch
import numpy as np
import os
import copy
from copy import deepcopy
from torch.utils.data import DataLoader

# 导入您项目中的模块
from FJSP_Env import FJSP, DFJSP_GANTT_CHART
from mb_agg import g_pool_cal, aggr_obs
from Params import configs
from models.PPO_Actor_Attention import Job_Actor, Mch_Actor # 确保路径正确
from uniform_instance import FJSPDataset

def validate(vali_set, batch_size, policy_jo, policy_mc):
    """
    验证函数：加载模型，执行调度，并生成甘特图
    """
    policy_job = copy.deepcopy(policy_jo)
    policy_mch = copy.deepcopy(policy_mc)
    policy_job.eval()
    policy_mch.eval()

    def eval_model_bat(bat, i):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()

            # 1. 确保保存图片的文件夹存在
            figure_path = './FJSP_FIGURE/'
            if not os.path.exists(figure_path):
                os.makedirs(figure_path)

            env = FJSP(n_j=configs.n_j, n_m=configs.n_m)

            # 2. 仅对第一个 batch (i==0) 初始化甘特图对象
            # 这样可以避免验证大量数据时生成成千上万张图
            if i == 0:
                print(f"正在为第 {i} 批次生成甘特图对象...")
                gantt_chart = DFJSP_GANTT_CHART(configs.n_j, configs.n_m)
            else:
                gantt_chart = None

            device = torch.device(configs.device)
            
            # 图池化准备
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                     n_nodes=configs.n_j * configs.n_m,
                                     device=device)

            # 环境重置
            adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)

            j = 0
            # 准备 Tensor
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            pool = None

            while True:
                # --- CPU/GPU 内存优化策略 ---
                cpu_sparse_adj = deepcopy(adj).to_sparse()
                env_adj_cpu = aggr_obs(cpu_sparse_adj, configs.n_j * configs.n_m)
                env_fea_cpu = torch.from_numpy(np.copy(fea)).float()
                env_fea_cpu = deepcopy(env_fea_cpu).reshape(-1, env_fea_cpu.size(-1))
                env_candidate_cpu = torch.from_numpy(np.copy(candidate)).long()
                env_mask_cpu = torch.from_numpy(np.copy(mask))
                env_mch_time_cpu = torch.from_numpy(np.copy(mch_time)).float()

                # 数据上 GPU
                env_adj_gpu = env_adj_cpu.to(device)
                env_fea_gpu = env_fea_cpu.to(device)
                env_candidate_gpu = env_candidate_cpu.to(device)
                env_mask_gpu = env_mask_cpu.to(device)
                env_mch_time_gpu = env_mch_time_cpu.to(device)

                # Job Actor 决策
                action, a_idx, log_a, action_node, _, mask_mch_action, hx = policy_job(x=env_fea_gpu,
                                                                                       graph_pool=g_pool_step,
                                                                                       padded_nei=None,
                                                                                       adj=env_adj_gpu,
                                                                                       candidate=env_candidate_gpu,
                                                                                       mask=env_mask_gpu,
                                                                                       mask_mch=env_mask_mch,
                                                                                       dur=env_dur,
                                                                                       a_index=0,
                                                                                       old_action=0,
                                                                                       mch_pool=pool,
                                                                                       old_policy=True,
                                                                                       T=1,
                                                                                       greedy=True) # 验证时使用贪婪策略
                
                # Machine Actor 决策
                pi_mch, pool = policy_mch(action_node, hx, mask_mch_action, env_mch_time_gpu)
                _, mch_a = pi_mch.squeeze(-1).max(1)

                # --- 环境执行一步 ---
                # 【关键】将 gantt_chart 传入 step 函数
                # 注意：这要求您的 FJSP_Env.py 的 step 函数已经修改为接受 gantt_chart 参数
                adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time = env.step(
                    action.cpu().numpy(), 
                    mch_a, 
                    gantt_chart=gantt_chart
                )

                j += 1
                if env.done():
                    break
            
            # 计算最终 Makespan
            cost = env.mchsEndTimes.max(-1).max(-1)
            C_max.append(cost)

            # 3. 保存图片并清理内存
            if i == 0:
                # 遍历 Batch 中的每一个环境结果进行保存 (通常 Batch=4 就存4张)
                # 由于 env.step 中是实时画图，此时 plt 中已经有了图像
                # 注意：这里我们简单地将当前的 plt 上下文保存。
                # 如果 Batch > 1，由于 DFJSP_GANTT_CHART 可能在一个图上画了多次，
                # 或者它可能只处理了 Batch 中的某一个（取决于 Env 实现）。
                # 我们假设 Env 中做好了 `if i==0` 的控制。
                
                save_name = os.path.join(figure_path, f'Validation_Result.png')
                plt.savefig(save_name, dpi=300, bbox_inches='tight')
                print(f"甘特图已保存至: {save_name}")
                
                # 强制关闭，释放内存
                plt.close('all')

        return torch.tensor(cost)

    # 对整个验证集循环
    totall_cost = torch.cat([eval_model_bat(bat, i) for i, bat in enumerate(vali_set)], 0)
    return totall_cost

# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 设置模型路径 (请修改为您实际的路径)
    model_save_path = 'my_experiment_Attention/FJSP_J10M10/best_value100/'
    
    # 自动推导权重文件路径
    job_path = os.path.join(model_save_path, 'policy_job.pth')
    mch_path = os.path.join(model_save_path, 'policy_mch.pth')

    # 检查文件是否存在
    if not (os.path.exists(job_path) and os.path.exists(mch_path)):
        print(f"Error: 模型文件未找到，请检查路径: {model_save_path}")
        exit()

    print(f"--- 加载模型: {model_save_path} ---")
    print(f"问题规模: {configs.n_j} Jobs x {configs.n_m} Machines")
    
    device = torch.device(configs.device)

    # 2. 初始化网络结构
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

    # 3. 加载权重
    # weights_only=False 是为了兼容旧版 PyTorch 保存格式，如果是新版建议 True
    try:
        policy_job.load_state_dict(torch.load(job_path, map_location=device))
        policy_mch.load_state_dict(torch.load(mch_path, map_location=device))
    except Exception as e:
        print(f"加载权重失败: {e}")
        print("尝试添加 weights_only=False (针对 PyTorch 2.4+)")
        policy_job.load_state_dict(torch.load(job_path, map_location=device, weights_only=False))
        policy_mch.load_state_dict(torch.load(mch_path, map_location=device, weights_only=False))

    print("模型加载成功！")

    # 4. 准备验证数据
    # 为了只生成少量图片用于查看，我们将 batch_size 设为 1，总数量设为 1
    # 这样只会生成一张最清晰的图
    test_batch_size = 1
    dataset_size = 1
    
    print(f"生成测试数据: Batch Size={test_batch_size}, Count={dataset_size}")
    validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 1, dataset_size)
    valid_loader = DataLoader(validat_dataset, batch_size=test_batch_size)

    # 5. 运行验证
    print("开始运行调度并绘图...")
    mean_makespan = validate(valid_loader, test_batch_size, policy_job, policy_mch).mean()

    print("---------------------------------------------------------")
    print(f"验证完成!")
    print(f"平均 Makespan: {mean_makespan.item():.4f}")
    print(f"请查看 ./FJSP_FIGURE/ 文件夹下的 Validation_Result.png")
    print("---------------------------------------------------------")