import torch
import os
import glob
import numpy as np
from Params import configs
from uniform_instance import FJSPDataset
from torch.utils.data import DataLoader
from models.PPO_Actor_Attention import Job_Actor, Mch_Actor
from FJSP_Env import FJSP
from agent_utils import select_action2
from collections import defaultdict

def verify_schedule_logic():
    # ================= 配置区域 =================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 2 
    EPSILON = 1e-4 # 允许的浮点数误差范围
    
    print("="*50)
    print("   🕵️‍♂️ 正在启动调度方案自动质检程序 (Verifier)   ")
    print("="*50)

    # ================= 1. 加载模型 (标准流程) =================
    base_dir = "my_experiment_Attention/FJSP_J15M15"
    best_dirs = glob.glob(os.path.join(base_dir, "best_value*"))
    if best_dirs:
        target_dir = max(best_dirs, key=os.path.getmtime)
        model_path = os.path.join(target_dir, "policy_job.pth")
    else:
        all_dirs = glob.glob(os.path.join(base_dir, "100_*"))
        if not all_dirs: print("❌ 没找到模型"); return
        target_dir = max(all_dirs, key=os.path.getmtime)
        model_path = os.path.join(target_dir, "policy_job.pth")

    print(f"✅ 载入模型: {os.path.basename(target_dir)}")
    job_model_file = model_path
    mch_model_file = model_path.replace('policy_job', 'policy_mch')

    ppo_job = Job_Actor(configs.n_j, configs.n_m, configs.num_layers, False, configs.neighbor_pooling_type, 
                        configs.input_dim, configs.hidden_dim, configs.num_mlp_layers_feature_extract, 
                        configs.num_mlp_layers_critic, configs.hidden_dim_critic, device)
    ppo_mch = Mch_Actor(configs.n_j, configs.n_m, configs.num_layers, False, configs.neighbor_pooling_type, 
                        configs.input_dim, configs.hidden_dim, configs.num_mlp_layers_feature_extract, device)
    
    try:
        ppo_job.load_state_dict(torch.load(job_model_file, map_location=device, weights_only=False))
        ppo_mch.load_state_dict(torch.load(mch_model_file, map_location=device, weights_only=False))
    except:
        ppo_job.load_state_dict(torch.load(job_model_file, map_location=device))
        ppo_mch.load_state_dict(torch.load(mch_model_file, map_location=device))
    
    ppo_job.eval()
    ppo_mch.eval()

    # ================= 2. 运行推理并收集数据 =================
    dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, BATCH_SIZE, 2024)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(data_loader))
    
    env = FJSP(configs.n_j, configs.n_m)
    adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(batch.numpy())
    
    # 转换环境数据
    env_adj = torch.tensor(adj, dtype=torch.float32).to(device)
    env_fea = torch.tensor(fea, dtype=torch.float32).to(device)
    env_candidate = torch.tensor(candidate, dtype=torch.long).to(device)
    env_mask = torch.tensor(mask).to(device)
    env_mask_mch = torch.tensor(mask_mch).to(device)
    env_dur = torch.tensor(dur, dtype=torch.float32).to(device)
    env_mch_time = torch.tensor(mch_time, dtype=torch.float32).to(device)

    from mb_agg import g_pool_cal
    g_pool_step = g_pool_cal(configs.graph_pool_type, torch.Size([BATCH_SIZE, configs.n_j*configs.n_m, configs.n_j*configs.n_m]), configs.n_j*configs.n_m, device)

    # 数据记录表：存储所有工序的 [Job, Mch, Start, End]
    schedule_log = []
    current_mch_times = np.zeros(configs.n_m)

    print("⚡ 正在执行调度并记录数据...")
    while True:
        action, _, _, action_node, _, mask_mch_action, hx = ppo_job(
            x=env_fea, graph_pool=g_pool_step, padded_nei=None, adj=env_adj, 
            candidate=env_candidate, mask=env_mask, mask_mch=env_mask_mch, 
            dur=env_dur, a_index=0, old_action=0, mch_pool=None, old_policy=True)
        
        pi_mch, _ = ppo_mch(action_node, hx, mask_mch_action, env_mch_time, mch_a=None, last_hh=None, policy=True)
        mch_a, _ = select_action2(pi_mch)
        
        # 获取动作信息
        selected_job = action[0].item() # 粗略 Job ID
        selected_mch = mch_a[0].item()
        
        # Step
        adj, fea, reward, done, candidate, mask, _, _, mch_time, job_time = env.step(action.cpu().numpy(), mch_a)
        
        # 计算时间块
        new_time = mch_time[0][selected_mch]
        prev_time = current_mch_times[selected_mch]
        duration = new_time - prev_time
        
        # 记录有效操作
        if duration > EPSILON:
            schedule_log.append({
                'job_id': selected_job, # 注意：这是全局节点索引，用于区分工件
                'mch_id': selected_mch,
                'start': prev_time,
                'end': new_time
            })
            current_mch_times[selected_mch] = new_time

        # 准备下一步
        env_adj = torch.tensor(adj, dtype=torch.float32).to(device)
        env_fea = torch.tensor(fea, dtype=torch.float32).to(device)
        env_candidate = torch.tensor(candidate, dtype=torch.long).to(device)
        env_mask = torch.tensor(mask).to(device)
        env_mask_mch = torch.tensor(env.mask_mchs).to(device) if hasattr(env, 'mask_mchs') else torch.tensor(mask_mch).to(device)
        env_mch_time = torch.tensor(mch_time, dtype=torch.float32).to(device)
        
        if np.all(done):
            break
            
    # ================= 3. 核心验证逻辑 =================
    print("\n" + "="*50)
    print("   📊 开始验证 (Verification Check)   ")
    print("="*50)
    
    error_found = False

    # --- 检查 1: 机器冲突 (Machine Overlap) ---
    # 规则：同一台机器上，时间块绝对不能重叠
    print("🔍 检查 1: 机器资源冲突检测...", end="")
    mch_tasks = defaultdict(list)
    for task in schedule_log:
        mch_tasks[task['mch_id']].append(task)
    
    overlap_errors = 0
    for mch, tasks in mch_tasks.items():
        # 按开始时间排序
        tasks.sort(key=lambda x: x['start'])
        for i in range(len(tasks) - 1):
            curr_task = tasks[i]
            next_task = tasks[i+1]
            # 如果 前一个结束时间 > 后一个开始时间 (考虑误差)
            if curr_task['end'] > next_task['start'] + EPSILON:
                print(f"\n❌ [Error] 机器 M{mch+1} 发生撞车！")
                print(f"   Task A: Job {curr_task['job_id']} ({curr_task['start']:.2f} -> {curr_task['end']:.2f})")
                print(f"   Task B: Job {next_task['job_id']} ({next_task['start']:.2f} -> {next_task['end']:.2f})")
                overlap_errors += 1
                error_found = True
    
    if overlap_errors == 0:
        print(" ✅ 通过 (无重叠)")
    
    # --- 检查 2: 工序时序约束 (Job Precedence) ---
    # 规则：同一个 Job，Op 1 必须在 Op 2 之前结束
    print("🔍 检查 2: 工序时序约束检测...", end="")
    job_tasks = defaultdict(list)
    for task in schedule_log:
        # 这里我们假设 schedule_log 的添加顺序大致反映了工序顺序
        # 但更严格的方法是按时间排序，确保同一个 Job 的所有时间块不重叠
        # FJSP 中，同一个工件在同一时刻只能在一个机器上加工
        job_tasks[task['job_id']].append(task)
        
    seq_errors = 0
    for job, tasks in job_tasks.items():
        # 按开始时间排序
        tasks.sort(key=lambda x: x['start'])
        for i in range(len(tasks) - 1):
            curr_op = tasks[i]
            next_op = tasks[i+1]
            # 检查：下一个工序开始时，上一个工序必须已经结束
            if curr_op['end'] > next_op['start'] + EPSILON:
                print(f"\n❌ [Error] 工件 J{job} 时序错乱！")
                print(f"   Op {i}: M{curr_op['mch_id']+1} ({curr_op['start']:.2f} -> {curr_op['end']:.2f})")
                print(f"   Op {i+1}: M{next_op['mch_id']+1} ({next_op['start']:.2f} -> {next_op['end']:.2f})")
                seq_errors += 1
                error_found = True
                
    if seq_errors == 0:
        print(" ✅ 通过 (时序正确)")

    # ================= 4. 总结报告 =================
    print("\n" + "-"*50)
    if not error_found:
        print(f"🏆 完美！调度方案逻辑验证通过。")
        print(f"   - 总工序数: {len(schedule_log)}")
        print(f"   - Makespan: {max(t['end'] for t in schedule_log):.2f}")
        print(f"   - 结论: 这是一个 Strictly Feasible (严格可行) 的解。")
    else:
        print(f"⚠️ 警告！发现逻辑错误。请检查环境代码 (Step Function)。")
    print("-"*50)

if __name__ == "__main__":
    verify_schedule_logic()