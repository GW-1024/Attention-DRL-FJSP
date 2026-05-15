import argparse

parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')

# --- 1. 基础环境参数 ---
parser.add_argument('--device', type=str, default="cuda", help='Device: cuda or cpu')
parser.add_argument('--n_j', type=int, default=15, help='Number of jobs')
parser.add_argument('--n_m', type=int, default=15, help='Number of machines')
parser.add_argument('--low', type=int, default=-99, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--rewardscale', type=float, default=0., help='Reward scale')
parser.add_argument('--init_quality_flag', type=bool, default=False, help='Flag of init quality')

# --- 2. 归一化参数 ---
parser.add_argument('--et_normalize_coef', type=int, default=1000)
parser.add_argument('--dr_normalize_coef', type=int, default=100)
parser.add_argument('--wkr_normalize_coef', type=int, default=1)

# --- 3. 网络参数 (关键：Hidden Dim & Layers) ---
parser.add_argument('--input_dim', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
parser.add_argument('--num_layers', type=int, default=3, help='Num of Attention/GNN layers')
parser.add_argument('--num_heads', type=int, default=8, help='Attention 头数 (补回了这一行)') # 【关键修复】
parser.add_argument('--num_mlp_layers_feature_extract', type=int, default=3)
parser.add_argument('--num_mlp_layers_actor', type=int, default=3)
parser.add_argument('--hidden_dim_actor', type=int, default=128)
parser.add_argument('--num_mlp_layers_critic', type=int, default=2)
parser.add_argument('--hidden_dim_critic', type=int, default=32)
parser.add_argument('--neighbor_pooling_type', type=str, default='average')
parser.add_argument('--graph_pool_type', type=str, default='average')
parser.add_argument('--Init', type=bool, default=True)

# --- 4. PPO 训练参数 ---
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=4) # 建议服务器上开大点
parser.add_argument('--max_updates', type=int, default=2000) # 实验演示用 2000
parser.add_argument('--ppo_step', type=int, default=3)
parser.add_argument('--k_epochs', type=int, default=4)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--eps_clip', type=float, default=0.2)
parser.add_argument('--vloss_coef', type=float, default=1)
parser.add_argument('--ploss_coef', type=float, default=2)
parser.add_argument('--entloss_coef', type=float, default=0.01)
parser.add_argument('--num_ins', type=int, default=3200, help='Number of training instances')
parser.add_argument('--decayflag', type=bool, default=False, help='lr decayflag')

# --- 5. 实验控制参数 ---
parser.add_argument('--seed', type=int, default=200, help='Global random seed')
parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
parser.add_argument('--model_dir', type=str, default='./saved_models', help='Directory to save models')

# --- 6. 学习率衰减参数 ---
parser.add_argument('--decay_step_size', type=int, default=500, help='decay_step_size')
parser.add_argument('--decay_ratio', type=float, default=0.96, help='decay_ratio')

# 生成全局配置对象
configs = parser.parse_args()