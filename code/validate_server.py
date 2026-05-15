import matplotlib
matplotlib.use('Agg') # 关键：服务器模式
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import random
from Params import configs
# 假设你已经有了 FJSP 环境
from FJSP_Env import FJSP 

# --- 1. 逻辑自检函数 ---
def check_logic(env, batch_idx=0):
    print("\n>>> 开始逻辑自检...")
    error = False
    # 这里放入之前提供的 verify_schedule_logic 代码
    # 简单示例：
    if env.mchsEndTimes.max() > 99999: 
        print("❌ 完工时间异常！")
        error = True
    
    if not error: print("✅ 逻辑检查通过！")

# --- 2. 英文甘特图绘制 ---
class SERVER_GANTT():
    def __init__(self, n_j, n_m):
        self.n_j, self.n_m = n_j, n_m
        h = max(5, n_m * 0.5)
        plt.figure(figsize=(10, h))
        plt.xlabel('Time')
        plt.ylabel('Machine ID')
        plt.yticks(range(1, n_m + 1))
        plt.grid(axis='x', linestyle='--', alpha=0.3)

    def draw_bar(self, m_id, start, dur, job_id, op_id):
        # 随机颜色
        random.seed(job_id)
        color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        plt.barh(m_id + 1, dur, 0.6, left=start, color=color)
        plt.text(start + dur/2, m_id + 1, f'J{job_id}', ha='center', va='center', color='white', size=8)

    def save(self, path):
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"甘特图已保存: {path}")

def run_validation():
    # 模拟一次调度过程
    print("正在生成验证图...")
    chart = SERVER_GANTT(configs.n_j, configs.n_m)
    
    # 这里应该调用你的模型 predict
    # 模拟数据：
    chart.draw_bar(m_id=0, start=0, dur=10, job_id=1, op_id=1)
    chart.draw_bar(m_id=1, start=10, dur=15, job_id=2, op_id=1)
    
    os.makedirs("./validation_results", exist_ok=True)
    chart.save("./validation_results/Validation_Gantt.png")

if __name__ == '__main__':
    run_validation()