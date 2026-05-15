import pandas as pd
import os

# 指定您的目标文件
target_file = "logs/log_15_15_seed200_20260126_151108.csv"

def analyze_log():
    if not os.path.exists(target_file):
        print(f"❌ 找不到文件: {target_file}")
        return

    try:
        # 读取 CSV
        df = pd.read_csv(target_file)
        
        # 清洗列名 (去掉可能存在的空格)
        df.columns = [c.strip() for c in df.columns]

        # 1. 计算总耗时 (Time 列求和)
        if 'Time' in df.columns:
            total_seconds = df['Time'].sum()
            total_minutes = total_seconds / 60
            print(f"⏱️  总耗时: {total_minutes:.2f} 分钟")
        else:
            print("⚠️ 未找到 'Time' 列")

        # 2. 找最佳 Makespan (Val_Makespan 的最小值)
        if 'Val_Makespan' in df.columns:
            # 过滤掉 0 或无效值 (如果有的话)
            valid_data = df[df['Val_Makespan'] > 0]
            if len(valid_data) > 0:
                best_make = valid_data['Val_Makespan'].min()
                print(f"🏆 最佳 Makespan: {best_make:.4f}")
            else:
                print("⚠️ 没有有效的验证数据")
        else:
            print("⚠️ 未找到 'Val_Makespan' 列")
            
        print(f"📄 文件名: {target_file}")

    except Exception as e:
        print(f"❌ 读取错误: {e}")

if __name__ == "__main__":
    analyze_log()