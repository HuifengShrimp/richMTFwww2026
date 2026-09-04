"""
将 KuaiRand-1K 的 csv 文件转换为 pkl 格式（加速后续读取）
从 xMTF/ 目录下运行:  python prepare_data.py --data_path ../KuaiRand-1K
"""
import os
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../KuaiRand-1K")
args = parser.parse_args()

data_path = args.data_path
data_type = data_path.split('-')[-1].lower()  # "1k"

csv_files = [
    f"log_standard_4_08_to_4_21_{data_type}.csv",
    f"log_standard_4_22_to_5_08_{data_type}.csv",
    f"user_features_{data_type}.csv",
    f"video_features_basic_{data_type}.csv",
    f"video_features_statistic_{data_type}.csv",
]

for fname in tqdm(csv_files, desc="Converting csv → pkl"):
    csv_path = os.path.join(data_path, "data", fname)
    pkl_path = csv_path.replace(".csv", ".pkl")
    if os.path.exists(pkl_path):
        print(f"  [skip] {pkl_path} already exists")
        continue
    print(f"  Reading {csv_path} ...")
    if "log_" in fname:
        df = pd.read_csv(csv_path, parse_dates=['date'])
    elif "video_features_basic" in fname:
        df = pd.read_csv(csv_path, parse_dates=['upload_dt'])
    else:
        df = pd.read_csv(csv_path)
    df.to_pickle(pkl_path)
    print(f"  Saved  {pkl_path}  ({len(df)} rows)")

print("\n✅ Done. All csv files converted to pkl.")

# 创建必要的输出目录
for d in ["./models", "./results", "./figs", "./predict/1k"]:
    os.makedirs(d, exist_ok=True)
    print(f"  mkdir {d}")
print("✅ Output directories ready.")
