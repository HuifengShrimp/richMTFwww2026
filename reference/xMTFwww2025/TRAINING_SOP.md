# xMTF 训练 SOP —— 傻瓜式上手指南

> 适用版本：xMTFwww2025 + PRM 重排  
> 流水线：**TwoTower 召回 → MMoE 精排 → PRM 重排 → top-8 曝光**  
> 预计总耗时：首次约 3–8 小时（主要取决于机器配置和 epochs 数量）

---

## 目录

1. [环境要求](#1-环境要求)
2. [第一步：克隆代码](#2-第一步克隆代码)
3. [第二步：下载数据集](#3-第二步下载数据集)
4. [第三步：创建虚拟环境 & 安装依赖](#4-第三步创建虚拟环境--安装依赖)
5. [第四步：数据预处理（CSV → PKL）](#5-第四步数据预处理csv--pkl)
6. [第五步：训练 Reward 模型](#6-第五步训练-reward-模型)
7. [第六步：构建全量候选集](#7-第六步构建全量候选集)
8. [第七步：PRM 重排 → 输出 top-8 曝光](#8-第七步prm-重排--输出-top-8-曝光)
9. [查看结果](#9-查看结果)
10. [常见报错 & 解决方案](#10-常见报错--解决方案)
11. [参数速查表](#11-参数速查表)

---

## 1. 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | macOS / Linux | macOS (Apple Silicon) / Linux |
| Python | 3.10 | 3.10 |
| 内存 | 32 GB | 64 GB |
| 磁盘 | 20 GB 空闲 | 50 GB 空闲 |
| GPU | 无需（CPU 可运行） | NVIDIA GPU 可加速 |
| conda | Anaconda / Miniconda | Anaconda |

---

## 2. 第一步：克隆代码

```bash
git clone https://github.com/HuifengShrimp/richMTFwww2026.git
cd richMTFwww2026
```

**进入工作目录（后续所有命令都从这里执行）：**

```bash
cd reference/xMTFwww2025/xMTF
```

---

## 3. 第二步：下载数据集

数据集放在 `xMTF/` 的同级目录下，即 `reference/xMTFwww2025/KuaiRand-1K/`。

### 方式 A：命令行下载（推荐）

```bash
# 回到 xMTFwww2025 目录
cd reference/xMTFwww2025

# 下载（压缩包约 1 GB，解压后约 4.3 GB）
wget https://chongming.myds.me:61364/data/KuaiRand-1K.tar.gz --no-check-certificate
tar -zxvf KuaiRand-1K.tar.gz
rm KuaiRand-1K.tar.gz   # 可选：删除压缩包节省空间
```

### 方式 B：手动下载

从 [Google Drive](https://drive.google.com/drive/folders/1BHcv3bgoXOOm2rjmgBYtpIqry883gwj3) 下载后，解压到 `reference/xMTFwww2025/KuaiRand-1K/`。

### 验证数据集结构

```bash
ls reference/xMTFwww2025/KuaiRand-1K/data/
```

应看到以下文件：

```
log_random_4_22_to_5_08_1k.csv
log_standard_4_08_to_4_21_1k.csv
log_standard_4_22_to_5_08_1k.csv
user_features_1k.csv
video_features_basic_1k.csv
video_features_statistic_1k.csv
```

✅ 6 个 CSV 文件都存在则数据集正常。

---

## 4. 第三步：创建虚拟环境 & 安装依赖

```bash
# 创建专用 conda 环境（只需做一次）
conda create -n xmtf python=3.10 -y

# 激活环境（每次新开终端都需要）
conda activate xmtf

# 进入代码目录
cd reference/xMTFwww2025/xMTF

# 安装依赖
pip install -r requirements.txt
```

### 验证安装

```bash
python -c "import tensorflow as tf; import deepctr; print('TF:', tf.__version__); print('deepctr:', deepctr.__version__)"
```

预期输出：
```
TF: 2.15.1
deepctr: 0.9.4
```

✅ 两行都正常输出则环境配置成功。

---

## 5. 第四步：数据预处理（CSV → PKL）

> **如果 `.pkl` 文件已存在（目录下有 `*_1k.pkl` 文件），跳过此步骤。**

PKL 是 CSV 的二进制版本，读取速度快 3–10 倍。

```bash
# 确保在 xMTF/ 目录下，且已激活 xmtf 环境
conda activate xmtf
cd reference/xMTFwww2025/xMTF

python prepare_data.py --data_path ../KuaiRand-1K
```

**预计耗时**：10–30 分钟（`video_features_statistic` 文件最大，3.1 GB）

完成后应看到：

```
Converting csv → pkl: 100%|████| 5/5
```

并自动创建 `models/`、`results/`、`predict/1k/` 目录。

---

## 6. 第五步：训练 Reward 模型

此步骤同时训练**双塔召回模型（TwoTower）**和 **MMoE 多任务精排模型**，为后续重排提供每个 (用户, 视频) 对的 reward 预测分。

```bash
conda activate xmtf
cd reference/xMTFwww2025/xMTF

python train.py \
    --data_path ../KuaiRand-1K \
    --split_date 2022-05-05 \
    --epochs 5 \
    --embedding_dim 16 \
    --num_experts 3 \
    --train_batch_size 512 \
    --test_batch_size 65536
```

### 训练过程说明

- 训练集：`2022-04-08` ~ `2022-05-05`，约 315 万条记录
- 测试集：`2022-05-06` ~ `2022-05-08`，约 37 万条记录
- 先训练 TwoTower（双塔召回），再训练 MMoE（精排）
- 每个 epoch 结束后自动保存最优权重（`val_loss` 最低）

### 预计耗时

| 机器类型 | 每 epoch 耗时 | 5 epochs |
|---------|-------------|---------|
| MacBook Pro M3 (CPU) | ~75 分钟 | ~6 小时 |
| Linux 服务器 (CPU, 32 核) | ~20 分钟 | ~2 小时 |
| GPU (A100) | ~5 分钟 | ~25 分钟 |

### 快速验证（只跑 1 epoch）

如果只想验证流程能跑通，可以先跑 1 epoch：

```bash
python train.py \
    --data_path ../KuaiRand-1K \
    --split_date 2022-05-05 \
    --epochs 1 \
    --embedding_dim 16 \
    --num_experts 3 \
    --train_batch_size 512
```

### 输出文件

训练完成后，以下文件将被保存：

```
models/prerank_1k.weights.h5        ← TwoTower 最优权重
models/fullrank_1k.weights.h5       ← MMoE 最优权重
results/prerank_predict_1k.pkl      ← TwoTower 对测试集的预测分
results/fullrank_predict_1k.pkl     ← MMoE 对测试集的预测分
```

并打印测试集指标，例如：

```
prerank test is_click AUC: 0.7234
prerank test is_like AUC: 0.8021
...
fullrank test is_click AUC: 0.7456
```

✅ 看到 AUC > 0.6 即为正常。

---

## 7. 第六步：构建全量候选集

用已训练的 TwoTower/MMoE 模型，对**每个用户 × 全量视频**打分，构建重排所需的候选集。

> ⚠️ **注意**：此步骤内存消耗较大（约 20–40 GB），建议在内存充足的机器上运行。

```bash
conda activate xmtf
cd reference/xMTFwww2025/xMTF

python test.py \
    --data_path ../KuaiRand-1K \
    --split_date 2022-05-05 \
    --embedding_dim 16 \
    --num_experts 3 \
    --test_batch_size 65536 \
    --per_select_num 1
```

### 进度说明

程序会把 1000 个用户分成 10 个 bucket，依次处理，终端会显示进度条：

```
100%|████████| 100/100 [05:32<00:00,  3.3s/it]   ← bucket 0
100%|████████| 100/100 [05:28<00:00,  3.3s/it]   ← bucket 1
...
```

**预计耗时**：约 1–4 小时（视机器配置）

### 输出文件

```
results/prerank_1k_0.pkl  ~  results/prerank_1k_9.pkl    ← 双塔召回打分（10个桶）
results/fullrank_1k_0.pkl ~  results/fullrank_1k_9.pkl   ← MMoE 精排打分（10个桶）
```

✅ 看到 `results/` 目录下有 20 个 pkl 文件即为成功。

---

## 8. 第七步：PRM 重排 → 输出 top-8 曝光

用 PRM（Personalized Re-ranking Model）对精排候选列表建模 item 间交互，最终为每个用户输出 top-8 曝光列表。

```bash
conda activate xmtf
cd reference/xMTFwww2025/xMTF

python prm_rerank.py \
    --data_path ../KuaiRand-1K \
    --candidate_num 50 \
    --expose_num 8 \
    --d_model 64 \
    --num_heads 4 \
    --num_layers 2 \
    --epochs 5 \
    --batch_size 64
```

### 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--candidate_num` | 50 | 每个用户从精排结果中取 top-N 送入 PRM |
| `--expose_num` | 8 | 最终曝光的 item 数量 |
| `--d_model` | 64 | Transformer hidden 维度 |
| `--num_heads` | 4 | 多头注意力头数 |
| `--num_layers` | 2 | Transformer Encoder 层数 |
| `--epochs` | 5 | PRM 训练轮数 |

**预计耗时**：约 5–20 分钟

### 输出文件

```
models/prm_1k.weights.h5              ← PRM 模型权重
results/rerank_1k_top8.csv            ← 每个用户的 top-8 曝光列表
```

### 输出格式（`rerank_1k_top8.csv`）

```csv
user_id,video_id,prm_score,expose_rank
0,4102931,0.8721,1
0,2891045,0.8634,2
0,1023847,0.8512,3
...
999,3847291,0.9012,8
```

每个用户恰好 8 行，`expose_rank` 表示曝光位置（1 = 最高位）。

---

## 9. 查看结果

### 查看评估指标

PRM 运行完后会自动打印对比精排基线和 PRM 重排的指标：

```
指标                    精排基线     PRM 重排      提升
────────────────────────────────────────────────────
NDCG@8               0.4523      0.4791     +5.92%
Precision@8          0.3812      0.4023     +5.54%
```

### 手动查看输出

```bash
# 查看某个用户的 top-8 曝光
python -c "
import pandas as pd
df = pd.read_csv('./results/rerank_1k_top8.csv')
print(df[df['user_id'] == 0])
"
```

### 查看各步骤产物

```bash
ls -lh models/      # 模型权重
ls -lh results/     # 所有结果文件
```

---

## 10. 常见报错 & 解决方案

### ❌ `ModuleNotFoundError: No module named 'tensorflow'`

```bash
# 忘记激活环境了
conda activate xmtf
```

---

### ❌ `FileNotFoundError: ... _1k.pkl`

```bash
# PKL 文件不存在，需要先做数据预处理
python prepare_data.py --data_path ../KuaiRand-1K
```

---

### ❌ `FileNotFoundError: 找不到精排结果文件: ./results/fullrank_1k_*.pkl`

```bash
# 跳过了 test.py，需要先构建候选集
python test.py --data_path ../KuaiRand-1K --split_date 2022-05-05 ...
```

---

### ❌ `ValueError: filepath provided must end in .weights.h5`

已修复（旧版 Keras 3 兼容问题）。如果仍报错，检查 `train.py` 中的模型保存路径是否以 `.weights.h5` 结尾。

---

### ❌ 内存不足（OOM / Killed）

```bash
# 减小 batch_size
python train.py ... --train_batch_size 128

# 或减小 test.py 的 test_batch_size
python test.py ... --test_batch_size 16384
```

---

### ❌ 训练太慢

```bash
# 快速验证流程用 1 epoch
python train.py ... --epochs 1

# test.py 内存紧张时也可只处理前几个 bucket（手动修改 test.py 第 271 行的 range(0, 10) 改为 range(0, 2)）
```

---

## 11. 参数速查表

### `train.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `KuaiRand-1K` | 数据集路径 |
| `--split_date` | `2022-05-05` | 训练/测试分割日期 |
| `--epochs` | `1` | 训练轮数（建议 5）|
| `--embedding_dim` | `16` | Embedding 维度 |
| `--num_experts` | `3` | MMoE Expert 数 |
| `--train_batch_size` | `256` | 训练 batch size |
| `--test_batch_size` | `65536` | 推理 batch size |
| `--seed` | `2023` | 随机种子 |

### `test.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `KuaiRand-1K` | 数据集路径 |
| `--split_date` | `2022-05-05` | 同 train.py |
| `--test_batch_size` | `65536` | 推理 batch size |
| `--per_select_num` | `1` | 每次处理的用户数（内存紧张时保持为 1）|

### `prm_rerank.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `../KuaiRand-1K` | 数据集路径 |
| `--candidate_num` | `50` | 送入 PRM 的候选数 |
| `--expose_num` | `8` | 最终曝光数 |
| `--d_model` | `64` | Transformer 隐层维度 |
| `--num_heads` | `4` | 注意力头数 |
| `--num_layers` | `2` | Encoder 层数 |
| `--epochs` | `5` | PRM 训练轮数 |
| `--batch_size` | `64` | PRM 训练 batch size |
| `--skip_train` | False | 跳过训练直接推理（需已有权重）|
| `--no_eval` | False | 跳过评估步骤 |

---

## 附：完整命令一览（复制即用）

```bash
# 0. 进入项目并激活环境
cd richMTFwww2026/reference/xMTFwww2025/xMTF
conda activate xmtf

# 1. 数据预处理（仅首次，有 pkl 文件可跳过）
python prepare_data.py --data_path ../KuaiRand-1K

# 2. 训练 Reward 模型（TwoTower + MMoE）
python train.py \
    --data_path ../KuaiRand-1K \
    --split_date 2022-05-05 \
    --epochs 5 \
    --embedding_dim 16 \
    --num_experts 3 \
    --train_batch_size 512

# 3. 构建全量候选集
python test.py \
    --data_path ../KuaiRand-1K \
    --split_date 2022-05-05 \
    --embedding_dim 16 \
    --num_experts 3 \
    --test_batch_size 65536

# 4. PRM 重排 → top-8 曝光
python prm_rerank.py \
    --data_path ../KuaiRand-1K \
    --candidate_num 50 \
    --expose_num 8 \
    --epochs 5
```

---

> 📌 如有问题，请确认：①环境已激活 (`conda activate xmtf`)；②当前目录在 `xMTF/`；③数据集路径正确。
