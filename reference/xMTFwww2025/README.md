# xMTF (WWW 2025) — 上手指南

本文档说明如何从零配置环境、准备数据、并逐步运行 xMTF 项目。

---

## 一、项目概览

xMTF 是一个面向快手短视频场景的**多任务推荐流水线系统**，完整实现了候选集→粗排→精排→重排→曝光的全链路排序流程，其中以 **MTF（Multi-Task Fusion）多任务融合 Transformer** 为核心创新。

```
候选集 (~7,097)
    ↓ TwoTower 双塔粗排     two_tower.py
top-500
    ↓ MMoE 多任务精排        mmoe.py
500 条精排分
    ↓ MTF 多任务融合         pipeline_demo.py（内置）
    └─ 把 7 个任务分作为 token，用 Transformer 建模任务间依赖，输出融合分
top-60（精排 × MTF 融合截断）
    ↓ PRM Transformer 重排  prm_rerank.py
    └─ 列表级上下文建模（60 个 item 互相感知），输出重排分
top-8 曝光
```

**一键运行：**

```bash
python pipeline_demo.py --data_path ../KuaiRand-1K --nrows 100000
```

---

## 二、项目目录结构

```
xMTFwww2025/
├── README.md                   # 本文档：环境配置 & 上手指南
├── PIPELINE_DOC.md             # 完整流水线说明（各阶段输入输出、模型结构）
├── TRAINING_SOP.md             # 傻瓜式训练 SOP（7步跑通全流程）
│
├── KuaiRand-1K/                # 数据集（需自行下载，不上传 git）
│   └── data/
│       ├── log_standard_4_08_to_4_21_1k.pkl    # 行为日志（前半月）
│       ├── log_standard_4_22_to_5_08_1k.pkl    # 行为日志（后半月）
│       ├── user_features_1k.pkl                # 用户特征（1000行）
│       ├── video_features_basic_1k.pkl         # 视频基础特征
│       └── video_features_statistic_1k.pkl     # 视频统计特征
│
└── xMTF/                       # 核心代码
    ├── requirements.txt        # 依赖列表（TF2.15.1 + deepctr 0.9.4）
    │
    │── pipeline_demo.py        # ★ 完整四阶段流水线（粗排→精排→MTF→PRM重排）
    ├── prepare_data.py         # 数据预处理（CSV → PKL，生成 data/ 目录）
    │
    ├── two_tower.py            # 粗排模型：TwoTower 双塔召回
    ├── mmoe.py                 # 精排模型：MMoE 多任务精排
    ├── prm_rerank.py           # 重排模型：PRM Transformer 列表级重排
    │
    ├── train.py                # 原始训练脚本（TwoTower + MMoE）
    ├── test.py                 # 原始推理脚本（生成候选集打分）
    ├── mfc_train_test.py       # MFC Transformer 训练 & 评估
    │
    ├── mfc.py                  # MFC 模型定义
    ├── simulator.py            # RL 环境模拟器
    ├── rule_agent.py           # 规则 Agent
    ├── td3.py                  # TD3 强化学习 Agent
    ├── ddpg.py                 # DDPG 强化学习 Agent
    └── util.py                 # 工具函数
```

> **推荐使用 `pipeline_demo.py`**：不需要 RL 环境，一个脚本跑通粗排→精排→MTF融合→PRM重排→top-8曝光完整链路，并输出各阶段的离线 AUC 评估。

---

## 二、数据集：KuaiRand-1K

### 2.1 数据集背景

KuaiRand 由快手（Kuaishou）发布，是**首个包含随机干预曝光**的序列推荐数据集。其最大特点是：在标准推荐流中随机插入约 0.37% 的随机曝光视频，提供了接近无偏的用户反馈观测，特别适合用于 off-policy evaluation 和强化学习研究。

- 论文：[KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos](https://arxiv.org/abs/2208.08696)
- GitHub：https://github.com/chongminggao/KuaiRand

**KuaiRand-1K** 是完整数据集（KuaiRand-27K）的子集，随机抽取 1,000 个用户，删除所有无关视频后保留。

| 指标 | KuaiRand-1K |
|------|------------|
| 用户数 | 1,000 |
| 视频数 | 4,369,953 |
| 标准日志交互数 | 11,713,045 |
| 随机曝光交互数 | 43,028 |
| 用户特征 | 30 维 |
| 视频特征 | 62 维 |
| 反馈信号 | 12 种 |
| 时间跨度 | 2022-04-08 ~ 2022-05-08 |

---

### 2.2 文件结构

```
KuaiRand-1K/
├── data/
│   ├── log_random_4_22_to_5_08_1k.csv      (2.9 MB)   ← 随机干预曝光日志
│   ├── log_standard_4_08_to_4_21_1k.csv    (368 MB)   ← 标准推荐日志（前两周）
│   ├── log_standard_4_08_to_4_21_1k.pkl    (733 MB)   ← 同上，pickle 格式（读取更快）
│   ├── log_standard_4_22_to_5_08_1k.csv    (481 MB)   ← 标准推荐日志（后两周）
│   ├── log_standard_4_22_to_5_08_1k.pkl    (965 MB)   ← 同上，pickle 格式
│   ├── user_features_1k.csv                (132 KB)   ← 用户特征
│   ├── user_features_1k.pkl                (215 KB)
│   ├── video_features_basic_1k.csv         (368 MB)   ← 视频基础特征
│   ├── video_features_basic_1k.pkl         (354 MB)
│   ├── video_features_statistic_1k.csv     (3.1 GB)   ← 视频统计特征（最大文件）
│   └── video_features_statistic_1k.pkl     (1.7 GB)
└── load_data_1k.py                                    ← 官方数据读取示例
```

> **关于 `.pkl` 文件**：pkl 是 Python Pickle 序列化格式，等价于 CSV 的二进制版本。它直接保存 DataFrame 对象（含数据类型信息），读取速度比 CSV 快 3-10 倍，文件大小也更小（统计特征从 3.1 GB 压缩到 1.7 GB）。`train.py` 直接读取 pkl 文件，如果 pkl 不存在需先运行 `prepare_data.py` 转换。

---

### 2.3 日志文件字段说明（`log_standard_*.csv` / `log_random_*.csv`）

每行代表一次用户-视频交互记录，共 **19 个字段**：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `user_id` | int64 | 用户 ID（已重新索引，范围 [0, 999]） |
| `video_id` | int64 | 视频 ID |
| `date` | int64 | 交互日期，格式 `YYYYMMDD`，例如 `20220421` |
| `hourmin` | int64 | 交互时刻，格式 `HHSS`，例如 `400` |
| `time_ms` | int64 | 毫秒级时间戳 |
| `is_click` | int64 | **点击**：双列 UI 表示点击；单列 UI 表示有效播放（`play_duration >= video_duration` if ≤7s，or `> 7s`）|
| `is_like` | int64 | 是否点击"喜欢"按钮 |
| `is_follow` | int64 | 是否关注了作者 |
| `is_comment` | int64 | 是否发表了评论 |
| `is_forward` | int64 | 是否转发了视频 |
| `is_hate` | int64 | 是否点击了"不喜欢" |
| `long_view` | int64 | **长观看**：`play_duration >= video_duration` if ≤18s，or `>= 18s` |
| `play_time_ms` | int64 | 用户实际观看时长（毫秒） |
| `duration_ms` | int64 | 视频总时长（毫秒） |
| `profile_stay_time` | int64 | 在作者主页停留的时长 |
| `comment_stay_time` | int64 | 在评论区停留的时长 |
| `is_profile_enter` | int64 | 是否进入了作者主页 |
| `is_rand` | int64 | **是否为随机干预曝光**（`log_random` 文件中全为 1）|
| `tab` | int64 | 场景标识 [0, 14]，`tab=1` 为首页推荐流（项目主要使用此场景） |

> **xMTF 使用的标签**（7 个）：`is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `long_view`, `play_time_label`
> 其中 `play_time_label = log(play_time_s + 1) / 7`，将播放时长归一化为连续回归标签。

---

### 2.4 用户特征字段说明（`user_features_1k.csv`）

共 **30 个字段**，1,000 行（每用户一行）：

| 字段名 | 说明 |
|--------|------|
| `user_id` | 用户 ID |
| `user_active_degree` | 活跃度，取值：`full_active` / `high_active` / `middle_active` / `UNKNOWN` |
| `is_lowactive_period` | 是否处于低活跃期 |
| `is_live_streamer` | 是否为直播主播 |
| `is_video_author` | 是否上传过视频 |
| `follow_user_num` | 关注用户数（原始值，训练时丢弃，改用 range 版本） |
| `follow_user_num_range` | 关注用户数区间，如 `"(0,10]"` |
| `fans_user_num` | 粉丝数（原始值） |
| `fans_user_num_range` | 粉丝数区间，如 `"[100,1k)"` |
| `friend_user_num` | 好友数（原始值） |
| `friend_user_num_range` | 好友数区间，如 `"0"` |
| `register_days` | 注册天数（原始值） |
| `register_days_range` | 注册天数区间，如 `"730+"` |
| `onehot_feat0` ~ `onehot_feat17` | 18 个加密特征（用户画像标签，具体含义未公开） |

---

### 2.5 视频基础特征字段说明（`video_features_basic_1k.csv`）

共 **12 个字段**，4,369,953 行（每视频一行）：

| 字段名 | 说明 |
|--------|------|
| `video_id` | 视频 ID |
| `author_id` | 作者 ID |
| `video_type` | 视频类型：`NORMAL` / `AD` |
| `upload_dt` | 上传日期，格式 `YYYY-MM-DD` |
| `upload_type` | 上传方式，如 `ShortImport` |
| `visible_status` | 视频当前可见状态 |
| `video_duration` | 视频时长（毫秒），训练时分 16 桶离散化 |
| `server_width` | 服务端视频宽度（像素） |
| `server_height` | 服务端视频高度（像素） |
| `music_id` | 背景音乐 ID |
| `music_type` | 背景音乐类型 |
| `tag` | 视频标签，逗号分隔，如 `"12,65"`（训练时未使用） |

---

### 2.6 视频统计特征字段说明（`video_features_statistic_1k.csv`）

共 **62 个字段**，数值为**一个月内每日每场景的平均统计值**，训练时全部分 16 桶离散化：

| 字段名 | 说明 |
|--------|------|
| `video_id` | 视频 ID |
| `counts` | 统计条数（同一视频在不同日期/场景的记录数） |
| `show_cnt` | 平均曝光次数 |
| `show_user_num` | 平均曝光用户数 |
| `play_cnt` / `play_user_num` | 平均播放次数/用户数 |
| `play_duration` | 平均总播放时长（毫秒） |
| `complete_play_cnt` / `complete_play_user_num` | 完播次数/用户数 |
| `valid_play_cnt` / `valid_play_user_num` | 有效播放次数/用户数 |
| `long_time_play_cnt` / `long_time_play_user_num` | 长观看次数/用户数 |
| `short_time_play_cnt` / `short_time_play_user_num` | 短观看次数/用户数 |
| `play_progress` | 平均播放进度比（`play_duration / video_duration`） |
| `comment_stay_duration` | 评论区停留总时长 |
| `like_cnt` / `like_user_num` | 点赞次数/用户数 |
| `click_like_cnt` / `double_click_cnt` | 双击点赞次数 |
| `cancel_like_cnt` / `cancel_like_user_num` | 取消点赞次数/用户数 |
| `comment_cnt` / `comment_user_num` | 评论次数/用户数 |
| `follow_cnt` / `follow_user_num` | 因该视频增加的关注次数/用户数 |
| `share_cnt` / `share_user_num` | 分享次数/用户数 |
| `download_cnt` / `download_user_num` | 下载次数/用户数 |
| `collect_cnt` / `collect_user_num` | 收藏次数/用户数 |
| ……（共 62 列） | 其余字段含义类似，详见 KuaiRand 官方 README |

---

## 三、环境配置

### 3.1 系统要求

- Python 3.10 或 3.12（推荐 3.10）
- 内存 ≥ 32 GB（`video_features_statistic` pkl 占 1.7 GB，全量数据合并后约 20 GB）
- 磁盘空间 ≥ 10 GB（存放 pkl 文件）

### 3.2 安装依赖

```bash
# 推荐使用 conda 管理环境
conda create -n xmtf python=3.10 -y
conda activate xmtf

# 安装 TensorFlow 2.18（当前版本）
pip install tensorflow==2.18.0

# 安装 DeepCTR（多任务模型库）
pip install deepctr==0.9.4

# 安装其他依赖
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

> **注意**：deepctr 官方只支持到 TF 2.x，本项目已针对 TF 2.18（Keras 3）做了兼容性修复（`tensorflow.python.keras` → `tensorflow.keras`，权重文件后缀 `.h5` → `.weights.h5`）。

---

## 四、数据准备

### 4.1 下载数据集

```bash
# 从官方服务器下载 KuaiRand-1K（压缩包约 1 GB）
wget https://chongming.myds.me:61364/data/KuaiRand-1K.tar.gz --no-check-certificate
tar -zxvf KuaiRand-1K.tar.gz
```

或从 Google Drive 手动下载：https://drive.google.com/drive/folders/1BHcv3bgoXOOm2rjmgBYtpIqry883gwj3

解压后将数据集放置到以下路径：

```
reference/xMTFwww2025/KuaiRand-1K/
```

### 4.2 转换 CSV → PKL（必须，仅需运行一次）

代码读取 pkl 格式（速度快），需先将 CSV 转换：

```bash
cd reference/xMTFwww2025/xMTF
python prepare_data.py --data_path ../KuaiRand-1K
```

`prepare_data.py` 会将全部 5 个 CSV 转为 pkl，并自动创建 `models/`、`results/`、`predict/1k/` 目录。转换耗时约 10-30 分钟（统计特征文件最大）。

> **如果 pkl 文件已存在**（已有 `.pkl` 后缀的文件），跳过此步骤。

---

## 五、逐步运行

### Step 1：训练 Reward 模型（`train.py`）

训练**双塔召回模型**（TwoTower）和 **MMoE 精排模型**，为后续 RL 提供 reward 预测。

```bash
cd reference/xMTFwww2025/xMTF

python train.py \
  --data_path ../KuaiRand-1K \
  --split_date 2022-05-05 \
  --epochs 5 \
  --embedding_dim 16 \
  --num_experts 3 \
  --train_batch_size 256 \
  --test_batch_size 65536 \
  --seed 2023
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `KuaiRand-1K` | 数据集根目录路径 |
| `--split_date` | `2022-05-05` | 训练/测试集分割日期（该日期之前为训练集） |
| `--epochs` | `1` | 训练轮数（正式训练建议 5） |
| `--embedding_dim` | `16` | Embedding 维度 |
| `--num_experts` | `3` | MMoE 专家网络数量 |
| `--train_batch_size` | `256` | 训练 batch size |

**输出文件**：
```
models/prerank_1k.weights.h5        ← TwoTower 最优权重
models/fullrank_1k.weights.h5       ← MMoE 最优权重
results/prerank_predict_1k.pkl      ← TwoTower 对测试集的预测结果
results/fullrank_predict_1k.pkl     ← MMoE 对测试集的预测结果
```

**训练集/测试集划分**：
- 训练集：`2022-04-08` ~ `2022-05-05`（约 315 万条，`tab=1` 首页推荐场景）
- 测试集：`2022-05-06` ~ `2022-05-08`（约 37 万条）

**预计时长**：1 epoch 约 75 分钟（M1/M2 Mac CPU），5 epoch 约 6 小时。

---

### Step 2：构建候选集（`test.py`）

使用训练好的模型对全量用户-视频对打分，构建 RL 环境所需的候选集。

```bash
python test.py \
  --data_path ../KuaiRand-1K \
  --split_date 2022-05-05 \
  --embedding_dim 16 \
  --num_experts 3 \
  --test_batch_size 65536
```

**输出文件**：
```
predict/1k/fullrank_set_train.pkl   ← 训练用候选集（用户-视频打分矩阵）
predict/1k/fullrank_set_test.pkl    ← 测试用候选集
```

---

### Step 3：训练 MFC 模型 + RL 评估（`mfc_train_test.py`）

训练列表级打分 Transformer（MFC），在 ESEnv 模拟器环境中做 RL 训练与评估。

```bash
# 训练
python mfc_train_test.py --data_path ../KuaiRand-1K --train

# 测试（推理）
python mfc_train_test.py --data_path ../KuaiRand-1K --test
```

**输出文件**：
```
models/1k/mfc_model/    ← MFC 模型权重
```

---

## 六、常见问题

### Q: 出现 `AttributeError: 'tuple' object has no attribute 'rank'`
**原因**：deepctr 0.9.4 使用 `tensorflow.python.keras`（TF1 遗留 API），与 TF 2.18 的 Keras 3 不兼容。  
**解决**：本项目已在 `two_tower.py`、`mmoe.py`、`train.py`、`test.py`、`mfc_train_test.py` 中将所有 `tensorflow.python.keras` 替换为 `tensorflow.keras`。

### Q: 出现 `ValueError: filepath provided must end in .weights.h5`
**原因**：Keras 3 要求 `save_weights_only=True` 时文件名以 `.weights.h5` 结尾。  
**解决**：已将 `train.py` 和 `test.py` 中的模型保存路径从 `.h5` 改为 `.weights.h5`。

### Q: 内存不足（OOM）
**解决**：减小 `--train_batch_size`（如改为 128），或减小 `--embedding_dim`（如改为 8）。

### Q: `prepare_data.py` 找不到文件
**确认** CSV 文件在 `KuaiRand-1K/data/` 目录下，文件名格式为 `*_1k.csv`（注意末尾的 `_1k`）。

---

## 七、项目文件说明

```
xMTFwww2025/
├── README.md                   # 本文档：环境配置 & 上手指南
├── PIPELINE_DOC.md             # 完整流水线说明（各阶段输入输出、模型结构）
├── TRAINING_SOP.md             # 傻瓜式训练 SOP（7步跑通全流程）
│
├── KuaiRand-1K/                # 数据集（需自行下载，不上传 git）
│   └── data/
│       ├── log_standard_4_08_to_4_21_1k.pkl    # 行为日志（前半月）
│       ├── log_standard_4_22_to_5_08_1k.pkl    # 行为日志（后半月）
│       ├── user_features_1k.pkl                # 用户特征（1000行）
│       ├── video_features_basic_1k.pkl         # 视频基础特征
│       └── video_features_statistic_1k.pkl     # 视频统计特征
│
└── xMTF/                       # 核心代码
    ├── requirements.txt        # 依赖列表（TF2.15.1 + deepctr 0.9.4）
    │
    ├── pipeline_demo.py        # ★ 完整四阶段流水线（粗排→精排→MTF→PRM重排）
    ├── prepare_data.py         # 数据预处理（CSV → PKL，生成 data/ 目录）
    │
    ├── two_tower.py            # 粗排模型：TwoTower 双塔召回
    ├── mmoe.py                 # 精排模型：MMoE 多任务精排
    ├── prm_rerank.py           # 重排模型：PRM Transformer 列表级重排
    │
    ├── train.py                # 原始训练脚本（TwoTower + MMoE）
    ├── test.py                 # 原始推理脚本（生成候选集打分）
    ├── mfc_train_test.py       # MFC Transformer 训练 & 评估
    │
    ├── mfc.py                  # MFC 模型定义
    ├── simulator.py            # RL 环境模拟器
    ├── rule_agent.py           # 规则 Agent
    ├── td3.py                  # TD3 强化学习 Agent
    ├── ddpg.py                 # DDPG 强化学习 Agent
    └── util.py                 # 工具函数
```

> **推荐使用 `pipeline_demo.py`**：不需要 RL 环境，一个脚本跑通粗排→精排→MTF融合→PRM重排→top-8曝光完整链路，并输出各阶段的离线 AUC 评估。
