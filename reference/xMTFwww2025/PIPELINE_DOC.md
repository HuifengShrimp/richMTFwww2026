# 小型推荐系统完整流水线说明文档

> 代码入口：`reference/xMTFwww2025/xMTF/pipeline_demo.py`
> 数据集：KuaiRand-1K（快手短视频真实用户行为数据）

---

## 整体架构

```
原始数据
   ↓ 特征预处理
候选集（~7,097 视频/用户）
   ↓ 粗排  TwoTower 双塔模型
 top-500
   ↓ 精排  MMoE 多任务模型（对500全打分）
  500 条精排分
   ↓ MTF   多任务融合（Transformer 融合7个任务分 → 1个最终分）
 top-60
   ↓ 重排  PRM（Transformer 建模列表级交互）
 top-8  ← 最终曝光
```

---

## 1. 原始数据集

### 1.1 数据集概述

**KuaiRand-1K** 是快手短视频平台的真实用户行为公开数据集，包含 991 个用户在 2022 年 4 月 8 日至 5 月 8 日（约一个月）内的完整曝光日志。

| 维度 | 数值 |
|------|------|
| 用户数 | 991 |
| 全量视频数 | 2,395,326 |
| 总交互记录数 | 约 703 万条 |
| 时间范围 | 2022-04-08 ～ 2022-05-08 |
| 每用户平均候选数 | 约 7,097 个视频 |

数据包含 **4 张表**，存储为 `.pkl` 格式（Pandas DataFrame 二进制序列化）：

---

### 1.2 表结构与样例

#### 表1：用户行为日志（log_standard）

记录每一次用户–视频曝光事件，含多维行为标签。

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `user_id` | int | 用户唯一标识 |
| `video_id` | int | 视频唯一标识 |
| `date` | datetime64 | 交互日期 |
| `tab` | int | 场景 Tab（1=首页推荐） |
| `play_time_ms` | int | 实际播放时长（毫秒） |
| `duration_ms` | int | 视频总时长（毫秒） |
| `is_click` | int | 是否点击（0/1） |
| `is_like` | int | 是否点赞（0/1） |
| `is_follow` | int | 是否关注作者（0/1） |
| `is_comment` | int | 是否评论（0/1） |
| `is_forward` | int | 是否转发（0/1） |
| `long_view` | int | 是否完整观看（0/1） |

**数据样例：**
```
user_id  video_id        date  is_click  is_like  is_follow  is_comment  is_forward  long_view  play_time_ms  duration_ms
0        3173641  2022-04-09         0        0          0           0           0          0          791        5200
0         889528  2022-04-09         0        0          0           0           0          0         2683       12000
1        2650549  2022-04-10         1        1          0           0           0          1         8900        9100
```

> 数据来源：两个 PKL 文件拼接
> - `log_standard_4_08_to_4_21_1k.pkl`（前半月）
> - `log_standard_4_22_to_5_08_1k.pkl`（后半月）

---

#### 表2：用户特征（user_features）

30 个字段，每个用户一行。

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `user_id` | int | 用户唯一标识 |
| `user_active_degree` | str | 活跃程度（full_active / 等级） |
| `is_lowactive_period` | int | 是否低活跃期 |
| `is_live_streamer` | int | 是否直播主播 |
| `is_video_author` | int | 是否视频创作者 |
| `follow_user_num_range` | str | 关注数区间（如 "500+"） |
| `fans_user_num_range` | str | 粉丝数区间 |
| `friend_user_num_range` | str | 互关好友数区间 |
| `register_days_range` | str | 注册天数区间（如 "730+"） |
| `onehot_feat1~17` | float | 脱敏后的用户行为特征 |

**数据样例：**
```
user_id  user_active_degree  is_live_streamer  follow_user_num_range  fans_user_num_range  register_days_range
0        full_active                 1                500+               [100,1k)                730+
1        full_active                 0              (250,500]              [10,100)               730+
```

---

#### 表3：视频基础特征（video_features_basic）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `video_id` | int | 视频唯一标识 |
| `author_id` | int | 作者 ID |
| `video_type` | str | 视频类型（NORMAL 等） |
| `upload_type` | str | 上传类型 |
| `visible_status` | str | 可见状态 |
| `music_id` | int | 背景音乐 ID |
| `music_type` | str | 音乐类型 |
| `video_duration` | float | 视频时长（毫秒） |
| `server_width` | int | 服务端宽度 |
| `server_height` | int | 服务端高度 |

---

#### 表4：视频统计特征（video_features_statistic）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `video_id` | int | 视频唯一标识 |
| `show_cnt` | float | 累计曝光次数 |
| `play_cnt` | float | 累计播放次数 |
| `like_cnt` | float | 累计点赞数 |
| ... | float | 其余约 60 个统计特征 |

---

## 2. 数据预处理

数据预处理分为 **五个子步骤**：数据清洗 → 衍生特征构造 → 连续特征离散化 → 类别特征编码 → 多表 Join → 训练/测试集划分。

---

### 2.1 数据清洗（在行为日志上操作）

**输入**：拼接后的两段行为日志（前半月 + 后半月），原始约 1100 万条。

```python
# Step 1：只保留首页推荐场景（tab=1），过滤掉搜索、关注等其他 Tab
data = data[data['tab'] == 1]

# Step 2：过滤无效播放记录
#   - play_time_ms=0：视频未被实际播放（纯曝光未点开）
#   - duration_ms=0：视频时长缺失，无法计算播放率
data = data[(data['play_time_ms'] > 0) & (data['duration_ms'] > 0)]

# Step 3：过滤播放倍率异常值
#   play_rate = play_time / duration
#   正常范围是 [0, 1]（完整播放=1），允许最多重播5次（≤5）
#   >5 认为是数据采集异常，予以剔除
data['play_rate'] = data['play_time_ms'] / data['duration_ms']
data = data[data['play_rate'] <= 5]
```

**清洗效果**：约 1100 万条 → **703 万条**有效记录。

---

### 2.2 衍生特征构造

```python
# 时长单位转换（毫秒 → 秒，更直观）
data['duration_s']  = data['duration_ms'] / 1000
data['play_time_s'] = data['play_time_ms'] / 1000

# play_time_label：回归任务的监督标签
# 为什么要 log + 归一化？
#   play_time_s 分布极度长尾（大多数人看几秒，少数人看几分钟）
#   log(x+1) 压缩长尾分布，使回归目标更平滑
#   除以7 是根据数据分位数归一化到 [0,1] 附近（log(e^7)≈7）
data['play_time_label'] = np.log(data['play_time_s'] + 1) / 7
```

**注意**：`play_time_label` 是精排/粗排中回归任务的标签，范围约 [0, 1]；
`play_time_s` 是预测结果还原时使用的原始秒数（反变换：`exp(label*7) - 1`）。

---

### 2.3 连续特征离散化（qcut 16分桶）

**为什么要分桶？**
深度学习推荐模型普遍使用稀疏特征 + Embedding，而不直接使用连续数值。
将连续值分桶后，每个桶对应一个可学习的 Embedding 向量，
比直接输入浮点数更容易学出非线性关系，且对异常值更鲁棒。

**分桶方式**：`pd.qcut`（等频分桶）—— 每个桶内样本数量大致相等，分 16 桶（0~15）。

涉及字段：

| 来源表 | 字段 | 说明 |
|--------|------|------|
| 视频基础特征 | `video_duration` | 视频时长（毫秒），分16桶 |
| 视频统计特征 | `show_cnt`, `play_cnt`, `like_cnt` 等约60个字段 | 全部分16桶 |

```python
# 视频时长分桶
vfb['video_duration'] = pd.qcut(vfb['video_duration'], q=16, labels=False, duplicates='drop')

# 视频统计特征全部分桶
for col in stat_dense_cols:  # show_cnt, play_cnt, like_cnt, ...
    vfs[col] = pd.qcut(vfs[col], q=16, labels=False, duplicates='drop')
```

分桶结果：连续浮点数 → 整数 [0, 15]，作为后续 Embedding 的输入索引。

---

### 2.4 类别特征 LabelEncoder 编码

**目的**：将所有字符串类别特征转换为从 0 开始的连续整数，作为 Embedding 表的查找索引。

```python
def fillna_df(df, features):
    for feat in features:
        df[feat] = LabelEncoder().fit_transform(
            df[feat].astype(str).fillna('-1')   # 缺失值用 '-1' 占位
        )
```

| 来源表 | 处理的字段 | 示例（处理前 → 处理后） |
|--------|----------|------------------------|
| 用户特征 | `user_active_degree`, `follow_user_num_range`, `fans_user_num_range`, `friend_user_num_range`, `register_days_range`, `is_lowactive_period`, `is_live_streamer`, `is_video_author`, `onehot_feat1~17` | `"full_active"` → `3`，`"500+"` → `7` |
| 视频基础特征 | `video_type`, `upload_type`, `visible_status`, `music_type`, `author_id`, `music_id`, `server_width`, `server_height`，以及分桶后的 `video_duration` | `"NORMAL"` → `1` |
| 视频统计特征 | 分桶后的所有统计字段（已经是整数 0~15，仍走 LabelEncoder 统一格式） | `7` → `7` |

**特别说明**：
- `follow_user_num`（精确关注数）在预处理时被**直接删除**，只保留区间型的 `follow_user_num_range`，因为精确数值存在隐私风险且区间特征已足够
- 缺失值统一填充为字符串 `'-1'`，LabelEncoder 会为其分配一个独立的编码，模型会学到"缺失"本身也是一种状态

---

### 2.5 多表 Left Join 合并

将用户特征、视频基础特征、视频统计特征全部拼接到行为日志上，形成一条完整的样本：

```
行为日志（user_id, video_id, is_click, ...）
    LEFT JOIN 用户特征表 ON user_id        ← 补充用户侧26个特征
    LEFT JOIN 视频基础特征表 ON video_id    ← 补充视频侧10个特征
    LEFT JOIN 视频统计特征表 ON video_id    ← 补充视频侧51个特征
    → 合并后每条记录 = 1次交互 + 用户画像 + 视频画像
```

Join 方式为 `LEFT JOIN`：行为日志为主表，特征表为副表。
若某个 video_id 在特征表中找不到（冷启动视频），对应特征列填 0。

```python
data = (data.merge(uf,  on='user_id',  how='left')
            .merge(vfb, on='video_id', how='left')
            .merge(vfs, on='video_id', how='left'))
data[sparse_features] = data[sparse_features].fillna(0).astype(int)
```

---

### 2.6 最终特征维度汇总

经过上述处理，每条样本包含以下稀疏特征：

| 类别 | 特征列表 | 数量 |
|------|----------|------|
| **用户侧** | `user_id`, `user_active_degree`, `is_lowactive_period`, `is_live_streamer`, `is_video_author`, `follow_user_num_range`, `fans_user_num_range`, `friend_user_num_range`, `register_days_range`, `onehot_feat1~17` | **26 个** |
| **视频基础侧** | `video_id`, `author_id`, `video_type`, `upload_type`, `visible_status`, `music_id`, `music_type`, `video_duration`（分桶）, `server_width`, `server_height` | **10 个** |
| **视频统计侧** | `show_cnt`, `play_cnt`, `like_cnt` 等约 51 个统计特征（全部分桶） | **51 个** |
| **合计** | — | **87 个稀疏特征** |

每个稀疏特征对应一张 Embedding 表，查表得到 8 维向量，
**模型 DNN 层的实际输入维度 = 87 × 8 = 696 维**。

---

### 2.7 训练/测试集划分

按时间顺序切分，**严格保证测试集在训练集之后**（避免未来数据泄露到训练）：

```
数据时间范围：2022-04-08 ～ 2022-05-08（约30天）
split_date = 数据起始日 + 30天 × 80% ≈ 2022-04-08 + 24天 = 2022-05-02

训练集：date ≤ split_date（前80%时间段）
测试集：date >  split_date（后20%时间段）
```

> 注：`split_date` 会根据实际数据日期范围自动计算，若用户手动指定的日期不在数据范围内，则自动回退为"数据日期 × 0.8"分割点。

---

## 3. 粗排阶段（TwoTower 双塔召回）

### 3.1 作用

候选集平均有 **7,097 个视频**，对全量做精排计算成本极高。
双塔模型能快速对所有候选打分，从中筛出最相关的 **top-500** 送入精排。

### 3.2 模型结构

```
用户侧特征（26个稀疏特征）
    → Embedding concat（26×8=208维）
    → DNN Bottom: 208→256→128
    → 7个任务各自 Tower DNN: 128→64
    → 用户塔向量 user_vec (64维)

视频侧特征（61个稀疏特征）
    → Embedding concat（61×8=488维）
    → DNN Bottom: 488→256→128
    → 7个任务各自 Tower DNN: 128→64
    → 视频塔向量 item_vec (64维)

打分：
    user_vec ⊙ item_vec  （逐元素乘积，64维）
    → Dense(1) → sigmoid/linear
    → 7个任务各一个输出
```

**关键设计**：用户塔和视频塔**分离编码**，线上推理时可预先存好所有视频的向量，用 ANN（近似最近邻）检索，速度远快于精排。

### 3.3 训练目标（7个任务）

**为什么是这7个任务？** 它们覆盖了用户的主要正向行为信号：
- 浅层互动：点击（is_click）
- 中层互动：点赞（is_like）、评论（is_comment）、转发（is_forward）
- 深层互动：关注作者（is_follow）
- 观看质量：完整观看（long_view）、实际播放时长（play_time_label）

| 任务名 | 含义 | 类型 | 损失函数 | 权重 |
|--------|------|------|----------|------|
| `is_click` | 用户是否点击该视频 | 二分类(0/1) | Binary CrossEntropy | 1.0 |
| `is_like` | 用户是否点赞 | 二分类(0/1) | Binary CrossEntropy | 1.0 |
| `is_follow` | 用户是否关注该视频作者 | 二分类(0/1) | Binary CrossEntropy | 1.0 |
| `is_comment` | 用户是否发表评论 | 二分类(0/1) | Binary CrossEntropy | 1.0 |
| `is_forward` | 用户是否转发 | 二分类(0/1) | Binary CrossEntropy | 1.0 |
| `long_view` | 用户是否完整观看 | 二分类(0/1) | Binary CrossEntropy | 1.0 |
| `play_time_label` | 归一化播放时长 log(t+1)/7 | 回归[0,1] | Huber Loss | 100.0 |

> Huber Loss 权重设为 100 是因为回归值本身很小（约 0.01 量级），不放大权重会被分类 loss 淹没。

### 3.4 输入输出

| 项目 | 说明 |
|------|------|
| **输入** | 训练：87个稀疏特征 dict，每特征 shape=(N,) |
| **输出（预测）** | 7个任务的预测分，shape 各为 (N,) |
| **输出字段** | `pre_is_click`, `pre_is_like`, `pre_is_follow`, `pre_is_comment`, `pre_is_forward`, `pre_long_view`, `pre_play_time_s` |

**粗排输出样例：**
```
user_id  video_id  pre_is_click  pre_is_like  pre_long_view  pre_play_time_s  _prerank_score
0        3173641      0.412        0.031          0.301            5.2             5.956
0        889528       0.389        0.028          0.278            4.8             5.495
0        2650549      0.445        0.038          0.335            6.1             6.918
```

### 3.5 粗排排序与截断

```
综合分 = pre_is_click×1 + pre_is_like×1 + pre_is_follow×1
       + pre_is_comment×1 + pre_is_forward×1 + pre_long_view×1
       + pre_play_time_s×1
（全1等权加权，各任务分贡献相同）

每用户按综合分降序排列 → 截取 top-500
```

> 候选集 **~7,097** → 粗排 → **top-500**（漏斗压缩比约 14:1）

---

## 4. 精排阶段（MMoE 多任务精排）

### 4.1 作用

对粗排筛出的 **500 个候选**进行精细建模。MMoE 使用共享 Expert 网络+独立 Gate，更好地建模多个任务之间的相互关联（如点赞和评论正相关、关注和转发正相关）。

### 4.2 模型结构与输入维度解析

**为什么输入是 696 维？**

精排的输入把用户特征和视频特征**拼接在一起**，一共 87 个稀疏特征，每个特征通过 Embedding 表映射为 8 维向量，最终 concat 得到：

```
87 个稀疏特征 × 8维 Embedding = 696 维稠密向量

拆解：
  用户侧   26 个特征 × 8 = 208 维
  视频基础  10 个特征 × 8 =  80 维
  视频统计  51 个特征 × 8 = 408 维
  合计：208 + 80 + 408     = 696 维
```

> 粗排 TwoTower：用户塔 208 维，视频塔 488 维，两者**分开**各自走 DNN，最后才点积。  
> MMoE 精排：696 维全部**合并**走同一个 DNN，用户–视频的交叉信息从输入层就开始建模，表达能力更强。

**完整模型流程：**

```
输入：87个稀疏特征（用户26 + 视频基础10 + 视频统计51）
    → 各特征查 Embedding 表（每特征 8 维）
    → Concat 拼接 → 696 维稠密向量 dnn_input

Expert 层（3个独立 DNN，对所有任务共享）：
    Expert_0: 696 → Dense(256, relu) → Dense(128, relu)  输出 128 维
    Expert_1: 696 → Dense(256, relu) → Dense(128, relu)  输出 128 维
    Expert_2: 696 → Dense(256, relu) → Dense(128, relu)  输出 128 维
    Stack → expert_concat，shape=(N, 3, 128)

Gate 层（每个任务独立 1 个 Gate）：
    Gate_k: dnn_input(696) → Dense(3, softmax) → (α₀, α₁, α₂)  三权重之和=1
    混合：mmoe_out_k = α₀×Expert_0 + α₁×Expert_1 + α₂×Expert_2   128 维

Tower 层（每个任务独立 1 个 Tower）：
    Tower_k: mmoe_out_k(128) → Dense(64, relu) → Dense(1) → sigmoid/linear
    → 任务 k 的预测分
```

**MMoE vs TwoTower 核心差异对比：**

| | TwoTower（粗排） | MMoE（精排） |
|--|--|--|
| 特征融合方式 | 用户塔、视频塔分离编码，最后点积 | 用户+视频特征拼接，统一建模 |
| 计算速度 | 快（可预存视频向量，ANN 检索） | 慢（每次都要完整前向） |
| 表达能力 | 弱（交叉信息只在最后一步） | 强（从输入层就有交叉） |
| 适用阶段 | 粗排（大候选集快速召回） | 精排（小候选集精细打分） |
- TwoTower：用户塔和视频塔分离，速度快，用于粗筛
- MMoE：用户+视频特征拼接后统一建模，表达能力更强，用于精排

### 4.3 训练目标

与粗排相同：7 个任务，同样的损失函数和权重配置。

### 4.4 输入输出

| 项目 | 说明 |
|------|------|
| **输入** | 粗排 top-500 的 87个稀疏特征，shape=(500_per_user × N_users,) |
| **输出（预测）** | 7个任务的预测分 |
| **输出字段** | `full_is_click`, `full_is_like`, `full_is_follow`, `full_is_comment`, `full_is_forward`, `full_long_view`, `full_play_time_s` |

**精排输出样例：**
```
user_id  video_id  full_is_click  full_is_like  full_long_view  full_play_time_s
0        225393        0.175          0.002          0.112            0.863
0        3749675       0.217          0.002          0.145            1.109
0        2868024       0.198          0.002          0.132            0.905
```

**1 epoch 测试集指标（参考）：**
```
is_click    AUC = 0.656
is_like     AUC = 0.752
is_comment  AUC = 0.827
long_view   AUC = 0.657
```

### 4.5 精排后 MTF 融合与截断

精排完成后，**MTF（Multi-Task Fusion）** 对 500 个候选的 7 维任务分进行融合，选出 top-60 送入重排。

#### MTF 模型结构

```
输入：(N, 7) — 每个候选的 7 个 MMoE 任务分

处理：
    ① 每个任务分 → Dense(32, relu) → 32维 token
       → 7个任务得到 7 个 token，shape=(N, 7, 32)
    ② Multi-Head Attention（4头）
       → 任务间自注意力，每个任务 token 感知其他6个任务的分数
    ③ Add & LayerNorm
    ④ 全局平均池化（沿任务维度）→ (N, 32)
    ⑤ Dense(1) → 最终融合分 mtf_score

输出：(N, 1) — 每个候选的融合排序分
```

**MTF 训练：**
- 损失：**ListMLE**（最大化正确排序的对数似然）
- **Soft label**：精排 500 个候选的"全1等权综合分"排序  
  Soft label 的含义：不用 0/1 硬标签（那需要知道"正确答案"），而是用精排自己的综合分作为排序参考，让 MTF 学习"在精排分已经有序的前提下，如何通过任务间交叉进一步优化排序"。
- 目标：让 MTF 学出的融合权重优于简单等权加和

**精排→重排截断分（final_score）：**
```
final_score = 0.5 × es_score(归一化) + 0.5 × mtf_score(归一化)

其中：
  es_score  = 7个任务分全1等权加和（等权综合分），归一化到 [0,1]
  mtf_score = MTF Transformer 融合分，归一化到 [0,1]

每用户按 final_score 降序 → 截取 top-60
```

这样设计的原因：mtf_score 是学习出来的，1 epoch 效果有限；es_score 是简单但稳定的基线。两者各占 0.5 做折中，防止 MTF 欠拟合时排序退化。

> 精排 500 → MTF → **top-60**（漏斗压缩比约 8:1）

---

## 5. 重排阶段（PRM 个性化重排）

### 5.1 作用

MTF 输出的 top-60 是**单点打分**——每个 item 独立计算得分，互相看不见。
但最终展示给用户的是一个**列表**，列表中 item 的组合质量（相关性、多样性）同样重要。

**PRM（Personalized Re-ranking Model，SIGIR 2019）** 用 Transformer 让每个候选 item
能感知整个列表中其余 59 个 item 的存在，从而计算出考虑列表上下文的重排分。

### 5.2 模型结构

```
输入：(B, 60, 7) — B个用户，每用户60个候选，每候选7维特征
     （7维 = full_is_click / full_is_like / full_is_follow /
             full_is_comment / full_is_forward / full_long_view / full_play_time_s）

处理：
① 特征投影 + 个性化向量（Personalized Vector）
    item_features → Linear(7→64, relu) → x (B, 60, 64)
    user_vec = mean_pool(item_features) → Linear(64, tanh) → pv (B, 60, 64)
    x = x + pv      ← 个性化注入

② Transformer Encoder × 2 层（item 间列表交互）
    Layer 1:
        Multi-Head Attention（4头）
        → item_i 能看到其他59个候选（无 mask，全可见）
        → Add & LayerNorm
        FFN: Dense(128, relu) → Dense(64)
        → Add & LayerNorm
    Layer 2（同上）

③ 输出层
    Dense(64→1) → squeeze → prm_score (B, 60)

输出：(B, 60) — 每个候选的重排分
```

### 5.3 训练目标

- 损失：**ListMLE（List Maximum Likelihood Estimation）**
- **训练标签**：真实行为标签加权组合 `1.0×is_click + 1.5×long_view + 2.0×is_like`
  - 有真实行为记录的候选（出现在测试集中）：用真实标签计算监督分
  - 无行为记录的候选（测试集中未曝光）：标签为 0（视为负样本）
- 目标：用 Transformer 对列表上下文建模，使排序结果直接向真实用户行为对齐

**ListMLE loss 公式解释：**
```python
# 直觉：label 分高的候选应该排在前面，这种排列的概率应该最大化
# loss = -log P(正确排序) = 负对数似然

# Step 1：按真实行为标签降序对预测分排列
idx = argsort(label, descending=True)
sorted_scores = gather(prm_scores, idx)       # 第1名的预测分, 第2名的...

# Step 2：计算每个位置的 log-softmax（分母是从该位置往后所有候选的 exp 分之和）
# 位置 i 的概率 = exp(score_i) / sum(exp(score_j), j>=i)
cumsum_exp = reverse(cumsum(reverse(exp(sorted_scores))))

# Step 3：所有位置对数概率之和取负 = ListMLE loss
loss = -mean( sum(sorted_scores - log(cumsum_exp + ε)) )
```

### 5.4 输入输出

| 项目 | 说明 |
|------|------|
| **输入** | shape=(B, 60, 7)，B=用户数，每用户60个候选的7维特征（MTF top-60 的精排分） |
| **输出** | shape=(B, 60)，每个候选的重排分 `prm_score` |
| **截断** | 每用户取 prm_score 最高的 **top-8** 作为最终曝光 |

**最终曝光输出样例：**
```
user_id  video_id  prm_score  expose_rank
0        1777686    -0.137        1
0        4282121    -0.214        2
0        265081     -0.214        3
0        913005     -0.218        4
...      ...         ...          ...
0        XXXXXXX    -0.351        8
```

> 重排候选 **60** → PRM → **top-8 曝光**（漏斗压缩比约 7.5:1）

---

## 6. 完整漏斗总览

```
阶段        候选数      筛选方式                        压缩比
─────────────────────────────────────────────────────────────
候选集       ~7,097     用户历史交互过的所有视频          —
    ↓
粗排 TwoTower   500     7任务分等权综合分 top-500        14:1
    ↓
精排 MMoE       500     全量打分（不截断，全部精排）       1:1
    ↓
MTF 融合         60     0.5×es_score + 0.5×mtf_score → top-60  8:1
    ↓
重排 PRM          8     列表级上下文交互→top-8            7.5:1
    ↓
最终曝光          8     透出给用户                        —
─────────────────────────────────────────────────────────────
总压缩比：~7,097 → 8（约 887:1）
```

---

## 7. 输出文件清单

| 文件路径 | 内容 |
|----------|------|
| `models/prerank_1k.weights.h5` | TwoTower 粗排模型权重 |
| `models/fullrank_1k.weights.h5` | MMoE 精排模型权重 |
| `models/mtf_1k.weights.h5` | MTF 多任务融合模型权重 |
| `models/prm_1k.weights.h5` | PRM 重排模型权重 |
| `results/fullrank_predict_1k.pkl` | 精排 500 候选的7任务打分 |
| `results/expose_1k_top8.csv` | **最终曝光列表**（每用户top-8） |

---

## 8. 运行命令

```bash
conda activate xmtf
cd reference/xMTFwww2025/xMTF

# 演示模式（10万条数据，1 epoch，约 2 分钟）
python pipeline_demo.py --data_path ../KuaiRand-1K --nrows 100000

# 全量训练（700万条数据，建议多 epoch）
python pipeline_demo.py --data_path ../KuaiRand-1K
```

---

## 9. 离线评估：AUC 指标

### 9.1 为什么用 AUC

推荐排序任务的目标是"排序质量"，而不是"预测准确率"。  
**AUC（Area Under ROC Curve）** 衡量的是：模型给正样本打的分高于负样本的概率。

- AUC = 0.5：排序完全随机，等于乱猜
- AUC = 1.0：正样本分数全部高于负样本，完美排序
- 工业界典型水平：粗排 0.60\~0.70，精排 0.70\~0.80

### 9.2 各阶段评估方案

| 阶段 | Ground Truth 标签 | 打分列 | 评估含义 |
|------|-------------------|--------|----------|
| 粗排（TwoTower） | `is_click`（主）, `is_like`, `long_view` | `pre_is_click`, `pre_is_like`, `pre_long_view` | 双塔召回质量——粗排分高的视频，真实点击率是否也更高 |
| 精排（MMoE） | 同上 | `full_is_click`, `full_is_like`, `full_long_view` | 精排打分精度 |
| MTF 融合分 | `is_click` | `final_score`（0.5×es+0.5×mtf） | 融合分是否优于单任务分 |
| 重排（PRM） | `is_click` | `prm_score` | 重排后 top-8 曝光的点击区分度 |

### 9.3 AUC 计算逻辑

```python
from sklearn.metrics import roc_auc_score

# 以精排 is_click AUC 为例：
# test_df 是测试集（含真实 is_click 标签）
# fullrank_df 是精排输出（含预测分 full_is_click）

eval_df = test_df.merge(fullrank_df, on=['user_id','video_id'], how='inner')
auc = roc_auc_score(eval_df['is_click'], eval_df['full_is_click'])
```

**注意事项：**
1. **Merge 方式是 inner join**：只评估"既出现在测试集中、又进入了该阶段候选"的样本。这会导致越靠后阶段的样本数越少（比如重排只有 top-8，样本量最小）
2. **正负样本不平衡**：KuaiRand 的点击率约 50%，相对均衡；但 is_follow（关注）的正样本极少（<1%），AUC 会不稳定
3. **AUC 要求正负样本都存在**：若某一阶段筛完后全是正样本或全是负样本，AUC 无法计算

### 9.4 推荐关注的核心指标

| 优先级 | 指标 | 原因 |
|--------|------|------|
| ⭐⭐⭐ | 精排 `is_click` AUC | 点击是最直接的用户意图信号，样本量大、稳定 |
| ⭐⭐⭐ | 精排 `long_view` AUC | 长观看是视频推荐最重要的深度行为指标 |
| ⭐⭐ | 精排 `is_like` AUC | 赞的质量信号强，但正样本少 |
| ⭐⭐ | MTF `final_score` vs `is_click` AUC | 验证 MTF 融合是否优于 es_score 基线 |
| ⭐ | 重排 `prm_score` vs `is_click` AUC | 重排样本量最小（top-8），仅供参考 |

### 9.5 pipeline_demo.py 中的自动评估

运行 `pipeline_demo.py` 后，脚本会在最后自动输出各阶段 AUC：

```
  ┌─ 粗排（TwoTower）AUC ─────────────────────────────────────────┐
  │  is_click    AUC = 0.6531
  │  is_like     AUC = 0.7041
  │  long_view   AUC = 0.6515
  └──────────────────────────────────────────────────────────────┘
  ┌─ 精排（MMoE）AUC ──────────────────────────────────────────────┐
  │  is_click    AUC = 0.6562
  │  is_like     AUC = 0.7523
  │  long_view   AUC = 0.6570
  │
  │  is_click (0.5×es+0.5×mtf 融合分) AUC = 0.6601
  └──────────────────────────────────────────────────────────────┘
  ┌─ 重排 top-8 AUC（PRM 重排分 vs 真实点击）────────────────────┐
  │  is_click    AUC = 0.xxxx  （样本数=240）
  └──────────────────────────────────────────────────────────────┘

  📌 AUC 解读参考：
     < 0.55  较差，排序接近随机
    0.55~0.65  一般，有一定区分度
    0.65~0.75  良好，工业界粗排典型水平（1 epoch 演示）
     > 0.75   优秀，精排/重排目标水平
  （注：本演示仅 1 epoch，正式训练需 10+ epoch，AUC 会显著提升）
```
