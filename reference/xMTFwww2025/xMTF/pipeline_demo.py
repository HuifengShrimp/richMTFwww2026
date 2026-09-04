"""
pipeline_demo.py — 完整推荐系统流水线演示
架构：候选集 → 粗排(TwoTower) → 精排(MMoE) → MTF融合 → 重排(PRM) → top-8 曝光

各阶段规模：
  候选集   : 用户历史交互过的所有视频（平均 ~7,097 个/用户）
  粗排输出 : top-500（TwoTower 打分后取综合分最高的 500 个）
  精排输出 : 500 个候选全部打分（MMoE 多任务）
  MTF 融合 : 对 500 个 MMoE 分数进行多任务融合 → 1 个最终分 → top-60
  重排输出 : PRM Transformer 对 60 个候选建模列表交互 → top-8 曝光

用法：
    conda activate xmtf
    cd reference/xMTFwww2025/xMTF
    python pipeline_demo.py --data_path ../KuaiRand-1K --nrows 100000
"""

import os, gc, sys, time, argparse, warnings
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error

from deepctr.feature_column import SparseFeat
from mmoe import MMOE
from two_tower import TwoTower

# ───────────────── 常量 ─────────────────────────────────────
PRERANK_TOPK  = 500   # 粗排 → 精排 候选数
RERANK_TOPK   = 60    # 精排 → 重排 候选数
EXPOSE_NUM    = 8     # 重排 → 曝光 最终数

# MMoE/TwoTower 的 7 个任务
LABELS      = ['is_click','is_like','is_follow','is_comment','is_forward','long_view','play_time_label']
TASK_TYPES  = ['binary','binary','binary','binary','binary','binary','regression']

# 精排综合分权重（用于粗排→精排筛选、以及无 MTF 时的基线）
SCORE_WEIGHTS = {'is_click':1.0,'is_like':1.0,'is_follow':1.0,
                 'is_comment':1.0,'is_forward':1.0,'long_view':1.0,'play_time_s':1.0}

DIVIDER = "\n" + "="*72 + "\n"
def banner(stage, title):
    print(DIVIDER + f"  STAGE {stage}: {title}\n" + "="*72)

def show_df(name, df, n=3):
    print(f"\n📋 {name}  shape={df.shape}")
    print(df.head(n).to_string())

def huber_loss(y_true, y_pred, delta=0.001):
    err = y_true - y_pred
    small = tf.abs(err) <= delta
    return tf.where(small, tf.square(err)/2, delta*(tf.abs(err)-0.5*delta))

def fillna_df(df, features):
    for feat in [x for x in features if x not in ['user_id','video_id']]:
        df[feat] = LabelEncoder().fit_transform(df[feat].astype(str).fillna('-1'))

def compute_rank_score(df, prefix=''):
    """综合精排分（用于粗排→精排筛选 & 无 MTF 基线对比）"""
    s = pd.Series(0.0, index=df.index)
    for col, w in SCORE_WEIGHTS.items():
        c = f"{prefix}{col}" if prefix else col
        if c in df.columns:
            s += df[c] * w
    return s


# ═══════════════════════════════════════════════════════════════
#   MTF（Multi-Task Fusion）模型
#   作用：接收 MMoE 输出的 7 维多任务分数，用 Transformer 建模
#         任务间依赖关系，输出 1 个融合后的最终分
# ═══════════════════════════════════════════════════════════════

class MTFusion(tf.keras.Model):
    """
    Multi-Task Fusion（MTF）
    
    输入  : (B, num_tasks=6) 的 MMoE 打分（去掉 play_time_s 连续值，只用6个离散任务分）
             + (B, 1) play_time_s 归一化后拼接 → (B, 7)
    处理  : 用小 Transformer Encoder 建模任务间相互依赖
             每个任务分视为一个 "token"
    输出  : (B, 1) 融合后的最终排序分
    
    为什么用 Transformer 而不是简单线性组合？
      不同任务分之间存在非线性关联（例如 like↑ 往往 comment↑），
      Transformer 的注意力机制能自适应地学习任务间的交叉权重。
    """
    def __init__(self, num_tasks=7, d_model=32, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.num_tasks = num_tasks
        # 每个任务分 → d_model 维 token
        self.token_proj = tf.keras.layers.Dense(d_model, activation='relu')
        # 自注意力：任务间交叉
        self.attn  = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.norm  = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # 最终输出分
        self.fc    = tf.keras.layers.Dense(1)

    def call(self, task_scores, training=False):
        """
        task_scores: (B, num_tasks)  MMoE 7个任务的输出分
        """
        # 每个任务分变成独立 token: (B, num_tasks, 1) → (B, num_tasks, d_model)
        x = tf.expand_dims(task_scores, axis=-1)              # (B, T, 1)
        x = self.token_proj(x)                                 # (B, T, d_model)
        # 任务间自注意力
        attn_out = self.attn(x, x, training=training)         # (B, T, d_model)
        x = self.norm(x + attn_out)                           # Add & Norm
        # 全局平均池化后输出分
        x = tf.reduce_mean(x, axis=1)                         # (B, d_model)
        return self.fc(x)                                      # (B, 1)


# ═══════════════════════════════════════════════════════════════
#   PRM（Personalized Re-ranking Model）
# ═══════════════════════════════════════════════════════════════

class PRMModel(tf.keras.Model):
    """
    PRM: Transformer Encoder 对候选列表 item 间上下文建模
    输入 : (B, N=60, feat_dim=7) 每个候选的 MMoE 打分
    输出 : (B, N)   每个候选的重排分
    """
    def __init__(self, feat_dim, d_model=64, num_heads=4, num_layers=2,
                 ffn_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers  = num_layers
        self.input_proj  = tf.keras.layers.Dense(d_model, activation='relu')
        self.pv_proj     = tf.keras.layers.Dense(d_model, activation='tanh')
        # 显式命名各层，避免 Sequential build 问题
        self.attn_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                               key_dim=d_model//num_heads,
                                               dropout=dropout,
                                               name=f'attn_{i}')
            for i in range(num_layers)
        ]
        self.attn_norms  = [tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'anorm_{i}')
                            for i in range(num_layers)]
        self.ffn_dense1  = [tf.keras.layers.Dense(ffn_dim, activation='relu', name=f'ffn1_{i}')
                            for i in range(num_layers)]
        self.ffn_dense2  = [tf.keras.layers.Dense(d_model, name=f'ffn2_{i}')
                            for i in range(num_layers)]
        self.ffn_norms   = [tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'fnorm_{i}')
                            for i in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, item_features, training=False):
        N = tf.shape(item_features)[1]
        user_vec = tf.tile(tf.reduce_mean(item_features, axis=1, keepdims=True), [1, N, 1])
        pv = self.pv_proj(user_vec)
        x  = self.input_proj(item_features) + pv
        for i in range(self.num_layers):
            attn_out = self.attn_layers[i](x, x, training=training)
            x = self.attn_norms[i](x + attn_out)
            ffn_out = self.ffn_dense2[i](self.ffn_dense1[i](x))
            x = self.ffn_norms[i](x + ffn_out)
        return tf.squeeze(self.output_layer(x), axis=-1)


# ═══════════════════════════════════════════════════════════════
#   主流程
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    type=str,  default="../KuaiRand-1K")
    parser.add_argument("--split_date",   type=str,  default="2022-05-05")
    parser.add_argument("--nrows",        type=int,  default=100000)
    parser.add_argument("--embedding_dim",type=int,  default=8)
    args = parser.parse_args()

    data_type = args.data_path.split('-')[-1].lower()
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # STAGE 0  加载数据
    # ──────────────────────────────────────────────────────────
    banner(0, "加载原始数据")
    t0 = time.time()

    df1  = pd.read_pickle(f"{args.data_path}/data/log_standard_4_08_to_4_21_{data_type}.pkl")
    df2  = pd.read_pickle(f"{args.data_path}/data/log_standard_4_22_to_5_08_{data_type}.pkl")
    uf   = pd.read_pickle(f"{args.data_path}/data/user_features_{data_type}.pkl")
    vfb  = pd.read_pickle(f"{args.data_path}/data/video_features_basic_{data_type}.pkl")
    vfs  = pd.read_pickle(f"{args.data_path}/data/video_features_statistic_{data_type}.pkl")

    data = pd.concat([df1, df2], ignore_index=True)
    data = data[(data['tab']==1) & (data['play_time_ms']>0) & (data['duration_ms']>0)]
    data['play_rate'] = data['play_time_ms'] / data['duration_ms']
    data = data[data['play_rate'] <= 5]
    data['duration_s']      = data['duration_ms'] / 1000
    data['play_time_s']     = data['play_time_ms'] / 1000
    data['play_time_label'] = np.log(data['play_time_s'] + 1) / 7
    uf.drop(['follow_user_num'], axis=1, inplace=True)
    vfb = vfb[vfb['video_duration'].notnull()]

    # 全量统计
    total_users  = data['user_id'].nunique()
    total_videos = data['video_id'].nunique()
    avg_cand     = data.groupby('user_id')['video_id'].nunique().mean()

    print(f"""
┌─────────────────────────────────────────────────────────────┐
│  KuaiRand-1K 全量数据统计                                    │
│                                                              │
│  用户数              : {total_users:>8,}                        │
│  视频数（全量候选集）: {total_videos:>8,}                      │
│  交互总条数          : {len(data):>8,}                         │
│  日期范围            : 2022-04-08 ~ 2022-05-08               │
│  每用户平均候选数    : {avg_cand:>8.0f}  ← 候选集大小         │
│                                                              │
│  流水线规模                                                  │
│  候选集  →  粗排(TwoTower) top-{PRERANK_TOPK}                    │
│  粗排    →  精排(MMoE)     {PRERANK_TOPK} 个全打分                │
│  精排    →  MTF融合        {PRERANK_TOPK} → top-{RERANK_TOPK}              │
│  重排    →  透出(PRM)      {RERANK_TOPK}  → top-{EXPOSE_NUM}               │
└─────────────────────────────────────────────────────────────┘""")

    # 取 nrows 条演示
    data = data.iloc[:args.nrows].copy()
    del df1, df2; gc.collect()

    # ──────────────────────────────────────────────────────────
    # STAGE 1  特征预处理 & 训练/测试集划分
    # ──────────────────────────────────────────────────────────
    banner(1, "特征预处理 & 训练/测试集划分")

    for col in ["video_duration"]:
        vfb[col] = pd.qcut(vfb[col], q=16, labels=False, duplicates='drop')
    stat_dense = [c for c in vfs.columns if c != 'video_id']
    for col in stat_dense:
        vfs[col] = pd.qcut(vfs[col], q=16, labels=False, duplicates='drop')

    user_sp  = (["user_id","user_active_degree","is_lowactive_period","is_live_streamer",
                 "is_video_author","follow_user_num_range","fans_user_num_range",
                 "friend_user_num_range","register_days_range"]
                + [f"onehot_feat{i}" for i in range(1,18)])
    item_sp_basic = ["video_id","author_id","video_type","upload_type","visible_status",
                     "music_id","music_type","video_duration","server_width","server_height"]
    item_sp_stat  = stat_dense
    item_sp = item_sp_basic + item_sp_stat
    sparse_features = user_sp + item_sp

    fillna_df(uf, user_sp)
    fillna_df(vfb, item_sp_basic)
    fillna_df(vfs, item_sp_stat)

    data = (data.merge(uf,  on='user_id', how='left')
                .merge(vfb, on='video_id', how='left')
                .merge(vfs, on='video_id', how='left'))
    data[sparse_features] = data[sparse_features].fillna(0).astype(int)

    # 自动计算分割日期
    dmin, dmax = data['date'].min(), data['date'].max()
    user_split = pd.Timestamp(args.split_date)
    split_date = user_split if dmin < user_split < dmax else (dmin + (dmax-dmin)*0.8)
    split_str  = str(split_date)[:10]

    train_df = data[data['date'] <= split_date].reset_index(drop=True)
    test_df  = data[data['date'] >  split_date].reset_index(drop=True)

    user_dnn_cols = [SparseFeat(f, data[f].max()+1, embedding_dim=args.embedding_dim) for f in user_sp]
    item_dnn_cols = [SparseFeat(f, data[f].max()+1, embedding_dim=args.embedding_dim) for f in item_sp]
    dnn_cols      = user_dnn_cols + item_dnn_cols

    train_input = {n: train_df[n].values for n in sparse_features}
    test_input  = {n: test_df[n].values  for n in sparse_features}

    print(f"""
  输入  : 原始日志 + 用户特征(30列) + 视频基础特征(12列) + 视频统计特征(62列)
  处理  : ① 稀疏特征 LabelEncoder 编码
          ② 连续特征 qcut 16分桶 → 离散化
          ③ 三张特征表 left join 合并到日志
          ④ 按 split_date={split_str} 划分

  输出:
    特征维度 : {len(user_sp)} 用户侧 + {len(item_sp)} 视频侧 = {len(sparse_features)} 维稀疏特征
    训练集   : {len(train_df):,} 条  {str(train_df['date'].min())[:10]} ~ {str(train_df['date'].max())[:10]}
    测试集   : {len(test_df):,} 条  {str(test_df['date'].min())[:10]} ~ {str(test_df['date'].max())[:10]}
""")

    # ──────────────────────────────────────────────────────────
    # STAGE 2  TwoTower 双塔召回（粗排）
    # ──────────────────────────────────────────────────────────
    banner(2, f"TwoTower 双塔召回（粗排）— 候选集 → top-{PRERANK_TOPK}")

    print(f"""
  【粗排的作用】
    候选集平均有 {avg_cand:.0f} 个视频，对全量做精排太贵。
    双塔模型把用户/视频分别编码成向量，点积度量相关性，速度快、可 ANN 检索。
    从候选集里快速筛出最相关的 top-{PRERANK_TOPK} 送入精排。

  模型结构:
    用户侧: {len(user_sp)}个稀疏特征 → embedding concat → DNN(256,128) → task_tower(64) → user_vec
    视频侧: {len(item_sp)}个稀疏特征 → embedding concat → DNN(256,128) → task_tower(64) → item_vec
    打分  : user_vec ⊙ item_vec（逐元素乘积）→ Dense(1) → sigmoid → pCTR/pLike/...
    共 7 个任务头（6 分类 + 1 回归）

  输入  : train_input  每个特征 shape=({len(train_df)},)
  标签  : {LABELS}
  损失  : 6×binary_crossentropy + 1×huber_loss，权重 1:1:...:100
""")

    t2 = time.time()
    model_prerank = TwoTower(
        user_dnn_feature_columns=user_dnn_cols,
        item_dnn_feature_columns=item_dnn_cols,
        bottom_dnn_hidden_units=(256,128), tower_dnn_hidden_units=(64,),
        task_types=TASK_TYPES, task_names=LABELS,
    )
    model_prerank.compile(
        optimizer="adam",
        loss=["binary_crossentropy"]*6 + [huber_loss],
        loss_weights=[1.0]*6 + [100.0],
    )
    print("  训练 TwoTower（1 epoch）...")
    model_prerank.fit(x=train_input, y=[train_df[l].values for l in LABELS],
                      batch_size=512, epochs=1, validation_split=0.1, verbose=1)
    model_prerank.save_weights(f"./models/prerank_{data_type}.weights.h5")

    # 对测试集打分（模拟对候选集打分）
    pred_pre = model_prerank.predict(test_input, batch_size=4096, verbose=0)
    prerank_df = test_df[['user_id','video_id']].copy()
    for i, l in enumerate(LABELS):
        prerank_df[f'pre_{l}'] = pred_pre[i]
    prerank_df['pre_play_time_s'] = np.exp(prerank_df['pre_play_time_label'] * 7) - 1

    # 粗排 → top-500：每用户取综合分最高的 PRERANK_TOPK 个
    prerank_df['_prerank_score'] = compute_rank_score(prerank_df, prefix='pre_')
    prerank_top500 = (prerank_df.groupby('user_id', group_keys=False)
                                .apply(lambda g: g.nlargest(PRERANK_TOPK, '_prerank_score'))
                                .reset_index(drop=True))

    print(f"""
  ✅ TwoTower 训练完成，耗时 {time.time()-t2:.1f}s

  输出（粗排结果）:
    models/prerank_{data_type}.weights.h5  ← 模型权重""")

    show_df(f"粗排 top-{PRERANK_TOPK} 候选（prerank_top500，每用户 500 行）",
            prerank_top500[['user_id','video_id','pre_is_click','pre_is_like','pre_long_view','pre_play_time_s','_prerank_score']])
    print(f"""
    总行数 : {len(prerank_top500):,}（{prerank_top500['user_id'].nunique()} 用户 × ~{PRERANK_TOPK} 候选）
    列说明 : user_id, video_id, pre_is_click(粗排CTR), pre_is_like, pre_long_view,
             pre_play_time_s, _prerank_score(综合分，用于筛选)""")

    # ──────────────────────────────────────────────────────────
    # STAGE 3  MMoE 多任务精排
    # ──────────────────────────────────────────────────────────
    banner(3, f"MMoE 多任务精排 — top-{PRERANK_TOPK} 候选全部打分")

    print(f"""
  【精排的作用】
    粗排召回了 {PRERANK_TOPK} 个候选，精排要对它们精细建模。
    MMoE（Multi-gate Mixture-of-Experts）通过多个 Expert 共享底层表示，
    每个任务有独立 Gate 自适应混合 Expert 输出，比单塔更好地建模多任务关联。

  模型结构:
    输入    : {len(sparse_features)} 维稀疏特征（用户 + 视频合并）
    Expert  : 3 个 DNN(256,128) 共享所有任务
    Gate    : 每任务独立 softmax Gate → 加权求和 Expert 输出
    Tower   : 每任务 DNN(64) → Dense(1) → 任务输出

  输入  : 精排 {PRERANK_TOPK} 候选的特征（同粗排同样的特征集）
  输出  : 7 个任务分 → [is_click, is_like, is_follow, is_comment, is_forward, long_view, play_time_s]
""")

    t3 = time.time()
    model_fullrank = MMOE(
        dnn_feature_columns=dnn_cols, num_experts=3,
        expert_dnn_hidden_units=(256,128), tower_dnn_hidden_units=(64,),
        task_types=TASK_TYPES, task_names=LABELS,
    )
    model_fullrank.compile(
        optimizer="adam",
        loss=["binary_crossentropy"]*6 + [huber_loss],
        loss_weights=[1.0]*6 + [100.0],
    )
    print("  训练 MMoE（1 epoch）...")
    model_fullrank.fit(x=train_input, y=[train_df[l].values for l in LABELS],
                       batch_size=512, epochs=1, validation_split=0.1, verbose=1)
    model_fullrank.save_weights(f"./models/fullrank_{data_type}.weights.h5")

    # 对粗排 top-500 打精排分
    # 由于 test 数据中不是所有用户都有 500 条，用 test_df 中属于粗排 top-500 的子集
    top500_keys = prerank_top500.set_index(['user_id','video_id']).index
    fullrank_input_df = test_df[
        pd.MultiIndex.from_arrays([test_df['user_id'], test_df['video_id']]).isin(top500_keys)
    ].reset_index(drop=True)

    # 如果 fullrank_input_df 为空（demo 数据不足），用整个 test_df
    if len(fullrank_input_df) == 0:
        fullrank_input_df = test_df.reset_index(drop=True)

    full_model_input = {n: fullrank_input_df[n].values for n in sparse_features}
    pred_full = model_fullrank.predict(full_model_input, batch_size=4096, verbose=0)

    fullrank_df = fullrank_input_df[['user_id','video_id']].copy()
    for i, l in enumerate(LABELS):
        fullrank_df[f'full_{l}'] = pred_full[i]
    fullrank_df['full_play_time_s'] = np.exp(fullrank_df['full_play_time_label'] * 7) - 1
    fullrank_df.to_pickle(f"./results/fullrank_predict_{data_type}.pkl")

    # 精排指标
    print(f"\n  ✅ MMoE 训练完成，耗时 {time.time()-t3:.1f}s\n")
    print(f"  输出（精排结果）:")
    print(f"    models/fullrank_{data_type}.weights.h5      ← 模型权重")
    print(f"    results/fullrank_predict_{data_type}.pkl    ← {len(fullrank_df):,} 行精排打分")

    for i, label in enumerate(LABELS):
        if i <= 5 and test_df[label].sum() > 0:
            try:
                auc = roc_auc_score(fullrank_input_df[label], pred_full[i])
                print(f"    {label:<20} AUC = {auc:.4f}")
            except:
                pass
        elif i == 6:
            mse = mean_squared_error(fullrank_input_df[label], pred_full[i])
            print(f"    {'play_time_label':<20} MSE = {mse:.6f}")

    show_df(f"精排输出（fullrank_predict，精排 {PRERANK_TOPK} 候选的打分）",
            fullrank_df[['user_id','video_id','full_is_click','full_is_like','full_long_view','full_play_time_s']])

    # ──────────────────────────────────────────────────────────
    # STAGE 4  MTF 多任务融合（精排后，选 top-60 进重排）
    # ──────────────────────────────────────────────────────────
    banner(4, f"MTF 多任务融合 — {PRERANK_TOPK} → top-{RERANK_TOPK}")

    print(f"""
  【MTF 的作用】
    MMoE 输出了 7 个任务分（CTR/Like/Follow/Comment/Forward/LongView/PlayTime），
    如何将它们融合为 1 个最终排序分？
    简单加权（如 1×CTR + 2×Like + ...）是静态的，无法捕捉任务间非线性交叉。
    MTF 用小 Transformer（把每个任务分视为 1 个 token）建模任务间依赖，
    学出自适应的任务融合权重，给出更优的最终分。

  模型结构:
    输入  : (N, 7) MMoE 的 7 个任务分
    处理  : ① 每个任务分 → Dense(32) → 7 个 token
            ② 多头自注意力（4头）→ 任务间依赖
            ③ Add&Norm → 全局平均池化 → Dense(1)
    输出  : (N, 1) 最终融合分

  训练方式: ListMLE loss，用精排综合分（静态加权分）作为 soft label 监督融合分的相对顺序
""")

    # 准备 MTF 输入：6 个二分类任务分 + play_time_s 归一化
    task_score_cols = ['full_is_click','full_is_like','full_is_follow',
                       'full_is_comment','full_is_forward','full_long_view','full_play_time_s']
    task_scores_np = fullrank_df[task_score_cols].values.astype(np.float32)

    # 归一化 play_time_s
    pt_max = task_scores_np[:, -1].max() + 1e-9
    task_scores_np[:, -1] /= pt_max

    # 计算精排综合分作为 soft label（用于 ListMLE 监督）
    fullrank_df['_rank_score'] = compute_rank_score(fullrank_df, prefix='full_')

    t4 = time.time()
    mtf_model = MTFusion(num_tasks=7, d_model=32, num_heads=4)
    _ = mtf_model(tf.zeros((1, 7)))  # build

    # ListMLE loss 训练
    optimizer = tf.keras.optimizers.Adam(1e-3)
    uid_list  = fullrank_df['user_id'].values
    score_np  = task_scores_np
    label_np  = fullrank_df['_rank_score'].values.astype(np.float32)

    unique_uids = fullrank_df['user_id'].unique()
    print(f"  MTF 训练（1 epoch，{len(unique_uids)} 个用户）...")
    total_loss = 0.0
    for uid in unique_uids:
        mask = uid_list == uid
        x_u  = tf.constant(score_np[mask])        # (n, 7)
        y_u  = tf.constant(label_np[mask])         # (n,)

        with tf.GradientTape() as tape:
            pred = tf.squeeze(mtf_model(x_u), -1)  # (n,)
            # ListMLE loss
            idx  = tf.argsort(y_u, direction='DESCENDING')
            sp   = tf.gather(pred, idx)
            cumexp = tf.reverse(tf.cumsum(tf.reverse(tf.exp(sp),[0])),[0])
            loss = -tf.reduce_sum(sp - tf.math.log(cumexp + 1e-9))
        grads = tape.gradient(loss, mtf_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, mtf_model.trainable_variables))
        total_loss += loss.numpy()

    print(f"  ListMLE loss = {total_loss/len(unique_uids):.4f}")
    mtf_model.save_weights(f"./models/mtf_{data_type}.weights.h5")

    # MTF 打分并选 top-60
    # 精排→重排截断分 = 0.5 × es_score（等权综合分）+ 0.5 × mtf_score
    # es_score 已经在上面计算（fullrank_df['_rank_score']）；先归一化到同一量纲再融合
    es_score_np  = fullrank_df['_rank_score'].values.astype(np.float32)
    es_min, es_max = es_score_np.min(), es_score_np.max()
    es_norm = (es_score_np - es_min) / (es_max - es_min + 1e-9)   # 归一化到 [0,1]

    mtf_scores = tf.squeeze(mtf_model(tf.constant(score_np)), -1).numpy()
    mtf_min, mtf_max = mtf_scores.min(), mtf_scores.max()
    mtf_norm = (mtf_scores - mtf_min) / (mtf_max - mtf_min + 1e-9)  # 归一化到 [0,1]

    fullrank_df['mtf_score']    = mtf_scores
    fullrank_df['final_score']  = 0.5 * es_norm + 0.5 * mtf_norm   # 融合分
    mtf_top60 = (fullrank_df.groupby('user_id', group_keys=False)
                             .apply(lambda g: g.nlargest(RERANK_TOPK, 'final_score'))
                             .reset_index(drop=True))

    print(f"""
  ✅ MTF 融合完成，耗时 {time.time()-t4:.1f}s

  输出:
    models/mtf_{data_type}.weights.h5  ← MTF 模型权重""")
    show_df(f"MTF 输出 top-{RERANK_TOPK}（每用户 60 行，送入 PRM 重排）",
            mtf_top60[['user_id','video_id','full_is_click','full_is_like','full_long_view','mtf_score','final_score']])
    print(f"""
    总行数    : {len(mtf_top60):,}（{mtf_top60['user_id'].nunique()} 用户 × ~{RERANK_TOPK} 候选）
    mtf_score : MTF Transformer 融合 7 个任务分后的排序分
    final_score: 0.5×es_score(归一化) + 0.5×mtf_score(归一化)  ← 实际用于截断 top-60 的分数""")

    # ──────────────────────────────────────────────────────────
    # STAGE 5  PRM 重排（top-60 → top-8 曝光）
    # ──────────────────────────────────────────────────────────
    banner(5, f"PRM 重排（Personalized Re-ranking）— top-{RERANK_TOPK} → top-{EXPOSE_NUM} 曝光")

    print(f"""
  【PRM 重排的作用】
    MTF 选出了 {RERANK_TOPK} 个候选，但这些候选是"单点打分"——每个 item 独立打分，
    没有考虑"如果这 {RERANK_TOPK} 个 item 放在一起展示，哪些组合能最大化用户体验"。
    例如：10 个相似的 item 排前10没有多样性；互补 item 组合效果更好。
    
    PRM 用 Transformer 对整个候选列表建模 item 间的上下文依赖：
    item_i 的最终得分受其他 {RERANK_TOPK-1} 个 item 影响（注意力感知上下文）。

  模型结构:
    输入  : (B, {RERANK_TOPK}, 7) 每个候选的 [is_click, is_like, ..., play_time_s]
    处理  : ① Linear(7→64) + Personalized Vector（用户偏好向量，用均值池化模拟）
            ② Transformer Encoder × 2（多头自注意力建模 item 间上下文）
            ③ Dense(64→1) → 重排分
    输出  : (B, {RERANK_TOPK}) 每候选的重排分 → 取 top-{EXPOSE_NUM} 作为曝光
""")

    # 为每个用户构建 (候选数, 7) 的特征矩阵
    feat_cols = ['full_is_click','full_is_like','full_is_follow',
                 'full_is_comment','full_is_forward','full_long_view','full_play_time_s']
    uid_arr, feat_arr, vid_arr = [], [], []
    for uid, grp in mtf_top60.groupby('user_id'):
        n = len(grp)
        if n < RERANK_TOPK:
            # pad
            pad = pd.DataFrame(np.zeros((RERANK_TOPK-n, len(feat_cols))), columns=feat_cols)
            grp_feat = pd.concat([grp[feat_cols], pad]).values.astype(np.float32)
            vid_row  = np.concatenate([grp['video_id'].values, np.zeros(RERANK_TOPK-n, dtype=int)])
        else:
            grp_feat = grp[feat_cols].values[:RERANK_TOPK].astype(np.float32)
            vid_row  = grp['video_id'].values[:RERANK_TOPK]
        uid_arr.append(uid)
        feat_arr.append(grp_feat)
        vid_arr.append(vid_row)

    feat_arr = np.stack(feat_arr)  # (n_users, 60, 7)
    vid_arr  = np.array(vid_arr)   # (n_users, 60)
    uid_arr  = np.array(uid_arr)   # (n_users,)

    # 归一化
    flat = feat_arr.reshape(-1, feat_arr.shape[-1])
    fmin = flat.min(axis=0, keepdims=True)
    fmax = flat.max(axis=0, keepdims=True)
    feat_norm = ((flat - fmin) / (fmax - fmin + 1e-9)).reshape(feat_arr.shape).astype(np.float32)

    # PRM 训练（1 epoch，ListMLE loss，直接用真实行为标签监督）
    t5 = time.time()
    feat_dim = feat_norm.shape[-1]
    prm_model = PRMModel(feat_dim=feat_dim, d_model=64, num_heads=4, num_layers=2)
    _ = prm_model(tf.zeros((1, RERANK_TOPK, feat_dim)))  # build

    # 真实行为标签：用 is_click + 1.5×long_view + 2×is_like 作为每个候选的监督分
    # 从 test_df 中查找每个候选在测试集中的真实行为标签，找不到则为 0（未曝光/负样本）
    gt_label_cols = ['is_click', 'long_view', 'is_like']
    gt_weights    = [1.0, 1.5, 2.0]
    gt_lookup = test_df.set_index(['user_id', 'video_id'])[gt_label_cols]

    label_mat = np.zeros((len(uid_arr), RERANK_TOPK), dtype=np.float32)
    for j, uid in enumerate(uid_arr):
        for k, vid in enumerate(vid_arr[j]):
            key = (int(uid), int(vid))
            if key in gt_lookup.index:
                row = gt_lookup.loc[key]
                label_mat[j, k] = sum(w * row[c] for c, w in zip(gt_label_cols, gt_weights))

    prm_opt = tf.keras.optimizers.Adam(1e-3)
    feat_t  = tf.constant(feat_norm)
    label_t = tf.constant(label_mat)
    print(f"  PRM 训练（1 epoch，{len(uid_arr)} 用户，batch_size=32）...")
    batch_size = min(32, len(uid_arr))
    n_batches  = max(1, len(uid_arr)//batch_size)
    ep_loss    = 0.0
    for b in range(n_batches):
        xb = feat_t[b*batch_size:(b+1)*batch_size]
        yb = label_t[b*batch_size:(b+1)*batch_size]
        with tf.GradientTape() as tape:
            scores = prm_model(xb, training=True)
            idx = tf.argsort(yb, axis=-1, direction='DESCENDING')
            ss  = tf.gather(scores, idx, batch_dims=1)
            cum = tf.reverse(tf.cumsum(tf.reverse(tf.exp(ss),[-1]),axis=-1),[-1])
            loss = -tf.reduce_mean(tf.reduce_sum(ss - tf.math.log(cum+1e-9), axis=-1))
        grads = tape.gradient(loss, prm_model.trainable_variables)
        prm_opt.apply_gradients(zip(grads, prm_model.trainable_variables))
        ep_loss += loss.numpy()
    print(f"  ListMLE loss = {ep_loss/n_batches:.4f}")
    prm_model.save_weights(f"./models/prm_{data_type}.weights.h5")

    # 推理：取 top-8 曝光
    all_scores = prm_model(feat_t, training=False).numpy()  # (n_users, 60)
    rows = []
    for j, uid in enumerate(uid_arr):
        user_scores = all_scores[j]
        top_idx = np.argsort(user_scores)[::-1][:EXPOSE_NUM]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({'user_id': int(uid), 'video_id': int(vid_arr[j][idx]),
                         'prm_score': float(user_scores[idx]), 'expose_rank': rank})
    expose_df = pd.DataFrame(rows)
    expose_df.to_csv(f"./results/expose_{data_type}_top{EXPOSE_NUM}.csv", index=False)

    print(f"""
  ✅ PRM 重排完成，耗时 {time.time()-t5:.1f}s

  输出:
    models/prm_{data_type}.weights.h5                  ← PRM 模型权重
    results/expose_{data_type}_top{EXPOSE_NUM}.csv     ← 最终曝光列表""")
    show_df(f"最终曝光列表（每用户 {EXPOSE_NUM} 行）", expose_df)

    # ──────────────────────────────────────────────────────────
    # 全流程汇总
    # ──────────────────────────────────────────────────────────
    print(DIVIDER + "  全流程汇总\n" + "="*72)
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  小型推荐系统流水线 — 各阶段输入/输出总览                       ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                  ║
  ║  候选集（原始交互）                                              ║
  ║    大小 : 全量约 {avg_cand:.0f} 个视频/用户（全量 {total_videos:,} 视频）     ║
  ║    来源 : 用户历史有过交互的视频                                 ║
  ║                                   ↓                              ║
  ║  Stage 2  TwoTower 双塔召回（粗排）                              ║
  ║    输入 : 候选集 (~{avg_cand:.0f} 视频) 的用户+视频特征            ║
  ║    输出 : top-{PRERANK_TOPK} 候选，7个任务分(CTR/Like/...）              ║
  ║    文件 : prerank_{data_type}.weights.h5                       ║
  ║                                   ↓ top-{PRERANK_TOPK}                    ║
  ║  Stage 3  MMoE 多任务精排                                       ║
  ║    输入 : {PRERANK_TOPK} 个候选的特征                                   ║
  ║    输出 : {PRERANK_TOPK} 个候选各7个任务分（is_click/like/...）           ║
  ║    文件 : fullrank_{data_type}.weights.h5                      ║
  ║           fullrank_predict_{data_type}.pkl                     ║
  ║                                   ↓ {PRERANK_TOPK} 候选                   ║
  ║  Stage 4  MTF 多任务融合（本文核心）                            ║
  ║    输入 : {PRERANK_TOPK} 候选 × 7个MMoE任务分                            ║
  ║    处理 : Transformer 建模任务间依赖 → 1个最终融合分             ║
  ║    输出 : {PRERANK_TOPK} 候选的 mtf_score → 选 top-{RERANK_TOPK}                 ║
  ║    文件 : mtf_{data_type}.weights.h5                           ║
  ║                                   ↓ top-{RERANK_TOPK}                     ║
  ║  Stage 5  PRM 重排                                              ║
  ║    输入 : {RERANK_TOPK} 个候选 × 7维特征，shape=({expose_df['user_id'].nunique()}, {RERANK_TOPK}, 7)          ║
  ║    处理 : Transformer 建模 item 间列表上下文交互                 ║
  ║    输出 : {RERANK_TOPK} → top-{EXPOSE_NUM} 曝光，共 {len(expose_df)} 行                  ║
  ║    文件 : expose_{data_type}_top{EXPOSE_NUM}.csv                        ║
  ║                                                                  ║
  ║  总耗时: {time.time()-t0:.0f}s（演示数据 {args.nrows:,} 条，1 epoch）             ║
  ╚══════════════════════════════════════════════════════════════════╝
""")

    # 最终生成文件清单
    print("  生成文件清单:")
    for f in ['models/', 'results/']:
        for fname in sorted(os.listdir(f'./{f.rstrip("/")}')):
            sz = os.path.getsize(f'./{f.rstrip("/")}/{fname}')
            print(f"    ./{f}{fname:<45} {sz/1024:.1f} KB")

    # ──────────────────────────────────────────────────────────
    # 离线 AUC 评估
    # ──────────────────────────────────────────────────────────
    banner("AUC", "离线评估 — 各阶段排序质量对比")
    print("""
  评估思路：
    用测试集中的真实行为标签（is_click / is_like / long_view）作为 ground truth，
    对比各阶段的打分在测试集上的 AUC，衡量"排序分越高，用户越喜欢"的程度。

    注意：AUC 衡量的是"排序质量"而非"绝对准确率"，
    AUC=0.5 等于随机，AUC=1.0 是完美排序。

  各阶段评估指标：
    粗排 TwoTower : is_click AUC（主指标），反映召回质量
    精排 MMoE     : is_click / is_like / long_view / is_comment AUC
    MTF 融合分    : final_score 与 is_click 的 AUC（融合分是否比单任务分更准）
    PRM 重排      : prm_score 与 is_click 的 AUC（重排后 top-8 的点击率提升）
""")

    eval_targets = ['is_click', 'is_like', 'long_view']
    # 用 test_df 作为评估基准（含真实 label）
    eval_df = test_df[['user_id','video_id'] + eval_targets].copy()

    # 粗排 AUC
    pr_eval = eval_df.merge(
        prerank_df[['user_id','video_id','pre_is_click','pre_is_like','pre_long_view']],
        on=['user_id','video_id'], how='inner')
    print("  ┌─ 粗排（TwoTower）AUC ─────────────────────────────────────────┐")
    for tgt, col in [('is_click','pre_is_click'),('is_like','pre_is_like'),('long_view','pre_long_view')]:
        if pr_eval[tgt].sum() > 0 and pr_eval[tgt].sum() < len(pr_eval):
            auc = roc_auc_score(pr_eval[tgt], pr_eval[col])
            print(f"  │  {tgt:<12} AUC = {auc:.4f}")
    print("  └──────────────────────────────────────────────────────────────┘")

    # 精排 AUC
    fr_eval = eval_df.merge(
        fullrank_df[['user_id','video_id','full_is_click','full_is_like','full_long_view','final_score']],
        on=['user_id','video_id'], how='inner')
    print("  ┌─ 精排（MMoE）AUC ──────────────────────────────────────────────┐")
    for tgt, col in [('is_click','full_is_click'),('is_like','full_is_like'),('long_view','full_long_view')]:
        if fr_eval[tgt].sum() > 0 and fr_eval[tgt].sum() < len(fr_eval):
            auc = roc_auc_score(fr_eval[tgt], fr_eval[col])
            print(f"  │  {tgt:<12} AUC = {auc:.4f}")
    print("  │")
    if fr_eval['is_click'].sum() > 0 and fr_eval['is_click'].sum() < len(fr_eval):
        auc_final = roc_auc_score(fr_eval['is_click'], fr_eval['final_score'])
        print(f"  │  is_click (0.5×es+0.5×mtf 融合分) AUC = {auc_final:.4f}")
    print("  └──────────────────────────────────────────────────────────────┘")

    # 重排 AUC（只评估进入 top-60 的候选）
    prm_eval = eval_df.merge(
        expose_df[['user_id','video_id','prm_score']],
        on=['user_id','video_id'], how='inner')
    print("  ┌─ 重排 top-8 AUC（PRM 重排分 vs 真实点击）────────────────────┐")
    if len(prm_eval) > 0 and prm_eval['is_click'].sum() > 0 and prm_eval['is_click'].sum() < len(prm_eval):
        auc_prm = roc_auc_score(prm_eval['is_click'], prm_eval['prm_score'])
        print(f"  │  is_click    AUC = {auc_prm:.4f}  （样本数={len(prm_eval)}）")
    else:
        print(f"  │  样本量不足（{len(prm_eval)} 条），无法计算 AUC")
    print("  └──────────────────────────────────────────────────────────────┘")

    print("""
  📌 AUC 解读参考：
     < 0.55  较差，排序接近随机
    0.55~0.65  一般，有一定区分度
    0.65~0.75  良好，工业界粗排典型水平（1 epoch 演示）
     > 0.75   优秀，精排/重排目标水平
  （注：本演示仅 1 epoch，正式训练需 10+ epoch，AUC 会显著提升）
""")

    print("\n✅  全流程演示完成！")
    print("="*72)
