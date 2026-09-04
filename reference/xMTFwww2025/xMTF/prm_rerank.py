"""
PRM: Personalized Re-ranking Model
论文：Personalized Re-ranking for Recommendation (SIGIR 2019, Alibaba)
https://arxiv.org/abs/1904.06813

核心思想：
  精排给出每个 item 的单点打分 p(y|u,v)，但忽略了 item 之间的交互（如多样性、互补性）。
  PRM 用 Transformer Encoder 对整个候选列表建模，让每个 item 能感知"上下文中其他 item 的存在"，
  从而学出更好的重排序。

在 xMTF 流水线中的位置：
  train.py (TwoTower + MMoE) →
  test.py  (cross_join 打分) →
  【prm_rerank.py】(PRM 重排 → top-8 曝光)

数据流：
  输入  : ./results/fullrank_{data_type}_{k}.pkl   ← test.py 输出的全量精排打分
  输出  : ./results/rerank_{data_type}_top8.csv    ← 每个用户最终曝光的 top-8 视频

用法：
  python prm_rerank.py \\
      --data_path ../KuaiRand-1K \\
      --candidate_num 50 \\
      --expose_num 8 \\
      --num_heads 4 \\
      --num_layers 2 \\
      --d_model 64 \\
      --dropout 0.1
"""

import os
import gc
import glob
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ─────────────────────────────────────────────────────────
#  1. Transformer Encoder Block (标准实现)
# ─────────────────────────────────────────────────────────

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """多头自注意力"""
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (B, heads, seq, depth)

    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        scaled = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled += (mask * -1e9)
        weights = tf.nn.softmax(scaled, axis=-1)

        output = tf.matmul(weights, v)                         # (B, heads, seq, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])      # (B, seq, heads, depth)
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(output)


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """单层 Transformer Encoder: Self-Attn + FFN + Add&Norm"""
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_dim, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        attn_out = self.attn(x, mask=mask)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)           # Add & Norm

        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out, training=training)
        x = self.norm2(x + ffn_out)            # Add & Norm
        return x


# ─────────────────────────────────────────────────────────
#  2. PRM 模型
# ─────────────────────────────────────────────────────────

class PRMModel(tf.keras.Model):
    """
    Personalized Re-ranking Model (PRM, SIGIR 2019)

    输入:
      item_features  : (B, N, feat_dim)  ← 候选列表的特征（精排分 + 个性化embedding）
    输出:
      scores         : (B, N)            ← 每个 item 的重排得分
    
    架构：
      Linear Projection → N × TransformerEncoderBlock → Dense(1) → squeeze
    """
    def __init__(self,
                 feat_dim: int,
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 ffn_dim: int = 128,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        # 输入投影：将原始特征映射到 d_model 维
        self.input_proj = tf.keras.layers.Dense(d_model, activation='relu')

        # 用户个性化向量（Personalized Vector，PRM 论文中的核心组件）
        # 这里用一个可学习的 Dense 层从精排分生成个性化向量
        self.pv_proj = tf.keras.layers.Dense(d_model, activation='tanh', name='personal_vec')

        # Transformer Encoder Stack
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, ffn_dim, dropout, name=f'encoder_{i}')
            for i in range(num_layers)
        ]

        # 输出层：每个 item 的重排分
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, item_features, training=False):
        """
        item_features: (B, N, feat_dim)
        """
        # 个性化向量：对每个列表取全局均值（模拟用户偏好）
        user_vec = tf.reduce_mean(item_features, axis=1, keepdims=True)   # (B, 1, feat_dim)
        user_vec = tf.tile(user_vec, [1, tf.shape(item_features)[1], 1])  # (B, N, feat_dim)
        pv = self.pv_proj(user_vec)                                        # (B, N, d_model)

        # 输入投影
        x = self.input_proj(item_features)   # (B, N, d_model)

        # PRM 核心：将个性化向量与 item 表示叠加（论文公式4）
        x = x + pv                            # (B, N, d_model)

        # Transformer Encoder 建模列表内上下文交互
        for block in self.encoder_blocks:
            x = block(x, training=training)  # (B, N, d_model)

        # 输出每个 item 的重排分
        scores = self.output_layer(x)         # (B, N, 1)
        scores = tf.squeeze(scores, axis=-1)  # (B, N)
        return scores


# ─────────────────────────────────────────────────────────
#  3. 训练/推理主流程
# ─────────────────────────────────────────────────────────

# 精排输出中使用的特征列（6个目标 + play_time_s）
SCORE_FEATURES = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'play_time_s']

# 精排综合分（用于从精排结果取 top-N 候选传给重排）
RANK_SCORE_WEIGHTS = {
    'is_click': 1.0,
    'is_like': 2.0,
    'is_follow': 4.0,
    'is_comment': 2.0,
    'is_forward': 2.0,
    'long_view': 1.5,
    'play_time_s': 0.01,
}


def compute_fullrank_score(df: pd.DataFrame) -> pd.Series:
    """计算精排综合分，用于取 top-candidate_num"""
    score = pd.Series(0.0, index=df.index)
    for col, w in RANK_SCORE_WEIGHTS.items():
        if col in df.columns:
            score += df[col] * w
    return score


def load_fullrank_results(results_dir: str, data_type: str) -> pd.DataFrame:
    """加载 test.py 输出的全量精排打分（多个 bucket）"""
    files = sorted(glob.glob(os.path.join(results_dir, f'fullrank_{data_type}_*.pkl')))
    if not files:
        raise FileNotFoundError(
            f"找不到精排结果文件: {results_dir}/fullrank_{data_type}_*.pkl\n"
            "请先运行 test.py 生成精排候选集"
        )
    print(f"找到 {len(files)} 个精排结果文件，开始加载...")
    dfs = []
    for f in tqdm(files, desc="加载精排结果"):
        dfs.append(pd.read_pickle(f))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"精排总行数: {len(df):,}，用户数: {df['user_id'].nunique():,}，视频数: {df['video_id'].nunique():,}")
    return df


def build_candidate_lists(fullrank_df: pd.DataFrame, candidate_num: int) -> tuple:
    """
    从精排全量打分中，为每个用户取 top-candidate_num 作为重排候选。
    
    返回：
      user_ids   : (n_users,)
      feat_array : (n_users, candidate_num, feat_dim)  ← 归一化后的特征
      video_ids  : (n_users, candidate_num)            ← 对应的 video_id
    """
    print(f"为每个用户构建 top-{candidate_num} 候选列表...")
    fullrank_df['_rank_score'] = compute_fullrank_score(fullrank_df)

    user_ids_list = []
    feat_list = []
    video_ids_list = []

    for uid, group in tqdm(fullrank_df.groupby('user_id'), desc="构建候选列表"):
        # 取 top-candidate_num
        topk = group.nlargest(candidate_num, '_rank_score')
        if len(topk) < candidate_num:
            # 不足则 pad 0（实际运行时 KuaiRand-1K 每用户视频数足够）
            pad = candidate_num - len(topk)
            pad_df = pd.DataFrame(
                np.zeros((pad, len(topk.columns))),
                columns=topk.columns
            )
            topk = pd.concat([topk, pad_df], ignore_index=True)

        # 特征：7维精排分
        feats = topk[SCORE_FEATURES].values.astype(np.float32)  # (candidate_num, 7)

        user_ids_list.append(uid)
        feat_list.append(feats)
        video_ids_list.append(topk['video_id'].values)

    feat_array = np.stack(feat_list, axis=0)      # (n_users, candidate_num, 7)
    video_ids_arr = np.array(video_ids_list)       # (n_users, candidate_num)
    user_ids_arr = np.array(user_ids_list)         # (n_users,)

    return user_ids_arr, feat_array, video_ids_arr


def normalize_features(feat_array: np.ndarray) -> np.ndarray:
    """
    对每个特征维度做 min-max 归一化，保证 Transformer 输入数值稳定。
    feat_array: (n_users, N, feat_dim)
    """
    n_users, N, feat_dim = feat_array.shape
    flat = feat_array.reshape(-1, feat_dim)
    f_min = flat.min(axis=0, keepdims=True) + 1e-9
    f_max = flat.max(axis=0, keepdims=True) + 1e-9
    flat_norm = (flat - f_min) / (f_max - f_min + 1e-9)
    return flat_norm.reshape(n_users, N, feat_dim).astype(np.float32)


def train_prm(model: PRMModel, feat_array: np.ndarray, epochs: int = 5, batch_size: int = 64,
              learning_rate: float = 1e-3):
    """
    无监督自监督训练 PRM（对比精排分排序）：
      用 ListMLE loss（最大似然列表排序）让模型学习精排分的相对顺序，
      同时通过 Transformer 捕捉 item 间的上下文相关性。

    说明：
      PRM 论文中是有监督训练（用真实点击标签）。这里采用"蒸馏精排分"的方式：
      把精排综合分作为 soft label，用 ListMLE loss 监督重排模型，
      让模型在学习精排分排序的同时，通过注意力机制发现 item 间的上下文关系。
    """
    n_users, N, feat_dim = feat_array.shape
    feat_tensor = tf.constant(feat_array)  # (n_users, N, feat_dim)

    # soft label：精排综合分（最后一列是 play_time_s，权重更低）
    # 使用 is_click 作为主要排序信号（也可以用加权综合分）
    # SCORE_FEATURES = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'play_time_s']
    weights_arr = np.array([
        RANK_SCORE_WEIGHTS[f] for f in SCORE_FEATURES
    ], dtype=np.float32)
    labels = (feat_array * weights_arr[np.newaxis, np.newaxis, :]).sum(axis=-1)  # (n_users, N)
    labels_tensor = tf.constant(labels, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def list_mle_loss(scores, labels):
        """
        ListMLE Loss：最大化正确排列的概率（ICML 2008）
        等价于 softmax cross-entropy over sorted positions
        """
        # 按 label 降序排列
        indices = tf.argsort(labels, axis=-1, direction='DESCENDING')
        sorted_scores = tf.gather(scores, indices, batch_dims=1)
        # LogSumExp trick: 对每个位置计算 log(softmax 分母)
        # ListMLE: sum_i [ score[i] - log(sum_{j>=i} exp(score[j])) ]
        n = tf.shape(sorted_scores)[1]
        cumsum_exp = tf.reverse(
            tf.cumsum(tf.reverse(tf.exp(sorted_scores), axis=[1]), axis=1),
            axis=[1]
        )
        log_cumsum = tf.math.log(cumsum_exp + 1e-9)
        loss = -tf.reduce_mean(tf.reduce_sum(sorted_scores - log_cumsum, axis=-1))
        return loss

    print(f"\n开始 PRM 训练（epochs={epochs}, batch_size={batch_size}）...")
    n_batches = (n_users + batch_size - 1) // batch_size
    for epoch in range(epochs):
        total_loss = 0.0
        # shuffle
        idx = np.random.permutation(n_users)
        for b in range(n_batches):
            batch_idx = idx[b * batch_size: (b + 1) * batch_size]
            x_batch = tf.gather(feat_tensor, batch_idx)        # (B, N, feat_dim)
            y_batch = tf.gather(labels_tensor, batch_idx)      # (B, N)

            with tf.GradientTape() as tape:
                scores = model(x_batch, training=True)         # (B, N)
                loss = list_mle_loss(scores, y_batch)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy()

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")

    return model


def rerank_and_expose(model: PRMModel, user_ids: np.ndarray,
                      feat_array: np.ndarray, video_ids: np.ndarray,
                      expose_num: int = 8, batch_size: int = 256) -> pd.DataFrame:
    """
    用训练好的 PRM 对候选列表重排，取 top-expose_num 作为最终曝光列表。
    
    返回：每行是一个 (user_id, video_id, prm_score, expose_rank) 记录。
    """
    print(f"\n开始 PRM 推理，最终曝光 top-{expose_num}...")
    n_users = len(user_ids)
    rows = []
    for i in tqdm(range(0, n_users, batch_size), desc="PRM 推理"):
        x_batch = feat_array[i: i + batch_size]                    # (B, N, feat_dim)
        scores = model(x_batch, training=False).numpy()            # (B, N)

        for j in range(len(x_batch)):
            uid = user_ids[i + j]
            vids = video_ids[i + j]                                 # (N,)
            user_scores = scores[j]                                 # (N,)
            # 取 top-expose_num
            top_indices = np.argsort(user_scores)[::-1][:expose_num]
            for rank, idx in enumerate(top_indices, start=1):
                rows.append({
                    'user_id': int(uid),
                    'video_id': int(vids[idx]),
                    'prm_score': float(user_scores[idx]),
                    'expose_rank': rank,
                })

    result_df = pd.DataFrame(rows)
    print(f"重排完成，共 {len(result_df):,} 行（{n_users} 用户 × {expose_num} 曝光）")
    return result_df


# ─────────────────────────────────────────────────────────
#  4. 评估：NDCG@k、Precision@k
# ─────────────────────────────────────────────────────────

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    relevances = np.asarray(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    gains = relevances / np.log2(np.arange(2, len(relevances) + 2))
    return float(gains.sum())


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(result_df: pd.DataFrame, fullrank_df: pd.DataFrame, expose_num: int = 8):
    """
    用精排分（is_click）作为 relevance，评估 PRM 重排的 NDCG@8。
    对比基线：直接用精排综合分取 top-8（不经过 PRM）。
    """
    print("\n评估 PRM 重排效果（以精排 is_click 分为相关性标签）...")
    fullrank_df['_rank_score'] = compute_fullrank_score(fullrank_df)

    ndcg_prm_list, ndcg_base_list = [], []
    prec_prm_list, prec_base_list = [], []

    for uid, group in tqdm(fullrank_df.groupby('user_id'), desc="评估"):
        topk_base = group.nlargest(expose_num, '_rank_score')[['video_id', 'is_click']].reset_index(drop=True)

        prm_user = result_df[result_df['user_id'] == uid].sort_values('expose_rank')
        if len(prm_user) == 0:
            continue

        # 获取 PRM 输出的 video_id 对应的 is_click 值
        vid_to_click = group.set_index('video_id')['is_click'].to_dict()
        prm_relevances = [vid_to_click.get(vid, 0) for vid in prm_user['video_id']]
        base_relevances = topk_base['is_click'].tolist()

        ndcg_prm_list.append(ndcg_at_k(prm_relevances, expose_num))
        ndcg_base_list.append(ndcg_at_k(base_relevances, expose_num))
        prec_prm_list.append(np.mean(prm_relevances[:expose_num]))
        prec_base_list.append(np.mean(base_relevances[:expose_num]))

    print(f"\n{'指标':<20} {'精排基线':>12} {'PRM 重排':>12} {'提升':>10}")
    print("-" * 56)
    for name, base_list, prm_list in [
        (f"NDCG@{expose_num}", ndcg_base_list, ndcg_prm_list),
        (f"Precision@{expose_num}", prec_base_list, prec_prm_list),
    ]:
        base_mean = np.mean(base_list)
        prm_mean = np.mean(prm_list)
        improvement = (prm_mean - base_mean) / (base_mean + 1e-9) * 100
        print(f"{name:<20} {base_mean:>12.4f} {prm_mean:>12.4f} {improvement:>+9.2f}%")


# ─────────────────────────────────────────────────────────
#  5. 入口
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PRM Personalized Re-ranking")
    parser.add_argument("--data_path", type=str, default="../KuaiRand-1K")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="test.py 输出的精排结果目录")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="重排结果输出目录")
    parser.add_argument("--candidate_num", type=int, default=50,
                        help="每个用户送入 PRM 的候选数（从精排 topN 中取）")
    parser.add_argument("--expose_num", type=int, default=8,
                        help="最终曝光 item 数")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Transformer hidden size")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Transformer 多头数")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Transformer Encoder 层数")
    parser.add_argument("--ffn_dim", type=int, default=128,
                        help="Transformer FFN 中间层维度")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5,
                        help="PRM 训练轮数")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip_train", action='store_true',
                        help="跳过训练，直接加载已保存权重做推理")
    parser.add_argument("--no_eval", action='store_true',
                        help="跳过评估步骤（加载精排结果做评估比较耗内存）")
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    # 随机种子
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_type = args.data_path.split('-')[-1].lower()
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = f"./models/prm_{data_type}.weights.h5"

    # ── Step 1: 加载精排结果
    fullrank_df = load_fullrank_results(args.results_dir, data_type)

    # ── Step 2: 构建候选列表
    user_ids, feat_array, video_ids = build_candidate_lists(fullrank_df, args.candidate_num)

    # ── Step 3: 特征归一化
    feat_array_norm = normalize_features(feat_array)
    feat_dim = feat_array_norm.shape[-1]
    print(f"特征维度: {feat_dim}，候选数: {args.candidate_num}，用户数: {len(user_ids)}")

    # ── Step 4: 建立/加载 PRM 模型
    model = PRMModel(
        feat_dim=feat_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    # 触发模型 build（明确指定 input_shape）
    dummy = tf.zeros((1, args.candidate_num, feat_dim))
    _ = model(dummy)

    if args.skip_train and os.path.exists(model_save_path):
        print(f"加载已有权重: {model_save_path}")
        model.load_weights(model_save_path)
    else:
        # ── Step 5: 训练
        model = train_prm(
            model, feat_array_norm,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        model.save_weights(model_save_path)
        print(f"模型权重已保存: {model_save_path}")

    # ── Step 6: 重排推理，输出 top-8 曝光列表
    result_df = rerank_and_expose(
        model, user_ids, feat_array_norm, video_ids,
        expose_num=args.expose_num,
        batch_size=256,
    )

    output_path = os.path.join(args.output_dir, f"rerank_{data_type}_top{args.expose_num}.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n重排结果已保存: {output_path}")
    print(result_df.groupby('expose_rank').size().rename("用户数").to_string())

    # ── Step 7: 评估（可选，对比精排基线和 PRM 重排）
    if not args.no_eval:
        evaluate(result_df, fullrank_df, expose_num=args.expose_num)

    print("\n✅ PRM 重排完成！")
