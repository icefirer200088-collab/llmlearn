"""
第 03 课：位置编码（正弦 / RoPE）+ 多头注意力，numpy 实现
运行：python multihead_position.py
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------- 1. 正弦位置编码 ----------------
def sinusoidal_pe(T, d):
    pos = np.arange(T)[:, None]                 # [T,1]
    i = np.arange(d // 2)[None, :]              # [1,d/2]
    angle = pos / (10000 ** (2 * i / d))        # [T,d/2]
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


# ---------------- 2. RoPE ----------------
def rope(x, positions, base=10000):
    """x: [T, d]，把每一对维度 (2i, 2i+1) 旋转 pos * θ_i"""
    T, d = x.shape
    theta = base ** (-np.arange(0, d, 2) / d)            # [d/2]
    ang = positions[:, None] * theta[None, :]            # [T, d/2]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out


# ---------------- 3. 多头注意力 ----------------
def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    T, d_model = X.shape
    d_head = d_model // n_heads
    # 投影后 reshape 成 [n_heads, T, d_head]，每个 head 独立计算
    split = lambda M: M.reshape(T, n_heads, d_head).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)        # [h, T, T]
    if causal:
        S = S + np.triu(np.full((T, T), -np.inf), k=1)
    A = softmax(S)
    Z = A @ V                                             # [h, T, d_head]
    Z = Z.transpose(1, 0, 2).reshape(T, d_model)          # 拼接回 [T, d_model]
    return Z @ Wo, A


if __name__ == "__main__":
    # —— 正弦 PE：相似度只和距离有关 ——
    pe = sinusoidal_pe(50, 64)
    print("正弦 PE 形状:", pe.shape)
    print("PE[10]·PE[13] =", round(pe[10] @ pe[13], 3), "  PE[30]·PE[33] =", round(pe[30] @ pe[33], 3), " (距离都是 3，点积相同)")

    # —— RoPE：旋转后点积只取决于相对位置 ——
    q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
    for m, n in [(5, 2), (105, 102), (1005, 1002)]:
        d = rope(q, np.array([m]))[0] @ rope(k, np.array([n]))[0]
        print(f"RoPE: m={m:<5} n={n:<5} m-n={m-n}  点积 = {d:.6f}")

    # —— 多头注意力 ——
    T, d_model, h = 6, 32, 4
    X = rng.normal(size=(T, d_model)) + sinusoidal_pe(T, d_model)
    Ws = [rng.normal(size=(d_model, d_model)) / np.sqrt(d_model) for _ in range(4)]
    out, A = multi_head_attention(X, *Ws, n_heads=h)
    print(f"\n多头注意力: 输入 {X.shape} -> 输出 {out.shape}, 注意力权重 {A.shape} (h, T, T)")
    for i in range(h):
        print(f"head {i} 最后一个 token 的注意力分布:", A[i, -1])
    print("参数量: 4 个 d_model×d_model 矩阵 =", 4 * d_model * d_model, "（和单头一样多！多头不增加参数，只是把空间切开）")
