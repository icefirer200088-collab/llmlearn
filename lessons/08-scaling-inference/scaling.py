"""
第 08 课：Scaling Law 计算 + 权重量化实验
运行：python scaling.py
"""
import numpy as np

# ---------------- 1. Chinchilla scaling law ----------------
# Hoffmann et al. 2022 拟合的参数：L(N, D) = E + A/N^α + B/D^β
E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(N, D):
    return E + A / N ** ALPHA + B / D ** BETA


def compute_optimal(C):
    """给定算力 C ≈ 6ND，在对数网格上搜索使 loss 最小的 N"""
    Ns = np.logspace(6, 13, 4000)
    Ds = C / (6 * Ns)
    L = loss(Ns, Ds)
    i = L.argmin()
    return Ns[i], Ds[i], L[i]


# ---------------- 2. 量化 ----------------
def quantize(w, bits, group=None):
    """对称 absmax 量化。group=None 表示整个张量共用一个刻度。"""
    qmax = 2 ** (bits - 1) - 1
    g = len(w) if group is None else group
    blocks = w.reshape(-1, g)
    scale = np.abs(blocks).max(1, keepdims=True) / qmax
    q = np.clip(np.round(blocks / scale), -qmax - 1, qmax)       # 存储的整数
    return (q * scale).reshape(w.shape)                           # 反量化回浮点


if __name__ == "__main__":
    print("—— Chinchilla 最优分配 ——")
    print(f"{'算力 C':>10} {'最优 N':>10} {'最优 D':>10} {'D/N':>6} {'loss':>6}")
    for c in [1e19, 1e21, 1e23, 1e25]:
        N, D, L = compute_optimal(c)
        print(f"{c:10.0e} {N / 1e9:9.2f}B {D / 1e9:9.0f}B {D / N:6.0f} {L:6.3f}")

    print("\n—— 同样算力下，'小模型 + 多数据' 推理更便宜 ——")
    for name, N, D in [("GPT-3", 175e9, 300e9), ("Chinchilla", 70e9, 1.4e12), ("Llama-3 8B", 8e9, 15e12)]:
        print(f"{name:<12} N={N / 1e9:5.0f}B D={D / 1e12:5.2f}T  算力 {6 * N * D:.1e}  预测 loss {loss(N, D):.3f}  推理每 token ≈ {2 * N / 1e9:.0f} GFLOPs")

    print("\n—— 量化误差（1 万个 N(0, 0.02) 权重 + 2 个离群值）——")
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, 10240)
    w[[5, 5000]] = [0.4, -0.35]
    for bits in [8, 4, 3]:
        for group in [None, 128, 32]:
            wq = quantize(w, bits, group)
            rel = np.sqrt(((w - wq) ** 2).mean() / (w ** 2).mean())
            zeros = (wq == 0).mean()
            print(f"INT{bits}  分组={str(group or '无'):<4}  相对误差 {rel * 100:6.2f}%   被量化成 0 的比例 {zeros * 100:5.1f}%")

    print("\n—— 模型大小（GB）——")
    for p in [7e9, 70e9, 405e9]:
        print(f"{p / 1e9:4.0f}B 参数:  FP16 {p * 2 / 1024 ** 3:7.1f}   INT8 {p / 1024 ** 3:7.1f}   INT4 {p / 2 / 1024 ** 3:7.1f}   训练(Adam) {p * 16 / 1024 ** 3:8.0f}")
