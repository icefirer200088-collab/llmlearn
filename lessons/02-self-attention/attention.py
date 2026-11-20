"""
第 02 课：用 numpy 从零实现 Self-Attention
运行：python attention.py
"""
import numpy as np

np.set_printoptions(precision=2, suppress=True)
rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # 减去最大值，防止 exp 溢出（数值稳定技巧）
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(X, Wq, Wk, Wv, causal=False):
    """
    X:  [T, d_model]  T 个 token 的向量
    W*: [d_model, d_k]
    返回 输出 Z [T, d_k] 和 注意力权重 A [T, T]
    """
    Q, K, V = X @ Wq, X @ Wk, X @ Wv           # 1. 线性投影
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)                  # 2. 打分 + 缩放
    if causal:                                  # 3.（可选）因果掩码：不能看未来
        S = np.where(np.triu(np.ones_like(S), k=1) == 1, -np.inf, S)
    A = softmax(S, axis=-1)                     # 4. 每行归一化成概率
    return A @ V, A                             # 5. 加权求和


if __name__ == "__main__":
    tokens = ["小猫", "坐", "在", "垫子", "上"]
    T, d_model, d_k = len(tokens), 8, 4
    X = rng.normal(size=(T, d_model))
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) / np.sqrt(d_model) for _ in range(3))

    Z, A = self_attention(X, Wq, Wk, Wv)
    print("注意力权重 A（行 = 谁在看，列 = 被看的）:")
    print("        " + "  ".join(f"{t:>4}" for t in tokens))
    for t, row in zip(tokens, A):
        print(f"{t:>4}  ", row)
    print("每行之和:", A.sum(axis=1))
    print("\n输出 Z 形状:", Z.shape, "（每个 token 得到一个融合了上下文的新向量）")

    _, A_causal = self_attention(X, Wq, Wk, Wv, causal=True)
    print("\n加因果掩码后（上三角全为 0，GPT 生成文本时就是这样）:")
    print(A_causal)

    # —— 为什么要除以 √d ——
    print("\n不同维度下 q·k 的标准差（q、k 每维 ~ N(0,1)）:")
    for d in [4, 64, 512]:
        q, k = rng.normal(size=(10000, d)), rng.normal(size=(10000, d))
        s = (q * k).sum(-1)
        print(f"  d={d:<4} 未缩放 std={s.std():6.2f}   缩放后 std={(s / np.sqrt(d)).std():.2f}")
