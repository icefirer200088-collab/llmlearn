"""
第 04 课：用 numpy 拼出一个完整的 Transformer Block，并叠成多层
运行：python block.py
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class Block:
    def __init__(self, d, n_heads):
        s = 1 / np.sqrt(d)
        self.h = n_heads
        self.Wqkv = rng.normal(size=(d, 3 * d)) * s      # Q、K、V 三个矩阵合在一起
        self.Wo = rng.normal(size=(d, d)) * s
        self.W1 = rng.normal(size=(d, 4 * d)) * s
        self.W2 = rng.normal(size=(4 * d, d)) * s / 2
        self.ln1 = (np.ones(d), np.zeros(d))
        self.ln2 = (np.ones(d), np.zeros(d))

    def attn(self, x):
        T, d = x.shape
        dh = d // self.h
        q, k, v = np.split(x @ self.Wqkv, 3, axis=-1)
        q, k, v = (m.reshape(T, self.h, dh).transpose(1, 0, 2) for m in (q, k, v))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(dh) + np.triu(np.full((T, T), -np.inf), 1)
        z = softmax(s) @ v
        return z.transpose(1, 0, 2).reshape(T, d) @ self.Wo

    def ffn(self, x):
        return gelu(x @ self.W1) @ self.W2

    def __call__(self, x):
        x = x + self.attn(layer_norm(x, *self.ln1))   # 子层 1：注意力 + 残差
        x = x + self.ffn(layer_norm(x, *self.ln2))    # 子层 2：FFN + 残差
        return x

    def n_params(self):
        return sum(m.size for m in (self.Wqkv, self.Wo, self.W1, self.W2)) + 4 * len(self.ln1[0])


if __name__ == "__main__":
    T, d, h, L = 5, 64, 4, 6
    x = rng.normal(size=(T, d))
    blocks = [Block(d, h) for _ in range(L)]

    print(f"输入形状 {x.shape}")
    for i, b in enumerate(blocks):
        x = b(x)
        print(f"  第 {i+1} 层后: 形状 {x.shape}, 残差流平均范数 {np.linalg.norm(x, axis=-1).mean():.2f}")
    print("形状始终不变，所以可以无限叠加。")

    p = blocks[0].n_params()
    print(f"\n每层参数: {p:,}  ≈ 12·d² = {12 * d * d:,}")

    # 证明因果性：改动最后一个 token，前面 token 的输出不变
    x0 = rng.normal(size=(T, d))
    x1 = x0.copy(); x1[-1] += 10
    y0, y1 = blocks[0](x0), blocks[0](x1)
    print("\n改动最后一个 token 后，各位置输出变化量:", np.abs(y0 - y1).sum(-1).round(3))
    print("前 4 个位置为 0 —— 因果掩码保证了'看不到未来'。")
