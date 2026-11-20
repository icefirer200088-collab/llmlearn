"""
第 05 课：用 numpy 从零写一个 mini-GPT（前向 + 手写反向传播 + Adam + 采样）
不依赖 PyTorch。运行：python mini_gpt.py          （约 1~2 分钟）
                       python mini_gpt.py --check  （只做梯度检验）

结构：token embedding + 位置 embedding → 1 个 Transformer Block（单头因果注意力 + ReLU FFN，Pre-LN）
      → 最终 LayerNorm → 输出层 → softmax 交叉熵
训练语料：Python 自带的《Python 之禅》（import this）
"""
import sys, io, codecs, contextlib
import numpy as np

# ---------------- 数据 ----------------
with contextlib.redirect_stdout(io.StringIO()):
    import this
TEXT = codecs.decode(this.s, "rot13")
chars = sorted(set(TEXT))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in TEXT])


# ---------------- 工具函数 ----------------
def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ln_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    sig = np.sqrt(x.var(-1, keepdims=True) + eps)
    xh = (x - mu) / sig
    return g * xh + b, (xh, sig, g)


def ln_bwd(dy, cache):
    xh, sig, g = cache
    dg = (dy * xh).reshape(-1, xh.shape[-1]).sum(0)
    db = dy.reshape(-1, xh.shape[-1]).sum(0)
    dxh = dy * g
    dx = (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True)) / sig
    return dx, dg, db


# ---------------- 模型 ----------------
def init_params(V, T, d, seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s, scale=0.02: r.normal(size=s) * scale
    return {
        "E": n(V, d), "P": n(T, d),
        "g1": np.ones(d), "b1": np.zeros(d),
        "Wq": n(d, d, scale=d ** -0.5), "Wk": n(d, d, scale=d ** -0.5), "Wv": n(d, d, scale=d ** -0.5), "Wo": n(d, d, scale=d ** -0.5),
        "g2": np.ones(d), "b2": np.zeros(d),
        "W1": n(d, 4 * d, scale=d ** -0.5), "c1": np.zeros(4 * d), "W2": n(4 * d, d, scale=(4 * d) ** -0.5), "c2": np.zeros(d),
        "gf": np.ones(d), "bf": np.zeros(d),
        "Wout": n(d, V, scale=d ** -0.5),
    }


def forward(p, idx, targets=None):
    """idx: [B, T] 的 token id。返回 logits、loss、反向传播需要的缓存。"""
    B, T = idx.shape
    d = p["E"].shape[1]
    x0 = p["E"][idx] + p["P"][:T]                                  # [B,T,d]
    # --- 注意力子层 ---
    h, c1 = ln_fwd(x0, p["g1"], p["b1"])
    q, k, v = h @ p["Wq"], h @ p["Wk"], h @ p["Wv"]
    s = q @ k.transpose(0, 2, 1) / np.sqrt(d)
    s = s + np.triu(np.full((T, T), -1e9), 1)                      # 因果掩码
    a = softmax(s)
    z = a @ v
    x1 = x0 + z @ p["Wo"]
    # --- FFN 子层 ---
    h2, c2 = ln_fwd(x1, p["g2"], p["b2"])
    u = h2 @ p["W1"] + p["c1"]
    r = np.maximum(u, 0)
    x2 = x1 + r @ p["W2"] + p["c2"]
    # --- 输出 ---
    h3, c3 = ln_fwd(x2, p["gf"], p["bf"])
    logits = h3 @ p["Wout"]                                        # [B,T,V]
    cache = (idx, h, c1, q, k, v, a, z, h2, c2, u, r, h3, c3)
    if targets is None:
        return logits, None, cache
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(B)[:, None], np.arange(T)[None, :], targets] + 1e-12).mean()
    return logits, loss, (cache, probs)


def backward(p, targets, cache_probs):
    (idx, h, c1, q, k, v, a, z, h2, c2, u, r, h3, c3), probs = cache_probs
    B, T = idx.shape
    d = p["E"].shape[1]
    g = {}
    # 交叉熵 + softmax 的梯度：probs - onehot
    dlog = probs.copy()
    dlog[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1
    dlog /= B * T
    g["Wout"] = np.einsum("btd,btv->dv", h3, dlog)
    dx2, g["gf"], g["bf"] = ln_bwd(dlog @ p["Wout"].T, c3)
    # FFN 子层（残差：梯度一路直通 + 分支）
    dx1 = dx2.copy()
    g["W2"] = np.einsum("btf,btd->fd", r, dx2); g["c2"] = dx2.sum((0, 1))
    du = (dx2 @ p["W2"].T) * (u > 0)
    g["W1"] = np.einsum("btd,btf->df", h2, du); g["c1"] = du.sum((0, 1))
    dh2, g["g2"], g["b2"] = ln_bwd(du @ p["W1"].T, c2)
    dx1 += dh2
    # 注意力子层
    dx0 = dx1.copy()
    g["Wo"] = np.einsum("btd,bte->de", z, dx1)
    dz = dx1 @ p["Wo"].T
    da = dz @ v.transpose(0, 2, 1)
    dv = a.transpose(0, 2, 1) @ dz
    ds = a * (da - (da * a).sum(-1, keepdims=True)) / np.sqrt(d)  # softmax 的反向
    dq = ds @ k
    dk = ds.transpose(0, 2, 1) @ q
    g["Wq"] = np.einsum("btd,bte->de", h, dq)
    g["Wk"] = np.einsum("btd,bte->de", h, dk)
    g["Wv"] = np.einsum("btd,bte->de", h, dv)
    dh = dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
    dh0, g["g1"], g["b1"] = ln_bwd(dh, c1)
    dx0 += dh0
    # embedding：哪些行被用到，梯度就加到哪些行上
    g["E"] = np.zeros_like(p["E"])
    np.add.at(g["E"], idx, dx0)
    g["P"] = np.zeros_like(p["P"])
    g["P"][:T] = dx0.sum(0)
    return g


# ---------------- 梯度检验：确认手写反向传播是对的 ----------------
def grad_check():
    r = np.random.default_rng(1)
    p = init_params(V, 8, 16, seed=3)
    for key in p:  # 放大参数，让检验更灵敏
        p[key] = p[key] + r.normal(size=p[key].shape) * 0.1
    idx = r.integers(0, V, size=(2, 8)); tgt = r.integers(0, V, size=(2, 8))
    _, _, cp = forward(p, idx, tgt)
    g = backward(p, tgt, cp)
    print("梯度检验（解析梯度 vs 数值梯度的相对误差，应 < 1e-5）:")
    for key in ["Wout", "W1", "Wq", "Wk", "Wv", "Wo", "g1", "E", "P"]:
        errs = []
        for _ in range(5):
            i = tuple(r.integers(0, s) for s in p[key].shape)
            if key == "E": i = (idx[0, r.integers(0, 8)],) + i[1:]
            old = p[key][i]
            p[key][i] = old + 1e-5; lp = forward(p, idx, tgt)[1]
            p[key][i] = old - 1e-5; lm = forward(p, idx, tgt)[1]
            p[key][i] = old
            num = (lp - lm) / 2e-5
            errs.append(abs(num - g[key][i]) / max(1e-8, abs(num) + abs(g[key][i])))
        print(f"  {key:<5} 最大相对误差 {max(errs):.2e}")


# ---------------- 训练 ----------------
def generate(p, start, n, T, temperature=0.8, seed=0):
    r = np.random.default_rng(seed)
    ids = [stoi[c] for c in start]
    for _ in range(n):
        ctx = np.array([ids[-T:]])
        logits, _, _ = forward(p, ctx)
        pr = softmax(logits[0, -1] / temperature)
        ids.append(r.choice(V, p=pr))
    return "".join(itos[i] for i in ids)


def train(steps=1500, B=16, T=32, d=64, lr=3e-3, seed=0):
    r = np.random.default_rng(seed)
    p = init_params(V, T, d, seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    s = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2 = 0.9, 0.99
    n_params = sum(v.size for v in p.values())
    print(f"语料 {len(TEXT)} 字符，词表 {V}，模型参数 {n_params:,}")
    print(f"随机猜的 loss = ln({V}) = {np.log(V):.2f}\n")
    for it in range(1, steps + 1):
        ix = r.integers(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix])
        y = np.stack([data[i + 1:i + T + 1] for i in ix])          # 目标 = 输入右移一位
        _, loss, cp = forward(p, x, y)
        g = backward(p, y, cp)
        for k in p:                                                # Adam
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            s[k] = b2 * s[k] + (1 - b2) * g[k] ** 2
            p[k] -= lr * (m[k] / (1 - b1 ** it)) / (np.sqrt(s[k] / (1 - b2 ** it)) + 1e-8)
        if it == 1 or it % 250 == 0:
            print(f"step {it:5d}  loss {loss:.3f}   样本: {generate(p, 'Beautiful ', 40, T)!r}")
    return p


if __name__ == "__main__":
    grad_check()
    if "--check" not in sys.argv:
        print()
        p = train()
        print("\n最终生成:\n" + generate(p, "Simple is", 300, 32, temperature=0.6, seed=1))
