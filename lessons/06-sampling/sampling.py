"""
第 06 课：解码策略（贪心 / 温度 / top-k / top-p）+ KV Cache
运行：python sampling.py
"""
import time
import numpy as np

rng = np.random.default_rng(0)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ---------------- 1. 解码策略 ----------------
def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=rng):
    if temperature == 0:                                  # 贪心
        return int(np.argmax(logits))
    probs = softmax(logits / temperature)
    order = np.argsort(-probs)
    keep = np.zeros_like(probs, dtype=bool)
    if top_k is not None:
        keep[order[:top_k]] = True
    else:
        keep[:] = True
    if top_p is not None:
        cum = np.cumsum(probs[order])
        cutoff = np.searchsorted(cum, top_p) + 1            # 累计概率刚超过 p 的最小集合
        nucleus = np.zeros_like(keep); nucleus[order[:cutoff]] = True
        keep &= nucleus
    probs = np.where(keep, probs, 0)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


# ---------------- 2. KV Cache ----------------
class TinyAttnLM:
    """单层单头注意力，用于对比有无 KV Cache 的计算量（不需要训练，只看速度和结果是否一致）"""
    def __init__(self, V=1000, d=256, seed=0):
        r = np.random.default_rng(seed)
        self.E = r.normal(size=(V, d)) * 0.1
        self.Wq, self.Wk, self.Wv = (r.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
        self.Wout = r.normal(size=(d, V)) / np.sqrt(d)
        self.d = d

    def step_no_cache(self, ids):
        x = self.E[ids]                                     # 每次都把全部 token 重新算一遍
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        a = softmax(q[-1] @ k.T / np.sqrt(self.d))
        return (a @ v) @ self.Wout

    def step_with_cache(self, new_id, cache):
        x = self.E[new_id]                                  # 只算新 token
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        cache["k"].append(k); cache["v"].append(v)
        K, V = np.array(cache["k"]), np.array(cache["v"])
        a = softmax(q @ K.T / np.sqrt(self.d))
        return (a @ V) @ self.Wout


if __name__ == "__main__":
    words = ["好", "不错", "晴朗", "热", "冷", "棒", "糟糕", "差", "蓝", "香蕉"]
    logits = np.array([4.0, 3.3, 2.8, 2.5, 2.2, 1.5, 1.0, 0.8, -1.0, -3.0])
    print('上文："今天天气真"  ——  各策略采样 1000 次的结果分布\n')
    for name, kw in [("贪心", dict(temperature=0)), ("T=1", dict()), ("T=0.5", dict(temperature=0.5)),
                     ("T=2", dict(temperature=2)), ("top-k=3", dict(top_k=3)), ("top-p=0.9", dict(top_p=0.9))]:
        c = np.bincount([sample(logits, **kw) for _ in range(1000)], minlength=len(words))
        print(f"{name:<10}", "  ".join(f"{w}:{n:3d}" for w, n in zip(words, c) if n))

    print("\n—— KV Cache ——")
    lm, n = TinyAttnLM(), 400
    ids = list(rng.integers(0, 1000, size=n))
    t0 = time.time(); out1 = [lm.step_no_cache(ids[:i + 1]) for i in range(n)]; t1 = time.time() - t0
    cache = {"k": [], "v": []}
    t0 = time.time(); out2 = [lm.step_with_cache(ids[i], cache) for i in range(n)]; t2 = time.time() - t0
    print(f"生成 {n} 个 token：不缓存 {t1:.2f}s，KV Cache {t2:.2f}s，加速 {t1 / t2:.1f}×")
    print("两种方式的输出完全一致:", np.allclose(out1, out2))
    print(f"计算 K/V 的次数：不缓存 {n * (n + 1) // 2:,}，有缓存 {n:,}")
