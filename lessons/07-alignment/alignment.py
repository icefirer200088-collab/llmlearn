"""
第 07 课：奖励模型（Bradley-Terry）与 DPO 的最小实现
用一个"只有 4 种回答"的玩具策略，看清偏好优化在做什么。
运行：python alignment.py
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)

RESPONSES = {
    "A": "听起来今天挺不容易的。愿意说说发生了什么吗？",
    "B": "可以试试运动、听音乐或者早点睡觉。",
    "C": "哦。",
    "D": "研究表明 97% 的坏心情会在 3 分钟内自动消失。（编造的事实）",
}
NAMES = list(RESPONSES)
PREFS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # (好, 差)
sigmoid = lambda x: 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------- 1. 奖励模型 ----------------
def train_reward_model(prefs, n, steps=500, lr=0.5):
    """每个回答一个分数 r。最大化 Σ log σ(r_好 − r_差)。"""
    r = np.zeros(n)
    for _ in range(steps):
        g = np.zeros(n)
        for w, l in prefs:
            c = 1 - sigmoid(r[w] - r[l])     # d/d(r_w - r_l) of log σ
            g[w] += c; g[l] -= c
        r += lr * g / len(prefs)
        r -= r.mean()                         # 分数只有相对意义
    return r


# ---------------- 2. DPO ----------------
def dpo(prefs, ref_probs, beta=0.5, steps=300, lr=0.5):
    """策略 π = softmax(θ)。损失 −log σ(β[(logπ_w − logπref_w) − (logπ_l − logπref_l)])"""
    theta = np.log(ref_probs).copy()
    log_ref = np.log(ref_probs)
    for t in range(steps + 1):
        p = softmax(theta)
        lp = np.log(p)
        g, loss = np.zeros_like(theta), 0.0
        for w, l in prefs:
            z = beta * ((lp[w] - log_ref[w]) - (lp[l] - log_ref[l]))
            loss -= np.log(sigmoid(z))
            # d logπ_i / dθ = onehot(i) − p
            d_lp = (np.eye(len(p))[w] - p) - (np.eye(len(p))[l] - p)
            g += -(1 - sigmoid(z)) * beta * d_lp
        theta -= lr * g / len(prefs)
        if t % 100 == 0:
            kl = (p * (lp - log_ref)).sum()
            print(f"  step {t:3d}  loss {loss / len(prefs):.3f}  KL {kl:.3f}  π = {p}")
    return softmax(theta)


if __name__ == "__main__":
    for k, v in RESPONSES.items():
        print(f"{k}: {v}")
    print("偏好:", ", ".join(f"{NAMES[w]}≻{NAMES[l]}" for w, l in PREFS))

    r = train_reward_model(PREFS, 4)
    print("\n奖励模型学到的分数:", {k: round(float(v), 2) for k, v in zip(NAMES, r)})
    print("例：P(A 优于 B) =", round(float(sigmoid(r[0] - r[1])), 3))

    ref = np.array([0.15, 0.40, 0.25, 0.20])
    print("\n参考模型（SFT 后）π_ref =", ref)
    for beta in [0.1, 1.0]:
        print(f"\nDPO, β = {beta}:")
        dpo(PREFS, ref, beta=beta)

    # RLHF 的最优策略有闭式解：π*(y) ∝ π_ref(y)·exp(r(y)/β)  —— 这正是 DPO 的出发点
    beta = 1.0
    opt = ref * np.exp(r / beta); opt /= opt.sum()
    print(f"\nRLHF 理论最优（β={beta}）π* ∝ π_ref·exp(r/β) =", opt)
