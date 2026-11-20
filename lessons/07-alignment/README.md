# 第 07 课 · 预训练 → SFT → RLHF / DPO

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. 三个阶段

| 阶段 | 数据 | 目标 | 产物 |
|---|---|---|---|
| 预训练 | 万亿 token 的网页、书、代码 | 预测下一个 token | 基座模型（会续写，懂知识） |
| SFT 指令微调 | 几万~几十万条"指令 → 回答" | 预测下一个 token（只在回答上算 loss） | 会按对话格式回答问题 |
| 偏好对齐 | 人类比较"哪个回答更好" | 提高好回答概率、降低差回答概率 | 更有用、诚实、无害 |

一个常见比喻：预训练是"读遍图书馆"，SFT 是"学会怎么答题"，对齐是"学会什么样的答案更好"。

## 2. SFT

- 数据按**对话模板**拼接：`<|system|>…<|user|>…<|assistant|>…<|end|>`
- 训练方法和预训练一样，但用户部分的 token 不算 loss（loss mask）
- 模型要学会输出 `<|end|>`，知道什么时候停
- 经验：**质量远比数量重要**（LIMA：1000 条精选数据就能有不错的效果）

## 3. 奖励模型与 RLHF

**奖励模型**：Bradley-Terry 假设 `P(A ≻ B) = σ(r(A) − r(B))`，在人类比较数据上训练打分器 r。

**RLHF（PPO）**：
1. 模型对 prompt 生成回答
2. 奖励模型打分
3. 用 PPO 更新模型，最大化 `E[r(y)] − β·KL(π ‖ π_ref)`
4. KL 项防止模型为了刷分跑偏（"reward hacking"），比如学会堆砌讨好的废话

需要同时维护策略、参考模型、奖励模型、价值模型 4 个网络，工程难度大。

## 4. DPO（Direct Preference Optimization，2023）

关键观察：上面那个带 KL 约束的目标有**闭式最优解**

$$\pi^*(y) \propto \pi_{\text{ref}}(y)\,\exp(r(y)/\beta)$$

反过来，奖励可以用策略表示：`r(y) = β·log(π(y)/π_ref(y)) + 常数`。代入 Bradley-Terry 就得到 DPO 损失：

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\left[\log\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\right]\right)$$

不需要奖励模型、不需要采样，就是一个普通的监督学习损失。

`alignment.py` 在一个只有 4 种回答的玩具策略上实现了奖励模型和 DPO：

```
奖励模型学到的分数: {'A': 5.01, 'B': 1.58, 'C': -1.58, 'D': -5.01}
DPO, β = 1.0:  step 300  π = [0.875 0.12 0.005 0.]
```

## 5. 更新的做法

- **RLAIF / Constitutional AI**：用 AI 按一套写明的原则来做比较标注，减少人工
- **面向推理的强化学习**：数学、代码等有标准答案的任务，直接用"答案对不对"作为奖励（如 GRPO），让模型学会长链条思考
- **KTO、ORPO、SimPO** 等 DPO 变体

## 6. 动手练习

1. 在 `viz.html` 里分别用 β=0.1 和 β=2 训练 DPO，比较 KL 曲线。
2. 在 `alignment.py` 里加一组**矛盾**的偏好（比如 `(2, 0)`：有人觉得"哦。"比 A 好），重新训练，β 的作用是否更明显？
3. 思考：如果奖励模型有漏洞（比如偏爱长回答），RLHF 会发生什么？

## 7. 延伸阅读

- Ouyang et al., 2022 · *Training language models to follow instructions with human feedback*（InstructGPT）
- Rafailov et al., 2023 · *Direct Preference Optimization*
- Bai et al., 2022 · *Constitutional AI: Harmlessness from AI Feedback*
- Zhou et al., 2023 · *LIMA: Less Is More for Alignment*

---
⬅️ [06 采样与生成](../06-sampling/) ｜ [目录](../../README.md) ｜ [08 Scaling 与推理](../08-scaling-inference/) ➡️
