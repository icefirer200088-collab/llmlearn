# 第 03 课 · 多头注意力与位置编码

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. Self-Attention 不知道顺序

注意力的计算只有矩阵乘法和 softmax，把输入的行打乱，输出也只是跟着打乱，每个 token 的结果**完全不变**（数学上叫"置换等变"）。所以"狗咬人"和"人咬狗"里，"狗"得到的向量一模一样。

解决办法：把**位置信息**注入进去。

## 2. 正弦位置编码（2017，原版 Transformer）

$$PE_{(pos,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d}}\right),\quad PE_{(pos,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

- 直接**加**到 embedding 上：`x = embedding + PE[pos]`
- 低维变化快、高维变化慢，像一个多指针的时钟
- `PE[a]·PE[b]` 只和 `|a−b|` 有关，模型可以借此推断相对距离

另一种做法是**可学习的位置 embedding**（GPT-2）：直接再开一张 `[最大长度, d]` 的表让模型自己学。缺点是超过训练长度就没法用。

## 3. RoPE（旋转位置编码，2021）

Llama、Qwen、Mistral、DeepSeek 等几乎所有现代开源模型都用 RoPE。

- 不加到 embedding 上，而是在算注意力前，把 q 和 k **按位置旋转**：位置 m 的向量旋转 `m·θ`
- 旋转不改变长度，两个旋转后向量的点积只和**夹角差 (m−n)θ** 有关 → 天然的**相对位置**编码
- d 维向量切成 d/2 对，每对用不同频率 `θᵢ = 10000^(−2i/d)`
- 长上下文扩展（NTK、YaRN 等）就是在调这些频率

运行 `multihead_position.py` 可以验证：位置 (5,2)、(105,102)、(1005,1002) 的点积完全相同。

## 4. 多头注意力

一组 W_Q/W_K/W_V 只能学到一种"关注模式"。但语言里同时存在很多关系：相邻搭配、指代、句法、长距离依赖……

做法：

1. 把 `d_model` 切成 `h` 份，每份 `d_head = d_model / h`
2. 每个 head 独立做注意力
3. 拼接所有 head 的输出，再乘 `W_O` 混合

```
MultiHead(X) = Concat(head₁, …, head_h) · W_O
headᵢ = Attention(X·W_Qⁱ, X·W_Kⁱ, X·W_Vⁱ)
```

**参数量和单头一样**（4 个 `d_model × d_model` 矩阵），只是把空间切开，让不同子空间各自专注。

研究者在训练好的模型里确实观察到了分工明确的 head：前一个 token head、induction head（第 09 课会讲）、句法 head、attention sink 等。

## 5. 常见配置

| 模型 | d_model | heads | d_head | 位置编码 |
|------|---------|-------|--------|----------|
| GPT-2 small | 768 | 12 | 64 | 可学习 |
| Llama-2 7B | 4096 | 32 | 128 | RoPE |
| Llama-3 70B | 8192 | 64（KV 8 组，GQA） | 128 | RoPE |

GQA（分组查询注意力）：多个 query head 共享一组 K/V，以减小 KV Cache —— 第 06 课会讲。

## 6. 动手练习

1. 在 `viz.html` 的 RoPE 实验中，固定 m−n = 3，反复点"同时右移"，验证点积不变。
2. 修改 `multihead_position.py`：去掉 `X` 里的 `sinusoidal_pe`，再把 X 的行打乱，验证多头注意力的输出也只是行被打乱。
3. 思考：为什么正弦编码用 10000 作为底数？换成 100 会怎样？

## 7. 延伸阅读

- Su et al., 2021 · *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Ainslie et al., 2023 · *GQA: Training Generalized Multi-Query Transformer Models*
- Anthropic · *A Mathematical Framework for Transformer Circuits*

---
⬅️ [02 Self-Attention](../02-self-attention/) ｜ [目录](../../README.md) ｜ [04 Transformer Block](../04-transformer-block/) ➡️
