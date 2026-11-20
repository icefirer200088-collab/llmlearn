# 第 04 课 · Transformer Block

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. 一个 Block 长什么样（Pre-LN，GPT-2 起的标准）

```
x = x + MultiHeadAttention(LayerNorm(x))   # token 之间交流
x = x + FFN(LayerNorm(x))                  # 每个 token 独立加工
```

就这两行。整个 GPT = embedding → N 个 Block → LayerNorm → 输出层。

## 2. 四个零件各自的作用

### 残差连接（Residual）

`x = x + f(x)`：子层只学"增量"。

- 把整个网络想象成一条**残差流**（residual stream），每一层都从流里读、再往流里写
- 反向传播时梯度可以沿着 `+` 直接回到底层，所以能训练 100+ 层
- 去掉残差，深层 Transformer 基本训练不动

### LayerNorm

对**每个 token 自己的向量**做标准化：`γ·(x−μ)/σ + β`。

- 防止数值随层数越来越大或越来越小
- Pre-LN（先 LN 再进子层）比原论文的 Post-LN 更稳定，不需要 warmup 也能训
- Llama 等模型用 **RMSNorm**：只除以均方根、不减均值，更省计算

### 多头因果注意力

第 02、03 课的内容。**唯一**让 token 之间交换信息的地方。

### FFN（前馈网络）

`W₂ · GELU(W₁·x)`，中间宽度通常为 `4·d`。

- 对每个 token 独立计算，不跨 token
- 占了每层约 **2/3 的参数**
- 被认为是存储"知识"的主要地方（key-value memory 的观点）
- Llama 用 **SwiGLU**：`W₂·(SiLU(W₁x) ⊙ W₃x)`，效果更好

## 3. 参数量估算

每层 ≈ `12·d²`（注意力 4d² + FFN 8d²），总参数 ≈ `12·L·d² + V·d`。

| 模型 | d | L | 估算 |
|---|---|---|---|
| GPT-2 small | 768 | 12 | 124M |
| GPT-3 | 12288 | 96 | 175B |

用 `viz.html` 里的计算器试试。

## 4. 代码

`block.py` 用 numpy 实现了完整的 Block，并验证了：

1. 叠 6 层后形状不变
2. 每层参数 = 12·d²
3. 改动最后一个 token，前面位置的输出**完全不变**（因果性）

```bash
python block.py
```

## 5. 动手练习

1. 在 `block.py` 里去掉残差（`x = self.attn(...)`），叠 24 层，观察残差流范数如何变化。
2. 把 LayerNorm 换成 RMSNorm 实现一遍。
3. 思考：为什么 FFN 要先升维再降维？只用一个 `d×d` 矩阵不行吗？

## 6. 延伸阅读

- Xiong et al., 2020 · *On Layer Normalization in the Transformer Architecture*（Pre-LN vs Post-LN）
- Geva et al., 2021 · *Transformer Feed-Forward Layers Are Key-Value Memories*
- Shazeer, 2020 · *GLU Variants Improve Transformer*

---
⬅️ [03 多头 + 位置编码](../03-multihead-position/) ｜ [目录](../../README.md) ｜ [05 mini-GPT](../05-mini-gpt/) ➡️
