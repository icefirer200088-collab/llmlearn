# 第 05 课 · 从零写一个 mini-GPT

> 👉 先打开 [`viz.html`](viz.html)，点"开始训练"，看 loss 下降、生成的文本从乱码变得像样。

## 1. 整个 GPT 的前向传播

```
token ids [B, T]
  → token embedding + 位置 embedding     [B, T, d]
  → N × Transformer Block                [B, T, d]
  → 最终 LayerNorm
  → 输出层 W_out                          [B, T, V]   每个位置一个"下一个 token"的分数（logits）
  → softmax                              [B, T, V]   概率分布
```

## 2. 训练目标：预测下一个 token

- 输入 `x = text[i : i+T]`，目标 `y = text[i+1 : i+T+1]`（右移一位）
- 有因果掩码，所以一段长度为 T 的文本同时产生 **T 个训练样本**
- 不需要人工标注 → **自监督学习**。互联网上所有文本都是训练数据

## 3. 损失函数：交叉熵

$$\mathcal{L} = -\frac{1}{BT}\sum \log p_\theta(y_t \mid x_{\le t})$$

- 完全随机猜：loss = ln(V)。本课语料 V = 45，所以起点是 3.81
- loss 的 e 次方叫**困惑度（perplexity）**：相当于"模型在几个选项之间犹豫"
- softmax + 交叉熵的梯度非常简洁：`∂L/∂logits = p − onehot(y)`

## 4. 反向传播与优化

`mini_gpt.py` 手写了每一个算子的反向传播：

| 算子 | 前向 | 反向（核心） |
|---|---|---|
| 线性层 | `y = xW` | `dW = xᵀ·dy`，`dx = dy·Wᵀ` |
| ReLU | `max(0,u)` | `du = dy · (u>0)` |
| softmax（注意力） | `a = softmax(s)` | `ds = a ⊙ (da − Σ(da⊙a))` |
| LayerNorm | `γ·x̂ + β` | `dx = (dx̂ − mean(dx̂) − x̂·mean(dx̂⊙x̂)) / σ` |
| 残差 | `x + f(x)` | 梯度**同时**流向两条路 |
| Embedding | `E[idx]` | 梯度累加到被用到的行 |

并用**数值梯度检验**确认每个梯度都对（相对误差 < 1e-5）：

```bash
python mini_gpt.py --check
```

优化器用 **Adam**：给每个参数维护梯度的一阶矩（动量）和二阶矩（自适应步长）。

## 5. 运行

```bash
python mini_gpt.py
```

约 20 秒，loss 从 4.4 降到 ~0.2，最后生成：

```
Simple is better than complicated.
Flat is better than nested.
Sparse is better than do explain, it's a bad idea.
```

注意它**过拟合**了：856 个字符的语料，5.7 万个参数，基本是背下来的。真正的预训练用的是万亿 token。

## 6. 从 mini-GPT 到真 GPT 还差什么？

| | 本课 | GPT-3 |
|---|---|---|
| 层数 | 1 | 96 |
| d_model | 64 | 12288 |
| 头数 | 1 | 96 |
| 上下文 | 32 | 2048 |
| 词表 | 45 个字符 | 5 万 BPE token |
| 训练数据 | 856 字符 | 3000 亿 token |
| 框架 | numpy 手写梯度 | PyTorch 自动求导 + 数千张 GPU |

**原理完全一样**，区别只在规模和工程。

## 7. 动手练习

1. 在 `viz.html` 里把学习率拉到最大，观察 loss 曲线震荡；拉到最小，观察下降很慢。
2. 修改 `mini_gpt.py`：把上下文 T 从 32 改成 4，loss 能降到多少？说明了什么？
3. 再加一层 Block（提示：把 Block 封装成函数，前向时依次调用，反向时倒序调用）。
4. 进阶：读 Karpathy 的 nanoGPT，用 PyTorch 重写本课模型，对比代码量。

## 8. 延伸阅读

- Andrej Karpathy · *Let's build GPT: from scratch, in code, spelled out*（视频）
- Andrej Karpathy · nanoGPT / micrograd
- Kingma & Ba, 2014 · *Adam: A Method for Stochastic Optimization*

---
⬅️ [04 Transformer Block](../04-transformer-block/) ｜ [目录](../../README.md) ｜ [06 采样与生成](../06-sampling/) ➡️
