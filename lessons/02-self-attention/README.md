# 第 02 课 · Self-Attention

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. 要解决什么问题？

第 01 课之后，每个 token 都有了一个向量，但这个向量**和上下文无关**："苹果"在"吃苹果"和"苹果发布会"里是同一个向量。

我们希望每个 token 能根据句子里的其他 token **更新自己**。Self-Attention 就是做这件事的。

## 2. 直觉：软查字典

| | 普通字典 | 注意力 |
|---|---|---|
| 查询 | 一个 key，精确匹配 | 一个 query 向量 |
| 匹配 | 相等 / 不等 | 和每个 key 算点积（相似度） |
| 取出 | 一个 value | 所有 value 按 softmax 权重加权平均 |

因为全程都是可导的（没有"取最大"这种离散操作），所以可以用梯度下降学习。

## 3. 公式

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

具体步骤（`T` 个 token，每个是 `d_model` 维）：

1. **投影**：`Q = X·W_Q`，`K = X·W_K`，`V = X·W_V`。同一个 token 扮演三个角色：
   - Q（query）：我在找什么？
   - K（key）：我能提供什么？
   - V（value）：如果你选中我，我给你什么内容？
2. **打分**：`S = Q·Kᵀ`，得到 `T×T` 矩阵，`S[i][j]` 是 token i 对 token j 的关注程度。
3. **缩放**：除以 `√d_k`。
4. **归一化**：对每一行做 softmax，得到注意力权重 `A`，每行和为 1。
5. **汇总**：`Z = A·V`，每个 token 的新向量是所有 V 的加权平均。

## 4. 为什么要除以 √d？

如果 q、k 每一维都是 N(0,1)，那么 `q·k` 的方差是 `d`。d = 512 时，分数的标准差约 22，softmax 后几乎变成 one-hot，梯度接近 0，训练就卡住了。除以 `√d` 把方差拉回 1。

运行 `attention.py` 最后一段可以亲眼看到这个数字。

## 5. 因果掩码（Causal Mask）

GPT 这类模型是"从左到右"生成文本的，训练时第 i 个位置**不能偷看**后面的词。做法是把 `S` 的上三角设为 `-∞`，softmax 之后这些位置权重就是 0。在 `viz.html` 里勾选"因果掩码"可以看到效果。

## 6. 复杂度

`S` 是 `T×T`，所以计算量和显存都是 **O(T²)**。上下文从 4K 扩到 1M，注意力矩阵会大 6 万倍 —— 这就是第 08 课要讲的长上下文优化（FlashAttention 等）的出发点。

## 7. 动手练习

1. 在 `viz.html` 里把 d 调到 64、取消"除以 √d"，观察注意力矩阵变得多"尖"。
2. 修改 `attention.py`：把同一句话的 token 顺序打乱，输出 Z 会怎么变？（提示：只是行的顺序跟着变，**内容完全一样**。这说明 Self-Attention 根本不知道顺序 → 第 03 课。）
3. 思考：为什么要用三个不同的矩阵 W_Q、W_K、W_V，而不是直接用 X·Xᵀ？

## 8. 延伸阅读

- Vaswani et al., 2017 · *Attention Is All You Need*
- Jay Alammar · *The Illustrated Transformer*
- 3Blue1Brown · *Attention in transformers, visually explained*

---
⬅️ [01 Tokenization](../01-tokenization-embedding/) ｜ [目录](../../README.md) ｜ [03 多头 + 位置编码](../03-multihead-position/) ➡️
