# 第 06 课 · 采样与生成、KV Cache

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. 自回归生成

```python
ids = prompt
while True:
    logits = model(ids)[-1]          # 只取最后一个位置的输出
    next_id = decode(logits)         # 解码策略
    if next_id == EOS: break
    ids.append(next_id)
```

模型本身只给概率分布，**怎么从分布里挑一个**是解码策略的事。

## 2. 解码策略

| 策略 | 做法 | 特点 |
|---|---|---|
| 贪心 | 每次取概率最大的 | 确定、可复现；容易重复、呆板 |
| 温度 T | `softmax(logits / T)` | T<1 更保守，T>1 更随机 |
| Top-k | 只在前 k 个里采样 | 截掉长尾的离谱候选 |
| Top-p（核采样） | 只在累计概率达到 p 的最小集合里采样 | 候选数随分布自适应：模型很确定时只剩 1~2 个，不确定时保留很多 |

常见组合：`temperature=0.7, top_p=0.9`。代码题、数学题偏向低温度；写诗、头脑风暴偏向高温度。

运行 `sampling.py` 可以看到每种策略采样 1000 次的分布：

```
贪心         好:1000
T=0.5      好:717  不错:163  晴朗:56 ...
T=2        好:246  不错:192 ... 香蕉:10
top-p=0.9  好:442  不错:226  晴朗:138  热:108  冷:86
```

其他策略：beam search（保留多条候选序列，翻译任务常用）、min-p、重复惩罚（repetition penalty）等。

## 3. KV Cache

生成第 t 个 token 时，注意力要用到前面所有 token 的 K、V。由于因果掩码，**前面 token 的 K、V 永远不会变**，所以可以缓存起来：

- 不缓存：每步重新算全部 K/V，总共 `n(n+1)/2` 行
- 缓存：每步只算新 token 的 1 行，总共 `n` 行

`sampling.py` 验证：两种方式输出**完全一致**，KV Cache 快了数倍（序列越长差得越多）。

### 代价：显存

```
KV Cache 大小 = 2 × 层数 × KV 头数 × 头维度 × 序列长度 × batch × 字节数
```

Llama-2 7B，FP16，4K 上下文：`2×32×32×128×4096×2 B = 2 GB`（每个请求！）。

于是有了一系列优化：

- **GQA / MQA**：多个 query 头共享一组 K/V（Llama-3 70B 只有 8 组 KV 头），缓存缩小 4~8 倍
- **KV Cache 量化**：用 INT8 / INT4 存
- **PagedAttention（vLLM）**：像操作系统管理内存页一样管理 KV Cache，减少碎片
- **MLA（DeepSeek-V2）**：把 K/V 压缩到低维潜空间再缓存

## 4. Prefill 与 Decode

推理分两个阶段：

1. **Prefill**：一次性处理整个 prompt，并行计算，填满 KV Cache —— 计算密集
2. **Decode**：一次生成一个 token —— 访存密集（每步都要读全部权重和 KV Cache）

这就是为什么"首 token 延迟"和"每秒生成 token 数"是两个不同的指标。

## 5. 动手练习

1. 在 `viz.html` 里设 T=0.05（贪心），反复点"一直生成到结束"，结果每次都一样吗？
2. 给 `sample()` 加上 **min-p**：只保留概率 ≥ `min_p × 最大概率` 的候选。
3. 计算 Llama-3 70B 在 128K 上下文、batch=8 时 KV Cache 需要多大显存。

## 6. 延伸阅读

- Holtzman et al., 2019 · *The Curious Case of Neural Text Degeneration*（top-p 的论文）
- Kwon et al., 2023 · *Efficient Memory Management for LLM Serving with PagedAttention*（vLLM）
- Shazeer, 2019 · *Fast Transformer Decoding: One Write-Head is All You Need*（MQA）

---
⬅️ [05 mini-GPT](../05-mini-gpt/) ｜ [目录](../../README.md) ｜ [07 对齐](../07-alignment/) ➡️
