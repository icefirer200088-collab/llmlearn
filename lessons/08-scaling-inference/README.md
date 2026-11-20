# 第 08 课 · Scaling Law 与高效推理

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. Scaling Law

Kaplan et al.（2020）发现：loss 和参数量 N、数据量 D、算力 C 之间是平滑的**幂律关系**，跨越 7 个数量级都成立。这意味着可以用小模型的实验**预测**大模型的效果，再决定要不要花几千万美元去训练。

Chinchilla（Hoffmann et al., 2022）拟合出：

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},\quad C \approx 6ND$$

- `E` 是不可约的 loss（语言本身的熵）
- 给定算力，有一个最优的 N 和 D 分配
- 结论：参数和数据应**大致等比例**增长，经验值每个参数约 20 个 token。GPT-3（175B / 300B token）严重"欠训练"

运行 `scaling.py` 可以看到不同算力下的最优分配。

### 为什么现在的小模型都用超多数据？

Chinchilla 最优只考虑了**训练**成本。但模型训练一次、推理上亿次。Llama-3 8B 用 15T token（每参数近 2000 个 token），训练时"浪费"了一些算力，换来一个推理便宜得多、效果接近大模型的小模型。

## 2. 显存账本

| 场景 | 每参数字节 | 7B 模型 |
|---|---|---|
| 推理 FP16/BF16 | 2 | 13 GB |
| 推理 INT8 | 1 | 6.5 GB |
| 推理 INT4 | 0.5 | 3.3 GB |
| 训练（Adam，混合精度） | ~16 | ~104 GB |

再加上 KV Cache（第 06 课）和激活值。训练大模型需要把参数、梯度、优化器状态切分到多张卡上（ZeRO / FSDP、张量并行、流水线并行）。

## 3. 量化

`q = round(w / s)`，`s = max|w| / (2^(b−1) − 1)`。

- **离群值**是最大敌人：一个大权重把刻度 s 撑大，其余权重被挤成 0
- **分组量化**：每 32~128 个权重一个刻度（GPTQ、AWQ、GGUF 的 Q4_K 等都是这个思路）
- 更聪明的方法：GPTQ 用二阶信息补偿误差；AWQ 保护对激活影响大的通道
- 经验：INT8 几乎无损，INT4 轻微损失，INT3 以下损失明显

`scaling.py` 的结果：INT4 不分组时 85% 的权重变成 0，相对误差 75%；分 32 一组后误差降到 11%。

## 4. MoE（混合专家）

把 FFN 换成 N 个专家 + 一个路由器，每个 token 只激活 top-k 个专家。

- 总参数大（容量大），激活参数小（计算少）
- Mixtral 8×7B：总 46.7B，激活 12.9B
- DeepSeek-V3：总 671B，激活 37B
- 难点：负载均衡（别让所有 token 都挤到同一个专家）、显存仍需放下全部参数

## 5. 长上下文与 FlashAttention

注意力矩阵是 T×T：128K 上下文时一个头就要 32 GB。

- **FlashAttention**：分块计算 + 在线 softmax，不在显存里存完整的 T×T 矩阵，结果完全相同，显存 O(T)，还更快（减少显存读写）
- **位置编码外推**：RoPE 频率缩放（NTK、YaRN），让短上下文训练的模型用于长上下文
- **稀疏/线性注意力、滑动窗口**：降低 O(T²) 本身

## 6. 其他推理加速

- **投机解码（Speculative Decoding）**：小模型先猜几个 token，大模型一次验证，结果和大模型单独生成一致
- **连续批处理（Continuous Batching）**：不同请求动态拼批
- **蒸馏**：用大模型的输出训练小模型

## 7. 动手练习

1. 用 `scaling.py` 算一下：如果你只有一张 H100 跑一个月（约 1e21 FLOPs 的有效算力），最优模型多大？
2. 在 `viz.html` 的量化实验里，找出"INT4 + 不分组 + 有离群值"和"INT4 + 32 分组"的误差差多少倍。
3. 实现 AWQ 的简化版思路：先把某些通道乘以一个系数再量化，最后除回来。

## 8. 延伸阅读

- Kaplan et al., 2020 · *Scaling Laws for Neural Language Models*
- Hoffmann et al., 2022 · *Training Compute-Optimal Large Language Models*（Chinchilla）
- Dao et al., 2022 · *FlashAttention*
- Frantar et al., 2022 · *GPTQ*；Lin et al., 2023 · *AWQ*
- Fedus et al., 2021 · *Switch Transformers*；Jiang et al., 2024 · *Mixtral of Experts*

---
⬅️ [07 对齐](../07-alignment/) ｜ [目录](../../README.md) ｜ [09 Prompt 与 ICL](../09-prompt-icl/) ➡️
