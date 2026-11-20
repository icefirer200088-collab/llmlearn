# 学习路线图

整体分四个阶段：**看懂 → 写出 → 训好 → 用好**。

## 阶段一：看懂 Transformer（01–04）

- **01 Tokenization 与 Embedding**：字符 / 词 / 子词三种切分；BPE 合并算法；token id → 向量查表；余弦相似度。
- **02 Self-Attention**：Q·Kᵀ 打分 → 缩放 → softmax → 加权求和 V；可视化注意力热力图。
- **03 多头注意力 + 位置编码**：多头 = 多个子空间并行看；正弦位置编码、RoPE 的旋转直觉。
- **04 Transformer Block**：Pre-LN 结构、残差流（residual stream）、FFN、因果掩码。

## 阶段二：写出一个 GPT（05–06）

- **05 mini-GPT**：用 numpy 手写前向 + 反向传播，在莎士比亚小语料上训练字符级模型，可视化 loss 曲线。
- **06 采样与生成**：贪心 / 温度 / top-k / top-p 的分布变化动画；KV Cache 为什么能加速。

## 阶段三：从基座到助手（07–08）

- **07 对齐**：预训练、指令微调 SFT、奖励模型、PPO 与 DPO 的直觉对比。
- **08 Scaling 与推理工程**：Scaling Law、量化、MoE、FlashAttention 与长上下文。

## 阶段四：用好 LLM，走向 Agent（09–13）

- **09 Prompt / ICL**：few-shot、思维链、为什么有效。
- **10 RAG**：切块、向量化、检索、重排、拼接上下文。
- **11 Tool Use**：函数签名 → 模型输出结构化调用 → 执行 → 回填结果。
- **12 ReAct Agent**：从零写一个能思考-行动-观察的循环。
- **13 多 Agent / 记忆 / 评估**：规划、长期记忆、子 Agent 协作、Agent 评测方法。
