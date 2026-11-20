# 第 09 课 · Prompt 与 In-Context Learning

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. Prompt 的本质

对模型来说，Prompt 只是一串 token，它做的仍然是"预测接下来最可能的内容"。写 Prompt 就是**设计一个上下文，让你想要的答案成为最可能的续写**。

一个好 Prompt 的常见组成：

| 组件 | 作用 |
|---|---|
| 角色 / 系统提示 | 设定身份、语气、边界 |
| 背景信息 | 模型不知道你知道的事：用途、受众、约束 |
| 任务说明 | 清楚、具体、无歧义 |
| 示例（few-shot） | 展示任务和答案的格式 |
| 思考要求 | 给模型"草稿纸" |
| 输出格式 | JSON / 列表 / 只输出一个词…… |

实用原则：

- **像给一个聪明但不了解情况的新同事写说明**：缺的背景要补上
- 示例要覆盖各种情况，格式一致；示例格式往往比数量更重要
- 指令之间不要互相矛盾（"一步步分析"和"只输出类别"冲突）
- 复杂任务拆成多步，或用 XML 标签分隔不同部分

## 2. In-Context Learning（上下文学习）

GPT-3 论文（2020）的核心发现：不更新任何参数，只在 prompt 里放几个例子，大模型就能完成新任务。

- **zero-shot**：只给任务描述
- **few-shot**：再给几个"输入 → 输出"示例

### 一种机制：Induction Head

Anthropic 的研究（Olsson et al., 2022）找到了一个具体电路：

1. **前一 token 头**：在每个位置写入"我前面的 token 是谁"
2. **归纳头**：当前 token 是 A 时，找"前一个 token 是 A"的位置，把那里的 token B 复制过来

实现了 `[A][B] … [A] → [B]` 的模式补全。研究发现归纳头在训练中**突然形成**，同时 ICL 能力突然提升。

`icl.py` 用算法实现了这个机制，让它从上下文中"学会"一个编造的密码表：

```
上下文: 苹果 → X7 ; 香蕉 → Q2 ; 西瓜 → M9 ; 香蕉 →
预测下一个 token: Q2
```

## 3. 思维链（Chain-of-Thought）

"让我们一步步思考"能显著提升推理题的准确率。直觉：

- 每生成一个 token，模型做的计算量是固定的
- 直接给答案 = 在一次前向计算里完成所有推理
- 写出中间步骤 = 每一步多一次完整计算，且中间结果在上下文里可以"回看"

今天的**推理模型**（先长时间思考再回答）就是把这个思路通过强化学习（第 07 课）做到了极致。

## 4. 动手练习

1. 在 `viz.html` 实验 02 里自己编一个有重复模式的序列，验证归纳头的预测。
2. 修改 `icl.py`，构造一个只匹配 1 个 token 会出错、匹配 2 个 token 就对的例子。
3. 如果你有 API key，拿同一个分类任务分别试 zero-shot、3-shot、3-shot + CoT，对比效果。

## 5. 延伸阅读

- Brown et al., 2020 · *Language Models are Few-Shot Learners*（GPT-3）
- Wei et al., 2022 · *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
- Olsson et al., 2022 · *In-context Learning and Induction Heads*
- Anthropic · Prompt engineering 文档

---
⬅️ [08 Scaling 与推理](../08-scaling-inference/) ｜ [目录](../../README.md) ｜ [10 RAG](../10-rag/) ➡️
