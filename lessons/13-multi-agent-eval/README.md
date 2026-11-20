# 第 13 课 · 多 Agent、记忆与评估

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. 多 Agent：Orchestrator–Workers

一个领队（orchestrator）拆分任务，多个 worker 并行执行，各自有**独立的上下文**，最后由领队汇总。

**好处**
- 并行，快
- 每个 worker 的上下文干净、聚焦，不会被其他子任务的信息干扰
- 总共能处理的信息量远超单个上下文窗口

**代价**
- token 消耗成倍增加（每个 worker 都要读任务说明、各自探索）
- 协调难：子任务描述不清，worker 会重复劳动或跑偏
- 不适合强依赖的任务（多个 Agent 同时改同一份代码容易冲突）

**适合**：广度优先的调研、可拆分的批量处理。

其他常见模式：**流水线**（A 的输出是 B 的输入）、**评审者–执行者**（一个写、一个挑错）、**路由**（按问题类型分给专门的 Agent）。

## 2. 记忆

上下文窗口是 Agent 的"工作记忆"，再大也会满。

| 策略 | 做法 | 风险 |
|---|---|---|
| 截断 | 只保留最近 N 条 | 早期的关键信息直接丢失 |
| 摘要 / 压缩 | 把旧对话压成摘要 | 摘要可能漏掉看似不重要的细节 |
| 长期记忆 | 把关键事实写到外部存储（文件、数据库、向量库），需要时检索 | 要判断"什么值得记"，检索可能不准 |
| 笔记 / 待办文件 | Agent 自己维护一个笔记文件，随时读写 | 需要模型有好的习惯 |

`viz.html` 的例子：用户一开始说了对花生过敏，截断策略下这条信息被丢掉，模型就可能推荐含花生的菜。

## 3. 评估

LLM 的输出是随机的，Agent 又是多步的，一步错可能步步错。**没有评估，就不知道改动是变好还是变坏。**

### pass@k vs pass^k

- **pass@k** = k 次里**至少成功一次**的概率。适合能自动验证、可以多试的场景（代码 + 单元测试）
- **pass^k** = k 次**全部成功**的概率。衡量**可靠性**，适合直接面向用户的 Agent

单次成功率 70% 的 Agent：pass@8 ≈ 99.99%，pass^8 ≈ 5.8%。

无偏估计（Codex 论文）：采样 n 次、c 次正确，`pass@k = 1 − C(n−c, k) / C(n, k)`。

### 怎么判定对错

1. **程序判定**（优先）：精确匹配、正则、单元测试、检查最终状态（文件是否创建、数据库是否更新）
2. **LLM 当评委**：开放式回答用清晰的评分标准（rubric）让另一个模型打分；要抽样人工检查评委本身准不准
3. **人工评估**：最贵也最可靠，用来校准前两种

### 实践建议

- 从 20~50 个**真实的**失败案例开始建评估集，不要等"完美"
- 既测**结果**（答案对不对），也看**过程**（有没有调用危险工具、绕了多少弯路、花了多少 token）
- 每次改 prompt、换模型、加工具，都跑一遍

## 4. 运行

```bash
python multi_agent_eval.py
```

## 5. 动手练习

1. 在 `multi_agent_eval.py` 里给 worker 加一个随机失败，让领队对失败的子任务重试。
2. 实现一个"Agent 自己维护笔记文件"的记忆策略。
3. 用第 12 课的 `react_agent.py --real` 跑 10 次同一个任务，统计 pass@1 和 pass^5。

## 6. 延伸阅读

- Anthropic · *How we built our multi-agent research system*
- Anthropic · *Effective context engineering for AI agents*
- Chen et al., 2021 · *Evaluating Large Language Models Trained on Code*（pass@k）
- Yao et al., 2024 · *τ-bench*（pass^k）
- Zheng et al., 2023 · *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*

---

## 🎓 结业

文字 → token → 向量 → 注意力 → 位置编码与多头 → Transformer → 训练 → 采样 → 对齐 → 规模 → Prompt → RAG → 工具 → Agent → 多 Agent、记忆与评估。

你已经从第一个矩阵乘法走到了一个能自主完成任务的系统。下一步：挑一个你真正想解决的问题，做一个 Agent，给它写评估，然后不断迭代。

⬅️ [12 ReAct Agent](../12-react-agent/) ｜ [目录](../../README.md)
