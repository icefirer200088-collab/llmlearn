# 第 12 课 · ReAct Agent

> 👉 先打开 [`viz.html`](viz.html)，挨个跑一遍三个任务，再勾选"不用工具，直接回答"对比。

## 1. 什么是 Agent

> **Agent = LLM + 工具 + 循环**

第 11 课的工具调用是"一问一答"。Agent 把它放进循环：模型自己决定下一步做什么、看结果、再决定，直到它认为任务完成。人只给目标，不规定步骤。

## 2. ReAct（Yao et al., 2022）

把**推理**（Reasoning）和**行动**（Acting）交替进行：

```
Thought: 需要两个城市的气温。先查北京。
Action: get_weather[北京]
Observation: {"city": "北京", "temp_c": 9}
Thought: 再查上海。
Action: get_weather[上海]
Observation: {"city": "上海", "temp_c": 18}
Thought: 温差 9°C，华氏温差 = 9 × 1.8
Action: calculator[(18 - 9) * 1.8]
Observation: 16.2
Final Answer: 温差 9°C，相当于 16.2°F 的温差。
```

- **Thought** 让模型先规划（第 09 课的思维链），比直接行动更准
- **Observation** 让模型基于真实信息推理，而不是编造
- 只靠推理不行动 → 幻觉；只行动不推理 → 乱撞

## 3. 实现要点

`react_agent.py` 约 100 行，就是一个完整可用的 Agent：

| 要点 | 做法 |
|---|---|
| 让模型只写到 Action 就停 | `stop_sequences=["Observation:"]`，否则它会自己编观察结果 |
| 解析行动 | 正则 `Action: (\w+)\[(.*)\]`；现代 API 直接用结构化的 tool_use |
| 出错处理 | 工具异常也作为 Observation 返回，让模型自己纠正 |
| 防止死循环 | 设置 `max_turns` |
| 安全 | 有副作用的工具（删除、付款、发送）先让人确认 |

```bash
python react_agent.py                    # 离线演示
python react_agent.py --real "北京和上海今天温差多少摄氏度？换算成华氏度的温差是多少？"
```

## 4. 从 ReAct 到现代 Agent

今天的 Agent（编程助手、深度研究、电脑操作等）在 ReAct 骨架上加了：

- **结构化工具调用**代替文本解析（第 11 课）
- **规划**：先列计划/待办清单，再逐项执行
- **反思**：完成后自查，发现问题再改
- **记忆管理**：上下文太长时压缩、摘要、存到外部（第 13 课）
- **更强的工具**：代码执行、文件系统、浏览器、MCP 服务器
- **推理模型**：经过 RL 训练，长时间思考再行动

一个经验法则：**能用固定流程（workflow）解决的就别用 Agent**。Agent 灵活，但更慢、更贵、更难预测；当步骤无法事先确定时才值得用。

## 5. 动手练习

1. 给 `react_agent.py` 加一个 `get_date` 工具，让 Agent 回答"距离元旦还有几天"。
2. 用 `--real` 运行，故意给一个工具无法完成的任务（如"查纽约天气"），观察模型如何处理失败。
3. 把文本格式的 ReAct 改写成第 11 课的结构化 tool_use 版本。

## 6. 延伸阅读

- Yao et al., 2022 · *ReAct: Synergizing Reasoning and Acting in Language Models*
- Shinn et al., 2023 · *Reflexion: Language Agents with Verbal Reinforcement Learning*
- Anthropic · *Building Effective Agents*

---
⬅️ [11 Tool Use](../11-tool-use/) ｜ [目录](../../README.md) ｜ [13 多 Agent、记忆与评估](../13-multi-agent-eval/) ➡️
