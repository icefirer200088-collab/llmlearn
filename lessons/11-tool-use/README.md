# 第 11 课 · Tool Use / Function Calling

> 👉 先打开 [`viz.html`](viz.html)，选一个问题，一步步点"下一步"。

## 1. 为什么需要工具

LLM 擅长理解和生成语言，但：

- **算不准**：大数乘法、精确统计
- **不知道实时信息**：天气、股价、今天的新闻
- **不能行动**：发邮件、查数据库、改文件

工具调用把这些能力"外包"给代码：**模型负责决定调用什么、参数是什么；你的程序负责执行**。

## 2. 完整流程

```
① 你把 [用户问题] + [工具定义] 发给模型
② 模型返回：一段文字 + tool_use 块 {name, input}      ← stop_reason = "tool_use"
③ 你的代码执行工具，得到结果
④ 把结果作为 tool_result 追加到 messages，再次调用模型
⑤ 模型根据结果给出最终回答（或继续调用工具）
```

以 Claude API 为例，`messages` 长这样：

```json
[
  {"role": "user", "content": "1234 × 5678 等于多少？"},
  {"role": "assistant", "content": [
      {"type": "text", "text": "我来用计算器算一下。"},
      {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "1234 * 5678"}}]},
  {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "t1", "content": "{\"result\": 7006652}"}]},
  {"role": "assistant", "content": [{"type": "text", "text": "结果是 7,006,652。"}]}
]
```

注意：**模型自己从不执行任何代码**，它只是输出了一段结构化文本。

## 3. 工具定义

```json
{
  "name": "get_weather",
  "description": "查询某个城市今天的天气，返回天气状况和温度（摄氏度）。",
  "input_schema": {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "城市名，例如 '上海'"}},
    "required": ["city"]
  }
}
```

这些定义会被放进 prompt。模型在后训练（第 07 课）中专门学过"什么时候、用什么格式输出工具调用"。

**写好工具描述**是最重要的工程实践：

- 说清楚**什么时候该用**、什么时候不该用
- 参数的格式、单位、例子
- 返回什么，出错时返回什么

## 4. 安全

- 永远不要对模型的输出直接 `eval()`（`tool_use.py` 里用 `ast` 写了安全的计算器）
- 有副作用的操作（付款、删除、发消息）应让用户确认
- 工具返回的内容（网页、邮件）可能包含恶意指令（**提示注入**）—— 把它们当作数据，而不是命令

## 5. 运行

```bash
python tool_use.py                       # 离线，规则模拟的模型
pip install anthropic
export ANTHROPIC_API_KEY=...
python tool_use.py --real "北京和广州今天哪里更热？温差是多少度的平方？"
```

用真实模型时，你会看到它自己决定先查两次天气、再调用计算器 —— 这已经是一个小 Agent 了。

## 6. MCP

当工具越来越多，每个应用都要重新写一遍集成很麻烦。**MCP（Model Context Protocol）** 是一个开放协议：工具提供方写一次 MCP 服务器，任何支持 MCP 的应用（Claude、IDE 等）都能直接使用。

## 7. 动手练习

1. 给 `tool_use.py` 加一个 `get_time` 工具，返回当前时间。
2. 让 `get_weather` 对未知城市返回错误，观察（真实）模型如何处理。
3. 思考：如果工具描述写得含糊，比如计算器的描述只写"计算"，会发生什么？

## 8. 延伸阅读

- Schick et al., 2023 · *Toolformer: Language Models Can Teach Themselves to Use Tools*
- Anthropic · Tool use 文档
- Model Context Protocol 官方文档（modelcontextprotocol.io）

---
⬅️ [10 RAG](../10-rag/) ｜ [目录](../../README.md) ｜ [12 ReAct Agent](../12-react-agent/) ➡️
