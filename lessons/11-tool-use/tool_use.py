"""
第 11 课：Tool Use（Function Calling）的完整循环
运行：python tool_use.py                    # 用规则模拟的模型，离线可跑
      python tool_use.py --real "问题"      # 用真实的 Claude API（需 pip install anthropic 并设置 ANTHROPIC_API_KEY）

核心循环：
    messages = [用户问题]
    while True:
        response = 模型(messages, tools)
        if 没有工具调用: 输出回答，结束
        执行每个工具调用 → 把结果作为 tool_result 追加到 messages
"""
import ast
import json
import operator
import os
import re
import sys

# ---------------- 1. 工具定义（名字 + 描述 + JSON Schema）----------------
TOOLS = [
    {
        "name": "calculator",
        "description": "计算一个数学表达式的精确结果。支持 + - * / ** 和括号。任何需要精确计算的数字运算都应使用此工具。",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "例如 '1234 * 5678'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "查询某个城市今天的天气，返回天气状况和温度（摄氏度）。",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，例如 '上海'"}},
            "required": ["city"],
        },
    },
]

WEATHER = {"上海": ("多云", 18), "北京": ("晴", 9), "广州": ("小雨", 24), "深圳": ("阴", 23)}


# ---------------- 2. 工具实现 ----------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def safe_eval(expr):
    """只允许数字和四则运算的安全求值（不要对模型输出直接用 eval！）"""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("不支持的表达式")
    return ev(ast.parse(expr, mode="eval").body)


def run_tool(name, args):
    try:
        if name == "calculator":
            return {"result": safe_eval(args["expression"])}
        if name == "get_weather":
            cond, temp = WEATHER[args["city"]]
            return {"city": args["city"], "condition": cond, "temp_c": temp}
        return {"error": f"未知工具 {name}"}
    except Exception as e:                       # 把错误也返回给模型，让它自己决定怎么办
        return {"error": str(e)}


# ---------------- 3. 规则模拟的“模型”（输出格式与 Claude API 相同）----------------
def fake_model(messages, tools):
    last = messages[-1]
    question = messages[0]["content"]
    if isinstance(last["content"], str):                       # 刚收到用户问题
        m = re.search(r"[\d\s+\-*/().]{3,}", question.replace("×", "*"))
        if m and re.search(r"\d\s*[*+\-/]\s*\d", m.group()):
            return [{"type": "text", "text": "我来用计算器算一下。"},
                    {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": m.group().strip()}}]
        cities = [c for c in WEATHER if c in question]
        if cities:
            return [{"type": "text", "text": "我来查一下天气。"}] + [
                {"type": "tool_use", "id": f"t{i}", "name": "get_weather", "input": {"city": c}} for i, c in enumerate(cities)]
        return [{"type": "text", "text": "你好！我可以帮你做精确计算或查询天气。"}]
    results = [json.loads(b["content"]) for b in last["content"]]  # 收到工具结果
    if "result" in results[0]:
        return [{"type": "text", "text": f"结果是 {results[0]['result']:,}。"}]
    return [{"type": "text", "text": "；".join(f"{r['city']}{r['condition']}，{r['temp_c']}°C" for r in results) + "。"}]


def real_model(messages, tools):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_tokens=1024, tools=tools, messages=messages,
    )
    out = []
    for b in resp.content:                       # 转成普通 dict，下一轮原样发回
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


# ---------------- 4. Agent 循环 ----------------
def chat(question, model=fake_model, max_turns=5):
    messages = [{"role": "user", "content": question}]
    print(f"👤 {question}")
    for _ in range(max_turns):
        content = model(messages, TOOLS)
        messages.append({"role": "assistant", "content": content})
        calls = [b for b in content if b["type"] == "tool_use"]
        for b in content:
            if b["type"] == "text" and b["text"]:
                print(f"🤖 {b['text']}")
        if not calls:
            break
        results = []
        for c in calls:
            out = run_tool(c["name"], c["input"])
            print(f"   🔧 {c['name']}({c['input']}) → {out}")
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": json.dumps(out, ensure_ascii=False)})
        messages.append({"role": "user", "content": results})
    return messages


if __name__ == "__main__":
    if "--real" in sys.argv:
        q = sys.argv[-1] if len(sys.argv) > 2 else "北京和广州今天哪里更热？温差是多少度的平方？"
        chat(q, model=real_model)
    else:
        for q in ["1234 × 5678 等于多少？", "北京和广州今天天气怎么样？", "你好"]:
            msgs = chat(q)
            print()
        print("一次完整对话的 messages 结构：")
        print(json.dumps(chat("1234 × 5678 等于多少？"), ensure_ascii=False, indent=2))
