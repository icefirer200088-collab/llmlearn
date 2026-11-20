"""
第 12 课：从零写一个 ReAct Agent
运行：python react_agent.py                      # 离线：用脚本化的假模型演示循环
      python react_agent.py --real "任务"        # 真实 Claude API（pip install anthropic，设置 ANTHROPIC_API_KEY）

ReAct 的输出格式（纯文本，任何能续写的模型都能用）：
    Thought: 我应该……
    Action: 工具名[参数]
    Observation: （由程序填入）
    ……重复……
    Final Answer: ……
"""
import ast
import json
import operator
import os
import re
import sys

# ---------------- 工具 ----------------
WEATHER = {"北京": ("晴", 9), "上海": ("多云", 18), "深圳": ("阴", 23)}
KB = {
    "会员积分": "每消费 1 元积 1 分。200 分兑换中杯美式，500 分兑换任意饮品，1000 分兑换星河限定马克杯。",
    "外卖配送": "配送范围 3 公里，起送价 25 元，配送费 4 元，满 50 元免配送费。",
    "营业时间": "每天 7:30 至 22:00。",
}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}


def calculator(expr):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("只支持数字和 + - * /")
    return str(round(ev(ast.parse(expr, mode="eval").body), 4))


def get_weather(city):
    if city not in WEATHER:
        raise ValueError(f"找不到城市“{city}”，请使用中文城市名，可选：{'、'.join(WEATHER)}")
    cond, t = WEATHER[city]
    return json.dumps({"city": city, "condition": cond, "temp_c": t}, ensure_ascii=False)


def search_docs(query):
    for k, v in KB.items():
        if k in query or query in k:
            return f"[{k}] {v}"
    return "没有找到相关文档。可搜索的主题：" + "、".join(KB)


TOOLS = {
    "get_weather": (get_weather, "查询中国城市今天的天气，参数是中文城市名"),
    "calculator": (calculator, "计算数学表达式，参数如 (18 - 9) * 1.8"),
    "search_docs": (search_docs, "搜索星河咖啡内部文档，参数是主题关键词"),
}

SYSTEM = """你是一个会使用工具解决问题的助手。可用工具：
{tools}

严格按以下格式输出，每次只输出一个 Thought 和一个 Action，然后停下等待 Observation：
Thought: 你对下一步的思考
Action: 工具名[参数]

当你已经能回答时，输出：
Thought: 我已经知道答案了
Final Answer: 最终答案"""


# ---------------- 模型 ----------------
def scripted_llm(script):
    """离线演示用：按顺序返回写好的“思考+行动”。真实 Agent 里这里是 LLM。"""
    it = iter(script)
    return lambda prompt: next(it)


def claude_llm(prompt):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_tokens=500,
        system=SYSTEM.format(tools="\n".join(f"- {k}: {d}" for k, (_, d) in TOOLS.items())),
        messages=[{"role": "user", "content": prompt}],
        stop_sequences=["Observation:"],          # 关键：让模型写完 Action 就停，不要自己编观察结果
    )
    return resp.content[0].text.strip()


# ---------------- Agent 循环 ----------------
def run_agent(task, llm, max_turns=8, verbose=True):
    transcript = f"任务：{task}\n"
    for turn in range(1, max_turns + 1):
        reply = llm(transcript)
        transcript += reply + "\n"
        if verbose:
            print(f"\n—— 第 {turn} 轮 ——\n{reply}")
        if "Final Answer:" in reply:
            return reply.split("Final Answer:")[-1].strip()
        m = re.search(r"Action:\s*(\w+)\[(.*)\]", reply)
        if not m:
            obs = "格式错误：请按 “Action: 工具名[参数]” 输出。"
        elif m.group(1) not in TOOLS:
            obs = f"没有工具 {m.group(1)}，可用：{', '.join(TOOLS)}"
        else:
            try:
                obs = TOOLS[m.group(1)][0](m.group(2).strip())
            except Exception as e:
                obs = f"错误：{e}"                    # 错误也作为观察返回，让模型自己纠正
        transcript += f"Observation: {obs}\n"
        if verbose:
            print(f"Observation: {obs}")
    return "超过最大轮数，任务未完成。"


if __name__ == "__main__":
    if "--real" in sys.argv:
        task = sys.argv[-1] if len(sys.argv) > 2 else "北京和上海今天温差多少摄氏度？换算成华氏度的温差是多少？"
        print("✅", run_agent(task, claude_llm))
    else:
        demo = scripted_llm([
            "Thought: Need the weather in Shenzhen.\nAction: get_weather[Shenzhen]",
            "Thought: 工具要求中文城市名，改用“深圳”重试。\nAction: get_weather[深圳]",
            "Thought: 23°C，换算华氏 = 23 × 1.8 + 32。\nAction: calculator[23 * 1.8 + 32]",
            "Thought: 我已经知道答案了\nFinal Answer: 深圳今天阴，23°C，即 73.4°F。",
        ])
        answer = run_agent("深圳今天多少华氏度？", demo)
        print("\n✅ 最终答案:", answer)
        print("\n（这里的 Thought/Action 是写好的脚本，Observation 是工具真实执行的结果。加 --real 可以让真实模型自己决定每一步。）")
