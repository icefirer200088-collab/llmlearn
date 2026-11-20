"""
第 13 课：多 Agent 编排 + 记忆管理 + 评估（pass@k / pass^k）
运行：python multi_agent_eval.py
"""
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor


# ================ 1. Orchestrator–Workers ================
def worker(subtask):
    """一个 worker Agent：在自己独立的上下文里完成子任务（这里用 sleep 模拟搜索和调用 LLM）"""
    time.sleep(subtask["cost"])
    return {"model": subtask["model"], "result": f"{subtask['model']} 的调研结果", "tokens": 6000}


def orchestrator(task, subtasks, parallel=True):
    print(f"领队：拆分任务「{task}」→ {len(subtasks)} 个子任务")
    t0 = time.time()
    if parallel:
        with ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
            results = list(pool.map(worker, subtasks))
    else:
        results = [worker(s) for s in subtasks]
    n_agents = len(subtasks) if parallel else 1
    # 规划 + 每个 Agent 读一遍任务说明 + 各子任务的搜索阅读 + 汇总
    tokens = 1200 + 2000 * n_agents + sum(r["tokens"] for r in results) + 2500
    print(f"  {'5 个 worker 并行' if parallel else '1 个 Agent 串行'}：耗时 {time.time() - t0:.2f}s，总 token ≈ {tokens:,}")
    return results


# ================ 2. 记忆管理 ================
class Memory:
    """三种策略：截断 / 摘要 / 长期记忆（关键事实存外部，按需检索）"""

    def __init__(self, window, strategy):
        self.window, self.strategy = window, strategy
        self.history, self.facts = [], {}

    def add(self, role, text):
        self.history.append((role, text))
        if self.strategy == "ltm":                    # 真实系统里由 LLM 判断哪些值得记
            if "我叫" in text: self.facts["姓名"] = text.split("我叫")[1].strip("。")
            if "过敏" in text: self.facts["过敏"] = text.split("对")[1].split("过敏")[0]

    def context(self, query):
        if self.strategy == "trunc":
            return self.history[-self.window:]
        recent = self.history[-(self.window - 1):]
        old = self.history[:-(self.window - 1)]
        if self.strategy == "sum":                    # 真实系统里由 LLM 生成摘要
            summary = "【摘要】" + "；".join(t for _, t in old if len(t) < 14)[:40]
            return [("system", summary)] + recent
        relevant = {k: v for k, v in self.facts.items() if k in "姓名过敏" and ("菜" in query or "名字" in query)}
        return [("system", f"【长期记忆】{relevant}")] + recent


# ================ 3. 评估 ================
def pass_at_k(n, c, k):
    """无偏估计（Chen et al., 2021）：从 n 次采样（c 次正确）中随机取 k 次，至少一次正确的概率"""
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n, c, k):
    """pass^k：随机取 k 次，全部正确的概率（衡量可靠性）"""
    return math.comb(c, k) / math.comb(n, k) if c >= k else 0.0


def run_eval(agent, cases, trials=20, k=5, seed=0):
    random.seed(seed)
    print(f"{'用例':<14}{'成功/尝试':>10}{'pass@1':>9}{f'pass@{k}':>9}{f'pass^{k}':>9}")
    for name, inp, check in cases:
        c = sum(check(agent(inp)) for _ in range(trials))
        print(f"{name:<14}{c:>5}/{trials:<4}{pass_at_k(trials, c, 1):>9.2f}{pass_at_k(trials, c, k):>9.2f}{pass_hat_k(trials, c, k):>9.2f}")


if __name__ == "__main__":
    print("=" * 20, "多 Agent", "=" * 20)
    subs = [{"model": m, "cost": c} for m, c in [("Llama", .32), ("Qwen", .26), ("Mistral", .22), ("DeepSeek", .36), ("Gemma", .24)]]
    orchestrator("调研 5 个开源模型", subs, parallel=False)
    orchestrator("调研 5 个开源模型", subs, parallel=True)

    print("\n" + "=" * 20, "记忆", "=" * 20)
    conv = ["你好，我叫林小雨。", "我对花生过敏。", "明天上海天气？", "周末去杭州玩有推荐吗？",
            "西湖附近有什么好吃的？", "坐高铁要多久？", "好的谢谢。"]
    q = "推荐一道适合我的菜？"
    for st in ["trunc", "sum", "ltm"]:
        m = Memory(window=4, strategy=st)
        for t in conv:
            m.add("user", t)
        ctx = " | ".join(t for _, t in m.context(q))
        ok = "花生" in ctx
        print(f"{st:<6} {'✅' if ok else '⚠️ 丢失过敏信息'}  上下文: {ctx}")

    print("\n" + "=" * 20, "评估", "=" * 20)

    def flaky_agent(x):                     # 一个模拟的 Agent：不同任务成功率不同
        p = {"天气": .95, "计算": .9, "换算": .6, "多步": .4}[x]
        return "OK" if random.random() < p else "WRONG"

    cases = [(f"{k}类任务", k, lambda out: out == "OK") for k in ["天气", "计算", "换算", "多步"]]
    run_eval(flaky_agent, cases)
    print("\n注意 pass@5 和 pass^5 的差距：能“试出来”不等于“靠得住”。")
