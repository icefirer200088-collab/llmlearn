"""
第 09 课：Prompt 模板 + 一个能"上下文学习"的极简模型（归纳头算法）
运行：python icl.py
"""


# ---------------- 1. Few-shot prompt 构造 ----------------
def build_prompt(task, examples, query, system=None, cot=False):
    parts = []
    if system:
        parts.append(system)
    parts.append(task)
    for x, y in examples:
        parts.append(f"输入：{x}\n输出：{y}")
    if cot:
        parts.append("先一步步分析，最后一行写“输出：<答案>”。")
    parts.append(f"输入：{query}\n输出：")
    return "\n\n".join(parts)


# ---------------- 2. 归纳头：[A][B] … [A] → [B] ----------------
def induction_predict(tokens):
    """找到最后一个 token 上一次出现的位置，预测它后面跟着的 token。"""
    cur = tokens[-1]
    for j in range(len(tokens) - 2, -1, -1):
        if tokens[j] == cur:
            return tokens[j + 1]
    return None


def induction_predict_fuzzy(tokens, n=3):
    """更强一点：匹配最近 n 个 token 组成的后缀（类似模型利用更长的上下文）。"""
    for k in range(min(n, len(tokens) - 1), 0, -1):
        suffix = tokens[-k:]
        for j in range(len(tokens) - k - 1, -1, -1):
            if tokens[j:j + k] == suffix:
                return tokens[j + k], k
    return None, 0


if __name__ == "__main__":
    print("—— Few-shot prompt ——")
    p = build_prompt(
        task="把英文单词翻译成中文。",
        examples=[("apple", "苹果"), ("cat", "猫"), ("book", "书")],
        query="water",
        system="你是一个简洁的翻译助手。",
    )
    print(p)

    print("\n—— 归纳头：只靠上下文就能“学会”一个从没见过的映射 ——")
    # 一个编造的密码表：模型参数里不可能有这个知识，只能从上下文里学
    ctx = "苹果 → X7 ; 香蕉 → Q2 ; 西瓜 → M9 ; 香蕉 →".split()
    print("上下文:", " ".join(ctx))
    nxt, k = induction_predict_fuzzy(ctx)
    print(f"预测下一个 token: {nxt}   （匹配了长度为 {k} 的后缀）")

    seq = "张三 的 电话 是 138 ， 李四 的 电话 是 159 ， 张三 的 电话 是".split()
    print("\n上下文:", " ".join(seq))
    print("只看 1 个 token 的归纳头预测:", induction_predict(seq), "  ← 错！张三的电话应该是 138")
    nxt, k = induction_predict_fuzzy(seq, n=4)
    print(f"匹配更长后缀的预测: {nxt}  （匹配了 {k} 个 token：“{' '.join(seq[-k:])}”）")
    print("\n这说明：上下文越长、匹配越精细，ICL 越准 —— 真实模型的注意力能做这种多 token 的模糊匹配。")
