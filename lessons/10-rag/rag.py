"""
第 10 课：从零实现一个最小 RAG 检索器（切块 + TF-IDF 向量 + 余弦相似度 + 拼 prompt）
只依赖 numpy。运行：python rag.py "会员积分可以换什么？"
真实项目把 embed() 换成神经网络 embedding 模型（如 bge、text-embedding-3），
把暴力搜索换成向量数据库（FAISS、Milvus、pgvector…），流程不变。
"""
import sys
import math
from collections import Counter
import numpy as np

DOCS = {
    "营业时间": "星河咖啡所有门店的营业时间为每天早上 7:30 至晚上 22:00。节假日营业时间不变，春节期间除夕下午 16:00 提前闭店，初一至初三正常营业。",
    "会员积分": "注册星河会员后，每消费 1 元积 1 分。积分可在小程序中兑换：200 分兑换一杯中杯美式，500 分兑换任意一杯饮品，1000 分兑换星河限定马克杯。积分有效期为 12 个月。",
    "退换政策": "饮品制作完成后如口味有问题，可在 30 分钟内凭小票到店免费重做一次。已开封的咖啡豆和周边商品不支持退换；未开封商品 7 天内可凭小票退款。",
    "外卖配送": "外卖配送范围为门店周边 3 公里以内，起送价 25 元。配送费 4 元，满 50 元免配送费。高峰时段（11:30-13:30）配送时间约 40 分钟。",
    "新品上市": "十一月新品为“桂花燕麦拿铁”，使用燕麦奶和自制桂花酱，限时供应至 12 月底，会员首杯八折。",
    "咖啡豆": "星河咖啡的意式拼配豆由 60% 巴西豆和 40% 埃塞俄比亚豆组成，中深烘焙，风味有坚果、可可和柑橘的味道。",
    "员工福利": "门店员工每个班次可免费领取一杯饮品，入职满一年可参加咖啡师进阶培训，培训费用由公司承担。",
}
STOP = set("的了是在和，。、：（）“”0123456789 -")


def chunk(text, size=60, overlap=10):
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def grams(s):
    c = [ch for ch in s if ch not in STOP]
    return c + [a + b for a, b in zip(c, c[1:])]


class TfidfIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        tfs = [Counter(grams(t)) for _, t in chunks]
        self.vocab = {g: i for i, g in enumerate(sorted({g for tf in tfs for g in tf}))}
        df = Counter(g for tf in tfs for g in tf)
        N = len(chunks)
        self.idf = np.array([math.log((N + 1) / (df[g] + 1)) + 1 for g in self.vocab])
        self.M = np.stack([self.embed_counts(tf) for tf in tfs])      # [n_chunks, vocab]

    def embed_counts(self, tf):
        v = np.zeros(len(self.vocab))
        for g, c in tf.items():
            if g in self.vocab:
                v[self.vocab[g]] = (1 + math.log(c)) * self.idf[self.vocab[g]]
        return v / (np.linalg.norm(v) or 1)

    def embed(self, text):
        return self.embed_counts(Counter(grams(text)))

    def search(self, query, k=2):
        scores = self.M @ self.embed(query)                               # 向量已归一化 → 点积 = 余弦
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]


def build_prompt(query, hits):
    ctx = "\n".join(f"[资料{j + 1}]（{title}）{text}" for j, (_, (title, text)) in enumerate(hits))
    return ("你是星河咖啡的客服助手。请只根据下面的资料回答问题，并注明引用了哪条资料。"
            "如果资料中没有答案，请回答“抱歉，我没有找到相关信息”。\n\n"
            f"{ctx}\n\n问题：{query}\n回答：")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "会员积分可以换什么？"
    chunks = [(title, c) for title, text in DOCS.items() for c in chunk(text)]
    index = TfidfIndex(chunks)
    print(f"{len(DOCS)} 篇文档 → {len(chunks)} 个块，词表 {len(index.vocab)} 维\n")
    hits = index.search(query, k=3)
    print(f"问题：{query}\n检索结果：")
    for s, (t, c) in hits:
        print(f"  {s:.3f}  [{t}] {c}")
    print("\n—— 拼好的 prompt ——\n" + build_prompt(query, hits[:2]))

    print("\n—— 关键词检索的局限 ——")
    for q in ["晚上几点关门？", "几点打烊？"]:
        s, (t, _) = index.search(q, k=1)[0]
        note = "  ← 相似度为 0，等于没找到，排第一只是并列时的顺序" if s == 0 else ""
        print(f"  {q:<10} → [{t}] 相似度 {s:.3f}{note}")
    print("“几点打烊”和资料没有任何共同的字（除了被过滤掉的停用字），TF-IDF 无能为力 —— 这就是需要语义 embedding 的原因。")
