# 第 10 课 · RAG：检索增强生成

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. 为什么需要 RAG

LLM 的知识有三个问题：

1. **过时**：停留在训练截止日期
2. **缺失**：不知道你公司的内部文档、你的个人笔记
3. **幻觉**：不知道的时候也可能编一个看似合理的答案

重新训练又贵又慢。RAG 的思路：**提问时把相关资料检索出来，放进上下文**，让模型"开卷考试"。

## 2. 流程

```
离线：文档 → 切块 → 每块用 embedding 模型变成向量 → 存入向量数据库
在线：问题 → 向量化 → 找最相似的 k 块 → 拼进 prompt → LLM 生成回答
```

### 切块（Chunking）

- 块太小：信息被切碎，缺上下文
- 块太大：混入无关内容，稀释相似度，浪费上下文窗口
- 常用 200~800 token，带 10%~20% 重叠；更好的做法是按段落、标题等**语义边界**切

### 向量化（Embedding）

用专门训练的 embedding 模型把文本变成一个向量，使**语义相近的文本向量也相近**（还记得第 01 课的"国王 − 男人 + 女人"吗？）。

本课为了能在浏览器和纯 numpy 里跑，用字符 n-gram 的 TF-IDF 代替。它只能做"字面匹配"：`rag.py` 里演示了"几点打烊"完全检索不到"营业时间"—— 语义 embedding 就是为了解决这个问题。

### 检索

余弦相似度：`cos(q, d) = q·d / (|q||d|)`。向量提前归一化后就是点积，可以用矩阵乘法一次算完。数据量大时用近似最近邻（ANN）索引：FAISS、HNSW 等。

### 生成

Prompt 里要说清楚：只依据资料回答、没有就说不知道、标注引用来源。

## 3. 进阶技巧

| 问题 | 方法 |
|---|---|
| 用户提问和文档用词不同 | 语义 embedding；查询改写；HyDE（先让 LLM 写个假想答案再去检索） |
| 召回了但排序不准 | 两阶段：向量召回 top-50 → 交叉编码器（reranker）重排出 top-5 |
| 专有名词、编号搜不到 | 混合检索：向量 + BM25 关键词 |
| 问题需要多条资料综合 | 多次检索、让 Agent 自己决定查什么（第 12 课） |
| 块丢了上下文（"它"指谁？） | 给每块加上文档标题/摘要（contextual retrieval） |

## 4. RAG vs 长上下文 vs 微调

- **长上下文**：资料不多时，直接全部塞进去最简单
- **RAG**：资料多、经常更新、需要引用来源
- **微调**：要改变模型的**行为/风格/格式**，而不是注入事实

## 5. 运行

```bash
python rag.py "会员积分可以换什么？"
python rag.py "外卖送多远"
```

## 6. 动手练习

1. 在 `viz.html` 里把块大小调到 20 和 160，分别问"会员积分可以换什么？"，对比检索到的内容。
2. 给 `rag.py` 加一个简单的"查询改写"：把"打烊"替换成"营业时间"这类同义词表，看能否解决检索失败。
3. 进阶：用 `sentence-transformers` 加载一个中文 embedding 模型替换 `TfidfIndex.embed`，再测"几点打烊？"。

## 7. 延伸阅读

- Lewis et al., 2020 · *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Gao et al., 2023 · *Precise Zero-Shot Dense Retrieval without Relevance Labels*（HyDE）
- Anthropic · *Introducing Contextual Retrieval*

---
⬅️ [09 Prompt 与 ICL](../09-prompt-icl/) ｜ [目录](../../README.md) ｜ [11 Tool Use](../11-tool-use/) ➡️
