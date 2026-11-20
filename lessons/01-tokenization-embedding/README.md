# 第 01 课 · Tokenization 与 Embedding

> 👉 先打开 [`viz.html`](viz.html) 玩一遍，再回来读这里。

## 1. 为什么要分词？

神经网络只能处理数字。所以 LLM 处理文本的第一步是：

```
"the lowest"  ──分词──▶  ["the", "_", "low", "est_"]  ──查编号──▶  [512, 3, 87, 140]  ──查表──▶  4 个向量
```

这个"切块"的单位就叫 **token**。

## 2. 三种切分方式

| 方式 | 例子 `lowest` | 优点 | 缺点 |
|------|--------------|------|------|
| 字符级 | `l o w e s t` | 词表小（几十~几千），永不 OOV | 序列很长，单个字符几乎没有语义 |
| 单词级 | `lowest` | 序列短，每个 token 有意义 | 词表爆炸；没见过的词只能 `[UNK]` |
| **子词级（BPE）** | `low est` | 常见词完整，罕见词拆开，平衡二者 | 需要先"训练"分词器 |

GPT、Llama、Claude 等现代模型都用子词分词（BPE 或其变体）。

## 3. BPE 算法

1. 把每个词拆成字符，末尾加词尾标记 `</w>`（页面里写作 `_`）；
2. 统计所有**相邻符号对**的出现次数；
3. 把出现最多的一对合并成一个新符号，加入词表；
4. 重复 2–3，直到词表达到目标大小（GPT-2 约 5 万次合并）。

训练得到的产物就是一张**有序的合并规则表**。编码新文本时，按同样的顺序依次应用这些规则即可。

运行 [`bpe.py`](bpe.py)：

```bash
python bpe.py
```

你会看到 `e+s → es`、`es+t → est`……模型自己"发现"了 `est`、`low`、`new` 这些词根，而我们从没告诉它英语语法。

### 关键细节：byte-level BPE

如果基础词表是"语料里出现过的字符"，那遇到生僻字、emoji 还是会 OOV。GPT-2 起的做法是让基础词表 = **256 个字节**，任何 UTF-8 文本都能被表示，彻底消灭 `[UNK]`。这也是为什么一个汉字有时会占 2~3 个 token。

## 4. Embedding：编号 → 向量

分词之后每个 token 有一个整数 id。模型内部有一个矩阵 **E**，形状为 `[词表大小, d_model]`：

```python
x = E[token_id]          # 就是取第 token_id 行，没有任何计算
```

- GPT-2 small：`E` 是 50257 × 768
- 这张表一开始是随机数，随训练被梯度更新
- 训练完后，语义相近的 token 向量方向相近 —— 这就是页面里"国王 − 男人 + 女人 ≈ 王后"的来源

**相似度**用余弦衡量：`cos(a, b) = a·b / (|a|·|b|)`。

> ⚠️ 注意：embedding 只表示"这个 token 是什么"，**不含位置信息**。"狗咬人"和"人咬狗"查出的是同一组向量。怎么解决？见第 03 课"位置编码"。

## 5. 动手练习

1. 在 `viz.html` 的实验 02 里把语料换成中文（用空格分隔词，比如 `机器 学习 机器 翻译 深度 学习`），看看 BPE 会合并出什么。
2. 修改 `bpe.py`，让它在合并次数 = 0、5、20 时分别输出 `"lowest newer"` 的 token 数，画出"合并次数 vs 序列长度"的关系。
3. 思考：词表越大越好吗？（提示：Embedding 矩阵大小、softmax 输出层的计算量、罕见 token 训练不足。）

## 6. 延伸阅读

- Sennrich et al., 2016 · *Neural Machine Translation of Rare Words with Subword Units*（BPE 用于 NLP 的原始论文）
- Andrej Karpathy · *Let's build the GPT Tokenizer*（视频 + minbpe 代码）
- OpenAI `tiktoken` 库：可以直接看 GPT 系列如何切分你的文本

---
⬅️ [返回目录](../../README.md) ｜ [02 Self-Attention](../02-self-attention/) ➡️
