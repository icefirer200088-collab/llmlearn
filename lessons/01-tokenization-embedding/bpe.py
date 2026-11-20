"""
第 01 课：从零实现 BPE（Byte Pair Encoding）分词器
只用 Python 标准库。运行：python bpe.py

核心思想一句话：
    反复找出语料里"出现次数最多的相邻符号对"，把它合并成一个新符号。
    合并 N 次，词表就多出 N 个子词。
"""
from collections import Counter

END = "</w>"  # 词尾标记：让 "low" 末尾的 w 和 "lower" 中间的 w 区分开


def pre_tokenize(text: str) -> Counter:
    """按空白切成单词，每个单词拆成字符序列 + 词尾标记，并统计词频。"""
    words = Counter(text.split())
    return Counter({tuple(w) + (END,): c for w, c in words.items()})


def count_pairs(vocab: Counter) -> Counter:
    """统计所有相邻符号对的出现次数（按词频加权）。"""
    pairs = Counter()
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge(vocab: Counter, pair: tuple) -> Counter:
    """把语料中所有出现的 pair 合并成一个符号。"""
    a, b = pair
    new_vocab = Counter()
    for symbols, freq in vocab.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_vocab[tuple(out)] += freq
    return new_vocab


def train(text: str, num_merges: int, verbose: bool = True):
    vocab = pre_tokenize(text)
    merges = []
    for step in range(1, num_merges + 1):
        pairs = count_pairs(vocab)
        if not pairs:
            break
        best, freq = pairs.most_common(1)[0]
        if freq < 2:  # 只出现一次的对没必要合并
            break
        vocab = merge(vocab, best)
        merges.append(best)
        if verbose:
            print(f"第 {step:2d} 次合并: {best[0]!r:>8} + {best[1]!r:<8} -> {''.join(best)!r:<10} (出现 {freq} 次)")
    return merges


def encode(word: str, merges: list) -> list:
    """用学到的合并规则，按学习顺序依次应用到新单词上。"""
    symbols = list(word) + [END]
    for a, b in merges:
        i, out = 0, []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


if __name__ == "__main__":
    corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
    print("训练语料:", corpus, "\n")
    merges = train(corpus, num_merges=10)

    print("\n用学到的规则切分新词（包括训练时没见过的词）:")
    for w in ["lowest", "newer", "wider", "slow"]:
        print(f"  {w:<8} -> {encode(w, merges)}")

    # —— Embedding：token id → 向量，本质就是查表 ——
    import random
    random.seed(0)
    # 完整词表 = 所有基础字符 + 词尾标记 + 每次合并产生的新符号
    base = sorted(set(corpus.replace(" ", ""))) + [END]
    tokens = base + ["".join(p) for p in merges]
    token_to_id = {t: i for i, t in enumerate(tokens)}
    dim = 4
    embedding = [[round(random.uniform(-1, 1), 2) for _ in range(dim)] for _ in tokens]

    print(f"\n词表（{len(tokens)} 个 token）:", token_to_id)
    print(f"Embedding 矩阵形状: {len(tokens)} x {dim}（每行是一个 token 的向量，训练时会被梯度更新）")
    for word in ["lowest", "slow"]:
        print(f"\n  '{word}' 的查表过程:")
        for t in encode(word, merges):
            if t in token_to_id:
                print(f"    {t!r:<10} id={token_to_id[t]:<3} 向量={embedding[token_to_id[t]]}")
            else:
                print(f"    {t!r:<10} 不在词表中！（这就是 OOV 问题，GPT 用 byte-level BPE 解决）")
