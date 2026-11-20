# llmlearn · 从零可视化学 LLM

> 从 Transformer 原理一路走到 Agent。每一课 = **讲义** + **可交互的可视化页面** + **能跑的最小代码**。

## 怎么学

1. 先读每课的 `README.md`，建立直觉；
2. 打开 `viz.html`（浏览器直接双击即可，无需安装任何东西），亲手拖动、输入、单步执行；
3. 运行 `*.py`，看懂每一行，再改一改。

所有代码只依赖 Python 标准库或 `numpy`，前期**不用 PyTorch**，确保你看到的每个数字都是自己算出来的。

## 课程路线

| # | 主题 | 你会搞懂 | 状态 |
|---|------|---------|------|
| 01 | [Tokenization 与 Embedding](lessons/01-tokenization-embedding/) | 文字如何变成数字；BPE；向量空间 | ✅ |
| 02 | Self-Attention | Q/K/V 是什么，注意力矩阵怎么算 | ⏳ |
| 03 | 多头注意力 + 位置编码 | 为什么要多头；模型怎么知道顺序 | ⏳ |
| 04 | Transformer Block | 残差、LayerNorm、FFN、因果掩码 | ⏳ |
| 05 | 从零写一个 mini-GPT | 前向传播 + 反向传播 + 训练循环 | ⏳ |
| 06 | 采样与生成 | 温度、top-k、top-p、KV Cache | ⏳ |
| 07 | 预训练 → SFT → RLHF / DPO | 一个"聊天模型"是如何炼成的 | ⏳ |
| 08 | Scaling 与高效推理 | Scaling Law、量化、MoE、长上下文 | ⏳ |
| 09 | Prompt 与 In-Context Learning | 为什么提示词有效 | ⏳ |
| 10 | RAG | 向量检索 + 生成 | ⏳ |
| 11 | Tool Use / Function Calling | 让模型调用外部工具 | ⏳ |
| 12 | Agent：ReAct 循环 | 思考-行动-观察，从零写一个 Agent | ⏳ |
| 13 | 多 Agent、记忆与评估 | 规划、记忆、协作、怎么评测 | ⏳ |

详细的每课目标见 [ROADMAP.md](ROADMAP.md)。

## 目录结构

```
llmlearn/
├── index.html                 # 课程首页（可用 GitHub Pages 托管）
├── ROADMAP.md
└── lessons/
    └── 01-tokenization-embedding/
        ├── README.md          # 讲义
        ├── viz.html           # 交互可视化
        └── bpe.py             # 从零实现 BPE
```

## 在线浏览

在仓库 Settings → Pages 里选择 `main` 分支根目录，即可通过 `https://<你的用户名>.github.io/llmlearn/` 访问所有可视化页面。
