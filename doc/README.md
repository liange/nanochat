# nanochat 中文技术文档

本目录是 nanochat 项目的中文技术文档，旨在讲清楚项目的**核心流程、核心类与核心设计**。

文档基于源码分析编写，所有结论均标注对应的源码位置。**引用格式是 `文件#符号`**（如 `nanochat/gpt.py#GPT.setup_optimizer`）而不是裸行号——符号名不会因为别人在文件上方插了几行代码而失效。

## 代码基准

| 项 | 值 |
|----|----|
| 基准 commit | `a2778de` |
| 对应版本特征 | `report` 模块与 WebUI 已删除；`MuonAdamW` 已合并单机/分布式；`gpt.py` 含推理成本建模 |

> **文档与代码不一致时以代码为准**，并按下面的流程修文档。

## 文档结构

| 文档 | 内容 |
|------|------|
| [01-架构总览.md](01-架构总览.md) | 项目定位、设计哲学、目录结构、核心概念（单一旋钮 `--depth`、精度管理、分布式、FA3/SDPA） |
| [02-核心流程.md](02-核心流程.md) | 端到端全流程：数据准备 → tokenizer → pretrain → SFT → RL → 评估/推理基准/对话 |
| [03-核心类与模块.md](03-核心类与模块.md) | 核心库各模块详解：GPT / MuonAdamW / Engine / Tokenizer / DataLoader / Checkpoint / Eval / FP8 / 沙箱执行 |
| [04-模型与训练机制.md](04-模型与训练机制.md) | **训练机制深入**：模型架构细节、Muon 六步内核、ZeRO-2 分布式优化器、精度/dtype、完整训练循环、数据加载、scaling laws 超参推导与调度 |
| [05-推理与评测.md](05-推理与评测.md) | **推理引擎深入**：KV Cache 布局、prefill + 多路并行采样、工具调用状态机、`infer_bench` / MBU / TTFT / TPOT、CORE / bpb / ChatCORE 评测 |

## 快速导航

- 想了解整体设计哲学与目录布局 → [01-架构总览](01-架构总览.md)
- 想跟着数据走一遍从零到对话的完整链路 → [02-核心流程](02-核心流程.md)
- 想查某个类/模块的职责与接口 → [03-核心类与模块](03-核心类与模块.md)
- **想搞懂训练是怎么跑起来的**（循环、梯度累积、调度、分布式通信、数据打包、超参推导）→ [04-模型与训练机制](04-模型与训练机制.md)
- **想搞懂推理引擎与评测**（KV cache、并行采样、工具调用、CORE/ChatCORE、成本基准）→ [05-推理与评测](05-推理与评测.md)

如果只读两篇，建议是 04 与 05——它们承载了 nanochat 最容易踩坑的部分（训练/推理一致性、mask 语义、分布式通信时机）。

## 阅读约定

- 所有命令默认从**仓库根目录**运行（nanochat 的入口都是 module）。
- 训练/评估脚本面向 Linux GPU 节点，CPU/MPS 仅作演示用途；`scripts/infer_bench.py` **只支持 CUDA 单卡**。
- 引用中的 `@Lnnn` 表示「该符号所在行」，是可选的附加校验；只有确实需要指认某一行时才用。
- 文中出现 `doc:no-check` 标记的地方是**故意列举已删除/不存在的路径**，校验脚本会跳过。

## 校验文档引用

文档里的每条引用都由 `check_citations.py` 校验（文件是否存在、符号是否还能找到）。改完文档请跑：

```bash
python doc/check_citations.py             # 校验，失败返回 1
python doc/check_citations.py --verbose   # 额外列出每条解析结果
python -m pytest tests/test_doc_citations.py -q
```

支持的引用语法：

| 写法 | 含义 |
|------|------|
| `nanochat/gpt.py#GPT.setup_optimizer` | **首选**：在该文件里解析这个符号 |
| `nanochat/gpt.py#GPT.setup_optimizer@L419` | 额外断言第 419 行落在该符号范围内 |
| `nanochat/gpt.py:419` | 旧的裸行号写法，只做范围检查（会随时间失效） |
| `nanochat/gpt.py:419 "def setup_optimizer"` | 裸行号 + 断言该行包含这段代码文本 |
| `README.md#Precision / dtype` | markdown 标题锚点（允许空格） |
| `check_citations.py` | 与本目录同级的文件（`doc/check_citations.py`） |

添加新内容时**优先用 `#符号`**。如果某个符号被重命名或删除，校验会立刻失败——这正是我们想要的信号，而不是让一条过期引用悄悄留在文档里。
