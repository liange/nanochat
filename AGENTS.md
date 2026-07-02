# AGENTS.md

nanochat 是一个极简的全栈 LLM 训练框架（tokenization → pretrain → SFT → RL → eval → inference → chat UI），目标是在单 GPU 节点上以 <$100 训练出 GPT-2 级模型。代码刻意保持精简、可读、可 fork，**避免**框架化的配置对象 / 模型工厂 / 冗长 if-else。改动请遵循同一风格。

## 环境与安装

- Python 3.10，依赖由 [uv](https://docs.astral.sh/uv/) 管理（见 `pyproject.toml`、`uv.lock`）。
- 安装时**必须且只能**选一个 extra：`uv sync --extra gpu`（CUDA）或 `uv sync --extra cpu`（CPU/MPS）。两者在 `tool.uv.conflicts` 中互斥。
- 开发依赖（pytest、matplotlib、ipykernel、transformers、python-dotenv）走 group：`uv sync --extra gpu --group dev`。
- torch 版本钉死 `2.9.1`，CUDA 走 `cu128` index。运行前激活 venv：`source .venv/bin/activate`。
- `runs/*.sh` 是 bash 脚本，面向 Linux GPU 节点（依赖 `curl`、`screen` 等）。

## 常用命令

所有训练 / 评估 / 聊天入口都是 **module**，须从仓库根运行。

测试（无需 GPU，用 mock 模型）：
```
python -m pytest tests/                       # 全部
python -m pytest tests/test_engine.py -v      # 单文件
python -m pytest -m "not slow"                # 跳过 slow marker（定义在 pyproject.toml）
```
`test_attention_fallback.py` 的 FA3 对比类在无 Hopper GPU 时自动 skip；`TestSDPAOnly` 按设备自适应 dtype，可在 CPU/CUDA/MPS 跑。

训练 / 评估 / 对话：
```
python -m nanochat.dataset -n 8               # 下载预训练数据分片
python -m scripts.tok_train                    # 训练 tokenizer
python -m scripts.tok_eval                     # 评估 tokenizer 压缩率
python -m scripts.base_train                  # 预训练（单 GPU）
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24   # 分布式
python -m scripts.base_eval                    # 评估 base 模型（CORE / bpb / 采样）
python -m scripts.chat_sft                      # SFT 微调
python -m scripts.chat_eval -- -i sft         # 评估 chat 模型
python -m scripts.chat_rl                       # RL
python -m scripts.chat_cli -p "Why is the sky blue?"   # CLI 对话
python -m scripts.chat_web                      # WebUI（ChatGPT 风格）
python -m nanochat.report reset|generate       # 重置 / 生成训练报告（写 report.md）
```
端到端 GPT-2 复现：`bash runs/speedrun.sh`（8×H100，约 3 小时）。CPU/MPS 演示：`bash runs/runcpu.sh`。

## 架构要点

- **单一复杂度旋钮 `--depth`**：Transformer 层数自动决定宽度、head 数、LR 调度、训练步数、weight decay 等所有超参。任何改动必须对**所有 depth** 都成立，不能只针对某一档。GPT-2 能力大约在 d24–d26。
- **包结构**：`nanochat/` 是核心库（`gpt.py` 模型、`engine.py` 推理、`tokenizer.py`、`optim.py` AdamW+Muon、`dataloader.py`、`core_eval.py`/`loss_eval.py`、`checkpoint_manager.py`、`report.py`、`flash_attention.py`、`fp8.py`、`execution.py`）；`scripts/` 是各阶段可执行入口；`tasks/` 是评测/训练任务（arc/gsm8k/mmlu/humaneval/smoltalk/spellingbee，经 `common.py` 组合成 mixture/sequence）；`runs/` 是 shell 编排；`tests/` 是 pytest；`dev/` 是实验脚本与日志。
- **`NANOCHAT_BASE_DIR`** 环境变量（默认 `~/.cache/nanochat`）决定所有中间产物（数据分片、tokenizer、checkpoint、报告）的存放位置。
- **DDP**：用 `torchrun` 启动；省略 `torchrun` 时自动切到单 GPU + 梯度累积，结果基本一致。分布式时务必 `export OMP_NUM_THREADS=1`。
- **`--` 分隔符**：`torchrun ... -m scripts.base_train -- --depth=24` 中 `--` 把 torchrun 自身参数与脚本参数隔开，不要漏。
- **`print0` / `log0`**：只在 rank 0 输出 / 落盘，避免多 rank 重复。
- **Inference 引擎**（`nanochat/engine.py`）：基于 KV cache，**只处理 token id 序列，不碰 tokenization**。生成有 seed 可复现；`temperature=0` 与 seed 无关地确定。修改模型前向时务必同步 Engine 的 KV cache 路径，否则推理与训练不一致（仓库有对照测试守住）。

## 精度与硬件陷阱（重要）

- **不使用 `torch.amp.autocast`**。精度由 `nanochat/common.py` 中的全局 `COMPUTE_DTYPE` 显式管理：master 权重 fp32，自定义 `Linear`（`nanochat/gpt.py`）在前向时 cast 到 `COMPUTE_DTYPE`；embedding 直接存 `COMPUTE_DTYPE` 省显存。
- **`NANOCHAT_DTYPE`** 环境变量可覆盖默认（`bfloat16`/`float16`/`float32`）。默认按硬件自动检测：SM 80+（A100/H100）→ bf16；前 Ampere（V100/T4）→ fp32；CPU/MPS → fp32。
- **Flash Attention 3** 仅在 Hopper（sm90）+ bf16 下可用；Blackwell / Ada / MPS / CPU 走 PyTorch SDPA 回退。`HAS_FA3`/`USE_FA3` 在 `nanochat/flash_attention.py` 中一次性解析，可用模块级 `_override_impl` 强制切换（测试用）。
- **滑动窗口注意力**（`--window-pattern`，默认 `SSSL`）依赖 FA3；无 FA3 时 SDPA 不支持滑动窗口，会严重拖慢，建议 `--window-pattern L`。
- **`--fp8`** 开启 FP8 训练（需 H100+ 与 torchao，默认 tensorwise scaling）；无 fp8 硬件省略即可，bf16 精度更高，可适当降低 `--target-param-data-ratio`。
- **`--device-batch-size`**：理想值 32（配合 seq 2048 × 8 GPU = 524288 总 batch）。OOM 时按 2 的幂递减（16/8/4/2/1），脚本自动算梯度累积补齐。
- **奇数 depth（如 d25）不推荐**：head 维度等 sizing 算不整齐。
- fp16 训练自动启用 `GradScaler`（`base_train.py`）；SFT 支持，RL 暂不支持。

## wandb

- `--run=dummy`（默认）禁用 wandb；`WANDB_RUN=xxx bash runs/speedrun.sh` 启用。非 rank 0 进程一律用 `DummyWandb`。
- 关键监控指标：`val_bpb`（按 step / time / flops）、`core_metric`、`train/mfu`、`train/tok_per_sec`。注意**更换数据集后 `val_bpb` 不再可比**。

## 约定

- 仓库未配置 lint / typecheck / formatter，**只有 pytest**。验证改动靠跑测试 +（若涉及训练）小规模 d12/d16 迭代验证。
- 提 PR 时**披露 LLM 贡献**（README Contributing 明确要求）；速度跑分类 PR 还需给出 `total_training_time` 与 CORE > 0.256525，并尽量在 GPT-2 附近多档 depth 验证。
- `.gitignore` 忽略 `.venv/`、`wandb/`、`report.md`、`eval_bundle/`、`CLAUDE.md`、`.env`（密钥放这里）。
