# PTCG non-AR V7 8 卡 PPO 训练包

该包训练的是同一个同步 PPO 策略，不是 8 个互不相干的单卡模型。每张卡独立收集官方 C++ 引擎对局，PPO 反向阶段通过 NCCL/DDP 平均梯度；只有 rank 0 保存 Best、定期 checkpoint、联赛快照和汇总指标。每个 GPU rank 默认再启动 4 个常驻 CPU 引擎 worker，由该 rank 的 GPU 集中批量推理，避免原实现由一个 Python 进程串行推进 128 个环境。

## 机器要求

- Linux x86_64，默认目标为 8 张 NVIDIA A100 40GB，驱动可支持目标 PyTorch CUDA runtime；
- Python 3.10 或 3.11；
- 至少约 20 GB 可用磁盘空间用于长期 checkpoint 和日志；
- 默认共 32 个 CPU rollout worker，建议至少 32 个物理 CPU 核心；少核机器可用 `--rollout-workers-per-rank 2 --rollout-envs-per-worker 64`。

## 一条命令启动

解压后进入目录：

```bash
chmod +x setup_and_run_8gpu.sh
./setup_and_run_8gpu.sh --output-dir runs/dragapult_8gpu_1m
```

如果当前 Python 已安装 CUDA PyTorch 和 `orjson`，脚本直接使用当前环境；缺少依赖时会在包内创建 `.venv` 并安装。

默认参数为 8 卡、每卡每轮 256 局、8,000 轮：

```text
2048 局/全局轮 × 8,000 轮 = 16,384,000 局
每卡 minibatch 4096，全局有效 minibatch 32768（针对 A100 40GB）
每卡 4 个 C++ 环境 worker × 每 worker 32 个环境
每 50 轮保存 checkpoints/update-XXXX.pt，默认共约 160 份
best.pt 只在两阶段晋级门通过后更新
last.pt 在完整训练结束时保存
```

实时监控：

```bash
tail -f runs/dragapult_8gpu_1m/train.log
watch -n 2 nvidia-smi
```

本地单卡正式基准（8 worker × 16 环境、10 ms 聚合）为 256/256 有效局、0 硬失败、1,763 SPS；旧串行实现约 463 SPS。实现同时按 checkpoint 哈希共享只读对手模型，并用等价的快速实时特征 collator；同机优化前的并行版约 912 SPS。这个数字只证明单卡并行 rollout 路径，8 卡总吞吐仍应以目标机器首轮 `aggregate_decisions_per_second` 为准。

4096 每卡 minibatch 已在 32GB 级别显存上跑通；A100 40GB 仍应在首轮监控峰值显存。若目标机同时被其他任务占用，可用 `--minibatch-size-per-rank 2048` 保守启动。

训练被中断后，从最近 checkpoint 继续，并把 `--updates` 设为最终累计轮数：

```bash
./setup_and_run_8gpu.sh \
  --output-dir runs/dragapult_8gpu_1m \
  --updates 8000 \
  --resume runs/dragapult_8gpu_1m/checkpoints/update-0100.pt
```

关键输出：

- `best.pt`：通过镜像初筛和对手池复筛的当前 Best；
- `last.pt`：最终策略；
- `checkpoints/update-XXXX.pt`：含模型、AdamW、联赛窗口和冷却状态，可完整续训；
- `metrics.jsonl`：每轮全局 8 卡 rollout/优化/晋级指标；
- `distributed_launch_manifest.json`：实际命令、BC 哈希和计划对局数。

默认最近环境为 2026-08-13 Top23：80% 真实比例并做窗口逆加权、10% BC 镜像、10% 历史池。8 卡每轮产生 2048 局，所以失败/晋级冷却自动换算为 3/8 个全局更新，分别约等于 5,000/15,000 局。

如果目标机 CPU 很多，可尝试每卡 8 个 worker：

```bash
./setup_and_run_8gpu.sh \
  --output-dir runs/dragapult_8gpu_1m \
  --rollout-workers-per-rank 8 \
  --rollout-envs-per-worker 16
```
