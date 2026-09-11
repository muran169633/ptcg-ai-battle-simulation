# PTCG AI Battle Challenge Simulation

基于 **Pokémon Trading Card Game（PTCG / 宝可梦集换式卡牌游戏）** 规则引擎打造的 AI 战斗智能体模拟项目。

该项目面向竞技对战场景，以**纯 JAX 规则引擎**驱动对局推演，结合真实对战数据完成**行为克隆（Behavior Cloning）**与 **PPO 强化学习**训练，最终生成可部署的单卡策略提交物，用于离线对战验证、联赛模拟与赔率分析。

---

## ✨ 特性

- **纯 JAX 规则引擎**：固定 shape 的 JAX PyTree 状态机、合法动作对齐官方 `SelectType` / `SelectContext`、`jit + vmap + lax.scan` 批量推演。
- **分布式 PPO 训练**：支持多卡（含 8×A100 目标）数据并行，NCCL/DDP 梯度平均，常驻 CPU 引擎 worker 并行采样。
- **行为克隆（BC）管线**：从真实/克隆对局数据构建行为策略，作为 PPO 的初始化或混合基线。
- **规则正确性**：官方 C++ 引擎逐决策差分验证，覆盖终局原因、效果栈、触发栈与 KO 队列。
- **提交打包**：一键生成 Kaggle/平台兼容的 `tar.gz` 提交物与校验清单（manifest、sha256、validation）。
- **实验追踪**：大量 `tools/` 下的分析、审计、构建脚本，留存完整实验记录与中文报告。

---

## 📁 目录结构

```
.
├── tools/                        # 核心代码：训练 / 评估 / 分析 / 打包脚本
│   └── local_seeded_engine/      # 离线种子规则引擎（导出用 C++ 片段）
├── data/decks/                   # 样本卡组定义（CSV）
├── dataset/                      # 原始数据集（本地保留，未上传）
├── artifacts/                    # 训练产物/checkpoint（本地保留，未上传）
├── submissions/                  # 提交打包产物（本地保留，未上传）
├── tests/                        # 单元测试
├── submission_templates/         # 可部署策略模板（PPO / Marnie / Alakazam 等）
├── notes/                        # 开发笔记与交接模板
├── reports/                      # 分析报告
└── *.md                          # 训练报告、架构说明、使用文档
```

> 大型数据产物（`data/`、`dataset/`、`artifacts/`、`submissions/`、`build/`）仅保留在本机，未包含在本仓库中。

---

## 🚀 快速开始

### 环境要求

- Python 3.10 / 3.11
- JAX + XLA（GPU 训练）
- PyTorch + CUDA（8 卡 PPO 训练目标为 NVIDIA A100 40GB）
- 官方 PTCG 离线引擎（用于差分验证，仅离线 oracle 使用，不进入训练热路径）

### 单卡 / 通用运行

先查看顶层 README 了解引擎架构：

- [`PTCG_JAX_ENGINE_README.md`](./PTCG_JAX_ENGINE_README.md) —— 纯 JAX 规则引擎架构说明

### 8 卡 PPO 训练包

依赖清单见 `portable_8gpu_requirements.txt`，一条命令启动（详见 [`PORTABLE_8GPU_README.md`](./PORTABLE_8GPU_README.md)）：

```bash
chmod +x setup_and_run_8gpu.sh
./setup_and_run_8gpu.sh --output-dir runs/dragapult_8gpu_1m
```

---

## 🧪 训练与提交流程

1. **行为克隆（BC）**：由真实对局数据训练基线策略。
2. **PPO 强化学习**：以 BC 策略为初始化，在自对弈 / 联赛环境中继续优化。
3. **评估与审计**：`tools/` 下的 `audit_*.py`、`analyze_*.py` 提供统计门禁与一致性校验。
4. **打包提交**：从 `submission_templates/` 生成符合平台契约的提交物。

---

## 📚 训练报告

项目保留了完整的中文训练与消融报告，例如：

- `PPO_TRAINING_REPORT.md` · `PPO_V2` · `PPO_V5` … `PPO_V9_QUOTA_REPORT.md`
- `BC_DATA_PIPELINE.md` · `BC_TRAINING_REPORT.md`
- `MARNIE_BC_RETRAIN_REPORT_20260813.md`
- `KAGGLE_SUBMISSION_REPORT.md`

---

## ⚠️ 免责声明

本项目为**技术研究与教育用途**的模拟练习，与 The Pokémon Company / Nintendo 无任何从属关系。Pokémon 相关名称与素材版权归其各自所有者所有。请勿将其用于商业用途或参与违反平台规则的比赛。

---

## 📄 License

仅供学习与研究使用。请在使用前遵守宝可梦官方对于其 IP 的使用规范。