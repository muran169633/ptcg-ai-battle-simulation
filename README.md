# PTCG AI Battle Challenge Simulation

基于 **Pokémon Trading Card Game（PTCG / 宝可梦集换式卡牌游戏）** 规则引擎打造的 AI 战斗智能体方案 —— 本方案为**银牌级别（Silver-medal）** 的竞技解。

方案核心是 **BC（行为克隆）+ PPO（强化学习）两阶段训练**：先用真实对战数据克隆专家决策，再在纯 JAX 规则引擎上以官方 C++ 引擎做强化学习优化，最终生成可部署的单卡提交物，用于离线对战、联赛模拟与赔率分析。

---

## 🏆 方案亮点（银牌方案）

- **两阶段范式**：真实对局专家克隆 → PPO 强化学习，而非单一监督学习。
- **纯 JAX 规则引擎**：固定 shape PyTree 批量推演，官方 C++ 引擎仅作离线 oracle 差分验证，不进入训练热路径。
- **可控的训练池组合**：训练池由「近期真实元分布 + 困难样本挖掘 + 历史/镜像池」按比例混合，兼顾分布一致性与抗局快泛化。
- **可审计的提交物**：一键打包 `tar.gz` + manifest + sha256 + validation，轻量可部署。

---

## 🧬 技术方案：BC + PPO

### 阶段一：行为克隆（BC）

用官方每日 episode 数据克隆**排行榜 Top 专家**的决策行为，作为 PPO 的初始化与安全基线。

**BC 数据池构成：**

- 来源：官方 Episodes Index **最近 7 天**每日对局，用当前排行榜 **Top-20 队伍**过滤专家决策；
- 时间切分（chronological）：前 5 天训练、第 6 天验证、第 7 天测试，衡量次日分布漂移；
- **样本权重**：胜局 `1.0`、败局前 60% `0.75`、败局后段 `0.25`；
- 正向对齐 `step[t-1].observation → step[t].action`，只保存 agent 可见 observation，避免隐藏信息泄漏；
- 生成轻量分片 ZIP（每 50k 决策一个 JSONL shard），按目标牌组/专家过滤，避免反复扫描完整数据。

**模型（Orbit 神经 BC）：** 实体 Transformer —— 全局决策状态、可见卡牌、玩家摘要、近期公开日志编码为实体；每个合法动作编码为 Option Token；联合注意 + Pointer actor 对合法选项打分；count head + masked logits 处理弹性多选；value head 预热后用终局胜负充当 PPO critic。

**门槛：** exact action-set accuracy（非仅 top-1）≥ `0.75` 才进入 PPO。

### 阶段二：PPO（尤以 league 强化学习）

以 BC actor 为初始化，在官方引擎上做终局奖励强化学习：

- **奖励**：严格终局稀疏奖励 —— `win=1`，`loss/draw=0`，非终局一律 `0`，不使用稠密奖励；
- **单学习方 league**：每局只有本策略控制一个 seat，另一方是冻结对手，避免双侧自对弈的期望梯度抵消；
- 保留 BC **KL anchor**，随训练衰减，避免初期破坏已学会的合法操作；
- 加入独立 BC replay（独立 AdamW），持续强化专家操作。

---

## 🎯 PPO 训练池构成

PPO 的强度高度依赖环境/对手池的分布。本项目两种主流训练配置的池构成如下：

### ① mode-AR（8×A100 分布训练，主力配置）

环境池按比例混合，每局独立抽样：

| 比例 | 构成 | 作用 |
|---|---:|---|
| **70%** | 近期 day meta（recent-day meta） | 贴近线上真实的元分布 |
| **20%** | 逆窗口困难样本挖掘（inverse-window hard mining） | 对抗瓶颈/压轴局面，提升鲁棒性 |
| **10%** | 当前策略自对弈（current-policy self-play） | 挖掘策略自身弱点 |

每轮 8 卡 2,048 局官方引擎对局；候选每 5 updates 评估一次、1000 局 champion gate，需胜率 `≥ 54%` 才替换 incumbent。

### ② league PPO（历史配置，V7/V2）

- **对手池**：原始 Marnie BC（永久保留）+ **跨牌组 BC**（Dragapult/Lumen、袋兽 Kangaskhan，提供牌组与局面分布多样性）+ **历史 PPO 快照**；
- **V7 环境池**：80% 真实比例（recent-day Top23，窗口逆加权）+ 10% BC 镜像 + 10% 历史池；
- BC 对手使用 canonical action order 评估；PPO rollout 保留随机采样顺序计算正确 ordered log-prob；
- 最终提交推理使用 canonical ascending action order。

> 训练池的核心思想：**少用随机噪声，多用近线真实分布 + 定向困难样本**，在有限 GPU 预算下把提升集中在有效局面。

---

## ✨ 特性

- **纯 JAX 规则引擎**：固定 shape 的 JAX PyTree 状态机、合法动作对齐官方 `SelectType` / `SelectContext`、`jit + vmap + lax.scan` 批量推演、官方引擎逐决策差分验证。
- **分布式 PPO**：8×A100 数据并行、NCCL/DDP 梯度平均、每 rank 常驻 CPU 引擎 worker 并行采样。
- **行为克隆管线**：实体 Transformer、exact accuracy 门槛、可复现的 shard 化 BC 数据。
- **提交打包**：一键生成平台兼容 `tar.gz` + manifest + sha256 + validation 校验。

---

## 📁 目录结构

```
.
├── tools/                        # 核心代码：训练 / 评估 / 分析 / 打包
│   └── local_seeded_engine/      # 离线种子规则引擎（导出用 C++ 片段）
├── data/decks/                   # 样本卡组定义（CSV）
├── dataset/                      # 原始数据集（本地保留，未上传）
├── artifacts/                    # 训练产物 / checkpoint（本地保留，未上传）
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
- 官方 PTCG 离线引擎（仅离线 oracle 差分验证，不进入训练热路径）

### 关键文档（本仓库保留的 5 份）

- [`PTCG_JAX_ENGINE_README.md`](./PTCG_JAX_ENGINE_README.md) —— 纯 JAX 规则引擎架构说明
- [`BC_DATA_PIPELINE.md`](./BC_DATA_PIPELINE.md) —— 行为克隆数据管线
- [`PORTABLE_8GPU_MODE_AR_README.md`](./PORTABLE_8GPU_MODE_AR_README.md) —— mode-AR 主力 8×A100 训练包
- [`PPO_V2_TRAINING_REPORT.md`](./PPO_V2_TRAINING_REPORT.md) —— league PPO 完整训练报告

### 8 卡 PPO 训练包

依赖见 `portable_8gpu_requirements.txt`，一条命令启动（详见 [`PORTABLE_8GPU_MODE_AR_README.md`](./PORTABLE_8GPU_MODE_AR_README.md)）：

```bash
chmod +x setup_and_run_8gpu_mode_ar.sh
./setup_and_run_8gpu_mode_ar.sh
```

---

## 🧪 训练与提交流程

1. **行为克隆（BC）**：由最近 7 天 Top-20 专家对局训练 Orbit 实体 Transformer。
2. **PPO 强化学习**：以 BC 为初始化，按上面池构成在官方引擎单学习方 league 继续优化。
3. **评估与审计**：`tools/` 下的 `audit_*.py`、`analyze_*.py` 提供统计门禁、KL/镜像/真实胜率三重评估与一致性校验。
4. **打包提交**：从 `submission_templates/` 生成符合平台契约的提交物。

---

## 📚 关键文档

本仓库精选保留 5 份完整、准确的文档（其余历史训练与提交报告仅保留在本机，未上传）：

| 文档 | 说明 |
|---|---|
| `README.md` | 总览与技术方案 |
| `PTCG_JAX_ENGINE_README.md` | 纯 JAX 规则引擎架构 |
| `BC_DATA_PIPELINE.md` | 行为克隆数据管线 |
| `PORTABLE_8GPU_MODE_AR_README.md` | mode-AR 主力 8×A100 训练包 |
| `PPO_V2_TRAINING_REPORT.md` | league PPO 完整训练报告 |

---

## ⚠️ 免责声明

本项目为**技术研究与教育用途**的模拟练习，与 The Pokémon Company / Nintendo 无任何从属关系。Pokémon 相关名称与素材版权归其各自所有者所有。请勿将其用于商业用途或参与违反平台规则的比赛。

---

## 📄 License

仅供学习与研究使用。请在使用前遵守宝可梦官方对于其 IP 的使用规范。