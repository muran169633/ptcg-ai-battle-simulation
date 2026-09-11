# PTCG 最近一周 BC 决策数据抽取

`tools/prepare_bc_week.py` 将官方每日 episode 数据转换为可直接供行为克隆使用的决策帧 ZIP。

## 为什么不能直接读取同一行的 action

官方 replay 中，某次 observation 对应的动作保存在下一环境步：

```text
steps[t - 1][seat].observation  ->  steps[t][seat].action
```

脚本已经按这个关系对齐。若使用 `steps[t].observation -> steps[t].action`，会产生大量 `minCount=1` 但标签为空的错误样本。

脚本只保存 agent 实际可见的 `observation`，不会复制 replay 的 `visualize`。后者可能包含双方完整牌库，作为 actor 输入会造成隐藏信息泄漏。

## 推荐：下载并处理最近 7 天

需要已登录的 Kaggle CLI：

```bash
python tools/prepare_bc_week.py \
  --download \
  --days 7 \
  --top-k 20 \
  --output data/bc_recent7_top20.zip
```

默认行为：

- 从官方 Episodes Index 解析最新七天；
- 下载七个每日 ZIP，但不将它们完整解压；
- 用当前排行榜 Top 20 队名过滤专家决策；
- 前五天为训练集，倒数第二天为验证集，最后一天为测试集；
- 胜局权重为 `1.0`，败局前 60% 为 `0.75`，败局后段为 `0.25`；
- 每 50,000 个决策生成一个 ZIP 内部 JSONL shard；
- 在 `manifest.json` 中记录来源、数量、分组、上下文和队伍统计。

每日 ZIP 缓存在 `data/episodes_cache/daily/`，后续重新抽取不会再次下载。加 `--refresh` 才会强制刷新。

## 已经下载了每日 ZIP

```bash
python tools/prepare_bc_week.py \
  --input-root /path/to/daily_zips \
  --days 7 \
  --top-k 20 \
  --output data/bc_recent7_top20.zip
```

脚本也支持 Kaggle Notebook 挂载的数据目录：

```bash
python tools/prepare_bc_week.py \
  --input-root /kaggle/input \
  --days 7 \
  --teams-file /kaggle/input/my-top20/top20.csv \
  --output /kaggle/working/bc_recent7_top20.zip
```

`--teams-file` 支持：

- 每行一个队名；
- 含 `team_name` 列的 CSV；
- 含 `date,team_name` 的 CSV。存在 `date` 时会按每日历史 Top-K 精确过滤，优于使用当前排行榜。

若只是检查代码：

```bash
python tools/prepare_bc_week.py \
  --input /path/to/one_episode.json \
  --date-end 2026-07-22 \
  --days 1 \
  --all-teams \
  --split-mode hash \
  --max-episodes 1 \
  --output /tmp/ptcg_bc_smoke.zip
```

## 输出记录

每行 JSON 包含：

- `episode_id`、日期、seat、队名和自身牌组 hash；
- 时间切分 `train/valid/test`；
- 与 observation 正确对齐的 `action`；
- `min_count/max_count/option_count` 和选择上下文；
- 最终胜负及 BC 样本权重；
- 仅 agent 可见的完整 `observation`。

多选动作保留原始索引列表，空选择仅在 `minCount=0` 时保留。

## 75% BC 门槛

后续训练应以 **exact action-set accuracy** 为主要门槛：

```text
set(predicted_action_indices) == set(expert_action_indices)
```

不要只报普通 top-1，因为该环境包含：

- `minCount=0` 的跳过动作；
- 一次选择多个目标的动作；
- 合法选项数量随状态变化的 masked action space。

建议同时保留两种评估：

- episode-hash：以完整 `episode_id` 分组做 80/10/10 划分，同一局绝不跨集合；
- chronological：前 5 天训练、第 6 天验证、第 7 天测试，衡量次日分布漂移。

进入 PPO 前建议同时满足：

- 测试日整体 exact accuracy `>= 0.75`；
- 非空动作 top-1 accuracy `>= 0.75`；
- 每个主要 `select_context` 有独立结果；
- agent 在官方引擎 smoke matches 中无非法动作和异常。

本次实测结果见 `BC_TRAINING_REPORT.md`。旧 LightGBM 的最后一天
chronological test 为 `74.1594%`；Orbit V5 神经 Actor 已将同一严格
测试日提升到 `76.3039%`，因此可以进入 PPO 准备阶段。

达到选定门槛后再用终局 `win=1, loss=0` 做 PPO，并保留 BC KL anchor，
避免 PPO 初期破坏已学会的合法操作。

## PPO 已完成

2026-07-23 已从 Orbit V5 BC Actor 完成 200 轮官方引擎 PPO。环境奖励
严格为非终局 `0`、终局胜者 `1`、败者和平局 `0`，没有使用稠密奖励。

最终模型：

```text
artifacts/ppo_marnie_terminal01_v1/final.pt
```

修正有序多选推理后，update 200 对冻结 BC 的两批独立复评合计
`731/292/1`，总胜率 `71.3867%`，1,024 局无非法动作。完整超参数、
候选对战、代码审计和适用边界见 `PPO_TRAINING_REPORT.md`；机器可读结果
见 `artifacts/ppo_marnie_terminal01_v1/final_evaluation.json`。

## Orbit 风格神经 BC（推荐）

`tools/train_bc_orbit.py` 是可直接继承给 PPO 的实体 Transformer：

- 全局决策状态、每张可见卡牌、玩家摘要和近期公开日志分别编码为实体；
- 每个动态合法选项编码为独立 Option Token；
- Transformer 在实体和选项之间做联合注意力；
- Pointer actor 对合法选项打分；
- count head 处理 `minCount != maxCount` 的弹性多选；
- value head 用终局胜负预热，后续可直接作为 PPO critic。

固定数量选择严格取合法选项中的 Top-k。弹性选择先将 count logits
mask 到 `[minCount, maxCount]`，再选择对应数量的最高分选项，因此推理
不依赖验证集阈值。

为避免每个 epoch 重复扫描完整 17GB 解压数据，可先生成目标牌组和专家
的轻量分片 ZIP：

```bash
python tools/filter_bc_archive.py \
  --input data/bc_recent7_top20.zip \
  --output data/bc_marnie_luca_recent7.zip \
  --deck-hash c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af \
  --team-name Luca
```

正式训练命令：

```bash
python tools/train_bc_orbit.py \
  --data data/bc_marnie_luca_recent7.zip \
  --deck-hash c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af \
  --team-name Luca \
  --split-mode archive \
  --expected-train-rows 236029 \
  --epochs 8 \
  --batch-size 256 \
  --workers 8 \
  --categorical-dim 64 \
  --model-dim 128 \
  --layers 4 \
  --heads 4 \
  --output-dir artifacts/bc_marnie_luca_orbit_v5
```

2026-07-23 已完成结果：

| 时间切分 | 决策数 | Exact action-set | 固定数量 | 弹性多选 | 非空 Top-1 |
|---|---:|---:|---:|---:|---:|
| 7 月 21 日验证集 | 37,943 | 76.8073% | 78.9386% | 62.6610% | 77.8018% |
| 7 月 22 日测试集 | 35,719 | 76.3039% | 78.4069% | 62.0705% | 77.2647% |

模型共 5,146,643 个参数。最佳 checkpoint 为
`artifacts/bc_marnie_luca_orbit_v5/best.pt`，完整结果为
`artifacts/bc_marnie_luca_orbit_v5/summary.json`。

checkpoint 中的 PPO 接口：

- `policy_logits`：在 `option_mask` 上定义动态合法动作分布；
- `count_logits`：mask 到本帧允许选择的数量范围；
- `value_logits`：`sigmoid(value_logits)` 表示终局胜率；
- PPO 使用终局 `win=1, loss=0`，并对 advantage 做标准化；
- 初期保留对 BC checkpoint 的 KL anchor，之后逐渐衰减；
- 评估使用未见对局、历史 checkpoint 池和真实胜率，不再使用 BC
  accuracy 作为 PPO 强度指标。
