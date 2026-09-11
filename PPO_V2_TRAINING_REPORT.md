# PTCG Marnie League PPO V2 训练报告

## 结论

本轮没有继续使用旧版的双边 current self-play。最终方案是单学习方
league PPO：当前 Marnie 策略只控制一个 seat，另一方从冻结 BC、跨牌组
BC 和历史 PPO 快照中抽取。奖励严格为终局 `win=1`、`loss/draw=0`，
所有非终局奖励为 `0`。

正式训练原计划 60 updates，但在 update 20 后按验证结果早停。最终选择
update 15 与原始 BC 的权重插值：

```text
selected = 0.75 * PPO(update 15) + 0.25 * BC
```

选定 checkpoint：

```text
artifacts/ppo_marnie_v2_crossdeck15_soups/ppo-alpha-75.pt
```

SHA-256：

```text
3b3d6a67f6d88fa6f7495c192520c8a85adac86692583a5bbd903ea2f8d93ceb
```

## BC 数据规模

`data/bc_marnie_luca_recent7.zip` 按 `episode_id` 去重后的规模：

| 切分 | 完整 replay | 决策帧 |
|---|---:|---:|
| train | 2,571 | 236,029 |
| valid | 403 | 37,943 |
| test | 368 | 35,719 |
| 合计 | 3,342 | 309,691 |

只有训练集 2,571 局参与 BC 梯度更新；valid/test 的 771 局没有参与训练。
原始 Orbit BC 在最后一天 test 上的统一 evaluator exact action-set accuracy
为 `76.3067%`。

## 相比 PPO V1 的修正

- `league_probability=1.0`：每局只有一个可训练 seat，消除共享参数同时
  控制双方导致的期望梯度抵消。
- `opponent_sampling=per_game`：每局独立抽取冻结对手。
- 原始 Marnie BC 永久保留在对手池中；跨牌组 BC 和最新历史快照提供
  状态分布多样性。
- BC replay 使用独立 AdamW，不与 PPO optimizer 共享一、二阶动量。
- 每轮 BC replay 为 `4 x 256 = 1,024` 个专家决策；缓存共 10,240 帧，
  覆盖 Marnie 训练 ZIP 的全部 10 个 train shards。
- replay 前清理全模型旧梯度，并只裁剪 replay optimizer 管理的 actor
  参数。
- 冻结 BC 对手使用 canonical action order；PPO rollout 保留随机采样
  顺序以计算正确的 ordered log-prob。
- checkpoint 选择可同时评估所有永久对手；最终提交推理使用 canonical
  ascending action order。

## 正式 PPO 配置

| 参数 | 值 |
|---|---:|
| planned updates | 60 |
| evaluated early-stop update | 20 |
| environments | 32 |
| games target / update | 128 |
| PPO epochs | 2 |
| minibatch size | 1,024 |
| actor LR | `1e-5` cosine decay |
| critic LR | `5e-5` cosine decay |
| gamma | 1.0 |
| GAE lambda | 0.97 |
| clip ratio | 0.10 |
| entropy coefficient | 0.003 |
| BC KL coefficient | 0.03 |
| target KL | 0.008 |
| BC opponent probability | 0.50 |
| snapshot interval | 5 |
| max opponent pool | 4 |

冻结跨牌组对手：

- Dragapult/Lumen BC：chronological test exact `72.0494%`；
- Kangaskhan BC：episode-hash test exact `56.3972%`。

袋兽 BC 不作为主模型质量门槛，只用于提供不同牌组与局面分布；Marnie
策略始终由自身 chronological test 的 75% 门槛约束。

## 候选比较

所有 mirror 结果均为独立 1,024 局官方引擎对局，双方 deterministic
且使用 canonical action order。

| 候选 | test exact | Mirror W/L/D | 胜率 | invalid |
|---|---:|---:|---:|---:|
| PPO update 5 | 76.1667% | 521/502/1 | 50.8789% | 0 |
| PPO update 10 | 76.0772% | 537/487/0 | 52.4414% | 0 |
| PPO update 15 | 75.9708% | 552/472/0 | 53.9063% | 0 |
| PPO update 20 | 75.9512% | 546/477/1 | 53.3203% | 0 |
| 25% PPO + 75% BC | 76.2563% | 542/482/0 | 52.9297% | 0 |
| 50% PPO + 50% BC | 76.1639% | 532/491/1 | 51.9531% | 0 |
| **75% PPO + 25% BC** | **76.0968%** | **550/474/0** | **53.7109%** | **0** |

全 PPO update 15 与 75% PPO soup 仅相差 2 胜/1,024 局，但 soup 的
test exact 高 `0.1260` 个百分点。因此选择 soup，保留基本相同的 RL
收益并增加 BC 安全边际。

选定 soup 对其他牌组的复核：

| 对手 | 对局 | W/L/D | 胜率 | invalid |
|---|---:|---:|---:|---:|
| Dragapult BC | 512 | 453/57/2 | 88.4766% | 0 |
| Kangaskhan BC | 512 | 388/122/2 | 75.7813% | 0 |

## 产物

- 主 checkpoint：
  `artifacts/ppo_marnie_v2_crossdeck15_soups/ppo-alpha-75.pt`
- PPO 训练目录：
  `artifacts/ppo_marnie_v2_crossdeck60/`
- Dragapult BC：
  `artifacts/bc_dragapult_lumen_orbit_v1/best.pt`
- Kangaskhan BC：
  `artifacts/bc_kangaskhan_orbit_v1/best.pt`
- 统一 BC/PPO evaluator：
  `tools/evaluate_policy_bc.py`
- PPO 训练器：
  `tools/train_ppo.py`
- Kaggle 归档：
  `submissions/ptcg_ppo_league_v2_soup75.tar.gz`

归档大小为 `19,057,845` bytes，SHA-256：

```text
81931647ff6a15b7e9634b15ffe037cc58f355943fb63a3d2920a3f1b9839cf9
```

归档仅含顶层 `deck.csv`、`main.py`、`model.pt` 和
`policy_runtime.py`。从最终 tar 解包后的 exec loader、`weights_only`
加载和 32 局官方引擎自对局均通过；自对局为 `32 valid / 0 invalid`，
共 6,956 个决策。

## Kaggle 提交

用户确认后于 Kaggle 时间 `2026-07-23 23:36:42` 上传一次：

```text
submission_ref: 54938019
description: Marnie league PPO v2 soup75 exact76.10
status: SubmissionStatus.COMPLETE
initial_public_score: 600.0
```

没有在 `PENDING` 状态下重复提交。该比赛的公开分会随线上对局继续更新；
旧 v1 提交也曾从初始 `600.0` 上升到查询时的 `1064.6`，因此这里记录的
`600.0` 是刚完成 validation 时的初始快照。
