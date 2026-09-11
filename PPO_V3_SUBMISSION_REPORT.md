# PTCG PPO V3 提交报告

生成日期：2026-07-24

## 模型来源

本次提交使用长程 incumbent league PPO 的训练期最佳 checkpoint：

```text
artifacts/ppo_marnie_v3_incumbent600/best.pt
update: 440
SHA-256:
d3278052b2f13d0157b37ba8b43a21a0727233e093e8adc7963b4ee085f599de
```

训练从 v1 update 200 恢复，最终训练至 update 600。候选选择永久保留
冻结 v1，并使用 BC/v1 两个永久基线中的最低胜率作为训练期选择指标。

## 独立严格复评

update 440 与冻结 v1 update 200 使用官方引擎独立完成 2,048 局：

```text
1104 胜 / 941 负 / 3 平
win rate: 53.9063%
95% Wilson CI: [51.7421%, 56.0558%]
invalid: 0
```

双方均使用实际 PPO submission 的 ordered Plackett-Luce 动作顺序，没有
canonical sort。由于有效局为偶数且 invalid 为 0，learner seat 0/1 各
1,024 局；这里的 seat index 不直接解释为先手或后手。

完整机器可读结果：

```text
artifacts/ppo_marnie_v3_incumbent600/strict_eval_update440_vs_v1_2048.json
```

## 提交包

```text
submissions/ptcg_ppo_incumbent_v3_update440.tar.gz
大小: 19,078,220 bytes
SHA-256:
c50611a2d07b8f98d3bcfa72f87f98298d515f545361db4a4c79d0ef0bd66de8
```

归档顶层严格只有：

```text
main.py
deck.csv
model.pt
policy_runtime.py
```

提交侧 `model.pt` 移除了 optimizer、训练配置和 metrics。其 80 个
`model_state_dict` tensor 与源 checkpoint 逐一 `torch.equal`，没有量化
或近似转换。推理入口复用 v1 ordered PPO 行为，没有使用 v2 的
`action.sort()`。

从最终 tar.gz 解包后的 CPU 官方引擎测试：

```text
8 / 8 valid games
1,627 / 1,627 actions exactly match source checkpoint
invalid: 0
mean packaged inference: 3.03 ms / decision
```

## Kaggle 提交

用户明确授权后只提交一次：

```text
competition: pokemon-tcg-ai-battle
submission_ref: 54950660
file: ptcg_ppo_incumbent_v3_update440.tar.gz
description: Marnie PPO incumbent v3 update440
submitted_at: 2026-07-24 12:08:10.910 UTC
status: SubmissionStatus.COMPLETE
initial_public_score: 600.0
```

初始 600.0 是 validation episode 完成后的起始快照，公开分会随线上对局
继续变化。UTC 当日团队配额提交后为 `1 / 5`，剩余 4 次。
