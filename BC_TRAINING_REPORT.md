# BC 训练与验证报告

生成日期：2026-07-23

## 最终结论

本轮针对 **Marnie’s Grimmsnarl ex + Froslass** 主牌表、当前榜首
`Luca` 的行为数据训练候选动作排序 BC。

主牌表 hash：

```text
c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af
```

主要指标使用完整动作集合 exact match：

```text
set(predicted_action_indices) == set(expert_action_indices)
```

| 评估 | 模型/训练范围 | 评估决策数 | Exact accuracy | 固定数量 | 可变数量 |
|---|---|---:|---:|---:|---:|
| 7 月 21 日时间外 valid | Orbit V5 / 7 月 16–20 日 | 37,943 | **76.8073%** | 78.9386% | 62.6610% |
| 7 月 22 日时间外 test | Orbit V5 / 7 月 16–20 日 | 35,719 | **76.3039%** | 78.4069% | 62.0705% |
| Episode-hash valid | hash train | 30,696 | 75.0261% | 76.0034% | 68.4330% |
| Episode-hash test | hash train+valid | 27,650 | **75.1754%** | 76.4040% | 67.0433% |
| 7 月 22 日时间外 test | LightGBM / 7 月 16–21 日 | 35,719 | 74.1594% | 74.9430% | 68.8560% |

因此：

- 神经 Actor 自身在严格日期验证和测试上均通过 75% 门槛；
- 完全隔离的 7 月 22 日测试结果为 76.3039%，不是 episode-hash 泄漏；
- 该 Actor 与 PPO 共用模型结构，可直接继承权重；
- 弹性多选仍是后续重点，但总体门槛已达成。

## 最终神经模型

Orbit 风格实体/选项 Transformer V5：

```text
artifacts/bc_marnie_luca_orbit_v5/best.pt
```

模型共 5,146,643 个参数，训练数据为：

- 7 月 16–20 日：236,029 个训练决策；
- 7 月 21 日：37,943 个验证决策；
- 7 月 22 日：35,719 个最终测试决策；
- 测试集只在按验证集选择最佳 checkpoint 后评估。

模型接口：

- `policy_logits`：动态合法选项 Pointer actor；
- `count_logits`：弹性多选数量分布；
- `value_logits`：终局胜率 critic；
- 非空动作测试日 Top-1：77.2647%；
- 测试日 count accuracy：99.9272%。

完整摘要：

```text
artifacts/bc_marnie_luca_orbit_v5/summary.json
```

## 历史 LightGBM 教师

Episode-hash 最终教师：

```text
artifacts/bc_marnie_luca_lgbm_final/teacher.txt
SHA-256:
dc05474f90022aad7f347e7756a95047876411d9bc6df8ed1d779018c85435d6
```

该模型使用：

- 251,345 个 hash-train 决策；
- 30,696 个 hash-valid 决策合并重训；
- 27,650 个 hash-test 决策仅作一次最终测试；
- 1,000 棵 LightGBM boosting trees；
- 固定动作阈值 `0.05`。

完整摘要：

```text
artifacts/bc_marnie_luca_lgbm_final/summary.json
```

时间切分教师和摘要：

```text
artifacts/bc_marnie_luca_lgbm_time_final/teacher.txt
artifacts/bc_marnie_luca_lgbm_time_final/summary.json
```

## 复现命令

先用 hash-valid 选择阈值：

```bash
python tools/train_bc_lgbm.py \
  --deck-hash c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af \
  --team-name Luca \
  --rounds 800 \
  --output-dir artifacts/bc_marnie_luca_lgbm
```

固定阈值后合并 hash train+valid，最终测试：

```bash
python tools/train_bc_lgbm.py \
  --deck-hash c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af \
  --team-name Luca \
  --rounds 1000 \
  --fixed-threshold 0.05 \
  --final-train-valid \
  --output-dir artifacts/bc_marnie_luca_lgbm_final
```

严格日期测试：

```bash
python tools/train_bc_lgbm.py \
  --deck-hash c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af \
  --team-name Luca \
  --split-mode archive \
  --rounds 1000 \
  --fixed-threshold 0.05 \
  --final-train-valid \
  --output-dir artifacts/bc_marnie_luca_lgbm_time_final
```

## 历史神经 BC 基线

候选条件注意力神经模型 V4 在修正后的 episode-hash valid 上为
`70.8073%`。当时只有 LightGBM 教师达到 75%，但它不能直接作为 PPO
actor 权重。

该问题已由 Orbit V5 解决，不再需要将 LightGBM 蒸馏为主 Actor。LightGBM
仅保留作为特征与数据质量对照；PPO 从
`artifacts/bc_marnie_luca_orbit_v5/best.pt` 初始化，并保留衰减的 BC
KL anchor。
