# PPO 训练与复评报告

生成日期：2026-07-23

## 最终结论

已从通过 75% 决策门槛的 Orbit V5 BC Actor 继续完成 200 轮 PPO。
最终选择 update 200，并固化为：

```text
artifacts/ppo_marnie_terminal01_v1/final.pt
SHA-256:
c454dcc8d18a2ab57ce4f39408420b8cc7c692f44e73ae8e9a68491a83e5b206
```

最终模型对冻结 BC 的两批独立复评合计为：

```text
731 胜 / 292 负 / 1 平
win rate = 71.3867%
95% Wilson CI = [68.5427%, 74.0709%]
invalid = 0 / 1024
```

update 200 与次优候选 update 180 直接交手为 `276/235/1`，
总胜率 `53.9063%`。该直连结果的置信区间仍包含 50%，所以保留
`checkpoints/update-0180.pt` 作为回退版本，不宣称它已被统计显著地淘汰。

以上均为同一 Marnie 牌组的 mirror match，不是线上排行榜胜率估计。

## 奖励函数

环境奖励严格只使用终局 0/1：

```text
非终局决策：0
最终胜者：1
最终败者：0
平局：0
```

没有加入伤害、奖赏卡、回合数、击倒数或启发式牌面分。BC KL、entropy
和 value loss 是优化正则项，不是 reward shaping。

每个座位独立反向计算 GAE。`gamma=1.0`，因此 value head 的终局胜负
BCE 目标仍表示未折扣胜率；`lambda=0.95` 用于降低 actor advantage 方差。

## PPO 设置

| 项目 | 设置 |
|---|---:|
| 总更新轮数 | 200 |
| 并行官方引擎 | 16 |
| 每轮目标完整对局 | 64，完成在途局后实际约 79 |
| PPO epochs | 4 |
| minibatch | 1,024 |
| clip ratio | 0.20 |
| target KL | 0.015 |
| actor LR | `3e-5 -> 3e-6` cosine |
| critic LR | `1e-4 -> 1e-5` cosine |
| entropy coefficient | 0.005 |
| value coefficient | 0.5 |
| gradient clipping | 0.5 |
| BC KL | `0.05 -> 0.005` 指数衰减 |
| 对手混合 | 80% 当前策略自博弈，20% 冻结 league |
| league | BC 永久保留，最多 8 个对手，每 10 轮快照 |
| 最长单局 | 1,000 个引擎决策 |

前 10 轮先完成小规模试跑，随后从 update 10 做一次学习率和 BC-KL
warm restart，继续训练到 update 200。最终规范化日志按 update 保留最后
一条记录：

```text
artifacts/ppo_marnie_terminal01_v1/metrics_canonical.jsonl
```

训练累计：

- 15,791 个完整有效对局；
- 3,239,016 次官方引擎决策；
- 2,912,305 个可训练策略决策；
- 12,610 个当前策略自博弈局；
- 3,181 个冻结 league 对局；
- 非法 observation、非法 Select、截断均为 0。

## 动作分布

PPO 没有把动作空间错误地简化成固定 top-1：

- count head 在每帧动态 mask 到 `[minCount, maxCount]`；
- BC 的 `0..16` count head 扩展为官方引擎允许的 `0..60`；
- 选项使用无放回的 ordered Plackett-Luce 分布；
- 训练、PPO ratio 重算和 BC KL 都保留已选前缀；
- rollout 与更新阶段关闭 dropout，旧/新 log-prob 可精确重算。

代码审计发现，最初的确定性推理会把 PPO 多选动作重新按索引升序排列，
从而丢失 `SKILL_ORDER` 的顺序语义。现已修正：

- PPO 确定性推理保留 Plackett-Luce greedy 顺序；
- 冻结 BC 因训练目标是 action set，继续使用专家 replay 的升序兼容模式；
- 最终 1,024 局和候选直连均在修正后重新运行；
- 合成动作测试验证 PPO 顺序 `[2,0]` 与 BC 兼容顺序 `[0,2]` 可被正确区分。

## 最终复评

| 对局 | 有效局 | 胜 / 负 / 平 | 胜率 | Invalid |
|---|---:|---:|---:|---:|
| BC 对自身校准 | 256 | 129 / 127 / 0 | 50.3906% | 0 |
| update 180 对 BC | 512 | 355 / 157 / 0 | 69.3359% | 0 |
| update 200 对 BC，批次 A | 512 | 361 / 151 / 0 | 70.5078% | 0 |
| update 200 对 BC，批次 B | 512 | 370 / 141 / 1 | 72.2656% | 0 |
| **update 200 对 BC，合计** | **1,024** | **731 / 292 / 1** | **71.3867%** | **0** |
| update 200 对 update 180 | 512 | 276 / 235 / 1 | 53.9063% | 0 |

训练期 checkpoint 内嵌的 `93/35` 是顺序审计前的 128 局诊断结果。最终
选择与汇报请以 `final_evaluation.json` 为准。

## 复现训练

实际使用的是 10 轮试跑加 190 轮续训：

```bash
python tools/train_ppo.py \
  --bc-checkpoint artifacts/bc_marnie_luca_orbit_v5/best.pt \
  --deck data/decks/marnie_grimmsnarl_froslass_luca.csv \
  --output-dir artifacts/ppo_marnie_terminal01_tune \
  --updates 10 \
  --environments 16 \
  --games-per-update 64 \
  --ppo-epochs 4 \
  --minibatch-size 1024 \
  --learning-rate 3e-5 \
  --value-learning-rate 1e-4 \
  --gamma 1.0 \
  --gae-lambda 0.95 \
  --clip-ratio 0.2 \
  --entropy-coefficient 0.005 \
  --bc-kl-start 0.05 \
  --bc-kl-end 0.005 \
  --league-probability 0.2 \
  --snapshot-interval 5 \
  --eval-interval 5 \
  --eval-games 64 \
  --checkpoint-interval 5 \
  --device cuda

python tools/train_ppo.py \
  --bc-checkpoint artifacts/bc_marnie_luca_orbit_v5/best.pt \
  --deck data/decks/marnie_grimmsnarl_froslass_luca.csv \
  --output-dir artifacts/ppo_marnie_terminal01_v1 \
  --updates 200 \
  --environments 16 \
  --games-per-update 64 \
  --ppo-epochs 4 \
  --minibatch-size 1024 \
  --learning-rate 3e-5 \
  --value-learning-rate 1e-4 \
  --gamma 1.0 \
  --gae-lambda 0.95 \
  --clip-ratio 0.2 \
  --entropy-coefficient 0.005 \
  --bc-kl-start 0.05 \
  --bc-kl-end 0.005 \
  --league-probability 0.2 \
  --snapshot-interval 10 \
  --eval-interval 10 \
  --eval-games 128 \
  --checkpoint-interval 10 \
  --resume artifacts/ppo_marnie_terminal01_tune/checkpoints/update-0010.pt \
  --skip-initial-eval \
  --device cuda
```

官方引擎不暴露洗牌和掷硬币 seed，因此相同命令不能逐局复现相同战绩，
需要依赖足够大的交替座位评估。

## 当前边界

本轮只对 Marnie’s Grimmsnarl ex + Froslass 镜像进行了训练和验证。
正式提交前仍需把 `final.pt` 接入 submission agent，并用其他主流牌组做
cross-deck league；当前 71.3867% 不能外推成线上胜率。

## Kaggle 提交包

`final.pt` 已接入 CPU 推理入口，并打包为：

```text
submissions/ptcg_ppo_terminal01_v1.tar.gz
```

包内 `main.py` 和 `deck.csv` 位于顶层，纯推理权重不含 optimizer。
解包验证、文件哈希、CPU 延迟和上传命令见 `KAGGLE_SUBMISSION_REPORT.md`。
