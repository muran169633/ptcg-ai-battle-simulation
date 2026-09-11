# PPO v6 controlled-ablation report

Date: 2026-07-26

## Decision

No v6 checkpoint is promoted, packaged, or submitted.

The locked candidate was Phase B2, update 465 (`GAE lambda=1.0`). Its
independent 1,024-game screen against the v3 incumbent finished:

- 488 wins, 535 losses, 1 draw;
- scored win rate: 47.6562%;
- Wilson 95% interval: 44.6116% to 50.7184%;
- invalid games: 0.

The screen was required to reach at least 51.5% before spending the 4,096-game
strict-evaluation budget. It failed that frozen gate. Combined with the two
earlier 256-game blocks, B2 finished 754-780-2 over 1,536 games (49.0885%,
Wilson 95% interval 46.5939% to 51.5878%).

The v3 `best.pt` at update 440 remains the incumbent.

## Infrastructure changes

`tools/train_ppo.py` now supports:

- an independent `--kl-reference-checkpoint`;
- continuation-local LR schedules (constant or cosine) and KL interpolation
  via `--schedule-start-update`;
- `--reset-optimizer-on-resume`;
- explicit named permanent-opponent weights plus a separate history weight;
- exact target training-game counts without environment-batch overshoot;
- evaluation concurrency capped by `--environments`;
- bounded BattleStart/invalid/truncated retries even after some valid games;
- rejection of reset-without-resume and missing resume checkpoints;
- resume-time filtering of future league snapshots and future `best.pt`.

The single-opponent evaluator continues to use the base BC opponent, not the
independent KL reference.

Validation completed:

- Python compilation passed.
- Eight CPU unit tests passed.
- Real-engine smoke tests verified exact game counts, capped concurrency,
  phase-local LR/KL, independent reference loading, and Adam reset to step 1.

Reproducible runners:

- `tools/run_ppo_v6_phase_a.sh`
- `tools/run_ppo_v6_phase_b.sh`

## Episode-disjoint replay split

`tools/split_bc_archive_by_episode.py` created:

`data/bc_marnie_valid24_hash75_25_seed20260820.zip`

It uses SHA-256 over `20260820:{episode_id}` and physically rewrites the
2026-07-24 archive partition into:

| Partition | Episodes | Rows | Shards |
|---|---:|---:|---:|
| train / replay75 | 1,888 | 217,425 | 9 |
| valid / gate25 | 626 | 71,774 | 3 |
| total | 2,514 | 289,199 | 12 |

Checks passed:

- episode-id overlap: 0;
- episode-UUID count matches episode count in both partitions;
- row total and episode total exactly match the original valid24 split;
- all rows remain dated 2026-07-24;
- member prefix, row `split`, and hash assignment agree;
- ZIP integrity passed.

The gate25 partition prevents direct Phase B replay/gate gradient overlap. It
is not a globally pristine holdout because the fresh BC epoch and earlier
analysis had already used full valid24. Test25 was not consulted during the
v6 Phase A/B choices, but it was viewed in the earlier v5 investigation and
therefore is not globally pristine either.

## Phase A: reference x opponent sampling

All four runs resumed v3 update 440 with a fresh optimizer and trained exactly
10 updates x 64 games. Shared settings included four PPO epochs, actor LR
`3e-6`, critic-head LR `1.5e-5`, constant KL coefficient `0.002`, no BC
replay, and no dynamic history snapshots.

The weighted pool used:

- fresh BC 10%;
- v1 update 200 10%;
- v3 update 440 45%;
- v4 alpha-075 10%;
- Dragapult 5%;
- Kangaskhan 20%.

The legacy pool retained v5's 15% base-BC probability and divided the
remaining mass uniformly across the five extra opponents.

| Run | KL reference | Sampling | Mean approx KL | Mean clip fraction | Mean reference KL |
|---|---|---|---:|---:|---:|
| A00 | fresh BC | legacy | 0.0002199 | 0.2694% | 0.476566 |
| A01 | v3 u440 | legacy | 0.0002577 | 0.3501% | 0.001909 |
| A10 | fresh BC | weighted | 0.0002599 | 0.3726% | 0.485467 |
| A11 | v3 u440 | weighted | 0.0002520 | 0.3401% | 0.002852 |

All four runs completed 640/640 valid games with zero invalid games. The
reference distances show that the same coefficient does not impose an
equivalent constraint: fresh-BC and v3 references differ by roughly two
orders of magnitude at the resumed policy.

### Frozen external screens

| Run | vs v3 u440, 256 | vs v3 u570, 128 | vs Kang, 128 |
|---|---:|---:|---:|
| A00 | 134-122-0, 52.3438% | 69-58-1, 53.9062% | 90-38-0, 70.3125% |
| A01 | 115-141-0, 44.9219% | 73-55-0, 57.0312% | 92-36-0, 71.8750% |
| A10 | 130-126-0, 50.7812% | 67-61-0, 52.3438% | 82-46-0, 64.0625% |
| A11 | 126-129-1, 49.2188% | 61-67-0, 47.6562% | 91-37-0, 71.0938% |

Hard weighting did not produce a repeatable mirror improvement and A10
showed a Kangaskhan regression. A00 was selected as the Phase B starting
point, not promoted as a final model.

### Full valid24 behavior

All values below use 289,199 rows and PPO policy order.

| Model | Set exact | Ordered exact | Top-1 | Value | Context 34 ordered |
|---|---:|---:|---:|---:|---:|
| v3 u440 | 70.4069% | 69.1735% | 71.6273% | 69.7029% | 50.7752% |
| A00 | 70.1728% | 68.8540% | 71.3744% | 70.2164% | 43.4662% |
| A01 | 70.3232% | 69.0901% | 71.5518% | 70.3716% | 51.0520% |
| A10 | 69.9449% | 68.7094% | 71.1652% | 70.4646% | 49.0587% |
| A11 | 70.4456% | 69.3011% | 71.6806% | 69.5497% | 57.8627% |

Behavior fidelity and win rate were not aligned. Imitation accuracy was kept
as a safety gate rather than the PPO selection objective.

## Phase B: replay and terminal credit

All Phase B branches resumed A00 update 450, reset the optimizer, and trained
15 updates x 96 games with the same frozen opponent pool.

| Run | Difference from B0 | Mean approx KL | Mean clip fraction | Valid games | Invalid |
|---|---|---:|---:|---:|---:|
| B0 | no replay, lambda 0.97 | 0.0004594 | 0.7707% | 1,440 | 0 |
| B1 | 1x256 replay/update, LR scale 0.10 | 0.0004139 | 0.6886% | 1,440 | 0 |
| B2 | no replay, lambda 1.00 | 0.0003316 | 0.5220% | 1,440 | 0 |

B1 consumed only replay75: 3,840 auxiliary row-exposures total. It did not
improve the winning-policy Pareto frontier.

### Common external matrix

| Model | vs v3 u440 block 1 | vs v3 u570 | vs Kang | Gate25 set exact | Gate25 value | Gate25 context 34 ordered |
|---|---:|---:|---:|---:|---:|---:|
| A00 start | 127-129-0, 49.6094% | 53-72-3, 41.4062% | 94-34-0, 73.4375% | 69.9710% | 70.4085% | 46.9136% |
| B0 | 118-138-0, 46.0938% | 70-58-0, 54.6875% | 90-37-1, 70.3125% | 69.7119% | 71.0675% | 35.5556% |
| B1 | 125-131-0, 48.8281% | 59-69-0, 46.0938% | 86-42-0, 67.1875% | 70.1438% | 69.2758% | 41.9753% |
| B2 | 124-132-0, 48.4375% | 68-60-0, 53.1250% | 89-39-0, 69.5312% | 70.0867% | 70.7387% | 47.1605% |

Because A00 had already received two independent 256-game u440 blocks, the
three Phase B branches received a second block before locking a candidate.

| Model | u440 combined 512 | Win rate | Wilson 95% interval |
|---|---:|---:|---:|
| A00 | 261-251-0 | 50.9766% | 46.6553% to 55.2833% |
| B0 | 232-280-0 | 45.3125% | 41.0515% to 49.6433% |
| B1 | 261-251-0 | 50.9766% | 46.6553% to 55.2833% |
| B2 | 266-245-1 | 51.9531% | 47.6271% to 56.2501% |

B2 was locked because it led the equal-budget u440 screen, had no gate25
regression relative to A00, and retained acceptable u570/Kang results. Its
subsequent independent 1,024-game failure is the final promotion decision;
the earlier 512 games were not used to switch to another candidate afterward.

## What was learned

1. The v5 continuation was under-updating. Phase A/B raised mean PPO KL by
   roughly 4x to 9x without instability (individual updates reached about
   15x), but stronger updates alone did not guarantee a stronger final policy.
2. Fixed 45% v3 / 20% Kang hard weighting was not supported.
3. Light unordered BC replay was not supported. It modestly improved set
   exact on gate25 while reducing value accuracy and cross-deck strength.
4. `GAE lambda=1.0` was directionally best at 512 games but failed the locked
   1,024-game screen. It is not a proven improvement.
5. The official engine is not seeded by the Python/Torch seed. Repeated
   128/256-game blocks can move sharply, so raw in-training `min` selection is
   unsuitable.

## Highest-value next changes

1. Preserve expert action order in replay. The current BC featurizer reduces
   actions to a sorted set, while context 34 is order-sensitive. Add ordered
   action-sequence targets and a Plackett-Luce replay loss before testing
   replay again.
2. Instrument separate actor, value, reference-KL, and replay gradient norms
   on the shared trunk. Then test `last transformer block + actor/count/value
   heads` against full-model PPO, with no replay in either branch.
3. Separate trust-region and imitation roles: use v3 as the trust-region
   reference and, only after ordered replay is fixed, use a very small recent
   BC auxiliary loss. A fixed coefficient against references whose starting
   KL differs by about 250x is not a controlled comparison.
4. Replace fixed hard weights with an uncertainty-aware opponent sampler only
   after per-opponent estimates use enough games. Do not spend a fixed third
   of the rollout budget on already-easy cross-deck opponents.
5. Keep a locked 1,024-game screen before any 4,096-game strict evaluation.
   Do not use test25 for further tuning; obtain the next dated battle archive
   for a genuinely fresh behavior confirmation.

## Artifacts

- Phase A runs: `artifacts/ppo_marnie_v6_phaseA_*`
- Phase A screens: `artifacts/ppo_marnie_v6_phaseA_screen/`
- Phase B runs: `artifacts/ppo_marnie_v6_phaseB_*`
- Phase B screens: `artifacts/ppo_marnie_v6_phaseB_screen/`
- Locked screen:
  `artifacts/ppo_marnie_v6_phaseB_screen/B2_locked_vs_v3u440_1024.json`
- Episode-disjoint archive:
  `data/bc_marnie_valid24_hash75_25_seed20260820.zip`

No v6 submission package was created and no external submission occurred.
