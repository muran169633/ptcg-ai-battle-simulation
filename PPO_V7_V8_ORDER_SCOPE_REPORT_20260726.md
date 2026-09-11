# PPO v7/v8 Ordered Replay and Trainable-Scope Report

Date: 2026-07-26

## Decision

Neither v7 nor v8 is promoted. The existing v3 update 440 remains the
incumbent.

- v7 `last_block_heads` was the only branch to pass the 512-game screen, but
  its locked independent 1,024-game result was only `515-508-1` (50.2930%),
  below the predeclared 528-win gate.
- v8 increased the scoped actor learning rate from `3e-6` to `1.2e-5` after a
  KL-only calibration. It passed the behavior gate but failed both the u440
  and u570 external screen gates.
- No 4,096-game strict evaluation was run.
- No submission package was built and no external submission was made.

## Implementation completed

### Ordered expert replay

`tools/train_bc_orbit.py` now:

- preserves the original expert action sequence in `action_sequence`;
- retains the historical sorted multi-hot set target separately;
- rejects duplicate or out-of-range selections instead of silently
  de-duplicating them;
- collates `action_sequences` without truncating sequences longer than the BC
  model's historical 16-action width.

`tools/train_ppo.py` now supports:

- `--bc-replay-loss set`;
- `--bc-replay-loss ordered`;
- `--bc-replay-loss hybrid_ordered`.

The tested `hybrid_ordered` mode applies ordered Plackett-Luce sequence NLL
only to context 34 and preserves the legacy normalized-set pointer, set BCE,
and flexible count loss elsewhere.

The restriction to context 34 is deliberate. The replay75 training split has:

| Statistic | Value |
|---|---:|
| Rows | 217,425 |
| Multi-action rows | 8,672 (3.988%) |
| Context 34 rows | 1,401 (0.644%) |
| Context 34 non-ascending labels | 280 |
| Non-context-34 multi-action rows | 7,271 |
| Non-context-34 non-ascending labels | 1,490 (20.49%) |
| Duplicate action rows | 0 |

Only context 34 is treated as order-sensitive by the runtime/evaluator.
Applying ordered loss to every multi-action row would therefore learn recorded
order in contexts evaluated as sets.

The context-34 labels also have an identifiability limitation: the observed
examples use repeated card ID 104, while the current option features do not
retain the differentiating serial. Strict context-34 accuracy is therefore a
diagnostic, not a promotion objective.

### PPO trainable scope

New `--trainable-scope` choices:

- `full`;
- `heads`;
- `last_block_heads`.

`last_block_heads` trains:

- `transformer.layers.3.*`;
- `transformer.norm.*`;
- actor query/key/residual heads;
- count head;
- value head.

It freezes the embeddings, input encoders, position/kind embeddings, and
Transformer layers 0-2.

| Scope | Actor parameters | Value parameters | Total trainable | Ratio |
|---|---:|---:|---:|---:|
| full | 5,135,678 | 16,641 | 5,152,319 | 100.00% |
| last_block_heads | 288,702 | 16,641 | 305,343 | 5.93% |

Additional safety and diagnostics:

- optimizers contain only scope-approved `requires_grad=True` parameters;
- clipping uses only optimizer-owned trainable parameters;
- checkpoints store actor/value optimizer parameter-name manifests;
- changing scope on resume without optimizer reset fails explicitly;
- PPO logs total, actor-group, and value-head gradient norms;
- replay logs selection, set-BCE, count, context-34 NLL, exposure count, and
  replay gradient norm.

## Validation

- Python compile checks passed.
- 16 unit tests passed.
- Numeric Plackett-Luce regression:
  - logits `[0, 2]`, expert `[1, 0]`: selection NLL `0.126928`;
  - logits `[0, 2]`, expert `[0, 1]`: selection NLL `2.126928`.
- Hybrid gating test confirms sequence swapping affects context 34 but not
  context 22.
- End-to-end featurizer/collate regression preserves
  `[1, 0, -1, -1, ...]`.
- A real CUDA replay smoke completed with finite loss/gradients.
- A real CUDA scoped PPO smoke completed with 4/4 valid games and verified
  checkpoint optimizer manifests of 24 actor tensors plus 4 value tensors.

## v7 experiment

All branches started from:

`artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt`

Common controls:

- v3 update 440 as KL reference;
- 15 updates x 96 games;
- actor LR `3e-6`, value-head LR `1.5e-5`;
- GAE lambda `0.97`;
- constant KL coefficient `0.002`;
- the same six-opponent legacy pool;
- optimizer reset;
- no historical snapshots added.

Branches:

| ID | Change |
|---|---|
| R0H0 | full scope, no replay |
| R1H0 | full scope, hybrid ordered replay |
| R0H1 | last block + heads, no replay |

### Training health

| Run | Valid games | Invalid | Early stop | Mean approx KL | Max clip fraction |
|---|---:|---:|---:|---:|---:|
| R0H0 | 1,440 | 0 | 0 | 0.0003824 | 1.2683% |
| R1H0 | 1,440 | 0 | 0 | 0.0004007 | 1.4833% |
| R0H1 | 1,440 | 0 | 0 | 0.0000285 | 0.0051% |

R0H1 was under-updating at the shared LR: its mean approximate KL was about
13.4 times lower than R0H0.

R1H0 consumed 3,840 replay row-exposures but only 17 context-34 exposures,
consistent with the audited 0.644% prevalence.

### Episode-disjoint gate25 behavior

All metrics use 71,774 valid rows and PPO policy order.

| Model | Set exact | Ordered exact | Value | Context 34 ordered |
|---|---:|---:|---:|---:|
| v3 u440 | 70.2385% | 69.0375% | 69.9432% | 54.8148% |
| R0H0 | 69.7955% | 68.5764% | 70.5924% | 57.5309% |
| R1H0 | 70.4461% | 69.1824% | 71.0229% | 50.3704% |
| R0H1 | 70.1145% | 68.9581% | 70.2845% | 60.2469% |

All branches passed the overall behavior safety limits. R1H0 failed its
mechanism check: context-34 ordered accuracy was 7.16 percentage points below
R0H0, not at least 3 points above it.

### External screen

| Model | vs v3 u440, 512 | Win rate | vs v3 u570, 128 | vs Kang, 128 |
|---|---:|---:|---:|---:|
| R0H0 | 260-250-2 | 50.7812% | 61-67-0 | 88-40-0 |
| R1H0 | 265-247-0 | 51.7578% | 65-63-0 | 78-50-0 |
| R0H1 | 272-240-0 | 53.1250% | 65-63-0 | 96-32-0 |

All evaluations had zero invalid games.

R1H0 failed the absolute u440 gate (needed at least 269 wins), the improvement
gate relative to R0H0, the ordered-replay mechanism check, and the Kang
regression limit.

R0H1 uniquely passed the screen:

- at least 269/512 u440 wins;
- 12 more wins than R0H0, exceeding the required +8;
- no u570 regression;
- eight more Kang wins than R0H0.

### Locked result

R0H1 was locked before the next evaluation.

| Evaluation | Result | Win rate | Wilson 95% |
|---|---:|---:|---:|
| Independent vs v3 u440, 1,024 | 515-508-1 | 50.2930% | 47.2352%-53.3485% |

The gate required at least 528 wins. R0H1 failed, so v7 was not promoted and
the 4,096-game strict evaluation was not run.

## v8 KL-only LR calibration

The v7 scoped branch had under-updated, so three three-update probes were run
from the same v3 u440 start. LR selection used only training KL, never H2H.

| Scoped actor LR | Games | Mean approx KL | Max clip fraction |
|---|---:|---:|---:|
| 6e-6 | 288 | 0.0000688 | 0.0371% |
| 9e-6 | 288 | 0.0001526 | 0.1855% |
| 1.2e-5 | 288 | 0.0002059 | 0.3541% |

All probes had zero invalid games and zero early stops. `1.2e-5` was the tested
LR closest to the full-model control's early-phase KL and was locked for the
full run.

### v8 full run health

- 1,440 valid games;
- 0 invalid games;
- 0 early stops;
- mean approximate KL `0.0001905`;
- max clip fraction `0.6748%`.

### v8 gate and external screen

| Model | Set exact | Ordered exact | Value | Context 34 ordered |
|---|---:|---:|---:|---:|
| v3 u440 | 70.2385% | 69.0375% | 69.9432% | 54.8148% |
| v8 scoped 1.2e-5 | 70.0853% | 68.9066% | 70.4210% | 59.7531% |

The behavior gate passed.

| Evaluation | Result | Gate |
|---|---:|---|
| vs v3 u440 block 1 | 141-115-0 | diagnostic |
| vs v3 u440 block 2 | 116-140-0 | diagnostic |
| vs v3 u440 combined | 257-255-0, 50.1953% | fail; needed 269 wins |
| vs v3 u570 | 56-72-0 | fail; needed at least 57 wins |
| vs Kang | 84-44-0 | pass; needed at least 82 wins |

All evaluations had zero invalid games. Because v8 failed the screen, no
locked 1,024-game evaluation was run.

## Conclusions

1. Ordered replay correctness is fixed, but the available order-sensitive
   supervision is too sparse and partly unidentifiable. Only 17 such examples
   were exposed in the controlled run, and the mechanism metric regressed.
2. Limiting PPO to the last Transformer block plus heads is technically sound
   and preserved behavior well. At `3e-6` it produced a promising 512-game
   screen, but this did not survive the locked 1,024-game test.
3. Increasing scoped LR to `1.2e-5` corrected most of the KL under-update, but
   did not improve robustness. It failed both the u440 and u570 screen gates.
4. More PPO update magnitude by itself is not the next lever. The remaining
   bottleneck is robust opponent exposure and learnable state/action identity,
   not a simple LR or imitation-strength deficit.
5. v3 update 440 remains the only supported incumbent. v4 remains
   submission-unverified and was used only as a frozen training opponent.

## Recommended next iteration

1. Ingest the next genuinely fresh battle archive before tuning another
   replay variant. Keep it episode-disjoint and reserve a new untouched gate.
2. Audit whether a player-visible, stable option identity can distinguish the
   repeated-card context-34 choices. Add such a feature only if it is available
   at inference and does not leak replay-only state.
3. Replace the fixed opponent mixture with a predeclared uncertainty-aware
   sampler, while keeping a minimum v3 mirror quota. Evaluate the sampler as a
   separate factor.
4. Do not run another broad LR sweep on the current archive. If scoped PPO is
   revisited, use a fresh-data replication and retain the locked
   512 -> 1,024 -> 4,096 promotion sequence.

## Artifacts

- Code:
  - `tools/train_bc_orbit.py`
  - `tools/train_ppo.py`
  - `tests/test_train_ppo_unit.py`
- Runners:
  - `tools/run_ppo_v7_order_scope.sh`
  - `tools/run_ppo_v7_screen.sh`
  - `tools/run_ppo_v8_scope_lr_probe.sh`
  - `tools/run_ppo_v8_scope_calibrated.sh`
  - `tools/run_ppo_v8_screen.sh`
- v7 runs: `artifacts/ppo_marnie_v7_*`
- v7 screen: `artifacts/ppo_marnie_v7_screen_u455/`
- v8 probes: `artifacts/ppo_marnie_v8_scope_probe_*`
- v8 full run: `artifacts/ppo_marnie_v8_lastblockheads_lr12e6_u455/`
- v8 screen: `artifacts/ppo_marnie_v8_screen_u455/`
