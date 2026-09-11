# PTCG v4 submission, PPO v9-v17, and v18 feasibility report

Experiment date: 2026-07-26  
Final refresh: 2026-07-27 04:18 CST

## Decision

- The exact v4 package was submitted once, as authorized.
- v4 is not the new incumbent. Its live public score moved from the initial
  `600.0` validation snapshot to `858.1`, below the current v1 resubmission
  (`906.7`) and v3 (`993.7`) snapshots.
- Updated records through 2026-07-25 were incorporated into a fresh BC model.
- v9-v17 all failed at least one predeclared promotion or independent
  confirmation gate. v3 update 440 therefore remains the incumbent.
- v16 was the only strict-screen passer, but its apparent v3 advantage did not
  replicate in an independent 1,024-game confirmation.
- v17 removed the small-screen selection stage and ran 2,048 fresh games
  against each core opponent. It also failed both core promotion gates.
- A predeclared public-observation separability audit then failed the go/no-go
  gate for a stateless v18 multi-head router. v18 was not implemented or
  trained.
- No v9-v17 package was produced and no later model was submitted.
- A fresh official Episodes Index download still ends at 2026-07-25. There is
  no day26 archive available for the untouched final gate.

## v4 submission

Archive:

`submissions/ptcg_ppo_marnie_v4_soup75.tar.gz`

SHA-256:

`7f98dfc79d3258c86ae1864c505658efd6e7adeabe5d2d1fabd23088c5937450`

Pre-submit package validation completed 32/32 games with 6,443/6,443 action
matches.

Kaggle result:

| Field | Value |
|---|---|
| Submission ref | `55004569` |
| Submitted UTC | `2026-07-26 15:10:40` |
| Status | `COMPLETE` |
| Initial score snapshot | `600.0` |
| Score at final refresh | `858.1` |

The score is time-varying and is not treated as a matched model comparison.
At the same final refresh, the visible scores were:

| Package | Public score |
|---|---:|
| v4 soup75 | 858.1 |
| latest v1 resubmission | 906.7 |
| v3 update 440 | 993.7 |
| league v2 soup75 | 762.9 |
| original v1 submission | 1052.8 |

## Updated data and fresh BC

Official index refresh:

- 40 rows;
- latest date: 2026-07-25;
- no 2026-07-26 daily package.

The 2026-07-24 source has an integrity discrepancy: its manifest claims 4,445
episodes, but its ZIP contains 4,444 episode JSON files and is missing episode
`87841523`. A normal `unzip -tq` check does not detect this member-set
mismatch.

The deterministic roll-forward archive is:

`data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip`

SHA-256:

`a3b9d572bfc0a784b5b140d3dbe09314252c46dd0d9c1afa3423236cda54543c`

| Split | Decisions | Episodes |
|---|---:|---:|
| train | 1,185,094 | 11,597 |
| valid (day25 hash20) | 34,204 | 328 |
| total | 1,219,298 | 11,925 |

Train/valid episode overlap is zero. The source archive SHA-256 is
`f0e57e...`; the roll-forward tool records the full value in its output
manifest.

Fresh BC:

`artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt`

SHA-256:

`61e592ff9821f17747b206eb2e5dc30ffc741932fdd86d3ec620ab7e137d7c92`

On the same 34,204-row valid split:

| Model | Set exact | Ordered exact | Value | Count |
|---|---:|---:|---:|---:|
| fresh BC | 80.7099% | 80.1310% | 72.8044% | 99.3714% |
| old BC | 78.7481% | 78.2277% | 70.7812% | 99.3685% |
| v3 u440 | 70.8075% | 69.5387% | 68.3136% | 99.3129% |

The updated records therefore improved the supervised anchor under the same
scorer.

## Exact-quota sampler implementation

`tools/train_ppo.py` now supports:

- `legacy`, `fixed`, and `adaptive` opponent quota modes;
- exact per-update valid-game quotas;
- same-opponent replacement for invalid games;
- Beta(2,2) failure posteriors with draw counted as half a failure;
- stable largest-remainder allocation with caps;
- deterministic independent quota schedule RNG;
- quota plans, actual counts, posterior state, and checkpoint resume audit.

The legacy default remains unchanged.

An explicit `--reset-opponent-quota-on-resume` safety switch was added after
the v9-to-v10 continuation audit. It:

- requires both `--resume` and `--reset-optimizer-on-resume`;
- validates the saved exact-quota state before discarding it;
- initializes the new controller from the current CLI;
- records the reset and old/new modes in the run artifacts.

Validation:

- Python compile checks passed;
- full repository tests after the router-feasibility additions: 59 passed;
- real CUDA quota smoke: 12/12 valid, planned quotas equal actual quotas.

## Shared v9 audit

The same v3-u440 learner was evaluated for 64 games against each frozen
opponent before either branch trained:

| Opponent | W-L-D | Invalid |
|---|---:|---:|
| fresh BC | 32-32-0 | 0 |
| v1 u200 | 38-26-0 | 0 |
| v3 u440 | 31-33-0 | 0 |
| v4 soup75 | 41-23-0 | 0 |
| Dragapult BC | 61-3-0 | 0 |
| Kangaskhan BC | 45-19-0 | 0 |

These values seeded only the UQ posterior. They were not counted as external
screen games.

## v9 controlled experiment

Both branches:

- started from v3 update 440;
- trained updates 441-455;
- used 15 x 96 games;
- used actor LR `3e-6`, value LR `1.5e-5`, GAE lambda `0.97`;
- used v3 as KL reference with coefficient `0.002`;
- used full trainable scope and no BC replay.

The sole branch difference was the opponent quota controller.

| Branch | Games | Invalid | Early stop | Mean KL | Max KL | Max clip |
|---|---:|---:|---:|---:|---:|---:|
| FQ | 1,440 | 0 | 0 | 0.0004406 | 0.0009314 | 1.7759% |
| UQ | 1,440 | 0 | 0 | 0.0003907 | 0.0010820 | 2.0552% |

Checkpoint hashes:

- FQ u455:
  `8fefd82c483d0b4e0cf45d8766874a3d746e6f537452eec5a7d41eb1918dc260`
- UQ u455:
  `8420b350cb2f4a0f42c554f7e0f8dc9baa198158bf1fcc397562e0de6d6c7618`

Total opponent games:

| Branch | BC | v1 | v3 | v4 | Dragapult | Kang |
|---|---:|---:|---:|---:|---:|---:|
| FQ | 180 | 180 | 540 | 180 | 180 | 180 |
| UQ | 207 | 186 | 567 | 180 | 135 | 165 |

UQ's dynamic residual allocations were:

- update 441: `[5, 5, 6, 4, 1, 3]`;
- update 444: `[6, 5, 5, 4, 1, 3]`;
- updates 447/450/453: `[6, 4, 6, 4, 1, 3]`.

The maximum half-L1 shift from uniform residual allocation was 4, below the
predeclared mechanism threshold of 6. The adaptive-sampler mechanism gate
therefore failed independently of downstream scores.

### v9 absolute screen

Behavior thresholds were v3-u440 minus one percentage point for set, ordered,
and value, plus count >=99%. H2H thresholds were:

- v3 u440: at least 269/512;
- v3 u570: at least 57/128;
- Kang: at least 82/128;
- v4: at least 64/128;
- fresh BC: at least 64/128;
- zero invalid games.

| Branch | Set | Ordered | Value | vs v3 u440 | vs v3 u570 | vs Kang | vs v4 | vs BC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FQ | 70.4011% | 69.0972% | 69.2697% | 248-263-1 | 63-65-0 | 89-39-0 | 82-46-0 | 68-60-0 |
| UQ | 70.6906% | 69.5737% | 69.4539% | 283-227-2 | 60-68-0 | 91-37-0 | 77-51-0 | 57-71-0 |

FQ failed the v3-u440 gate. UQ passed the mirror gate but failed the fresh-BC
gate and its sampler mechanism gate. Since neither passed all absolute gates,
the predeclared UQ-vs-FQ paired match was not run.

## v10/v11 bridge ablation

Both rescue runs:

- started from the locked UQ u455 checkpoint;
- trained only updates 456-461;
- used exact fixed quotas per update:
  `BC 32 / v3 48 / v1 4 / v4 4 / Dragapult 4 / Kang 4`;
- used actor LR `3e-6`, KL coefficient `0.004`, and no replay;
- used identical seed, schedule, and all other training parameters.

The sole intended difference was the KL reference:

| Run | KL reference | Games | Invalid | Mean KL | Max KL | Final anchor KL |
|---|---|---:|---:|---:|---:|---:|
| v10 | v3 u440 | 576 | 0 | 0.0003913 | 0.0007489 | 0.0092112 |
| v11 | UQ u455 | 576 | 0 | 0.0004779 | 0.0007871 | 0.0025335 |

Checkpoint hashes:

- v10 u461:
  `518d67726d8fad00b75b54b19e50634b6413e1ed04ea2fa4dfd8eee2948559b3`
- v11 u461:
  `1f44a1e7e987be43ad3bcfed01463b06d708fed935cb7369422bd6bc93e94e51`

### v10/v11 absolute screens

| Run | Set | Ordered | Value | vs v3 u440 | vs v3 u570 | vs Kang | vs v4 | vs BC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v10 | 70.7403% | 69.6030% | 69.7111% | 242-269-1 | 64-64-0 | 88-40-0 | 81-47-0 | 70-58-0 |
| v11 | 70.6175% | 69.5474% | 69.5503% | 244-266-2 | 55-73-0 | 80-48-0 | 89-39-0 | 72-56-0 |

Both successfully repaired the UQ fresh-BC failure, but both lost the UQ
mirror advantage. v11 additionally missed the v3-u570 and Kang gates by two
wins each. Changing the KL reference alone therefore did not solve the
multi-opponent interference.

## v12 constrained multi-opponent PPO

v12 implemented the planned objective change:

- opponent-stratified advantage normalization;
- exact BC/v3 quotas and per-opponent seat balancing;
- a constrained BC-primary/v3-guard objective;
- per-opponent losses, frozen-rollout guard audit, gradient cosine, and dual
  state logging.

The formal run started from UQ u455 and trained updates 456-467. It processed
12 x 96 = 1,152 valid games with zero invalid replacements. Its endpoint
SHA-256 was
`3ac20ebe89d20ebedf9b053f53b7918327b9d31f3dd65690e935dd6624cfec57`.

The absolute cascade stopped at fresh BC:

| Gate | Result | Threshold |
|---|---:|---:|
| behavior set/ordered/value/count | all passed | v3-minus-1pp / 99% |
| fresh BC | 62-65-1 / 128 | at least 64 wins |

The key failure was objective mismatch rather than an inactive optimizer:

- all 12 frozen-rollout guard surrogates were positive
  (`0.00433` to `0.00641`), so the guard floor was never violated;
- the dual therefore decreased from `1.0` to `0.9958`;
- 7/12 first-minibatch BC/v3 gradient cosines were negative;
- the guard reused the rollout batch already optimized for four epochs and
  did not measure absolute greedy win rate;
- relative to UQ on 34,204 fixed behavior states, v12 changed 2.79% of ordered
  greedy actions but gained only 15 net expert-exact rows;
- mean UQ-to-v12 KL was only `0.004345`, while p99 was `0.05897` and the
  maximum was `0.33244`.

Small mean KL and a positive same-batch surrogate therefore did not protect
the high-impact tail decisions needed for the absolute v3 gate.

## v13 policy-soup diagnostic

v13 interpolated the rejected UQ and FQ endpoints at FQ weights 0.25, 0.50,
and 0.75. This was a diagnostic only; candidates were screened in a fixed
cascade and were never eligible for direct submission.

| FQ weight | fresh BC | v3 u440 | Result |
|---:|---:|---:|---|
| 0.25 | 62-65-1 / 128 | not run | BC gate failed |
| 0.50 | 63-65-0 / 128 | not run | BC gate failed |
| 0.75 | 64-64-0 / 128 | 249-263-0 / 512 | mirror gate failed |

All behavior gates passed, but no interpolation cleared both core opponents.
Static parameter interpolation did not remove the BC/v3 Pareto trade-off.

## v14-v16 optimization changes

The PPO implementation was extended with:

- a consistent learner policy temperature for rollout sampling, PPO
  likelihoods, entropy, and reference KL;
- head-only training scopes;
- symmetric PCGrad for BC/v3 policy gradients;
- asymmetric guard-priority PCGrad that projects only the BC gradient and
  preserves the complete v3 guard gradient;
- per-minibatch conflict, projection coefficient, and post-projection dot
  audits.

Default PPO behavior remains scalar objective at temperature 1. The additions
are covered by unit tests and real CUDA smoke runs.

| Run | Main change | Train games | PCGrad conflicts | Endpoint SHA-256 |
|---|---|---:|---:|---|
| v14 | full-scope constrained PPO, T=0.5 | 1,024 | n/a | `8ba22d90443279589df392a166ad8ddf69295bb97c2333c4c0b3d39dfd5b7f0d` |
| v15 | heads-only symmetric PCGrad | 768 | 90/160 (56.25%) | `1ae5be50ca08d152d3bb5f88cdcf4359dc0ec8bb06b40bef3f9c78d1d394a1ce` |
| v16 | short heads-only guard-priority PCGrad | 384 | 38/82 (46.34%) | `f2e7009fa619e46911c581b1dc54734566061c8b55c7ee0a2894de0f4475b8ba` |

All runs had exact per-opponent seat balance and zero invalid replacements.
For v15 and v16, every recorded minimum post-projection core dot was positive,
so the implemented gradient surgery met its local first-order mechanism gate.

### v14-v16 strict screens

| Run | Behavior | fresh BC | v3 u440 | Other absolute gates | Result |
|---|---|---:|---:|---|---|
| v14 | all passed | 63-65-0 / 128 | not run | not run | stopped at BC |
| v15 | all passed | 74-54-0 / 128 | 250-261-1 / 512 | all passed | mirror failed |
| v16 | all passed | 65-63-0 / 128 | 277-233-2 / 512 | all passed | strict-screen pass |

v16 was the first and only strict-screen passer, but that screen was not
treated as final evidence. Its locked endpoint was evaluated on two new
512-game blocks per core opponent:

| Opponent | Independent result | Win rate | Wilson 95% | Gate |
|---|---:|---:|---:|---|
| v3 u440 | 528-495-1 / 1,024 | 51.56% | [48.50%, 54.61%] | failed: required 538 wins |
| fresh BC | 530-493-1 / 1,024 | 51.76% | [48.70%, 54.81%] | passed: required 512 wins |

The v3 screen rate of 54.10% did not replicate. v16 is therefore
`screen-selected, confirmation not replicated`, not a demonstrated
improvement over v3.

## v17 variance-reduced confirmation

v17 was predeclared as the final small-architecture PPO attempt. It restarted
from UQ u455 and kept the v16 mechanism and weights unchanged. The only
training-design change was:

- games per update: 128 -> 256;
- PPO epochs: 2 -> 1.

This kept optimizer work approximately constant while doubling independent
games and eliminating second-epoch rollout reuse. There was one seed, one
fixed u458 endpoint, no intermediate selection, and no preliminary
head-to-head screen.

Formal training audit:

| Field | Value |
|---|---:|
| valid games | 768 |
| transitions | 79,649 |
| optimizer batches | 79 |
| guard-PCGrad conflicts | 43/79 (54.43%) |
| minimum post-dot, BC / v3 | 0.02812 / 0.06114 |
| invalid replacements | 0 |
| endpoint SHA-256 | `b02fcbb315c18b97a2641887853d2d9be80d36d3be26a3e84e4ad502d51852d3` |

The 34,204-row behavior safety gate passed:

| Set | Ordered | Hybrid | Value | Count |
|---:|---:|---:|---:|---:|
| 70.7081% | 69.6000% | 70.1497% | 69.4129% | 99.3100% |

The script then ran all eight interleaved 512-game blocks before aggregating
either opponent. Each core opponent required at least 1,076 wins out of
2,048, a Wilson 95% lower bound above 50%, exactly 2,048 valid games, and zero
invalid games.

| Opponent | Independent result | Win rate | Wilson 95% | Gap to win gate |
|---|---:|---:|---:|---:|
| v3 u440 | 1053-992-3 / 2,048 | 51.42% | [49.25%, 53.58%] | -23 wins |
| fresh BC | 1063-983-2 / 2,048 | 51.90% | [49.74%, 54.06%] | -13 wins |

Both core gates failed, even though all 4,096 games were valid. The secondary
v3-u570, Kang, and v4 blocks were correctly skipped. v17 was not packaged or
submitted.

## v18 stateless-router feasibility gate

After v17, a source and checkpoint audit found that UQ, v15, and v16 share all
66 non-head tensors exactly. Only 14 head tensors differ, so a shared-backbone
multi-head runtime would be technically compact. However, such a router is
useful only if legal public observations identify the relevant policy regime
early enough.

A router feature was therefore frozen before collection:

- `public_router_features(observation)` is stateless and takes only the
  serialized agent observation;
- 32 bounded dense values plus 128 keyed-BLAKE2b signed-hash buckets;
- self/opponent-relative normalization without raw absolute seat;
- explicit whitelist over current state, selection, visible cards/options,
  and current logs;
- explicit exclusion of search tokens, serial fields, unknown fields,
  training metadata, deck hashes, and opponent hand content.

Feature implementation SHA-256:

`9c0a752792176b4a178715df8ea2fa24b180a93ef4447fab0af53d4cb675eaa2`

The fixed probe was v3 u440. Fresh BC and v3 u440 trajectories were launched
in interleaved class/seat order; fresh BC used canonical ordering. For each
trajectory, exactly one feature was captured at the learner's first selection
after the opponent's first, second, and third normal turns. The second-turn
snapshot was the only main metric.

Collection and split:

| Field | Value |
|---|---:|
| trajectories per class | 512 |
| train/test per class | 256 / 256 |
| test rows at each prefix | 512 |
| learner seat 0/1 per class | 256 / 256 |
| learner first/second per class | 256 / 256 |
| fresh-BC replacements | 2/514 (0.39%) |
| v3 replacements | 1/513 (0.19%) |
| train/test game overlap | 0 |

The locked classifier was
`StandardScaler + LogisticRegression(C=1, L2, lbfgs, threshold=0.5)`.
The primary k=2 gate required at least 333/512 correct, Wilson 95% lower bound
above 60%, each class recall at least 60%, each learner-order accuracy at least
60%, and a 1,000-permutation p-value below 0.01.

| Snapshot | Correct | Accuracy | Wilson 95% | BC recall | v3 recall |
|---|---:|---:|---:|---:|---:|
| after opponent turn 1 | 262/512 | 51.17% | [46.85%, 55.48%] | 38.67% | 63.67% |
| after opponent turn 2 (primary) | 286/512 | 55.86% | [51.53%, 60.10%] | 57.03% | 54.69% |
| after opponent turn 3 | 303/512 | 59.18% | [54.87%, 63.35%] | 67.19% | 51.17% |

At the primary prefix, learner-first accuracy was 48.83% and learner-second
accuracy was 62.89%. The label-permutation p-value was `0.002997`, indicating
some real public signal, but its strength and stability were far below the
predeclared router gate. k=3 was diagnostic only and could not rescue k=2.

The stateless public-observation router feasibility gate therefore failed.
No v18 router was implemented, trained, packaged, or submitted. An alternative
dense schema proposed after this result was deliberately not run, because
changing features after observing the locked test would create a post-hoc
feature sweep.

The frozen research feature also has one known deployment issue: its whitelist
hashes a terminal `RESULT` log field as an absolute seat. The audit snapshots
are all nonterminal k=1-3 states, so that terminal-only field does not affect
this feasibility result. The module must nevertheless not be reused as a
production router without a new feature version, relative-result encoding, and
a complete seat-swap test.

## Interpretation

The combined evidence does not support another small LR, KL, loss-weight,
update-count, or soup sweep on the same day25 gate:

- v9 UQ showed a mirror/BC trade-off;
- v10 and v11 repaired BC but reversed the mirror advantage;
- v12 showed that a positive same-batch surrogate and low mean KL do not guard
  absolute greedy performance;
- v13 showed that static endpoint interpolation preserves the trade-off;
- v15/v16 show that gradient projection can find a more balanced policy, but
  46-56% of minibatches still contain task conflict;
- v16 demonstrated small-screen winner's curse;
- v17 removed that selection stage and still produced only 51-52% point
  estimates whose Wilson intervals cross 50%.
- the v18 feasibility audit found statistically detectable but operationally
  insufficient early public-state separability.

The official agent observation does not expose an opponent, team, checkpoint,
or policy identifier. Training `opponent_name` is used only for sampling and
loss grouping and is not a legal deployment feature. A future conditional
policy must therefore use only public board state and public within-game
history, with explicit leakage tests.

The stateless router feasibility gate has now been run and failed. The next
justified direction requires genuinely new information or representation:

1. a new official day package, kept untouched until a candidate is locked;
2. a separately versioned public-history encoder after validating the real
   agent lifecycle and fixing current log loss/truncation;
3. broader max-min/CVaR population training only if evaluated without reusing
   the exhausted day25 selection gate.

Until such a candidate clears all matched gates and a new-day holdout, v3
remains the only justified incumbent. v16/v17 are research artifacts, not
submission candidates.

## Key artifacts

- `tools/train_ppo.py`
- `tools/run_ppo_v9_shared_audit.sh`
- `tools/run_ppo_v9_quota_branches.sh`
- `tools/run_ppo_v9_screen.sh`
- `tools/run_ppo_v10_bc_bridge.sh`
- `tools/run_ppo_v10_screen.sh`
- `tools/run_ppo_v11_uq_anchor_bridge.sh`
- `tools/run_ppo_v13_soup_screen.sh`
- `tools/run_ppo_v14_lowtemp.sh`
- `tools/run_ppo_v15_pcgrad_heads.sh`
- `tools/run_ppo_v16_guard_pcgrad.sh`
- `tools/run_ppo_v16_confirmation.sh`
- `tools/run_ppo_v17_variance_reduced.sh`
- `tools/run_ppo_v17_confirmation.sh`
- `tools/ppo_router.py`
- `tools/audit_public_opponent_separability.py`
- `artifacts/ppo_marnie_v9_screen_u455/summary.json`
- `artifacts/ppo_marnie_v10_screen_u461/summary.json`
- `artifacts/ppo_marnie_v11_screen_u461/summary.json`
- `artifacts/ppo_marnie_v15_screen_u461/summary.json`
- `artifacts/ppo_marnie_v16_confirmation_u458/summary.json`
- `artifacts/ppo_marnie_v17_confirmation_u458/summary.json`
- `artifacts/v18_router_separability_k2_seed20261180/summary.json`
