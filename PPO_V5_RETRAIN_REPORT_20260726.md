# PTCG V5 updated-record retraining report

Date: 2026-07-26 (Asia/Shanghai)

## Decision

- The updated episode records through 2026-07-25 were downloaded, validated, merged, and used for a fresh BC model plus an 80-update PPO continuation from the accepted v3 update 440.
- The best new PPO candidate was update 510, but it did **not** prove that it is stronger than v3:
  - 4,096 games vs v3: 2,063 wins, 2,027 losses, 6 draws, 0 invalid.
  - Scored win rate: 50.3662%.
  - Wilson 95% CI: 48.8354% to 51.8963%.
  - Promotion gate required the lower bound to exceed 50%; it failed.
- v3 update 440 therefore remains the local incumbent.
- No v5 submission archive was produced, and no Kaggle submission or upload was made.
- The existing v4 archive remains unsubmitted locally. Its online effect is unknown; its local evidence is mixed and does not justify replacing v3.

## Updated data

The Episodes Index cache was refreshed to Kaggle index version 40, covering records through 2026-07-25.

New daily archives:

- 2026-07-24: 4,445 indexed episodes; archive size 741,776,272 bytes.
- 2026-07-25: 4,547 indexed episodes; archive size 743,874,944 bytes.
- Both archives passed `unzip -tq`.

The current Kaggle leaderboard endpoint returned HTTP 401 with the locally available credential, so the team filter was not claimed to be a live 2026-07-26 Top-50 refresh. The rebuild used the frozen `data/top50_snapshot_bc_recent7_20260724.txt` list. That list exactly matches the whitelist used by the 2026-07-16 to 2026-07-23 base archive, so the base and supplement use one consistent 50-team filter.

Final Marnie archive:

`data/bc_marnie_top50_20260716_25.zip`

| Split | Dates | Decisions |
|---|---|---:|
| train | 2026-07-16 to 2026-07-23 | 765,492 |
| valid / PPO replay | 2026-07-24 | 289,199 |
| test | 2026-07-25 | 164,607 |
| total | 2026-07-16 to 2026-07-25 | 1,219,298 |

Full-row audit results:

- ZIP directory prefix, row-level split, and manifest date policy mismatches: 0.
- Episodes crossing dates or splits: 0.
- Duplicate decision keys: 0.
- All rows use the target Marnie deck hash.
- The 23 teams actually present are all in the common 50-team whitelist.
- PPO replay explicitly reads only `valid/` (2026-07-24); no 2026-07-25 test shard entered PPO gradients.

The merged manifest has one provenance blemish: `filtered_from` was inherited from the old base and only names the archive through 2026-07-23. Its `merged_from` and `sources` fields correctly enumerate both source archives and all dates, so this is a metadata issue rather than data leakage.

## Fresh BC model

Checkpoint:

`artifacts/bc_marnie_top50_train23_valid24_orbit_v2/best.pt`

SHA-256:

`bf60cf12071a00ec15b7d1f73e0c35c1692d45ec646455a0dd0eb04ea894c672`

Best epoch: 7 of 8, selected only on 2026-07-24 validation data.

| Metric | valid24 | test25 |
|---|---:|---:|
| exact action set | 79.8367% | 78.5908% |
| fixed cardinality | 81.9742% | 80.7821% |
| flexible cardinality | 65.2814% | 64.0468% |
| count accuracy | 99.6393% | 99.3609% |
| nonempty top-1 | 80.7589% | 79.6115% |
| value accuracy | 70.2796% | 70.7722% |

Both validation and test exceed the 75% exact-action-set gate.

The test25 metric was computed once after BC epoch selection. It did not enter training, but it is no longer a completely unseen holdout. PPO checkpoint selection did not use test25.

## PPO retraining

Output:

`artifacts/ppo_marnie_v5_recent24_v3resume520`

Training configuration:

- Resume: v3 update 440.
- New updates: 441 through 520.
- Fresh BC anchor: the model above.
- Replay split: 2026-07-24 validation shards only.
- Frozen opponents: new BC, v1, v3, v4 soup75, Dragapult BC, and Kangaskhan BC.
- Eight 128-game-per-opponent evaluation points: updates 450, 460, ..., 520.
- Training artifacts: 80 metric rows and 16 scheduled checkpoints.
- Invalid games across all eight in-training six-opponent evaluations: 0.

The training CLI gained an explicit `--bc-replay-split {train,valid,test}` option. Its default remains `train`; this run deliberately used `valid` because 2026-07-25 remained separate from replay.

The built-in `best.pt` remained the original update 440 because its initial 128-game minimum score was not exceeded. Fixed-node review shortlisted updates 470, 480, and 510 without consulting test25.

### Locked-candidate screening

| Candidate | vs v3, 512 games | Wilson 95% CI | Invalid |
|---|---:|---:|---:|
| update 470 | 258-254-0, 50.3906% | 46.0731%–54.7024% | 0 |
| update 480 | 252-259-1, 49.2188% | 44.9103%–53.5388% | 0 |
| update 510 | 288-224-0, 56.2500% | 51.9223%–60.4847% | 0 |

Update 510 was then locked; later strict results were not used to return to another checkpoint.

Update 510 vs v1 screening:

- 272 wins, 238 losses, 2 draws, 0 invalid.
- Scored win rate 53.1250%.
- Wilson 95% CI 48.7953% to 57.4081%.
- This showed no clear regression but did not prove superiority over v1.

### Strict update 510 vs v3

| Block | Wins-losses-draws | Win rate | Wilson 95% CI | Invalid |
|---|---:|---:|---:|---:|
| block 1, 2,048 | 1,019-1,027-2 | 49.7559% | 47.5929%–51.9197% | 0 |
| block 2, 2,048 | 1,044-1,000-4 | 50.9766% | 48.8117%–53.1378% | 0 |
| combined, 4,096 | 2,063-2,027-6 | 50.3662% | 48.8354%–51.8963% | 0 |

The predefined combined gate required at least 2,111 wins and a Wilson lower bound above 50%. Update 510 had 2,063 wins and failed.

## Latest test25 behavior comparison

All three PPO checkpoints were evaluated over the same 164,607 test25 decisions after update 510 had already been locked.

| Model | Exact action set | Ordered exact | Nonempty top-1 | Value accuracy | Count accuracy |
|---|---:|---:|---:|---:|---:|
| v3 update 440 | 70.7206% | 69.5584% | 71.9936% | 68.7200% | 99.3111% |
| v4 soup75 | 76.5709% | 75.1280% | 77.6064% | 66.6211% | 99.3512% |
| v5 update 510 | 71.1306% | 69.9916% | 72.4165% | 68.9296% | 99.3153% |

Update 510 improved over v3 by only about 0.41 percentage points in exact action set and 0.21 points in value accuracy. This is useful evidence of slight adaptation to the newer behavior distribution, but it did not translate into a statistically demonstrated head-to-head improvement.

## v4 interpretation

The exact local v4 package is:

`submissions/ptcg_ppo_marnie_v4_soup75.tar.gz`

SHA-256:

`7f98dfc79d3258c86ae1864c505658efd6e7adeabe5d2d1fabd23088c5937450`

Existing strict local evidence:

- vs v3, 2,048 games: 795 wins, 1,252 losses, 1 draw; 38.82%, CI 36.73%–40.95%.
- vs v1, 2,048 games: 824 wins, 1,224 losses; 40.23%, CI 38.13%–42.37%.
- vs Dragapult BC, 512 games: 91.80%.
- vs Kangaskhan BC, 512 games: 82.81%.

The latest test25 result confirms that v4 imitates recent Top-50 behavior much better than v3, while the strict Marnie mirror evidence says it is much weaker than v3. It is therefore a specialized trade-off, not a justified incumbent replacement. No online effect can be claimed because it has not been submitted and no live score exists.

## Artifacts and submission boundary

Key evidence:

- `artifacts/ppo_marnie_v5_recent24_v3resume520/posttrain_eval/screen512/`
- `artifacts/ppo_marnie_v5_recent24_v3resume520/posttrain_eval/strict/update-0510_vs_v3_combined_4096.json`
- `artifacts/ppo_marnie_v5_recent24_v3resume520/posttrain_eval/test25/`

Current accepted local package remains:

`submissions/ptcg_ppo_incumbent_v3_update440.tar.gz`

SHA-256:

`c50611a2d07b8f98d3bcfa72f87f98298d515f545361db4a4c79d0ef0bd66de8`

Because no new candidate passed the promotion gate, creating a v5 submission package would falsely imply that it is the new best candidate. No such package was created. No Kaggle submission or upload was performed.
