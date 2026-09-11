# Marnie BC retraining report (2026-08-13)

## Frozen data and protocol

- Exact deck: Marnie / Grimmsnarl
- Deck hash: `c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af`
- Source: recent 14-day Top50 archive through 2026-08-11
- Train: 1,247,999 decisions, 12,072 episodes; dates 2026-07-29 through 2026-08-09 plus 2026-08-11
- Validation: 29,231 decisions, 315 episodes; date 2026-08-10
- Episode overlap: zero
- Environment: `/home/xxc/miniconda3/envs/my_project_env/bin/python`, PyTorch 2.8.0+cu128, RTX 5090
- Model: 5,146,643 parameters
- Archive SHA-256: `03b097825b79ca0c30c89cfc089cfec438a4c5fa34f07a0e4c3f1e577b482af4`

## Completed runs

| Run | Initialization | Epochs | Best epoch | Train exact at its maximum | Independent validation set exact | Independent ordered exact | Result |
|---|---|---:|---:|---:|---:|---:|---|
| Scratch Marnie specialist | Random | 12 | 10 | 80.685% | 76.918% | 75.735% | Rejected |
| General BC to Marnie specialist | Current 14-day Top50 general BC | 12 | 7 | 81.965% | 77.038% | 75.769% | Rejected |
| Current 14-day Top50 general BC | Random general BC lineage | 17 total | 5 of continuation | n/a | 77.274% | 76.152% | Better than both new specialists |
| Previous recent-7-day Marnie specialist | Recent-7-day general BC | 8 | 8 | n/a | 79.915% | 78.646% | Retained BC incumbent |

The independent evaluator differs from the trainer metric by only a few tied decisions. All comparisons above use the same archive, split, deck filter, evaluator, and policy-order setting.

## Diagnosis

Choosing the most frequent exact deck was correct. The failure came from treating every occurrence of that deck as an equally useful demonstration. The 1.248M training decisions combine 32 teams. Training accuracy rose above 81%, while date-isolated validation stayed near 77%, which is direct evidence of demonstrator-policy conflict and date shift rather than insufficient optimization.

More same-deck rows therefore did not improve BC. The previous recent-7-day specialist remains about 2.88 percentage points better than the best new specialist under the matched validation protocol.

## Next data design

Keep Marnie / Grimmsnarl as the main deck, but select a coherent demonstrator cohort inside that deck. Rank teams using forward-date accuracy, persistence across recent dates, adequate episode count, and policy agreement. Train a specialist on the top coherent cohort and retain the current general BC as the initialization and regression gate. Do not run PPO until the new specialist exceeds the 79.915% matched BC incumbent.

No Kaggle submission was made from these runs.
