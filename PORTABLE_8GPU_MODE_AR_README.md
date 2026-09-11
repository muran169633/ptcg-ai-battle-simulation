# PTCG mode-AR update40 PPO: portable 8x A100 package

This package trains one synchronized policy on eight GPUs. Its immutable
starting point is the stronger submitted Dragapult mode-AR PPO update40, not
the weaker non-AR BC included in the older portable package.

## Start

```bash
unzip ptcg_mode_ar_u40_8gpu_ppo_20260816.zip
cd ptcg_mode_ar_u40_8gpu_ppo_20260816
chmod +x setup_and_run_8gpu_mode_ar.sh
./setup_and_run_8gpu_mode_ar.sh
```

Defaults target 8x NVIDIA A100 40GB:

- 8 synchronized NCCL ranks;
- 4 official-engine rollout workers per rank, 32 workers globally;
- 8,000 global updates;
- 256 official-engine games per rank/update, 2,048 globally;
- 16,384,000 planned training games;
- 4,096 PPO rows per rank minibatch, 32,768 effective globally;
- 70% recent-day meta, 20% inverse-window hard mining, 10% current-policy self-play;
- candidate snapshot every 5 updates and full checkpoint every 50 updates;
- 1,000-game champion gate every 5 updates;
- candidate must reach 54% to replace the incumbent;
- failed candidates are saved, then weights and Adam state roll back to the incumbent.

The initial `best.pt` is parameter-identical to submitted update40. It remains
the deployable best until a candidate passes the gate.

## Monitoring

```bash
tail -f runs/dragapult_mode_ar_u40_8gpu.train.log
tail -f runs/dragapult_mode_ar_u40_8gpu/metrics.jsonl
nvidia-smi
```

Checkpoints are written only by rank 0:

```text
runs/dragapult_mode_ar_u40_8gpu/best.pt
runs/dragapult_mode_ar_u40_8gpu/checkpoints/update-0050.pt
runs/dragapult_mode_ar_u40_8gpu/candidates/update-0010.pt
runs/dragapult_mode_ar_u40_8gpu/champions/
```

## Continue from a saved checkpoint

The trainer intentionally starts a new audited output directory and a fresh
optimizer. Pass a safe rank-0 checkpoint as the next anchor:

```bash
./setup_and_run_8gpu_mode_ar.sh \
  --output-dir runs/dragapult_mode_ar_u40_8gpu_phase2 \
  --anchor-checkpoint runs/dragapult_mode_ar_u40_8gpu/checkpoints/update-8000.pt
```

Do not reuse an existing output directory. Candidate files are pre-gate
snapshots; use `best.pt`, a promoted champion, or a post-rollback periodic
checkpoint as a continuation anchor.
