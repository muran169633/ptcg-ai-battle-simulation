#!/usr/bin/env bash
set -uo pipefail

repo_root="/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation"
run_root="${repo_root}/artifacts/ppo_gold_duration16_cosine_tailconsensus5_ow8_ctx34q2_ordered_seed20260756_20260730"
log_path="${run_root}.log"
exitcode_path="${run_root}.exitcode"
pid_path="${run_root}.pid"

printf '%s\n' "$$" > "${pid_path}"
PYTHONUNBUFFERED=1 \
  /home/xxc/miniconda3/bin/python \
  "${repo_root}/tools/run_ppo_duration16_cosine_20260730.py" \
  > "${log_path}" 2>&1
status=$?
printf '%s\n' "${status}" > "${exitcode_path}"
exit "${status}"
