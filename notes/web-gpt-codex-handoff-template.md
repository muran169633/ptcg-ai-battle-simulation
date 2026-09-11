# 网页 GPT 思考 + Codex 写代码 落地模板（PTCG 用）

这个模板用于你先在网页 GPT 里做策略思考，再交给我（Codex）做代码实现。目标是每次只改“可执行”内容，避免模糊对话导致方向偏移。

## 一、先写网页思考稿（你在网页版 GPT 里完成）

- 目标：本轮要解决什么问题？（例如：提升 U477 vs u472 对抗表现）
- 现状：现有瓶颈是什么？（胜率、统计区间、对手、退化迹象）
- 假设：改变 A/B/C 哪个环节会带来收益？
- 证据：可引用最近哪些对战日志/实验结果？
- 风险：可能出现什么副作用？（过拟合/退化/校验失败）
- 预期收益：想要看到哪些明确数值变化？
- 收敛条件：多少胜率/CI 条件/门控条件达到就算成功？
- 回退方案：失败时回滚到哪个模型（版本号）

## 二、生成给 Codex 的结构化包（可直接落地）

用这个命令生成一个任务单（先填参数）：

```bash
python scripts/web_gpt_codex_bridge.py new \
  --title "对 S16 跑道的 v1/v3/BC 对战升级" \
  --goal "把 update-477 的 BC+PPO 方案做可复验对比并固定在 1024 局以上" \
  --hypothesis "BC 主导训练先收紧，PPO 仅做轻量优化; 数据仍采用最新两周 top20/top50，分别做消融核验; 使用 ordered Plackett-Luce replay 与 v3/v1 统一评估口径" \
  --metric "以 256/512 条件下 vs v3 的胜率 CI 下界>50% 为主，优先关注 1024 局均值" \
  --output notes/webgpt_<日期>.json
```

生成后再补齐 JSON 字段：
- `problem_statement`
- `experiment_plan`
- `data_scope`
- `notes_for_codex`（给 Codex 的具体指令）

## 三、交给 Codex 的执行规则

1) 只改动命名明确、可追踪的产物目录（`artifacts/`、`submissions/`、对应 `run_.../.py`、`tools/...`）。
2) 实验必须有「本地严格评测」与「门控报告」：
   - v3/v1 对照
   - 空任务统计必须可复用
   - 需要说明 CI/置信区间与每个对手的分层胜率
3) 每次失败必须输出对照项（哪个指标退化、可能的根因、回退步骤）。
4) 全流程保留可复现输入：命令、数据路径、随机数种子、脚本 hash、结果文件位置。

## 四、你确认后直接发我：

- 1 个已填写的 `notes/*.json`（由上一步创建）
- 你要执行的优先级（立即 / 本周 / 观察）
- 是否允许我顺序尝试 2-3 个小改动（推荐）

我会按文件先做：

- 执行 `validate` 与 `render`：

```bash
python scripts/web_gpt_codex_bridge.py validate --plan notes/xxx.json
python scripts/web_gpt_codex_bridge.py render --plan notes/xxx.json
```

- 按 `tasklist.md` 把代码改动落到位，并同步回填结果到该 json 的 notes 和状态里。

## 五、自动化一键流程（推荐）

直接用 `auto` 一条命令即可：

```bash
python scripts/web_gpt_codex_bridge.py auto \
  --title "对战升级路线" \
  --goal "拉升 update-477 在 1024 局上对 v3 的胜率" \
  --hypothesis "先固定数据版本; 先跑 BC 门控再做 PPO 调参; 评估用席位平衡并发 256+ 局" \
  --metric "CI 下界 > 50% 且优于当前候选" \
  --output notes/20260806_u477_upgrade.json
```

这会自动输出：

- `notes/20260806_u477_upgrade.json`
- `notes/20260806_u477_upgrade.tasklist.md`

然后你只需把 JSON 发我，我就按同一份文件继续执行下去。

如果你已经有网页端整理好的 JSON 决策稿，也可以直接导入：

```bash
python scripts/web_gpt_codex_bridge.py ingest \
  --spec /path/to/web_draft.json \
  --output notes/20260806_u477_upgrade.json
```

说明：当前环境无法自动读取你的浏览器会话，必须把网页结论以文件/文本贴入本地后再导入。

## 六、文件输出建议

- 计划单：`notes/<日期>-<slug>.json`
- Codex 任务单：`notes/<日期>-<slug>.tasklist.md`
- 结果日志：按你已有的 `artifacts/...` 目录放评估产物
- 最终结论：更新到 `notes/` 下的同前缀文件里，避免散落
