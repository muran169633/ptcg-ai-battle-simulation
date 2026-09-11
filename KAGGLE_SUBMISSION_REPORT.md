# Kaggle PPO 提交包报告

生成日期：2026-07-23

## 可上传文件

```text
submissions/ptcg_ppo_terminal01_v1.tar.gz
SHA-256:
2148a1ed34e3fc2e0ae7495ce1ad740f649c8178e1ce9fb755b406ea48473889
大小:
19,059,397 bytes
```

官方要求提交 `.tar.gz`，其中 `main.py` 必须位于压缩包顶层，并包含
`deck.csv`。本包顶层恰好包含：

```text
main.py
deck.csv
model.pt
policy_runtime.py
```

没有外层目录，也没有打入 `__pycache__`、训练数据、optimizer 或官方引擎。
Kaggle 会把这些文件放在 `/kaggle_simulations/agent/`；入口使用
导入的兄弟模块 `policy_runtime.__file__` 定位牌组和权重。这里不能直接
依赖顶层 `main.py` 的 `__file__`，因为 Kaggle agent loader 使用 `exec`
加载入口时不会注入它。

## 模型来源

`model.pt` 来自 PPO update 200：

```text
artifacts/ppo_marnie_terminal01_v1/final.pt
```

提交模型移除了 Adam optimizer 和训练日志，将文件从 61,927,871 bytes
缩小到 20,629,575 bytes。80 个 `model_state_dict` tensor 已逐一使用
`torch.equal` 对照，权重未做 FP16、量化或近似压缩。

提交侧模型保持：

- 61 类 count head，对应 `0..60`；
- 动态 `[minCount, maxCount]` mask；
- ordered Plackett-Luce 逐步 greedy；
- `SKILL_ORDER` 动作不按索引重新排序；
- CPU lazy loading，初始牌组请求不会加载模型。

## 本地验证

- 从原提交目录导入并完成官方引擎自博弈：214 decisions，非法动作 0；
- 提交动作与原训练模型整局逐决策比较：202/202 完全相同；
- 对不同牌组随机策略进行 16 局合法性测试：16 局完成，非法动作 0；
- 从最终 `.tar.gz` 解压到全新临时目录，并用 Kaggle 相同的
  `env={}; exec(compile(...), env)` 方式加载后完成官方引擎整局：
  215 decisions，非法动作 0；
- CPU 首次模型调用约 43 ms，后续决策中位数约 2.9 ms；
- `main.py` 与 `policy_runtime.py` 已通过 `py_compile`。

引擎随机洗牌，完整对局的决策数和结果每次可能不同。

## Kaggle 提交结果

已于 2026-07-23 上传并完成 Kaggle Validation Episode：

```text
submission_ref: 54931100
description: Marnie PPO terminal01 update200
status: SubmissionStatus.COMPLETE
public_score: 600.0
```

实际上传命令：

```bash
kaggle competitions submit \
  pokemon-tcg-ai-battle \
  -f submissions/ptcg_ppo_terminal01_v1.tar.gz \
  -m "Marnie PPO terminal01 update200"
```

本次只上传一次，没有在 `PENDING` 状态下重复提交。Kaggle 随后将记录从
`SubmissionStatus.PENDING` 更新为 `SubmissionStatus.COMPLETE`。
