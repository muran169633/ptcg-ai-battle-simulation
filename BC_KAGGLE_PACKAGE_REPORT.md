# 纯 BC Kaggle 基线包报告

生成日期：2026-07-24

## 产物

未上传 Kaggle。可供后续手动提交的文件为：

```text
submissions/ptcg_bc_orbit_v5.tar.gz
SHA-256:
4113c591bbc310c596ea485e9f46cba1823cd50105848fd15bd8475ec531fc8b
大小:
19,060,490 bytes
```

压缩包只有四个顶层成员：

```text
main.py
deck.csv
model.pt
policy_runtime.py
```

没有外层目录，也没有包含训练数据、optimizer、`kaggle.json` 或
`__pycache__`。

## 模型和推理行为

权重来自：

```text
artifacts/bc_marnie_luca_orbit_v5/best.pt
SHA-256:
259f0c9ad44a3aa71d38305a1f6a21188f0fdf9fd7166fefdc5ed3e8482c0968
```

提交侧精简权重：

```text
submissions/ptcg_bc_orbit_v5/model.pt
SHA-256:
1c7ff437743b0f699655e683bfdae1e4c542b3818dc8f41876751473621eca8a
```

原 checkpoint 与精简模型的 80 个 `model_state_dict` tensor 已逐一使用
`torch.equal` 比较，差异为 0。BC count head 保持原始 17 类，即
`0..16`，其权重形状为 `[17, 128]`。

推理使用确定性 BC 策略：

- 固定数量选择直接服从 `minCount == maxCount`；
- 弹性数量在 17 类 count head 中施加合法区间 mask；
- 选项按 pointer logit 贪心无放回选取；
- 返回前按选项索引升序排列，保持 BC 专家回放的 canonical order；
- 若以后出现弹性选择且 `minCount > 16`，安全选择合法的最小数量；
- 使用 `policy_runtime.__file__` 定位同目录资源，兼容 Kaggle 对
  `main.py` 的 `exec()` 加载方式。

牌组文件与训练使用的 Marnie’s Grimmsnarl ex + Froslass 牌表完全一致，
共 60 张：

```text
deck.csv SHA-256:
92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d

canonical deck hash:
c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af
```

## 本地验证

- `main.py` 和 `policy_runtime.py` 完成无落盘字节码的语法编译；
- 使用 Kaggle 等价的 `env={}; exec(compile(...), env)` 成功加载；
- 打包策略与原始冻结 BC 策略连续 442 次决策完全一致；
- 从最终 `.tar.gz` 解压至全新临时目录后，完成 16 局官方引擎自对局；
- 共执行 3,336 次决策，非法动作 0，引擎选择错误 0；
- 所有多选动作均为 canonical ascending order；
- CPU 平均决策耗时约 2.98 ms，首次模型加载调用约 28.12 ms；
- tar 成员、gzip 完整性和 SHA-256 sidecar 均已校验通过。

该包是用于隔离比较 PPO 是否真实带来线上收益的纯 BC 基线，不包含任何
PPO 权重或 PPO 推理逻辑。
