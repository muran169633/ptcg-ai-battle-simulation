# PTCG 纯 JAX 规则引擎

## 完成状态

| 项目 | 完成结果 |
|---|---|
| 状态机 | 固定 shape 的 JAX PyTree |
| 合法动作 | 与官方 `SelectType` / `SelectContext` 对齐 |
| 卡牌与牌区 | 全部通过定长索引数组管理 |
| 效果系统 | opcode 解释器与 JAX 特例 handler |
| 触发和时序 | 显式效果栈、触发栈、KO 队列 |
| 随机规则 | 每局独立 JAX key；差分使用 random tape |
| 终局规则 | 奖品牌、无宝可梦、无法抽牌及官方结束原因 |
| 批量执行 | `jit + vmap + lax.scan` |
| C++ 依赖 | 仅限离线 oracle，不进入训练热路径 |
| 规则验证 | 官方引擎逐决策差分 |
| 编译管理 | 固定 shape profile 与持久化 XLA 缓存 |

## 运行架构

```text
官方本地规则数据
        |
        v
离线规则编译器 --------------------+
        |                           |
        v                           v
定长 rule artifacts          coverage manifest
        |
        v
纯 JAX 规则运行时
  reset_batch
  step_batch
  rollout_chunk
        |
        v
JAX policy + PPO update

官方 C++ 引擎
        |
        v
canonical trace + random tape
        |
        +----------> 与 JAX 逐决策差分
```

规则数据和运行时代码彼此分离：

- 卡牌数据被编译为设备常驻数组；
- 通用卡牌效果被编译为 opcode；
- 少量无法数据化的行为由静态 JAX handler 执行；
- 官方 C++ 不参与任何生产 step；
- 训练过程中不会解析 CSV、JSON、规则文本或 Python 对象。

## 代码结构

```text
ptcg_jax/
  config.py
  enums.py
  types.py
  reset.py
  step.py
  auto_advance.py
  actions.py
  observation.py
  rng.py
  rules/
    zones.py
    setup.py
    turn.py
    pokemon.py
    energy.py
    trainer.py
    combat.py
    status.py
    modifiers.py
    timing.py
    terminal.py
  effects/
    opcodes.py
    interpreter.py
    handlers.py
    tables.py
  compiled/
    loader.py
    schema.py
  parity/
    canonical.py
    trace.py
    compare.py
  tests/
  benchmarks/

tools/
  build_ptcg_jax_tables.py
  audit_ptcg_rule_coverage.py
  export_ptcg_cpp_traces.py
  compare_ptcg_jax_trace.py
  measure_ptcg_jax_compile.py
```

## 定长状态

### 容量

官方引擎确认的基础容量已经直接进入状态 schema：

| 对象 | 固定容量 |
|---|---:|
| 玩家 | 2 |
| 单副牌 | 60 |
| 默认备战区 | 5 |
| 最大备战区 | 8 |
| 初始奖品牌 | 6 |
| 全局卡实例池 | 128 |
| 单次选择序列 | 60 |

`MAX_OPTIONS`、`MAX_EFFECT_STACK`、`MAX_TRIGGER_STACK`、`MAX_KO_QUEUE` 和 `MAX_INTERNAL_OPS` 保存在 shape profile 中。它们来自官方源码静态审计与全量 trace 峰值审计。运行时达到容量上限会返回确定错误码，不会截断、覆盖或跳过规则。

### GameState

完整真实状态是单个 JAX PyTree：

```python
class GameState(NamedTuple):
    # 对局状态
    turn: jax.Array
    phase: jax.Array
    active_player: jax.Array
    first_player: jax.Array
    result: jax.Array
    finish_reason: jax.Array

    # 当前官方 Select 边界
    select_type: jax.Array
    select_context: jax.Array
    select_player: jax.Array
    select_min: jax.Array
    select_max: jax.Array
    option_data: jax.Array
    option_mask: jax.Array

    # 卡实例主表
    card_id: jax.Array
    card_owner: jax.Array
    card_zone: jax.Array
    card_zone_index: jax.Array
    card_damage: jax.Array
    card_flags: jax.Array
    card_counters: jax.Array

    # 有序牌区
    zone_cards: jax.Array
    zone_length: jax.Array

    # 规则执行栈
    selected: jax.Array
    selected_count: jax.Array
    effect_stack: jax.Array
    effect_stack_size: jax.Array
    trigger_stack: jax.Array
    trigger_stack_size: jax.Array
    ko_queue: jax.Array
    ko_queue_size: jax.Array
    scratch: jax.Array

    # 随机与诊断
    rng_key: jax.Array
    random_event_count: jax.Array
    transition_count: jax.Array
    error_code: jax.Array
```

实际手牌数量、option 数量、效果栈深度和牌组内容都是数组值，不是 static argument，因此不会因不同牌组或对局状态触发 XLA 重编译。

### 状态不变量

每次原子规则转移均保持：

- 每个有效卡实例只存在于一个合法位置；
- `zone_cards`、`card_zone` 和 `card_zone_index` 双向一致；
- 双方卡牌总数守恒；
- 有序牌区保留官方顺序；
- 活跃区、备战区和附属引用有效；
- 效果栈、触发栈和 KO 队列没有悬空引用；
- 终局状态不会再产生玩家决策；
- 私有状态不会泄露到对手 observation。

debug profile 在每个内部转移后检查完整不变量；train profile 保留容量、索引和错误状态检查。

## 已重写的规则

### 开局

纯 JAX reset 完整处理：

- 卡组合法性；
- 洗牌；
- 起手抽牌；
- 基础宝可梦检查；
- mulligan 与对手补牌选择；
- 奖品牌设置；
- 先后手；
- 战斗宝可梦与备战宝可梦设置；
- 第一个官方决策边界。

### 回合状态机

回合状态机覆盖：

- 回合开始；
- 回合抽牌；
- 主阶段；
- 攻击阶段；
- 回合结束；
- Pokemon Checkup；
- 玩家切换；
- 回合级限制重置；
- 所有阶段对应的终局检查。

状态机不依赖 Python 分支。phase 和 context 通过数组枚举、`lax.cond` 与 `lax.switch` 分派。

### 卡牌移动和牌区

所有卡牌移动均由统一原语完成：

```text
remove_from_zone
insert_into_zone
move_card
move_cards_ordered
shuffle_zone
attach_card
detach_card
reveal_cards
restore_visibility
```

统一原语负责更新卡实例位置、牌区顺序、区域长度、公开状态、附属关系和相关触发。具体效果不能直接改写牌区数组，从而避免同一卡实例出现在多个区域。

覆盖的区域包括牌库、手牌、弃牌堆、奖品牌、战斗场、备战区、附属卡区域和引擎使用的临时区域。

### 宝可梦规则

JAX 运行时处理：

- 基础宝可梦上场；
- 进化及本回合上场限制；
- 退化；
- 战斗宝可梦与备战宝可梦交换；
- 撤退费用和撤退限制；
- HP、伤害指示物和状态继承；
- KO 收集与同时 KO；
- 奖品牌领取；
- KO 后新战斗宝可梦选择；
- 无可用宝可梦时的终局。

### 能量规则

能量求解器处理：

- 基本能量和特殊能量；
- 每回合一次手动附能；
- 效果附能；
- 能量移动和弃置；
- 撤退费用；
- 攻击费用；
- 多类型、任意类型和替代类型；
- 费用降低、增加、忽略和禁止效果。

费用检查输出确定的合法组合，不使用浮点数或启发式近似。

### 训练家牌

物品牌、支持者和竞技场均进入统一动作与效果管线，覆盖每回合次数、使用条件、目标选择、场地替换、持续 modifier 和离场清理。

### 攻击与伤害

攻击流水线固定为：

```text
检查攻击合法性
  -> 检查并支付费用
  -> 攻击前效果
  -> 计算基础伤害
  -> 替换效果
  -> 加减修正
  -> 弱点和抵抗
  -> 免疫与上下限
  -> 放置伤害
  -> 攻击后效果
  -> 触发收集
  -> KO 检查
```

伤害、伤害指示物、治疗、伤害移动、弱点、抵抗、免疫和持续伤害修正均使用整数张量。

### 特殊状态

中毒、灼伤、睡眠、麻痹和混乱全部在状态张量中显式表示。Pokemon Checkup 按官方顺序处理状态伤害、硬币结果、状态恢复、触发和 KO。

### 特性、触发和时序

主动特性、被动特性、once-per-turn、来源失效、来源离场、复制与持续效果均通过统一生命周期管理。

一次选择后的时序为：

```text
动作校验
  -> 支付费用
  -> 当前效果入栈
  -> 执行原子操作
  -> 收集新触发
  -> 同时触发排序
  -> 状态检查与 KO 队列
  -> 奖品牌和替补选择
  -> 恢复效果栈
  -> 下一个 Select / Checkup / 终局
```

每个效果记录 `timing_class`、`source_card`、`source_skill`、`owner`、`target` 和 `priority`，执行顺序不依赖 Python 调用顺序。

### 终局

终局检查与官方结束原因对齐，覆盖：

- 奖品牌取完；
- 对手没有可用战斗宝可梦；
- 回合开始无法抽牌；
- 投降；
- 同一原子边界出现多个终局条件；
- 官方错误和异常终止语义。

winner、finish reason、最后一个合法决策和 reward 在同一终局转移中生成。

## 合法动作

一次官方 `Select(...)` 对应一个环境 step：

```python
class Action(NamedTuple):
    indices: jax.Array  # int16 [MAX_SELECTION]
    count: jax.Array    # int16 []
```

每个决策点输出：

```text
select_type
select_context
select_player
select_min
select_max
option_count
option_data[MAX_OPTIONS, OPTION_FIELDS]
option_mask[MAX_OPTIONS]
```

动作校验依次检查数量范围、index 范围、mask、唯一性和顺序语义。非法动作返回明确错误码，并保证状态按官方错误语义保持不变。

option 比较使用稳定整数身份，不依赖本地化显示文本。option 内容、顺序、`select_min` 和 `select_max` 全部参与 C++ parity。

## 效果解释器

通用卡牌规则被编译为定长 opcode：

```text
DRAW
MOVE_CARD
SHUFFLE
DAMAGE
HEAL
MOVE_DAMAGE_COUNTER
ATTACH_ENERGY
DISCARD_ENERGY
SET_FLAG
ADD_COUNTER
CLEAR_STATE
PUSH_SELECTION
PUSH_TRIGGER
JUMP_IF
SWITCH_ACTIVE
DEVOLVE
TAKE_PRIZE
```

解释器通过有界 `lax.while_loop` 执行。所有 program、operand、跳转位置和有效 mask 都是定长数组。

无法由通用 opcode 完整表达的效果被映射到稳定 JAX handler ID。handler 通过分组 `lax.switch` 调用，仍经过统一效果栈、触发栈、modifier、KO 和终局流程。

规则覆盖清单为每个 card、attack、skill 和 effect 记录：

```json
{
  "rule_id": "effect-or-skill-id",
  "implementation": "opcode|special_handler|unreachable",
  "source_hash": "...",
  "unit_tests": 3,
  "fixture_traces": 12,
  "random_trace_hits": 842,
  "first_divergence_count": 0,
  "max_internal_ops_seen": 17
}
```

生产 coverage 中不存在 `unsupported` 或默认 no-op。`unreachable` 项具有静态引用证据。

## 随机规则

每个并行对局持有独立 JAX PRNG key。洗牌、硬币、随机目标和随机选牌都显式消费 key，不使用 Python 或 NumPy RNG。

官方 C++ 使用的随机算法与 JAX PRNG 不同，因此 parity 不依赖“相同整数 seed 应产生相同序列”的错误假设。离线验证使用 random tape：

```text
event_index
event_type
range_or_shape
sampled_value
source_card
source_effect
```

C++ 和 JAX 消费同一 tape，逐事件比较类型、范围、结果和消费位置。生产 JAX PRNG 另行通过确定性与分布测试。

## 玩家观察

`GameState` 保存完整真值，`observation_for_player(state, player)` 生成单方可见 observation。

观察层隐藏：

- 对手手牌内容；
- 双方未知牌库顺序；
- 未公开奖品牌；
- 已结束公开窗口的信息。

公开数量、展示窗口和检索产生的临时信息严格跟随官方状态。padding、stable identity、排序和 mask 不携带隐藏信息。

## 批量执行

单局规则函数保持纯函数形式，批量维由 `vmap` 添加：

```python
reset_batch = jax.jit(jax.vmap(reset))
step_batch = jax.jit(jax.vmap(step))
```

玩家动作完成后的自动推进使用：

```python
state = lax.while_loop(
    lambda s: (~needs_player_input(s)) & (~is_terminal(s)) & (~has_error(s)),
    advance_one_internal_transition,
    state,
)
```

完整 rollout 使用固定长度 `lax.scan`。已经终局的 batch 元素通过 done mask 保持静止，不改变整体 shape。

## 编译

### 编译单元

环境和训练被拆分为四个稳定 executable：

| 单元 | 内容 |
|---|---|
| `reset_batch` | 批量开局到首个决策边界 |
| `step_batch` | 一次动作和后续自动规则推进 |
| `rollout_chunk` | 固定长度环境—策略 scan |
| `ppo_update` | GAE、minibatch 和 PPO 更新 |

该拆分避免任意规则改动都重新编译完整训练图，也允许单独保存每个 executable 的编译报告。

### Shape profile

系统使用三个固定 profile：

| Profile | 用途 |
|---|---|
| `debug` | 小 batch、完整不变量和差分字段 |
| `eval` | 固定评估 batch、确定性策略 |
| `train` | 固定每设备 batch 和 rollout length |

profile 固定 `MAX_OPTIONS`、选择长度、栈容量、内部操作上限、rollout 长度、local batch、minibatch 数量和 dtype。

牌组、卡牌 ID、phase、option 数和实际栈深不是 static argument。更换牌组或进入新的规则路径不会生成新 executable。

### 编译缓存

持久化缓存键包含：

- JAX、JAXLIB 和 XLA 版本；
- 后端、GPU 和 compute capability；
- XLA flags；
- 源码版本；
- ruleset schema；
- shape profile 与 dtype。

相同配置的新进程直接加载缓存。不同 GPU、JAX 版本或 shape profile 不共享不兼容缓存。

### 编译时间

编译报告分别记录 rule build、Python tracing、lowering、XLA compile、cache load、device warm-up 和 warm execution。

冻结的生产时间边界为：

| 指标 | 时间边界 |
|---|---:|
| 规则 artifact 构建 | 不超过 60 秒 |
| `reset_batch` 冷编译 | 不超过 2 分钟 |
| `step_batch` 冷编译 | 不超过 5 分钟 |
| `rollout_chunk` 冷编译 | 不超过 10 分钟 |
| `ppo_update` 冷编译 | 不超过 10 分钟 |
| 全部生产单元累计冷编译 | 不超过 25 分钟 |
| 缓存命中新进程启动 | 不超过 60 秒 |
| device warm-up | 不超过 60 秒 |
| 单次正式训练意外重编译 | 0 次 |
| 编译与 warm-up 占训练墙钟 | 不超过 5% |

具体机器结果不硬编码在 README 中，而保存在 `reports/ptcg_jax_compile_*.json`，其中同时记录硬件、版本、StableHLO 大小、executable 大小和 peak memory。

JAX 异步计时统一调用 `block_until_ready()`：

```python
t0 = monotonic()
compiled = fn.lower(*args).compile()
compile_seconds = monotonic() - t0

t0 = monotonic()
out = compiled(*args)
jax.tree.map(lambda x: x.block_until_ready(), out)
execute_seconds = monotonic() - t0
```

空缓存、缓存命中新进程和 warm 进程分别测量，不把首次执行时间误报为稳定吞吐。

### 编译收益

环境报告使用以下公式计算 JAX 相对 C++ 的冷编译回本步数：

```text
break_even_steps = compile_seconds /
                  (1 / cpp_steps_per_second - 1 / jax_steps_per_second)
```

训练 manifest 同时记录总训练步数、回本步数和编译成本占比。吞吐统计只计算完整规则环境 step，不把单独的模型前向速度当作环境速度。

## 官方 C++ 逐决策差分

### Canonical trace

官方 oracle 在每个 `Select` 边界记录：

```text
engine_hash
ruleset_hash
seed
decision_index
random_tape_position
canonical_full_state
player_observation
select_type
select_context
select_min
select_max
ordered_options
selected_action
post_state
terminal_result
finish_reason
error_code
```

canonical state 不包含指针、容器地址、未初始化 padding 或本地化文本。

### 每一步比较

JAX 重放相同初始状态、动作和 random tape，并依次比较：

1. 当前 phase 和 active player；
2. 完整卡实例状态；
3. 所有有序牌区；
4. 效果栈、触发栈和 KO 队列；
5. 玩家 observation；
6. `SelectType`、`SelectContext` 和 min/max；
7. option 内容及顺序；
8. random tape 消费位置；
9. 动作后的完整状态；
10. winner、finish reason 和 error code。

比较器在第一个分叉停止，输出字段级 diff、卡牌/技能/effect ID、决策编号和可独立重放的最小 trace。

### 差分测试集合

持续验证包含：

- 每个状态原语的单元测试；
- 每个规则域的最小 C++ fixture；
- 固定 replay 逐步重放；
- 从官方合法 options 采样的随机差分；
- Marnie mirror；
- cross-deck 对局；
- 全卡池 card/attack/skill/effect 覆盖；
- 最大/最小选择数量、同时触发和容量边界；
- 非法动作错误语义；
- 完整对局终局分布回归。

生产构建满足：

```text
unsupported rules       = 0
option divergences      = 0
state divergences       = 0
random tape divergences = 0
terminal divergences    = 0
unexpected recompiles   = 0
```

相似胜率不代替逐决策零分叉；差分报告哈希与 ruleset hash 一起写入训练 manifest。

## 规则构建

```bash
python tools/build_ptcg_jax_tables.py \
  --output build/ptcg_jax_rules

python tools/audit_ptcg_rule_coverage.py \
  --rules build/ptcg_jax_rules \
  --report reports/ptcg_jax_rule_coverage.json
```

规则构建输出：

```text
build/ptcg_jax_rules/<ruleset_hash>/
  card_table.npz
  effect_programs.npz
  handler_map.json
  capacity_report.json
  coverage_manifest.json
  BUILD_INFO.json
```

相同输入连续构建两次得到相同 artifact 哈希。checkpoint、trace、benchmark 和训练数据都绑定相同 `ruleset_hash`。

## 差分验证

```bash
bash tools/build_ptcg_jax_oracle.sh \
  --output build/ptcg_jax_oracle/libcg_jax_oracle.so

python tools/export_ptcg_cpp_traces.py \
  --engine build/ptcg_jax_oracle/libcg_jax_oracle.so \
  --deck0 decks/marnie.csv \
  --deck1 decks/marnie.csv \
  --seeds 1000:1100 \
  --output data/ptcg_jax_traces/marnie_mirror_v1.jsonl

python tools/compare_ptcg_jax_trace.py \
  --trace data/ptcg_jax_traces/marnie_mirror_v1.jsonl \
  --stop-at-first-mismatch \
  --report reports/ptcg_jax_parity.json

pytest -q ptcg_jax/tests
```

## 编译与性能验证

```bash
python tools/measure_ptcg_jax_compile.py \
  --profile train \
  --cold-cache-report reports/ptcg_jax_compile_cold.json \
  --warm-cache-report reports/ptcg_jax_compile_cached.json

python -m ptcg_jax.benchmarks.bench_rollout \
  --profile train \
  --warmup 3 \
  --repeats 10 \
  --output reports/ptcg_jax_benchmark.json
```

## PPO 接入

生产路径在 JAX 中完成：

```text
batched reset
  -> observation
  -> policy forward
  -> ordered Plackett-Luce action
  -> batched step
  -> rollout scan
  -> GAE
  -> PPO minibatches
  -> optimizer update
```

PyTorch checkpoint 通过权重转换器进入 JAX policy，并在固定输入上对齐 logits、value、mask 后概率、ordered sampling、log-prob、advantage、KL、clip fraction 和单次 optimizer 更新。

环境 parity 与 PPO parity 使用独立报告，防止规则正确但训练数值漂移，或训练指标接近但环境规则错误。

## 运行清单

每个训练 run 保存：

```text
reports/
  ptcg_jax_rule_coverage.json
  ptcg_jax_parity.json
  ptcg_jax_compile_cold.json
  ptcg_jax_compile_cached.json
  ptcg_jax_benchmark.json

run_manifest.json
```

manifest 记录官方源码和 `libcg.so` 哈希、ruleset hash、JAX 源码版本、shape profile、JAX/JAXLIB/XLA、GPU、驱动、XLA flags、编译缓存命中、executable 签名、parity 报告、checkpoint 和训练数据版本。

## 最终结果

纯 JAX 目标路径已经完成为以下闭环：

```text
官方规则数据
  -> 可复现规则编译
  -> 定长 JAX 状态和规则执行
  -> jit/vmap/scan 批量 rollout
  -> 官方 C++ 逐决策零分叉验证
  -> 固定编译时间与缓存约束
  -> 端到端 JAX PPO
```

状态机、合法动作、卡牌移动、效果和终局规则都由 JAX 张量运算执行。官方 C++ 不在生产运行路径中，只负责持续证明 JAX 行为没有偏离比赛规则。
