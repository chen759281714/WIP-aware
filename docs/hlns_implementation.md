# H-LNS 实现与正确性报告

## A. 实现概述

本轮新增独立 `HLNSDualPopulation`，不覆盖历史 Full。复用原双稳态种群的初始化、锦标赛、全局不可比工序对结构距离、最近邻严格相似替换、FE 终止与档案批量更新。只替换 P_S 的 specialized 分支；没有内层评价、搜索预算 B、贪心接受或候选束。

没有修改任何已有实验执行脚本、正式 benchmark、数据生成器、目标函数、encoder/decoder 或历史结果。没有启动正式性能实验、历史 pre-ablation/Core-PS Gate/CRNA 或任何新的机制可行性审计。

## B. 主机制

1. 正式 P_S 分支仍先抽取一次 `self.rng.random() < self.p_mut`，默认 p_mut=.1（任务中的 p_mu），该支路调用原 Global BasicMutation。
2. 其余默认 .9 专用支路只调用原 `diagnose_shortage()`、`high_influence_set()`；保持 rho=.8、原排序/并列处理、最小前缀和守恒校验。主算法释放集合为全部 H，不存在额外 R、截断规模或破坏比例。
3. `q=1/(1+sqrt(len(D)))`，主算法 D=H。q 与 p_mut 不同，不是改变数量配额，也没有 q_min/max、自适应或独立 OS/MS 参数。
4. OS 对全部 O 重新进行拓扑重构：约束仅为相邻 job precedence 边和父代 F 投影相邻链边。ready 按父代静态 rank 稳定排序，多候选时以 1-q 保留最早候选，以 q 按 `1/(1+rank-rank_star)` 归一化抽取其他 ready。单 ready 不抽随机数；无 stage 分块。
5. MS 按 encoder 工序顺序和实例机器枚举顺序遍历全部 D：以 1-q 保留原机器，以 q 均匀选择其他合法机器；单机器直接保持；F 的机器完全固定，不使用加工速度或负荷。
6. 同一 JointRepair 产生一个完整 OS/MS，校验工序覆盖、唯一性、工艺优先、F 投影、F 机器及关闭组件不变量。完整候选才提交一次原正式评价。
7. 不比较 child 与 selected parent 来提前接受/拒绝。原替换对象仍为 P_S 内结构距离最近者，严格 S(child)<S(nearest) 才替换。包括被 P_S 拒绝的子代在内，仍按原 round 批量档案流程参与筛选。

这里只实现联合候选生成与完整目标评价，不声称精确求解了 OS/MS 联合子问题。

### 回退与错误

- S=0、缺少已评价 diagnosis 数据或有效 H 为空：full-component 专用分支调用原 Global BasicMutation；记录原因和 global fallback 次数。
- 已验证的不变量被破坏：抛出 `HLNSDiagnosisError` 并记录轻量错误计数。未知接口/编程错误记录类型后原样抛出，不静默转换为随机回退。
- 正常 JointRepair 同码：只执行一次 H-restricted 原子回退，不重复抽样、不额外解码。
- 为避免 src 依赖实验审计脚本，`restricted_atomic_fallback()` 对原 `basic_mutation()` 的选择流做过滤：类型/工序限制于 D，最后位置/机器选择仍由原实现生成。没有复制第二份 BasicMutation。已与原验证 helper 在多种 D、仅 OS/仅 MS 可行场景上逐子代及 RNG 状态精确比对。
- 双组件 H 内原子动作耗尽时，允许最终 Global BasicMutation。单组件 H 内剩余维度耗尽时返回完整同码候选，不重新打开关闭维度，仍走一次正常评价。
- 单组件 S=0/H 不可用时，使用全局操作对象范围、仅剩余维度的原子过滤回退，无法变化则同码。这是优先保证“禁用组件不能被专用回退重新启用”的适配，不是新增搜索机制。
- 统一保留的 10% Global BasicMutation 支路不受单组件 repair 开关限制；单组件开关控制 H-LNS 专用修复及其回退，而不是改变共同的基础变异机制。
- 重复/缺失工序、非法工艺/机器、骨架改变、拓扑不完整等编码错误直接抛出，不随机重试。

## C. 统一变体

|名称|MGS|P_M AC|区域|父代偏好|OS repair|MS repair|
|---|---|---|---|---|---|---|
|HLNS_FULL|是|是|真实 H|是|是|是|
|HLNS_WO_MGS|否|是|真实 H|是|是|是|
|HLNS_WO_HLNS|是|是|不诊断/不修复|不使用|不使用|不使用|
|HLNS_WO_AC|是|否|真实 H|是|是|是|
|HLNS_WO_DIAGNOSIS|是|是|uniform random D，大小等于真实 H|是|是|是|
|HLNS_WO_PARENT_BIAS|是|是|真实 H|否|是|是|
|HLNS_WO_OS_REPAIR|是|是|真实 H|是|否|是|
|HLNS_WO_MS_REPAIR|是|是|真实 H|是|是|否|

- 所有变体共享一个 class、一套原生 run()，没有八份进化循环。P_S 一律不用 archive guide。
- WO_DIAGNOSIS 仍计算真实 H 的大小，再从 encoder 的完整 O 列表均匀无放回抽取相同数量，不按 phi/H 身份加权；它不消融诊断耗时。
- WO_PARENT_BIAS 真正对全部 ready 均匀选择、对全部合法机器（含当前机器）均匀选择，绝不是 q=1。
- WO_HLNS 保留同一个 branch draw，两支均使用 Global BasicMutation；短程运行与独立 BasicPSReference 的 genotype/objectives/archive/RNG 精确一致。
- 历史 WO_MGS 同时关闭 AC；本轮不复用其开关耦合。新 WO_MGS 保留原 `_generate_pm_child()` 及其 .1 分支和 cooperation seed 选择，只有 `_vary_makespan_specialized()` 改为 BasicMutation，因此 archive seed 仍可生效。元数据为 `BasicMutationWithArchiveCooperation`，避免进入原基类关闭协作的 shortcut。
- WO_AC 只关闭 P_M archive-guided seed 通道；archive 仍更新和输出。
- `HLNSVariantConfig` frozen；registry 只读；未知名称、矛盾开关、非 bool 开关拒绝构造，不静默退化。

## D. 新增模块和接口

生产模块：

- `src/algorithms/hlns_repair.py`：`biased_topological_os_repair()`、`alternative_probabilities()`、`biased_machine_repair()`、`parent_deviation_probability()`、`restricted_atomic_fallback()`。
- `src/algorithms/hlns_dual_population.py`：`HLNSDualPopulation`、`select_destroy_set()`、`joint_repair()`、P_S 分支和只读轻量统计；继承原 run、selection、distance、archive 和 PM seed 管线。
- `src/algorithms/hlns_variants.py`：`HLNSVariantConfig`、只读 `HLNS_VARIANTS`、`get_hlns_config()`、`make_hlns_algorithm()`。

统一入口（构造不等于运行）：

```python
from src.algorithms.hlns_variants import make_hlns_algorithm

algorithm = make_hlns_algorithm(
    operations=operations,
    buffers=buffers,
    variant="HLNS_FULL",
    N=population_size,
    N_A=archive_size,
    FE_max=evaluation_budget,
    seed=seed,
)
```

本轮没有把该入口接入任何历史 runner。N/N_A/FE_max/rho/p_mut/T_coop 等沿用现有构造参数及默认值；旧 eta 构造参数仅作基类接口兼容，H-LNS 修复不使用 eta，不将其当作新增有效参数。

### 轻量统计

`algorithm.hlns_statistics` 返回快照，包含 calls、normal children、same-genotype/global fallback、unchanged children、OS/MS/both changed、parent shortage improvement、population replacement、mean H size/ratio 和异常原因计数。

Normal、same-genotype fallback、global fallback 描述不同阶段，不是完全互斥分组；例如同码且 H 原子耗尽可能同时计入 same-genotype/global。Parent improvement 仅描述所选父代，population replacement 描述正式最近邻接受，二者不会混用。

`collect_hlns_stats=False` 关闭可选的编码变化及评价/替换观察，基础回退/错误计数仍在；不消耗额外 RNG、不调用额外 decoder。只保留当前一个待评价/待替换子代引用，无逐子代大型日志。

## E. 正确性测试

新增测试：

- `tests/test_hlns_repair.py`：覆盖/唯一/工艺/F 骨架、跨 stage、无环拓扑、q 边界、ready 稳定顺序、权重归一化、父代和远距离正概率、无偏 ready、机器合法/保留/换机/顺序、受限原子适配的精确基因型及 RNG 等价性。
- `tests/test_hlns_variants.py`：8 配置、冻结/非法开关、真实 H 与同规模随机 D、WO_MGS 保留 AC、WO_AC 保留 archive、WO_HLNS 保留 draw、单组件修复及回退、无 H/无数据、phi/anchor/minimal-prefix 错误暴露。
- `tests/test_hlns_integration.py`：多个 OS 关系和全部 H MS 同时可改变、输入不变、完整组合零中间 decode、一次子代一次评价、无父代门槛、最近邻统计分离、拒绝子代档案资格、PM MGS/AC 原行为、无 SIS/PS guide、FE 终止、批量更新、同 seed 重现与观察器开关。

实际运行结果：

- 新增 **76 项**，原有 **345 项**，合计 **421 项全部通过**。
- Failures=0，errors=0，skipped=0。
- 6 个新 Python 模块 py_compile 通过。
- 28 次显式短程 smoke，全部为 6 工序等软件测试小实例，单次 FE_max<=24，实际总计 **505 FE**。含 8 变体各 3 次 18 FE 的复现/观察开关检查、24 FE 档案节奏、13 FE 奇数预算，以及两次 18 FE Basic-PS 对照。
- 其他单步 evaluate/decoder 测试和既有回归 fixture 不混入上述显式 smoke FE。没有正式性能实验，没有 10000 FE 运行。
- 完整统计及每次 smoke FE 见 `docs/hlns_correctness_certificate.json`。

测试复现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python3 -m unittest discover -s tests
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python3 -m unittest discover -s experiments/tests
```

以上仅运行软件测试，不调用任何实验 runner 的 main。

## F. 历史源码保护

修改前记录了 **155 个既有 Python 源码及正式实例文件**的 SHA256，全部存于 `docs/hlns_historical_sha256.json`；完成测试后逐一核对，155/155 完全一致。git tracked diff 为空，新增内容全部独立。

下表“修改前=修改后”，没有改写这些文件：

|文件|前后相同的 SHA256|
|---|---|
|src/algorithms/wip_graph_dual_population.py|40961aa219e32bc23673c47dc66ec393b08baa234e52c0e8f961900e1257022a|
|src/solution/encoder.py|d759879b9a6e285a5b7a2355e0306c517be7368538e5460436422a57c1015d40|
|src/solution/decoder.py|f0139652b6f236e2bf9dac338933af5257220a86339e7103b7dca529f59e3b10|
|experiments/core_ps_variants.py|ec8b473cf976dd304cf64bdbcab498ac2e23f130f8d0582db4be5af2158dc139|
|experiments/run_pre_ablation.py|8979d11aa95da305cfee0e52c7456d47d99fdef147fe7c05b8a60f9ab12e0f49|
|experiments/run_core_ps_integration_gate.py|3c52c3900397332b0b79b64e455abde71f5b376a2eab1c3402d6918f30196074|
|experiments/diagnostics/coupled_reoptimization_audit.py|31d4b6aeff879831c72b19204794a3390a7358257ab5531169813f0dec3fbe7a|

## G. 已知边界

- 本轮只验证软件正确性，没有性能/机制有效性证据，不声称新算法优于历史 Full、Core-PS 或 Global Sequential。
- 全局拓扑修复采用稳定 ready 列表及其排序操作，优先清晰正确，没有另做性能优化。
- RestrictedBasicMutation 适配依赖原 BasicMutation 的三次 choice 合约，已测试精确等价，若将来原方法改动，该适配会明确报错，不会静默变义。
- 单组件约束只作用于 H-LNS 专用修复/回退，共同 10% Global BasicMutation 仍完整保留；无 H 时按剩余维度做全局原子回退。这一优先级已明确，不隐式混合两种语义。
- 只开放约定的 8 个配置，不允许任意组合悄悄冒充某个已命名消融。
- 所有修复和原子回退仍可产生同码，特别是无可行动的单组件实例；不会通过额外解码或重复抽样掩盖它，仍使用一次原有评价路径。

## H. 最终确认

H-LNS 候选代码、8 个可直接构造的消融配置及软件正确性测试已经完成。历史 Full、encoder/decoder、Core-PS、实验脚本、benchmark、目标函数及历史结果没有修改；没有恢复 SIS/Lambda 或方向性供需搜索，没有 CP/MILP/束搜索或内层求解器。没有开展正式实验或新的机制审计。本轮在代码及测试交付后停止，等待下一轮单独安排实验。
