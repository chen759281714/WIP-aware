# 正式对比与消融实验

统一入口为 `experiments/run_formal_comparison.py`。运行、续跑和离线评价共用该入口。

## 默认配置和命令

默认扫描 `data/final_benchmark/instances` 的全部 96 个实例，验证完整 S1–S8 × P1–P4 × B1–B3 集合。
四种算法为 full、wo_mgs、wo_hlns、wo_ac，seeds 为 10–19，每次 40000 FE。
计划共 3840 次运行、153600000 FE。N=10 是每个种群规模。

`PROPOSED_PARAMS=None` 表示读取算法模块中冻结的 `DEFAULT_FULL_PARAMETERS`，解析 N_A=2N，
实际为 N=10、N_A=20、p_mut=0.35、rho=0.95、T_coop=3。
读取使用标准库 AST 提取已有常量，不导入优化器，不执行算法源码。
如需显式覆盖，给 `PROPOSED_PARAMS` 设置参数字典；省略 N_A 时随 N 联动。
这套参数是研究选定配置，不代表已证明全局最优。

```bash
python experiments/run_formal_comparison.py --mode run
python experiments/run_formal_comparison.py --mode analyze
python experiments/run_formal_comparison.py --mode all
```

也可从任意工作目录使用脚本的绝对路径启动。所有相对数据与输出路径相对于项目根目录解析。
run/all 在必要时使用 `PYTHONHASHSEED=0` 启动新解释器，spawn 工作进程继承此设置；单进程也遵守同一规则。
算法自身使用构造函数 seed 初始化私有随机流。

顶部配置可修改：执行模式、运行算法、分析算法、实例目录/子集、seed 集合、FE、参数覆盖、
单/多进程及 MAX_WORKERS、结果目录、续跑、收敛记录及间隔、绘图开关和代表实例。
参考点固定为 (1.1,1.1)。默认并行数 16。

分批运行时只改变 `SELECTED_ALGORITHMS`。全部批次完成后，将 `ANALYSIS_ALGORITHMS`
设为需要统一比较的完整集合，再执行 analyze。改变分析集合不启动优化。
不完整集合会生成 `analysis_validation_errors.json`，逐项列出缺失或无效的算法、实例、seed。

开发时使用 `SELECTED_INSTANCES`、短 FE 和独立临时结果目录；子集分析始终标记 subset。
测试中使用临时目录，不向正式结果根目录写入模拟数据。

## 注册与收敛记录

REGISTRY 保存真实模块、类、variant 和四项机制身份。Full 显式传入 HLNS_FULL；
三个消融类各自沿用其构造函数默认 variant。
运行后还检查禁用模块调用统计与 FE 统计，防止只更换显示名称。

未来外部算法需提供适配器，满足 run(diagnostic_callback)、A、n_evaluations、
目标档案、参数属性、机制身份和诊断数据合同，然后在 REGISTRY 注册。
当前只实现现有四种算法的合同，没有引入或选择外部算法。
新适配器也需要调整其专用参数、源码依赖和诊断校验；已有算法任务身份不包含其他参评算法列表。

复用原有 `run(diagnostic_callback=...)`：首次在初始化完成且档案更新后调用，之后在每轮
`update_archive(offspring)` 完成后调用。回调仅复制已有目标数值和 FE，不解码、不评价、不使用随机数。
没有修改任何算法源文件。

初始化为 2N FE，每个完整轮次再增加 2N FE，最后一轮可为剩余预算。
正式 N=10、间隔=2000 时，每个检查点恰好在档案更新边界，实际记录 2000…40000 共 20 次。
短实验自动生成 interval 的倍数以及最终 FE；最终 FE 即使不是整轮边界也会记录。
非最终检查点必须是 2N 的倍数且不早于初始化；否则拒绝配置。
原算法按整轮批量更新共享档案，因此此限制用于保证精确观察，不能用较晚档案冒充较早 FE。

## 保存、续跑与评价

每个任务保存到 `raw/<instance>/<algorithm>/seed_<seed>.json`。
保存任务身份、实际类/variant、实参和实际参数、真实 FE、耗时、完成时间、环境信息、
完整原始目标档案（保留不同基因型对应的相同目标点）、快照及现有机制统计。
不保存染色体或解码明细。最终快照必须逐项等于最终档案。

任务指纹包含参数、FE、seed、实例哈希、相关算法及 solution 源码哈希、hash seed 和检查点配置。
结果另有完整载荷校验和。跳过前验证指纹、FE、参数、机制、档案和所有快照。
存在但损坏或冲突的文件直接报错，绝不静默覆盖。SKIP_COMPLETED=False 也不允许覆盖既有结果；
需要重新实验时选择新的结果目录。缺失任务可正常重跑。

MODE、worker 数、绘图/指标代码和分析算法列表不进入优化指纹。
修改收敛记录配置会改变保存合同，因此需新的结果目录；不能从未保存的历史快照推测中间状态。
`config.json` 是最近批次配置，`configs/` 与 `manifests/` 保留各批次记录。
`.session.lock` 记录 PID、主机及时间；残留锁需人工核实旧进程已退出后处理，脚本不自动删除已有锁。

分析严格读取完整的指定集合，且不构造优化器。
每实例所有参评算法和 seeds 的最终档案共同决定两个目标的 min/max。
绝对跨度 <=1e-12 的维度视为退化，使用分母 1；最终常值映射到零，中间偏移仍保留。
其他维度按共同线性公式归一化，不裁剪中间超界值。

HV 为归一化目标空间二维精确面积并集，参考点 (1.1,1.1)，不再除以 1.21。
参考点外无法形成正面积矩形的点不贡献 HV；负归一化坐标保留并正常计算。
IGD 是最终档案联合非支配经验参考前沿到被测档案的欧氏最近距离均值。
经验前沿不是数学真实最优前沿。两个指标均去重筛选，非法/非有限物理目标和空档案会报错。

保存共同边界与诊断、原始/归一化经验前沿、单次指标、实例统计、总体和 S/P/B 分组统计、
逐次和汇总收敛指标。记录中间点超出单位区间及 HV 参考点的数量。
最佳 makespan 与最佳 shortage 分别取各自最小值，可能来自不同解。

实例内汇总使用 seed 样本标准差；总体和分组先计算实例内均值，再对实例等权平均，
其标准差是实例均值间的样本标准差。仅一个观测时 SD 留空。

每次评价按完整算法/实例/seed 集合、原始文件内容、归一化规则、指标代码版本生成独立 evaluation ID。
表格和图分别写入 `analysis/<evaluation_id>/`、`figures/<evaluation_id>/`，
`analysis/latest.json` 指向最近成功评价，防止不同评价尺度混表。
绘图默认仅代表实例，输出相同内容的 PNG/PDF 和总体 HV/IGD 柱状图。

## 轻量验证

```bash
python -m unittest tests.test_formal_comparison tests.test_full_parameter_defaults -v
```

测试包括人工指标、真实实例上的四算法短预算运行、真实 2000 FE 检查点、单/多进程一致性、
有无记录的最终基因型/目标/随机状态一致、解码次数不增加、精确快照、跳过和缺失恢复、
冲突保护、锁、分批执行、离线分析隔离、图表与评价集合隔离。
未运行 96 实例完整正式实验，也未进行显著性检验或重新调参。
