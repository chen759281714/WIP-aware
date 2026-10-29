# 四种外部对比算法：第一阶段实现与验证

本阶段仅新增算法、公共基础操作与测试。没有注册进正式实验脚本，也未运行正式对比、调参或计算正式 HV/IGD 排名。

## 文件及接口

- `comparison_common.py`：Solution、统一解码评价、FE、快照、Pareto 分层与拥挤距离。
- `comparison_operators.py`：重复工件序列转换、POX、OS 插入、随机 MS 重分配。
- `nsga2.py`：NSGAII，最终输出来自 population。
- `moead.py`：MOEAD，最终输出来自子问题 population。
- `spea2.py`：SPEA2，最终输出来自 archive。
- `mpmoga.py`：MPMOGA，最终输出来自 AUS archive。
- `tests/test_comparison_algorithms.py`：机制单测及真实实例轻量测试。

四种类均支持 `(operations, buffers, FE_max=40000, seed=None, crossover_probability=.9,
mutation_probability=.1, snapshot_interval_fe=2000)`，并使用私有 random.Random。
NSGAII、MOEAD、SPEA2 使用 population_size=100；SPEA2 使用 archive_size=100。
MOEAD 使用 neighborhood_size=20（不超过种群规模）、neighborhood_mating_probability=.9、max_replacements=2。
MPMOGA 使用 makespan_population_size=105、shortage_population_size=105、archive_size=210，不接收统一 population_size。

调用 `run(diagnostic_callback=None)` 返回 Solution 列表，每个对象包含 os_seq、ms_map、makespan、shortage。
公开 n_evaluations、algorithm_variant、diagnostics、A、snapshots。
A 为当前快照/最终状态的非支配目标去重副本，不反向参与搜索；不存在累计历史非支配档案。
回调读取 A、n_evaluations，snapshots 保存 fe、archive、archive_size。

## 机制与适配

NSGA-II：fronts 快速非支配分层、crowding、等级/拥挤距离锦标赛、父子联合 environmental。
重复目标按稳定输入索引处理；零跨度维度不贡献拥挤距离，完全常值层无人为边界奖励。

MOEA/D：均匀双目标权重、欧氏权重邻域、邻域/全局交配、Tchebycheff 标量化和邻域限次替换。
理想点和上界为本次运行已评价可行解的逐目标累计最小/最大值；尺度为两者之差，零跨度分母为 1。
边界零权重使用 1e-6。每次替换比较使用同一组当前边界，不使用离线归一化或参考前沿。
POX 产生两个子代，只评价第一个；第二个不参与选优。初始化阶段完成后建立在线边界。

SPEA2：fitness 返回 Strength、Raw Fitness、Density、总适应度及距离矩阵。
距离基于当前 population + archive 联合集的 min/max 归一化，零跨度分母为 1。
k=floor(sqrt(n))，限制到非自身近邻的有效范围；单个体密度为 0。
archive_selection 先保留 raw=0 个体，再补充；过满时迭代删除近邻距离有序向量字典序最小者，
每次删除后重新考虑剩余邻居，不使用 NSGA-II 拥挤距离截断。

### MPMOGA 对应关系

论文：Si-Chen Liu 等，Many-Objective Job-Shop Scheduling: A Multiple Populations for Multiple Objectives-Based Genetic Algorithm Approach，
IEEE Transactions on Cybernetics 53(3), 1460–1474，DOI 10.1109/TCYB.2021.3102642。
书目信息可核对 https://pubmed.ncbi.nlm.nih.gov/34516383/ 。
本地未发现论文全文 PDF；本次不声称逐行核对过 Algorithm 1/2 原文，具体复现依据为用户任务提供的详细机制说明。

| 机制 | 实现入口 | 保留与必要适配 |
|---|---|---|
| MPMO | run、两个 population 字段 | 五目标五种群改为两个单目标种群；总规模210保持，105/105为适配 |
| POX（III-C/Fig.3） | comparison_operators.pox | 两非空工件集合、保留位置与另一父代顺序填充；MS 按同一次划分逐工序键继承 |
| IM（III-C/Fig.4） | comparison_operators.insertion | 重复工件序列中 a≤b 的向右插入，再恢复显式工序编号 |
| AST（Algorithm 1） | MPMOGA.ast | 档案随机父代；交叉比较 p1,c1,c2，未交叉比较 p1,p2；只按本种群目标，平局保留 p1 |
| 种群变异 | mutate_population | 原变异概率.1；50/50 OS/MS是适配；只接受本目标严格改善 |
| AUS（Algorithm 2） | MPMOGA.aus | 遍历旧档案逐一 OS IM、不使用.1概率；候选联合、染色体去重、等级和拥挤距离选取 |

AUS 容量充足时可保留被支配解；正式输出才提取非支配目标点。
完整 OS–MS 染色体判重、单工件 POX 返回父代副本、MS 基础变异，均为明确的问题适配约定。
没有引入 MGS、H-LNS、NEH、VNS、WIP 引导或其他强化搜索。
210、.9、.1以及档案210按任务给出的论文实验参数保留；经典三算法默认100等是项目初始工程设置，不宣称原论文统一推荐。

## FE、部分代与快照

每次真实 decode 尝试前增加 FE，合法性检查先于 FE；analyze 直接复用现有定义，无评价缓存。
只有 decoder 明确抛出的“死锁：”无事件推进异常被记为不可行，其他异常原样报错。
不可行候选不进入选择；不以无穷大或伪造惩罚值替代。初始化可重试至预算耗尽；全不可行时报错。
有限缓冲可能造成真实死锁，但本次三个真实实例短预算测试未观察到，不能据此证明所有合法染色体均可行。

快照在当前评价处理后到达指定 FE 时记录，不增加评价和随机调用：

- 初始化途中：当前已完成评价的初始部分种群。
- NSGA-II / SPEA2 的未完成一代：上一份已提交种群/档案；不将尚未完成环境选择的子代提前混入输出。
- MOEA/D：当次子问题替换完成后的当前子问题群体。
- MPMOGA：尚未提交 AUS 时记录旧档案；AUS 完成后记录新档案。
- 终止时：整理已经评价的部分代，再记录最终原生解集；绝不额外评价。

同一 FE 若又完成了环境选择/档案提交，会用该 FE 的最终提交状态替换内部快照；回调可能在同一 FE
收到状态提交前、提交后两次通知。调用者如持久化应按 FE 覆盖，而不能假设回调每个 FE 恰好一次。
最终快照等于 run 返回结果。下一阶段注册时需适配这些算法状态语义，不能沿用 Full 的整轮假设。

## 验证

```bash
PYTHONPYCACHEPREFIX=/private/tmp/comparison-cache python3 -m unittest tests.test_comparison_algorithms -v
```

测试覆盖转换、POX/MS来源、单工件、不可变 MS、直接解码目标一致性、分层、拥挤距离、
MOEA/D 权重尺度替换上限、SPEA2 强度和距离截断、AST 两分支和单目标平局、
变异严格接受、AUS 全档案扰动、预算、异常区分、观察一致性与随机复现。
真实实例 S1_P1_B1、S4_P2_B3、S8_P4_B2，每算法每实例43 FE，单seed=12，另做同seed重复及回调对照。
实际解码调用次数与 FE 一致。测试汇总写到 `/private/tmp/comparison-smoke-summary.json`，未写正式结果目录。

本地 formal_comparison 结果目录未发现已有原始文件，因而无法核验服务器上的3840份结果。
本地保护文件哈希前后核对；没有修改这些文件或旧 baseline，也没有写入任何正式实验结果。
