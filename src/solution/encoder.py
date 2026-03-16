"""
编码模块（Encoder）

负责：
1. 定义 MS 的展开顺序
2. 将 MS_list 转换为 ms_map
3. 生成随机 OS / MS（后面 GA 初始化会用）

说明：
- OS 采用 job_id 重复序列编码
- MS 采用按 (job, op) 固定展开的一维列表编码
"""

import random
import sys
import os

# ===== 把项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'src' =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


class Encoder:
    """
    编码器类

    参数：
        operations : 调度实例中的 operations
        rng        : 随机数生成器（可选），用于保证实验可复现

    属性：
        operations      : 原始工序数据
        ms_index_order  : MS 的固定展开顺序
        rng             : 随机数生成器
    """

    def __init__(self, operations, rng=None):
        self.operations = operations
        self.rng = rng if rng is not None else random
        self.ms_index_order = self.build_ms_index_order()

    def build_ms_index_order(self):
        """
        构建 MS 编码的展开顺序。

        规则：
        按 job 顺序，再按 op 顺序展开：
            (J1,0),(J1,1),(J1,2),...
            (J2,0),(J2,1),(J2,2),...

        返回：
            list[(job_id, op_idx)]
        """
        order = []

        for job_id, ops in self.operations.items():
            for op_idx in range(len(ops)):
                order.append((job_id, op_idx))

        return order

    def build_ms_map(self, ms_list):
        """
        将 MS_list 转换为 ms_map。

        参数：
            ms_list : 机器选择列表

        返回：
            dict[(job, op)] = machine_id
        """
        if len(ms_list) != len(self.ms_index_order):
            raise ValueError("MS_list 长度与工序数量不一致")

        ms_map = {}

        for idx, (job, op) in enumerate(self.ms_index_order):
            machine = ms_list[idx]

            # 该工序的合法机器集合
            legal_machines = self.operations[job][op]["machines"].keys()

            if machine not in legal_machines:
                raise ValueError(
                    f"非法机器选择: {(job, op)} 不能选择 {machine}"
                )

            ms_map[(job, op)] = machine

        return ms_map

    def generate_random_os(self):
        """
        生成随机 OS。

        规则：
        - 每个 job 重复其工序数次
        - 然后整体随机打乱

        返回：
            os_seq : list[str]
        """
        os_seq = []

        for job_id, ops in self.operations.items():
            os_seq += [job_id] * len(ops)

        self.rng.shuffle(os_seq)

        return os_seq

    def generate_random_ms(self):
        """
        随机生成 MS_list。

        规则：
        - 对每个 (job, op)
        - 在其合法机器集合中随机选择一台

        返回：
            ms_list : list[str]
        """
        ms_list = []

        for job, op in self.ms_index_order:
            machines = list(self.operations[job][op]["machines"].keys())
            ms_list.append(self.rng.choice(machines))

        return ms_list

    def get_total_operations(self):
        """
        返回总工序数量。
        """
        return len(self.ms_index_order)

    def print_ms_index_order(self):
        """
        打印 MS 展开顺序（调试用）。
        """
        print("MS index order:")
        for idx, item in enumerate(self.ms_index_order):
            print(idx, "->", item)