from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any


class NoWIPScheduler:
    """
    完全忽略 WIP / buffer 约束的调度器。

    说明：
    - 保留 OS/MS 的解码方式
    - 仅考虑：
        1) 工件工序顺序约束
        2) 机器可用性约束
    - 不考虑：
        1) buffer_in / buffer_out
        2) blocking
        3) starving
        4) 缓冲区容量 / 在制品占用

    返回三元组：
    - makespan
    - schedule
    - buffer_trace（恒为空字典，保留接口兼容）
    """

    def __init__(self, operations: Dict[str, List[Dict[str, Any]]]):
        self.operations = operations
        self.machines = self._collect_machines()

    # =========================
    # 初始化 / 工具函数
    # =========================

    def _collect_machines(self) -> List[str]:
        """从所有工序中收集机器集合"""
        s = set()
        for job, ops in self.operations.items():
            for op in ops:
                for m in op["machines"].keys():
                    s.add(m)
        return sorted(s)

    def _validate_ms_map(self, ms_map: Dict[Tuple[str, int], str]):
        """
        检查 ms_map 是否完整且合法。

        要求：
        1. 必须覆盖所有工序 (job, op_idx)
        2. 不能包含无效工序键
        3. 为每道工序指定的机器必须属于该工序的合法机器集合
        """
        expected = []

        for job, ops in self.operations.items():
            for op_idx in range(len(ops)):
                expected.append((job, op_idx))

        missing = [key for key in expected if key not in ms_map]
        if missing:
            raise ValueError(
                f"ms_map 缺少以下工序的机器选择: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

        extra = [key for key in ms_map.keys() if key not in expected]
        if extra:
            raise ValueError(
                f"ms_map 包含无效工序键: "
                f"{extra[:10]}{'...' if len(extra) > 10 else ''}"
            )

        for (job, op_idx), m in ms_map.items():
            legal = self.operations[job][op_idx]["machines"].keys()
            if m not in legal:
                raise ValueError(
                    f"ms_map 为 {(job, op_idx)} 指定了非法机器 {m}"
                )

    def _choose_machine(
        self,
        job: str,
        op_idx: int,
        ms_map: Optional[Dict[Tuple[str, int], str]],
        t: int,
        machine_free_at: Dict[str, int],
    ) -> Optional[str]:
        """
        选择加工该工序的机器（支持阶段内多机）：

        模式1：若 ms_map is None
            - 自动选机
            - 只在当前时刻可启动的机器中选择
            - 优先选最早可用；若并列，再选加工时间短；仍并列选机器编号小

        模式2：若 ms_map 不为 None
            - 严格按 ms_map[(job, op_idx)] 指定机器
            - 若该机器此刻不可启动，则返回 None
        """
        op = self.operations[job][op_idx]
        eligible = list(op["machines"].keys())

        # ---- MS 模式：严格按指定机器 ----
        if ms_map is not None:
            m = ms_map[(job, op_idx)]

            if m not in op["machines"]:
                raise ValueError(f"ms_map 为 {job}-op{op_idx} 选择了无效机器 {m}")

            if machine_free_at[m] <= t:
                return m

            return None

        # ---- 自动选机模式 ----
        candidates = []
        for m in eligible:
            if machine_free_at[m] > t:
                continue

            pt = int(op["machines"][m])
            candidates.append((machine_free_at[m], pt, m))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return candidates[0][2]

    # =========================
    # 解码主过程
    # =========================

    def decode(
        self,
        os_seq: List[str],
        ms_map: Optional[Dict[Tuple[str, int], str]] = None
    ):
        """
        解码函数：给定 OS/MS，生成可执行调度并返回三元组：
          (makespan, schedule, buffer_trace)

        - makespan: 最大完工时间
        - schedule: 调度记录列表，每条记录包含：
            job, op, machine, start, end, release, buffer_in, buffer_out
          这里 release == end（因为无 blocking）
        - buffer_trace: 恒为空字典，保留接口兼容
        """
        if ms_map is not None:
            self._validate_ms_map(ms_map)

        # -------- 工件状态 --------
        job_next = {j: 0 for j in self.operations}      # 每个 job 下一道待加工工序编号
        job_done = {j: False for j in self.operations}  # 是否已完工

        # -------- 机器状态 --------
        machine_free_at = {m: 0 for m in self.machines}

        # -------- 正在加工事件 --------
        # (end_time, job_id, op_idx, machine)
        running: List[Tuple[int, str, int, str]] = []

        # -------- 调度结果 --------
        schedule: List[Dict[str, Any]] = []

        # -------- 为了兼容原接口 --------
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]] = {}

        # -------- 主循环控制 --------
        t = 0
        os_ptr = 0
        safety_iter = 0
        max_iter = 200000

        while not all(job_done.values()):
            safety_iter += 1
            if safety_iter > max_iter:
                raise RuntimeError("超过最大迭代次数：可能存在死锁或时间推进逻辑错误")

            # (1) 处理所有在当前时刻 t 完成的工序
            finished = [ev for ev in running if ev[0] == t]
            if finished:
                running = [ev for ev in running if ev[0] != t]
                for end_time, job, op_idx, m in finished:
                    self._finish_op(
                        t=t,
                        job=job,
                        op_idx=op_idx,
                        machine=m,
                        machine_free_at=machine_free_at,
                        schedule=schedule,
                        job_done=job_done,
                    )

            # (2) 在当前时刻尽可能多地启动可行工序
            started_any = True
            while started_any:
                started_any = False

                cand = self._select_startable(
                    os_seq=os_seq,
                    os_ptr=os_ptr,
                    t=t,
                    job_next=job_next,
                    job_done=job_done,
                    machine_free_at=machine_free_at,
                    ms_map=ms_map,
                )

                if cand is None:
                    break

                job, op_idx, m, new_ptr = cand
                ok, end_time = self._try_start(
                    job=job,
                    op_idx=op_idx,
                    machine=m,
                    t=t,
                    job_next=job_next,
                    schedule=schedule,
                    machine_free_at=machine_free_at,
                )
                os_ptr = new_ptr

                if ok:
                    running.append((end_time, job, op_idx, m))
                    started_any = True

            # (3) 时间推进
            if all(job_done.values()):
                break

            t_next = self._next_time(t, running)
            if t_next is None:
                raise RuntimeError(f"死锁：t={t} 时无法推进，但仍有工件未完成")
            t = t_next

        makespan = max((rec["release"] for rec in schedule), default=0)
        return makespan, schedule, buffer_trace

    # =========================
    # 启动 / 选择 / 完成
    # =========================

    def _select_startable(
        self,
        os_seq: List[str],
        os_ptr: int,
        t: int,
        job_next: Dict[str, int],
        job_done: Dict[str, bool],
        machine_free_at: Dict[str, int],
        ms_map: Optional[Dict[Tuple[str, int], str]],
    ) -> Optional[Tuple[str, int, str, int]]:
        """
        根据 OS 编码，从 os_ptr 开始循环扫描，选择当前时刻可启动的工序。

        可启动需要满足：
        - job 未完成
        - 对应下一道工序存在
        - 选定机器 machine_free_at <= t

        不检查：
        - buffer_in
        - buffer_out
        - starving
        - blocking
        """
        if not os_seq:
            return None

        n = len(os_seq)
        for k in range(n):
            idx = (os_ptr + k) % n
            job = os_seq[idx]

            if job_done.get(job, False):
                continue

            op_idx = job_next[job]
            if op_idx >= len(self.operations[job]):
                continue

            m = self._choose_machine(
                job=job,
                op_idx=op_idx,
                ms_map=ms_map,
                t=t,
                machine_free_at=machine_free_at,
            )
            if m is None:
                continue

            if machine_free_at[m] > t:
                continue

            new_ptr = (idx + 1) % n
            return job, op_idx, m, new_ptr

        return None

    def _try_start(
        self,
        job: str,
        op_idx: int,
        machine: str,
        t: int,
        job_next: Dict[str, int],
        schedule: List[Dict[str, Any]],
        machine_free_at: Dict[str, int],
    ) -> Tuple[bool, int]:
        """
        在时刻 t 启动工序：
        - 不做 buffer 取件
        - 计算 end_time
        - 更新 machine_free_at[machine] = end_time
        - 写入 schedule
        - job_next[job] += 1
        """
        op = self.operations[job][op_idx]
        pt = int(op["machines"][machine])
        end_time = t + pt

        machine_free_at[machine] = end_time

        schedule.append({
            "job": job,
            "op": op_idx,
            "machine": machine,
            "start": t,
            "end": end_time,
            "release": end_time,   # 无 blocking，因此 release == end
            "buffer_in": None,
            "buffer_out": None,
        })

        job_next[job] += 1
        return True, end_time

    def _finish_op(
        self,
        t: int,
        job: str,
        op_idx: int,
        machine: str,
        machine_free_at: Dict[str, int],
        schedule: List[Dict[str, Any]],
        job_done: Dict[str, bool],
    ):
        """
        工序加工结束时：
        - 机器立即释放
        - 更新 release = t
        - 若是最后一道工序，则标记 job_done
        """
        machine_free_at[machine] = t
        self._update_release_time(schedule, job, op_idx, t)

        if op_idx == len(self.operations[job]) - 1:
            job_done[job] = True

    def _update_release_time(
        self,
        schedule: List[Dict[str, Any]],
        job: str,
        op_idx: int,
        release_t: int
    ):
        """更新某道工序的释放时间 release"""
        for rec in reversed(schedule):
            if rec["job"] == job and rec["op"] == op_idx:
                rec["release"] = release_t
                return
        raise RuntimeError(f"未找到对应工序的调度记录：{job}-op{op_idx}")

    def _next_time(self, t: int, running: List[Tuple[int, str, int, str]]) -> Optional[int]:
        """推进到下一个加工完成事件时刻"""
        future = [ev[0] for ev in running if ev[0] > t]
        return min(future) if future else None