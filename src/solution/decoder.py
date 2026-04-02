from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set


@dataclass
class Buffer:
    """
    缓冲区（工段之间的在制品缓冲）
    - capacity: 最大容量
    - content : 当前缓冲区中存放的 job_id 集合（用于保证“取到的是本 job 的半成品”）
    """
    capacity: int
    content: Set[str]


class StageBufferWIPScheduler:
    """
    工段间 WIP（有限缓冲）调度器，支持 blocking / starving 机制。

    【OS 编码方案 A】
    - os_seq 是 job_id 的序列
    - 每出现一次 job_id，表示“尝试调度该 job 的下一道工序”
    - 调度器会从 os_ptr 开始循环扫描 os_seq，找到当前时刻可启动的工序就启动

    【核心机制】
    - starving：某道工序需要 buffer_in，但 buffer_in 中没有该 job 的半成品 -> 不能开工
    - blocking：某道工序加工结束后要放入 buffer_out，但 buffer_out 已满 -> 不能释放，机器被占用直到有空位

    【返回三元组】
    - makespan
    - schedule：每条记录包含 job/op/machine/start/end/release/buffer_in/buffer_out
    - buffer_trace：事件日志格式：
        buffer_trace[bid] = [(t, level, action, job), ...]
        action ∈ {"init","put","take"}
        level 为事件发生后的 level（len(content)）
    """

    def __init__(self, operations: Dict[str, List[Dict[str, Any]]], buffers: Dict[str, Dict[str, Any]]):
        """
        参数说明：
        operations[job] = 工序列表
            每个工序必须包含：
            - machines   : {machine_id: processing_time}
            - buffer_in  : 开工前需要取件的缓冲区（第一道工序为 None）
            - buffer_out : 完工后需要释放到的缓冲区（最后一道工序为 None）

        buffers[buffer_id] 必须包含：
            - capacity   : 缓冲区容量
            可选包含：
            - init_content: 初始 content（job_id 列表/集合），默认空
        """
        self.operations = operations
        self.buffers_def = buffers

        # 收集所有可能用到的机器
        self.machines = self._collect_machines()

        # 运行时缓冲区（每次 decode 前重置）
        self.buffers: Dict[str, Buffer] = {}

    # =========================
    # 初始化/工具函数
    # =========================

    def _collect_machines(self) -> List[str]:
        """从所有工序中收集机器集合"""
        s = set()
        for job, ops in self.operations.items():
            for op in ops:
                for m in op["machines"].keys():
                    s.add(m)
        return sorted(s)

    def _reset_buffers(self):
        """初始化（或重置）所有缓冲区状态"""
        self.buffers = {}
        for bid, bdef in self.buffers_def.items():
            cap = int(bdef["capacity"])
            init = bdef.get("init_content", [])
            init_set = set(init) if init is not None else set()
            self.buffers[bid] = Buffer(capacity=cap, content=set(init_set))


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

        # 1) 检查缺失
        missing = [key for key in expected if key not in ms_map]
        if missing:
            raise ValueError(
                f"ms_map 缺少以下工序的机器选择: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

        # 2) 检查多余键
        extra = [key for key in ms_map.keys() if key not in expected]
        if extra:
            raise ValueError(
                f"ms_map 包含无效工序键: "
                f"{extra[:10]}{'...' if len(extra) > 10 else ''}"
            )

        # 3) 检查机器是否合法
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
        blocked: Dict[str, Optional[Tuple[str, str, int]]],
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
            - 不允许自动切换到其他机器
        """
        op = self.operations[job][op_idx]
        eligible = list(op["machines"].keys())

        # ---- MS 模式：严格按指定机器 ----
        if ms_map is not None:
            m = ms_map[(job, op_idx)]

            if m not in op["machines"]:
                raise ValueError(f"ms_map 为 {job}-op{op_idx} 选择了无效机器 {m}")

            if blocked[m] is None and machine_free_at[m] <= t:
                return m

            return None

        # ---- 自动选机模式：从当前可启动的机器中选择 ----
        candidates = []
        for m in eligible:
            if blocked[m] is not None:
                continue
            if machine_free_at[m] > t:
                continue

            pt = int(op["machines"][m])
            candidates.append((machine_free_at[m], pt, m))

        if not candidates:
            return None

        # 按（最早可用时间、加工时间、机器编号）排序
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return candidates[0][2]

    def _log_buffer_event(
        self,
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
        bid: str,
        t: int,
        action: str,
        job: Optional[str],
    ):
        """
        记录缓冲区事件日志（事件发生后立刻记录）：
        - action: "init" / "put" / "take"
        - job   : 对应 job_id；init 时可为 None
        - level : 事件发生后的 level（len(content)）
        注意：同一时刻发生多次变化也要全部记录，不做覆盖。
        """
        level = len(self.buffers[bid].content)
        buffer_trace[bid].append((t, level, action, job))

    # =========================
    # 关键：解码主过程
    # =========================

    def decode(self, os_seq: List[str], ms_map: Optional[Dict[Tuple[str, int], str]] = None):
        """
        解码函数：给定 OS/MS，生成可执行调度并返回三元组：
          (makespan, schedule, buffer_trace)

        - makespan: 最大完工（release）时间
        - schedule: 调度记录列表，每条记录包含：
            job, op, machine, start, end, release, buffer_in, buffer_out
          其中 release 可能 > end（表示 blocking 造成的释放延迟）
        - buffer_trace: dict[buffer_id] = [(t, level, action, job), ...]
        """
        
        # 若提供了 ms_map，则必须是完整且合法的机器选择映射
        if ms_map is not None:
            self._validate_ms_map(ms_map)
        
        self._reset_buffers()

        # -------- 工件状态 --------
        job_next = {j: 0 for j in self.operations}      # 每个 job 下一道待加工工序编号
        job_done = {j: False for j in self.operations}  # 是否已完工

        # -------- 机器状态 --------
        machine_free_at = {m: 0 for m in self.machines}  # 机器最早可启动新工序的时间
        blocked: Dict[str, Optional[Tuple[str, str, int]]] = {m: None for m in self.machines}
        # blocked[m] = (job_id, buffer_out, op_idx)

        # -------- 正在加工事件 --------
        # (end_time, job_id, op_idx, machine)
        running: List[Tuple[int, str, int, str]] = []

        # -------- 调度结果 --------
        schedule: List[Dict[str, Any]] = []

        # -------- 缓冲区事件日志 --------
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]] = {}
        for bid in self.buffers.keys():
            buffer_trace[bid] = []
            # 记录初始状态
            self._log_buffer_event(buffer_trace, bid, 0, "init", None)

        # -------- 主循环控制 --------
        t = 0
        os_ptr = 0
        safety_iter = 0
        max_iter = 200000

        # ================== 主调度循环 ==================
        while not all(job_done.values()):
            safety_iter += 1
            if safety_iter > max_iter:
                raise RuntimeError("超过最大迭代次数：可能存在死锁或时间推进逻辑错误")

            # (1) 处理所有在当前时刻 t 加工结束的事件
            finished = [ev for ev in running if ev[0] == t]
            if finished:
                running = [ev for ev in running if ev[0] != t]
                for end_time, job, op_idx, m in finished:
                    self._finish_op_try_release(
                        t=t,
                        job=job,
                        op_idx=op_idx,
                        machine=m,
                        blocked=blocked,
                        machine_free_at=machine_free_at,
                        schedule=schedule,
                        job_done=job_done,
                        buffer_trace=buffer_trace
                    )

            # (2) 尝试解除 blocking
            self._release_blocked_if_possible(t, blocked, machine_free_at, schedule, buffer_trace)

            # (3) 在当前时刻尽可能多地启动可行工序
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
                    blocked=blocked,
                    ms_map=ms_map
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
                    buffer_trace=buffer_trace
                )
                os_ptr = new_ptr

                if ok:
                    running.append((end_time, job, op_idx, m))
                    # 启动后可能“取走 buffer_in”，立刻尝试解除上游 blocking
                    self._release_blocked_if_possible(t, blocked, machine_free_at, schedule, buffer_trace)
                    started_any = True

            # (4) 时间推进：跳到下一个加工完成事件时刻
            if all(job_done.values()):
                break

            t_next = self._next_time(t, running)
            if t_next is None:
                raise RuntimeError(f"死锁：t={t} 时无法推进，但仍有工件未完成")
            t = t_next

        makespan = max(rec["release"] for rec in schedule) if schedule else 0
        return makespan, schedule, buffer_trace

    # =========================
    # 启动/选择/完成/释放：核心逻辑
    # =========================

    def _select_startable(
        self,
        os_seq: List[str],
        os_ptr: int,
        t: int,
        job_next: Dict[str, int],
        job_done: Dict[str, bool],
        machine_free_at: Dict[str, int],
        blocked: Dict[str, Optional[Tuple[str, str, int]]],
        ms_map: Optional[Dict[Tuple[str, int], str]],
    ) -> Optional[Tuple[str, int, str, int]]:
        """
        根据 OS 编码，从 os_ptr 开始循环扫描，选择当前时刻可启动的工序：
        可启动需要满足：
        - job 未完成
        - 对应下一道工序存在
        - 选定机器未被阻塞 且 machine_free_at <= t
        - 若 buffer_in 不为空，则 buffer_in 中存在该 job 的半成品（否则 starving）
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

            op = self.operations[job][op_idx]
            m = self._choose_machine(
                job=job,
                op_idx=op_idx,
                ms_map=ms_map,
                t=t,
                machine_free_at=machine_free_at,
                blocked=blocked,
            )
            if m is None:
                continue

            if blocked[m] is not None:
                continue
            if machine_free_at[m] > t:
                continue

            buffer_in = op.get("buffer_in", None)
            if buffer_in is not None and job not in self.buffers[buffer_in].content:
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
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
    ) -> Tuple[bool, int]:
        """
        在时刻 t 启动工序：
        - 若有 buffer_in，则必须先取走该 job 的半成品（否则 starving）
        - 计算 end_time
        - 更新 machine_free_at[machine] = end_time
        - 写入 schedule（补充 buffer_in / buffer_out）
        - job_next[job] += 1
        """
        op = self.operations[job][op_idx]
        buffer_in = op.get("buffer_in", None)
        buffer_out = op.get("buffer_out", None)

        # 取件（starving 检查）
        if buffer_in is not None:
            if job not in self.buffers[buffer_in].content:
                return False, t
            self.buffers[buffer_in].content.remove(job)
            # 记录 take 事件
            self._log_buffer_event(buffer_trace, buffer_in, t, "take", job)

        pt = op["machines"][machine]
        end_time = t + int(pt)

        # 占用机器直到 end_time
        machine_free_at[machine] = end_time

        schedule.append({
            "job": job,
            "op": op_idx,
            "machine": machine,
            "start": t,
            "end": end_time,
            "release": end_time,
            "buffer_in": buffer_in,
            "buffer_out": buffer_out,
        })

        job_next[job] += 1
        return True, end_time

    def _finish_op_try_release(
        self,
        t: int,
        job: str,
        op_idx: int,
        machine: str,
        blocked: Dict[str, Optional[Tuple[str, str, int]]],
        machine_free_at: Dict[str, int],
        schedule: List[Dict[str, Any]],
        job_done: Dict[str, bool],
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
    ):
        """
        工序加工结束时：
        - 若 buffer_out 为 None：最后工序，工件完工，释放机器，标记 job_done
        - 否则尝试放入 buffer_out：
            - buffer_out 未满：put 成功，释放机器，更新 release
            - buffer_out 已满：blocking，机器继续占用
        """
        op = self.operations[job][op_idx]
        buffer_out = op.get("buffer_out", None)

        if buffer_out is None:
            machine_free_at[machine] = t
            self._update_release_time(schedule, job, op_idx, t)
            job_done[job] = True
            return

        buf = self.buffers[buffer_out]
        if len(buf.content) < buf.capacity:
            buf.content.add(job)
            # 记录 put 事件
            self._log_buffer_event(buffer_trace, buffer_out, t, "put", job)

            machine_free_at[machine] = t
            self._update_release_time(schedule, job, op_idx, t)
        else:
            blocked[machine] = (job, buffer_out, op_idx)

    def _release_blocked_if_possible(
        self,
        t: int,
        blocked: Dict[str, Optional[Tuple[str, str, int]]],
        machine_free_at: Dict[str, int],
        schedule: List[Dict[str, Any]],
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
    ):
        """
        若某被阻塞机器对应的缓冲区出现空位，则立刻释放：
        - put 到 buffer_out
        - machine_free_at[m] = t
        - 更新该工序 release = t
        """
        for m, blk in list(blocked.items()):
            if blk is None:
                continue
            job, buffer_out, op_idx = blk
            buf = self.buffers[buffer_out]
            if len(buf.content) < buf.capacity:
                buf.content.add(job)
                # 记录 put 事件（解除阻塞放入缓冲区）
                self._log_buffer_event(buffer_trace, buffer_out, t, "put", job)

                blocked[m] = None
                machine_free_at[m] = t
                self._update_release_time(schedule, job, op_idx, t)

    def _update_release_time(self, schedule: List[Dict[str, Any]], job: str, op_idx: int, release_t: int):
        """更新某道工序的释放时间 release（用于表示 blocking 延迟）"""
        for rec in reversed(schedule):
            if rec["job"] == job and rec["op"] == op_idx:
                rec["release"] = release_t
                return
        raise RuntimeError(f"未找到对应工序的调度记录：{job}-op{op_idx}")

    def _next_time(self, t: int, running: List[Tuple[int, str, int, str]]) -> Optional[int]:
        """推进到下一个加工完成事件时刻"""
        future = [ev[0] for ev in running if ev[0] > t]
        return min(future) if future else None
    
    def analyze(
        self,
        schedule: List[Dict[str, Any]],
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
        makespan: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        对 decode 的输出进行统计分析

        输入：
        - schedule：decode 返回的 schedule 列表
        - buffer_trace：decode 返回的 buffer 事件日志
            buffer_trace[bid] = [(t, level, action, job), ...]
        - makespan：可选；若不提供则从 schedule 里计算 max(release)

        输出：stats 字典，包含
        - makespan
        - blocking:
            - total_blocking_time
            - per_machine_blocking_time
            - per_op_blocking_time（每条记录的 blocking 时间，方便你定位是谁在堵）
        - machines:
            - per_machine_busy_time
            - per_machine_idle_time（机器跨度内空闲时间）
            - per_machine_span
            - per_machine_utilization（busy/span）
        - buffers:
            - per_buffer_avg_level（时间加权平均 level）
            - per_buffer_full_ratio（满载时间占比）
            - per_buffer_empty_ratio（空载时间占比）
            - per_buffer_horizon（统计区间长度）
        """
        # ---------- 1) makespan ----------
        if makespan is None:
            makespan = max((r["release"] for r in schedule), default=0)

        # ---------- 2) blocking 统计（release - end） ----------
        per_machine_blocking: Dict[str, int] = {}
        per_op_blocking: List[Dict[str, Any]] = []
        total_blocking = 0

        for r in schedule:
            blk = max(0, int(r["release"]) - int(r["end"]))
            total_blocking += blk
            m = r["machine"]
            per_machine_blocking[m] = per_machine_blocking.get(m, 0) + blk

            # 记录每条工序的 blocking，便于后续诊断/画图
            per_op_blocking.append({
                "job": r["job"],
                "op": r["op"],
                "machine": m,
                "end": r["end"],
                "release": r["release"],
                "blocking": blk,
                "buffer_out": r.get("buffer_out", None),
            })

        # ---------- 3) 机器 busy/idle/utilization ----------
        # 对每台机器收集加工区间（start,end）；注意 busy 只算加工时间，不算 blocking
        per_machine_intervals: Dict[str, List[Tuple[int, int, int]]] = {}  # (start, end, release)
        for r in schedule:
            m = r["machine"]
            per_machine_intervals.setdefault(m, []).append((int(r["start"]), int(r["end"]), int(r["release"])))

        per_machine_busy: Dict[str, int] = {}
        per_machine_span: Dict[str, int] = {}
        per_machine_idle: Dict[str, int] = {}
        per_machine_util: Dict[str, float] = {}

        for m in self.machines:
            intervals = per_machine_intervals.get(m, [])
            if not intervals:
                per_machine_busy[m] = 0
                per_machine_span[m] = 0
                per_machine_idle[m] = 0
                per_machine_util[m] = 0.0
                continue

            intervals_sorted = sorted(intervals, key=lambda x: x[0])
            busy = sum(e - s for s, e, _ in intervals_sorted)

            # 机器“跨度”建议用：从第一段开始到最后一段释放（release）
            # 因为 blocking 会占住机器资源，release 才是真正空出来
            span_start = intervals_sorted[0][0]
            span_end = max(rel for _, _, rel in intervals_sorted)
            span = max(0, span_end - span_start)

            idle = max(0, span - busy)
            util = (busy / span) if span > 0 else 0.0

            per_machine_busy[m] = busy
            per_machine_span[m] = span
            per_machine_idle[m] = idle
            per_machine_util[m] = util

        # ---------- 4) buffer 时间加权统计（平均占用/满载比例/空载比例） ----------
        # 注意：buffer_trace 是事件日志，level 表示“事件发生后的 level”
        # 我们按时间段积分：在相邻事件之间，level 保持不变
        per_buffer_avg_level: Dict[str, float] = {}
        per_buffer_full_ratio: Dict[str, float] = {}
        per_buffer_empty_ratio: Dict[str, float] = {}
        per_buffer_horizon: Dict[str, int] = {}
        per_buffer_full_time: Dict[str, int] = {}
        per_buffer_empty_time: Dict[str, int] = {}
        per_buffer_area: Dict[str, float] = {}
        # ===== 新增：WIP 下限统计 =====
        per_buffer_low_wip: Dict[str, int] = {}
        per_buffer_shortage_area: Dict[str, float] = {}
        per_buffer_below_low_time: Dict[str, int] = {}
        per_buffer_below_low_ratio: Dict[str, float] = {}

        total_shortage_area = 0.0
        total_below_low_time = 0

        for bid, events in buffer_trace.items():
            # 统计区间长度
            T = int(makespan)
            per_buffer_horizon[bid] = T

            cap = int(self.buffers_def[bid]["capacity"])
            low_wip = int(self.buffers_def[bid].get("low_wip", max(1, cap // 3)))
            per_buffer_low_wip[bid] = low_wip

            if T <= 0:
                per_buffer_avg_level[bid] = 0.0
                per_buffer_full_ratio[bid] = 0.0
                per_buffer_empty_ratio[bid] = 0.0
                continue

            if not events:
                # 极端情况：没有任何日志（正常不应发生）
                per_buffer_avg_level[bid] = 0.0
                per_buffer_full_ratio[bid] = 0.0
                per_buffer_empty_ratio[bid] = 1.0
                continue

            # 确保按时间排序；同一时刻多事件保持原顺序对积分无影响
            events_sorted = sorted(events, key=lambda x: x[0])

            # 从 t=0 开始的初始 level：取第一条事件的 level（一般是 init）
            cur_t = int(events_sorted[0][0])
            cur_level = int(events_sorted[0][1])

            # 如果第一条事件不是 t=0，则认为 [0, cur_t) 也保持 cur_level（通常不会发生）
            area = 0.0
            full_time = 0
            empty_time = 0
            shortage_area = 0.0
            below_low_time = 0

            # 先补 [0, cur_t)
            if cur_t > 0:
                dt = min(cur_t, T) - 0
                if dt > 0:
                    area += cur_level * dt
                    if cur_level >= cap:
                        full_time += dt
                    if cur_level == 0:
                        empty_time += dt

                    # ===== 新增：low WIP =====
                    gap = max(0, low_wip - cur_level)
                    shortage_area += gap * dt
                    if cur_level < low_wip:
                        below_low_time += dt
            # 再遍历事件段
            for i in range(len(events_sorted) - 1):
                t_i = int(events_sorted[i][0])
                level_i = int(events_sorted[i][1])
                t_j = int(events_sorted[i + 1][0])

                # 事件发生后的 level = level_i，在 [t_i, t_j) 保持
                seg_start = max(0, t_i)
                seg_end = min(T, t_j)
                dt = seg_end - seg_start
                if dt <= 0:
                    continue

                area += level_i * dt
                if level_i >= cap:
                    full_time += dt
                if level_i == 0:
                    empty_time += dt

                # ===== 新增：low WIP =====
                gap = max(0, low_wip - level_i)
                shortage_area += gap * dt
                if level_i < low_wip:
                    below_low_time += dt

            # 最后一条事件之后，延续到 T
            last_t = int(events_sorted[-1][0])
            last_level = int(events_sorted[-1][1])
            if last_t < T:
                dt = T - last_t
                area += last_level * dt
                if last_level >= cap:
                    full_time += dt
                if last_level == 0:
                    empty_time += dt

                # ===== 新增：low WIP =====
                gap = max(0, low_wip - last_level)
                shortage_area += gap * dt
                if last_level < low_wip:
                    below_low_time += dt

            per_buffer_avg_level[bid] = area / T
            per_buffer_full_ratio[bid] = full_time / T
            per_buffer_empty_ratio[bid] = empty_time / T
            per_buffer_full_time[bid] = full_time
            per_buffer_empty_time[bid] = empty_time
            per_buffer_area[bid] = area
            # ===== 新增：low WIP =====
            per_buffer_shortage_area[bid] = shortage_area
            per_buffer_below_low_time[bid] = below_low_time
            per_buffer_below_low_ratio[bid] = (below_low_time / T) if T > 0 else 0.0

            total_shortage_area += shortage_area
            total_below_low_time += below_low_time

        stats = {
            "makespan": makespan,
            "blocking": {
                "total_blocking_time": total_blocking,
                "per_machine_blocking_time": per_machine_blocking,
                "per_op_blocking_time": per_op_blocking,
            },
            "machines": {
                "per_machine_busy_time": per_machine_busy,
                "per_machine_idle_time": per_machine_idle,
                "per_machine_span": per_machine_span,
                "per_machine_utilization": per_machine_util,
            },
            "buffers": {
                "per_buffer_avg_level": per_buffer_avg_level,
                "per_buffer_full_ratio": per_buffer_full_ratio,
                "per_buffer_empty_ratio": per_buffer_empty_ratio,
                "per_buffer_full_time": per_buffer_full_time,
                "per_buffer_empty_time": per_buffer_empty_time,
                "per_buffer_area": per_buffer_area,
                "per_buffer_horizon": per_buffer_horizon,
            },
            "shortage": {
                "total_shortage_area": total_shortage_area,
                "total_below_low_time": total_below_low_time,
                "per_buffer_low_wip": per_buffer_low_wip,
                "per_buffer_shortage_area": per_buffer_shortage_area,
                "per_buffer_below_low_time": per_buffer_below_low_time,
                "per_buffer_below_low_ratio": per_buffer_below_low_ratio,
            }
        }
        return stats
