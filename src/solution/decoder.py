from __future__ import annotations

from dataclasses import dataclass, field
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


OpKey = Tuple[str, int]


@dataclass(frozen=True)
class ProvenanceEvent:
    event_id: int
    time: int
    kind: str
    job: str
    op_idx: int
    machine: str


@dataclass(frozen=True)
class ActiveDependency:
    src_event_id: int
    dst_event_id: int
    kind: str
    machine: Optional[str] = None
    buffer_id: Optional[str] = None


@dataclass(frozen=True)
class BufferStateEvent:
    event_id: int
    time: int
    buffer_id: str
    action: str
    job: Optional[str]
    level_before: int
    level_after: int
    cause_event_id: Optional[int] = None


@dataclass
class DecodeProvenance:
    events: Dict[int, ProvenanceEvent] = field(default_factory=dict)
    op_events: Dict[OpKey, Dict[str, int]] = field(default_factory=dict)
    active_dependencies: List[ActiveDependency] = field(default_factory=list)
    buffer_events: Dict[str, List[BufferStateEvent]] = field(default_factory=dict)
    _next_id: int = field(default=0, repr=False)
    _machine_release: Dict[str, int] = field(default_factory=dict, repr=False)
    _item_source: Dict[Tuple[str, str], int] = field(default_factory=dict, repr=False)

    def next_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def operation_event(self, time: int, kind: str, job: str, op_idx: int, machine: str) -> int:
        event_id = self.next_id()
        self.events[event_id] = ProvenanceEvent(event_id, time, kind, job, op_idx, machine)
        self.op_events.setdefault((job, op_idx), {})[kind] = event_id
        return event_id

    def edge(self, src: int, dst: int, kind: str, machine: Optional[str] = None,
             buffer_id: Optional[str] = None) -> None:
        self.active_dependencies.append(ActiveDependency(src, dst, kind, machine, buffer_id))


@dataclass(frozen=True)
class BlockedOperation:
    job: str
    buffer_out: str
    op_idx: int
    machine: str
    complete_event_id: Optional[int] = None


class StageBufferWIPScheduler:
    """Finite-buffer WIP scheduler with blocking and starving.

    OS is a precedence-preserving explicit operation-priority sequence:
    - every gene is a unique ``(job_id, op_idx)`` tuple;
    - every operation appears exactly once;
    - operations of the same job appear in technological order;
    - gene position is a static priority, not an actual start-time order.

    At each decision point the decoder checks only each job's next operation
    and starts the feasible operation with the smallest static priority rank.
    Actual start times are jointly determined by precedence, machine
    availability, exact-job buffer availability, and finite-buffer blocking.
    No cyclic OS pointer is used.

    The decoder returns ``(makespan, schedule, buffer_trace)``. Schedule
    records retain job/op/machine/start/end/release/buffer_in/buffer_out, and
    buffer traces retain ordered ``(time, level, action, job)`` events.
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
        self._expected_operations = [
            (job, op_idx)
            for job, ops in self.operations.items()
            for op_idx in range(len(ops))
        ]
        self._expected_operation_set = set(self._expected_operations)
        self._buffer_active_start_cache: Dict[str, int] = {}

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
        # 1) 检查缺失
        missing = [key for key in self._expected_operations if key not in ms_map]
        if missing:
            raise ValueError(
                f"ms_map 缺少以下工序的机器选择: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

        # 2) 检查多余键
        extra = [key for key in ms_map if key not in self._expected_operation_set]
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

    def _validate_os(
        self, os_seq: List[Tuple[str, int]]
    ) -> Dict[Tuple[str, int], int]:
        """Validate explicit OS genes and return their static priority ranks."""
        if not isinstance(os_seq, list):
            raise ValueError("OS must be a list")
        if len(os_seq) != len(self._expected_operations):
            raise ValueError(
                f"Invalid OS length: expected {len(self._expected_operations)}, "
                f"got {len(os_seq)}"
            )

        seen = set()
        for position, gene in enumerate(os_seq):
            if not isinstance(gene, tuple) or len(gene) != 2:
                raise ValueError(
                    f"OS gene {position} must be a (job_id, op_idx) tuple: {gene!r}"
                )
            job, op_idx = gene
            if job not in self.operations:
                raise ValueError(f"OS gene {position} has an unknown job: {job!r}")
            if not isinstance(op_idx, int) or isinstance(op_idx, bool):
                raise ValueError(
                    f"OS gene {position} has a non-integer op_idx: {gene!r}"
                )
            if op_idx < 0 or op_idx >= len(self.operations[job]):
                raise ValueError(
                    f"OS gene {position} is not a valid operation: {gene!r}"
                )
            if gene in seen:
                raise ValueError(f"OS contains a duplicate operation: {gene!r}")
            seen.add(gene)

        missing = [key for key in self._expected_operations if key not in seen]
        if missing:
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(f"OS is missing operations: {missing[:10]}{suffix}")

        next_expected = {job: 0 for job in self.operations}
        for position, (job, op_idx) in enumerate(os_seq):
            expected_op_idx = next_expected[job]
            if op_idx != expected_op_idx:
                raise ValueError(
                    f"OS violates precedence for {job} at position {position}: "
                    f"got {(job, op_idx)!r}, expected {(job, expected_op_idx)!r}"
                )
            next_expected[job] += 1

        return {op_key: position for position, op_key in enumerate(os_seq)}

    def _choose_machine(
        self,
        job: str,
        op_idx: int,
        ms_map: Optional[Dict[Tuple[str, int], str]],
        t: int,
        machine_free_at: Dict[str, int],
        blocked: Dict[str, Optional[BlockedOperation]],
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
        for m, processing_time in op["machines"].items():
            if blocked[m] is not None:
                continue
            if machine_free_at[m] > t:
                continue

            pt = int(processing_time)
            candidates.append((machine_free_at[m], pt, m))

        if not candidates:
            return None

        # 按（最早可用时间、加工时间、机器编号）排序
        return min(candidates)[2]

    def _log_buffer_event(
        self,
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
        bid: str,
        t: int,
        action: str,
        job: Optional[str],
        provenance: Optional[DecodeProvenance] = None,
        cause_event_id: Optional[int] = None,
        level_before: Optional[int] = None,
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
        if provenance is not None:
            if level_before is None:
                level_before = level
            event_id = provenance.next_id()
            provenance.buffer_events[bid].append(BufferStateEvent(
                event_id, t, bid, action, job, level_before, level, cause_event_id
            ))

    # =========================
    # 关键：解码主过程
    # =========================

    def decode(
        self,
        os_seq: List[Tuple[str, int]],
        ms_map: Optional[Dict[Tuple[str, int], str]] = None,
        return_provenance: bool = False,
    ):
        """
        解码函数：给定 OS/MS，生成可执行调度。默认返回三元组；
        return_provenance=True 时追加事实事件与 active dependency 记录。

        - makespan: 最大完工（release）时间
        - schedule: 调度记录列表，每条记录包含：
            job, op, machine, start, end, release, buffer_in, buffer_out
          其中 release 可能 > end（表示 blocking 造成的释放延迟）
        - buffer_trace: dict[buffer_id] = [(t, level, action, job), ...]
        """
        
        # Validate the complete static priority sequence before runtime setup.
        priority_rank = self._validate_os(os_seq)

        if ms_map is not None:
            self._validate_ms_map(ms_map)
        
        self._reset_buffers()
        provenance = DecodeProvenance() if return_provenance else None

        # -------- 工件状态 --------
        job_next = {j: 0 for j in self.operations}      # 每个 job 下一道待加工工序编号
        job_done = {j: False for j in self.operations}  # 是否已完工

        # -------- 机器状态 --------
        machine_free_at = {m: 0 for m in self.machines}  # 机器最早可启动新工序的时间
        blocked: Dict[str, Optional[BlockedOperation]] = {m: None for m in self.machines}
        # blocked[m] = (job_id, buffer_out, op_idx)

        # -------- 正在加工事件 --------
        # (end_time, job_id, op_idx, machine)
        running: List[Tuple[int, str, int, str]] = []

        # -------- 调度结果 --------
        schedule: List[Dict[str, Any]] = []
        schedule_index: Dict[Tuple[str, int], Dict[str, Any]] = {}

        # -------- 缓冲区事件日志 --------
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]] = {}
        for bid in self.buffers.keys():
            buffer_trace[bid] = []
            if provenance is not None:
                provenance.buffer_events[bid] = []
            # 记录初始状态
            self._log_buffer_event(buffer_trace, bid, 0, "init", None, provenance)

        # -------- 主循环控制 --------
        t = 0
        safety_iter = 0
        max_iter = 200000

        # ================== 主调度循环 ==================
        while not all(job_done.values()):
            safety_iter += 1
            if safety_iter > max_iter:
                raise RuntimeError("超过最大迭代次数：可能存在死锁或时间推进逻辑错误")

            # (1) 处理所有在当前时刻 t 加工结束的事件
            finished = []
            still_running = []
            for event in running:
                if event[0] == t:
                    finished.append(event)
                else:
                    still_running.append(event)
            if finished:
                running = still_running
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
                        buffer_trace=buffer_trace,
                        schedule_index=schedule_index,
                        provenance=provenance,
                    )

            # (2) 尝试解除 blocking
            self._release_blocked_if_possible(
                t, blocked, machine_free_at, schedule, buffer_trace, schedule_index,
                provenance=provenance,
            )

            # (3) 在当前时刻尽可能多地启动可行工序
            started_any = True
            while started_any:
                started_any = False

                cand = self._select_startable(
                    priority_rank=priority_rank,
                    t=t,
                    job_next=job_next,
                    job_done=job_done,
                    machine_free_at=machine_free_at,
                    blocked=blocked,
                    ms_map=ms_map
                )

                if cand is None:
                    break

                job, op_idx, m = cand
                ok, end_time = self._try_start(
                    job=job,
                    op_idx=op_idx,
                    machine=m,
                    t=t,
                    job_next=job_next,
                    schedule=schedule,
                    machine_free_at=machine_free_at,
                    buffer_trace=buffer_trace,
                    schedule_index=schedule_index,
                    provenance=provenance,
                )
                if ok:
                    running.append((end_time, job, op_idx, m))
                    # 启动后可能“取走 buffer_in”，立刻尝试解除上游 blocking
                    self._release_blocked_if_possible(
                        t, blocked, machine_free_at, schedule, buffer_trace, schedule_index,
                        provenance=provenance,
                        trigger_start_event_id=(
                            provenance.op_events[(job, op_idx)]["start"]
                            if provenance is not None else None
                        ),
                        trigger_buffer_id=self.operations[job][op_idx].get("buffer_in"),
                    )
                    started_any = True

            # (4) 时间推进：跳到下一个加工完成事件时刻
            if all(job_done.values()):
                break

            t_next = self._next_time(t, running)
            if t_next is None:
                raise RuntimeError(f"死锁：t={t} 时无法推进，但仍有工件未完成")
            t = t_next

        makespan = max(rec["release"] for rec in schedule) if schedule else 0
        if provenance is not None:
            return makespan, schedule, buffer_trace, provenance
        return makespan, schedule, buffer_trace

    # =========================
    # 启动/选择/完成/释放：核心逻辑
    # =========================

    def _select_startable(
        self,
        priority_rank: Dict[Tuple[str, int], int],
        t: int,
        job_next: Dict[str, int],
        job_done: Dict[str, bool],
        machine_free_at: Dict[str, int],
        blocked: Dict[str, Optional[BlockedOperation]],
        ms_map: Optional[Dict[Tuple[str, int], str]],
    ) -> Optional[Tuple[str, int, str]]:
        """Select the highest-priority feasible next operation.

        At most one next operation per unfinished job is inspected. A candidate
        must have an available selected machine and, when applicable, its exact
        job item in the input buffer.
        """
        best_candidate: Optional[Tuple[str, int, str]] = None
        best_rank = float("inf")
        for job in self.operations:
            if job_done.get(job, False):
                continue

            op_idx = job_next[job]
            if op_idx >= len(self.operations[job]):
                continue

            op_key = (job, op_idx)
            op = self.operations[job][op_idx]
            if ms_map is not None:
                machine = ms_map[op_key]
                if blocked[machine] is not None or machine_free_at[machine] > t:
                    continue
            else:
                machine = self._choose_machine(
                    job=job,
                    op_idx=op_idx,
                    ms_map=None,
                    t=t,
                    machine_free_at=machine_free_at,
                    blocked=blocked,
                )
            if machine is None:
                continue

            buffer_in = op.get("buffer_in", None)
            if buffer_in is not None and job not in self.buffers[buffer_in].content:
                continue

            rank = priority_rank[op_key]
            if rank < best_rank:
                best_rank = rank
                best_candidate = (job, op_idx, machine)

        return best_candidate

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
        schedule_index: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
        provenance: Optional[DecodeProvenance] = None,
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
        start_event_id = None
        if provenance is not None:
            start_event_id = provenance.operation_event(t, "start", job, op_idx, machine)
            prior_release = provenance._machine_release.get(machine)
            if prior_release is not None and provenance.events[prior_release].time == t:
                provenance.edge(prior_release, start_event_id, "machine", machine=machine)
            if buffer_in is not None:
                source = provenance._item_source.pop((buffer_in, job), None)
                if source is not None and provenance.events[source].time == t:
                    provenance.edge(source, start_event_id, "wip", buffer_id=buffer_in)

        if buffer_in is not None:
            level_before = len(self.buffers[buffer_in].content)
            self.buffers[buffer_in].content.remove(job)
            # 记录 take 事件
            self._log_buffer_event(buffer_trace, buffer_in, t, "take", job, provenance,
                                   start_event_id, level_before)

        pt = op["machines"][machine]
        end_time = t + int(pt)

        # 占用机器直到 end_time
        machine_free_at[machine] = end_time

        record = {
            "job": job,
            "op": op_idx,
            "machine": machine,
            "start": t,
            "end": end_time,
            "release": end_time,
            "buffer_in": buffer_in,
            "buffer_out": buffer_out,
        }
        schedule.append(record)
        if schedule_index is not None:
            schedule_index[(job, op_idx)] = record

        job_next[job] += 1
        return True, end_time

    def _finish_op_try_release(
        self,
        t: int,
        job: str,
        op_idx: int,
        machine: str,
        blocked: Dict[str, Optional[BlockedOperation]],
        machine_free_at: Dict[str, int],
        schedule: List[Dict[str, Any]],
        job_done: Dict[str, bool],
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
        schedule_index: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
        provenance: Optional[DecodeProvenance] = None,
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
        complete_id = None
        if provenance is not None:
            complete_id = provenance.operation_event(t, "complete", job, op_idx, machine)
            provenance.edge(provenance.op_events[(job, op_idx)]["start"],
                            complete_id, "processing")
        if buffer_out is None:
            machine_free_at[machine] = t
            self._update_release_time(schedule, job, op_idx, t, schedule_index)
            self._record_release(provenance, t, job, op_idx, machine)
            job_done[job] = True
            return

        buf = self.buffers[buffer_out]
        if len(buf.content) < buf.capacity:
            release_id = self._record_release(provenance, t, job, op_idx, machine)
            level_before = len(buf.content)
            buf.content.add(job)
            # 记录 put 事件
            self._log_buffer_event(buffer_trace, buffer_out, t, "put", job, provenance,
                                   release_id, level_before)
            if provenance is not None:
                provenance._item_source[(buffer_out, job)] = release_id

            machine_free_at[machine] = t
            self._update_release_time(schedule, job, op_idx, t, schedule_index)
        else:
            blocked[machine] = BlockedOperation(job, buffer_out, op_idx, machine, complete_id)

    def _record_release(self, provenance: Optional[DecodeProvenance], t: int,
                        job: str, op_idx: int, machine: str,
                        trigger_start_event_id: Optional[int] = None,
                        trigger_buffer_id: Optional[str] = None) -> Optional[int]:
        if provenance is None:
            return None
        release_id = provenance.operation_event(t, "release", job, op_idx, machine)
        provenance.edge(provenance.op_events[(job, op_idx)]["complete"],
                        release_id, "completion_release")
        if trigger_start_event_id is not None:
            provenance.edge(trigger_start_event_id, release_id, "unblocking",
                            buffer_id=trigger_buffer_id)
        provenance._machine_release[machine] = release_id
        return release_id

    def _release_blocked_if_possible(
        self,
        t: int,
        blocked: Dict[str, Optional[BlockedOperation]],
        machine_free_at: Dict[str, int],
        schedule: List[Dict[str, Any]],
        buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
        schedule_index: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
        provenance: Optional[DecodeProvenance] = None,
        trigger_start_event_id: Optional[int] = None,
        trigger_buffer_id: Optional[str] = None,
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
            job, buffer_out, op_idx = blk.job, blk.buffer_out, blk.op_idx
            buf = self.buffers[buffer_out]
            if len(buf.content) < buf.capacity:
                actual_trigger = (trigger_start_event_id if trigger_buffer_id == buffer_out else None)
                release_id = self._record_release(provenance, t, job, op_idx, m,
                                                  actual_trigger, buffer_out)
                level_before = len(buf.content)
                buf.content.add(job)
                # 记录 put 事件（解除阻塞放入缓冲区）
                self._log_buffer_event(buffer_trace, buffer_out, t, "put", job, provenance,
                                       release_id, level_before)
                if provenance is not None:
                    provenance._item_source[(buffer_out, job)] = release_id

                blocked[m] = None
                machine_free_at[m] = t
                self._update_release_time(schedule, job, op_idx, t, schedule_index)

    def _update_release_time(
        self,
        schedule: List[Dict[str, Any]],
        job: str,
        op_idx: int,
        release_t: int,
        schedule_index: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
    ):
        """更新某道工序的释放时间 release（用于表示 blocking 延迟）"""
        if schedule_index is not None:
            record = schedule_index.get((job, op_idx))
            if record is None:
                raise RuntimeError(f"未找到对应工序的调度记录：{job}-op{op_idx}")
            record["release"] = release_t
            return
        for rec in reversed(schedule):
            if rec["job"] == job and rec["op"] == op_idx:
                rec["release"] = release_t
                return
        raise RuntimeError(f"未找到对应工序的调度记录：{job}-op{op_idx}")

    def _next_time(self, t: int, running: List[Tuple[int, str, int, str]]) -> Optional[int]:
        """推进到下一个加工完成事件时刻"""
        future = [ev[0] for ev in running if ev[0] > t]
        return min(future) if future else None

    def _compute_buffer_active_start(self, bid: str) -> int:
        """计算缓冲区在实例层面的理论最早可供给时刻。"""
        cached = self._buffer_active_start_cache.get(bid)
        if cached is not None:
            return cached

        init_content = self.buffers_def[bid].get("init_content", [])
        if init_content:
            self._buffer_active_start_cache[bid] = 0
            return 0

        earliest_arrivals: List[int] = []
        for job, ops in self.operations.items():
            for op_idx, op in enumerate(ops):
                if op.get("buffer_out", None) != bid:
                    continue

                earliest = 0
                for k in range(op_idx + 1):
                    earliest += min(
                        int(pt) for pt in self.operations[job][k]["machines"].values()
                    )
                earliest_arrivals.append(earliest)

        if not earliest_arrivals:
            raise ValueError(f"缓冲区 {bid} 找不到任何供给工序（buffer_out == {bid}）")

        active_start = min(earliest_arrivals)
        self._buffer_active_start_cache[bid] = active_start
        return active_start
    
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
        per_buffer_active_start: Dict[str, int] = {}
        per_buffer_active_end: Dict[str, int] = {}
        per_buffer_active_horizon: Dict[str, int] = {}

        total_shortage_area = 0.0
        total_below_low_time = 0

        for bid, events in buffer_trace.items():
            # 统计区间长度
            T = int(makespan)
            per_buffer_horizon[bid] = T

            cap = int(self.buffers_def[bid]["capacity"])
            low_wip = int(self.buffers_def[bid].get("low_wip", max(1, (cap + 2) // 3)))
            per_buffer_low_wip[bid] = low_wip

            active_start = self._compute_buffer_active_start(bid)
            take_times = [int(t) for t, _, action, _ in events if action == "take"]
            active_end = max(take_times) if take_times else active_start
            active_horizon = max(0, active_end - active_start)

            per_buffer_active_start[bid] = active_start
            per_buffer_active_end[bid] = active_end
            per_buffer_active_horizon[bid] = active_horizon

            per_buffer_shortage_area[bid] = 0.0
            per_buffer_below_low_time[bid] = 0
            per_buffer_below_low_ratio[bid] = 0.0

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

                    shortage_seg_start = max(active_start, 0)
                    shortage_seg_end = min(active_end, min(cur_t, T))
                    shortage_dt = shortage_seg_end - shortage_seg_start
                    if shortage_dt > 0:
                        gap = max(0, low_wip - cur_level)
                        shortage_area += gap * shortage_dt
                        if cur_level < low_wip:
                            below_low_time += shortage_dt
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

                shortage_seg_start = max(active_start, seg_start)
                shortage_seg_end = min(active_end, seg_end)
                shortage_dt = shortage_seg_end - shortage_seg_start
                if shortage_dt > 0:
                    gap = max(0, low_wip - level_i)
                    shortage_area += gap * shortage_dt
                    if level_i < low_wip:
                        below_low_time += shortage_dt

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

                shortage_seg_start = max(active_start, last_t)
                shortage_seg_end = min(active_end, T)
                shortage_dt = shortage_seg_end - shortage_seg_start
                if shortage_dt > 0:
                    gap = max(0, low_wip - last_level)
                    shortage_area += gap * shortage_dt
                    if last_level < low_wip:
                        below_low_time += shortage_dt

            per_buffer_avg_level[bid] = area / T
            per_buffer_full_ratio[bid] = full_time / T
            per_buffer_empty_ratio[bid] = empty_time / T
            per_buffer_full_time[bid] = full_time
            per_buffer_empty_time[bid] = empty_time
            per_buffer_area[bid] = area
            # ===== 新增：low WIP =====
            per_buffer_shortage_area[bid] = shortage_area
            per_buffer_below_low_time[bid] = below_low_time
            per_buffer_below_low_ratio[bid] = (
                below_low_time / active_horizon if active_horizon > 0 else 0.0
            )

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
                "per_buffer_active_start": per_buffer_active_start,
                "per_buffer_active_end": per_buffer_active_end,
                "per_buffer_active_horizon": per_buffer_active_horizon,
            }
        }
        return stats
