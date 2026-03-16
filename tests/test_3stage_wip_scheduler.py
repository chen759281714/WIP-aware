import os
import sys

# ===== 把项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'src' =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.solution.decoder import StageBufferWIPScheduler


def build_toy_3stage_blocking_starving():
    """
    三工段串联 Toy：
      Stage0(M0) -> Buffer(B01) -> Stage1(M1) -> Buffer(B12) -> Stage2(M2)

    设计意图（可控地观察现象）：
    - Stage0 很快（1），Stage1 很慢（5） => B01 容易满，从而 Stage0 发生 blocking
    - Stage2 很快（1）                  => B12 经常空，从而 Stage2 发生 starving（等待 Stage1 出料）

    缓冲区容量：
    - B01 容量=1（容易满）
    - B12 容量=1（便于观察脉冲式 put/take）
    """
    operations = {
        "J1": [
            {"machines": {"M0": 1}, "buffer_in": None,  "buffer_out": "B01"},
            {"machines": {"M1": 5}, "buffer_in": "B01", "buffer_out": "B12"},
            {"machines": {"M2": 1}, "buffer_in": "B12", "buffer_out": None},
        ],
        "J2": [
            {"machines": {"M0": 1}, "buffer_in": None,  "buffer_out": "B01"},
            {"machines": {"M1": 5}, "buffer_in": "B01", "buffer_out": "B12"},
            {"machines": {"M2": 1}, "buffer_in": "B12", "buffer_out": None},
        ],
        "J3": [
            {"machines": {"M0": 1}, "buffer_in": None,  "buffer_out": "B01"},
            {"machines": {"M1": 5}, "buffer_in": "B01", "buffer_out": "B12"},
            {"machines": {"M2": 1}, "buffer_in": "B12", "buffer_out": None},
        ],
    }

    buffers = {
        "B01": {"capacity": 1},
        "B12": {"capacity": 1},
    }

    # OS 方案A：保证长度足够覆盖全部工序（3 jobs * 3 ops = 9 次“有效调度”）
    os_seq = ["J1", "J2", "J3"] * 20
    return operations, buffers, os_seq


def test_3stage_propagation():
    """
    三工段串联现象验证 + 指标统计输出
    """
    operations, buffers, os_seq = build_toy_3stage_blocking_starving()
    sch = StageBufferWIPScheduler(operations, buffers)
    makespan, schedule, buffer_trace = sch.decode(os_seq)

    print("\n===== 3-Stage Toy schedule (只打印 op0/op1) =====")
    print("makespan:", makespan)
    for r in schedule:
        if r["op"] in (0, 1):
            print(
                f"job={r['job']}, op={r['op']}, m={r['machine']}, "
                f"start={r['start']}, end={r['end']}, release={r['release']}"
            )

    print("\n===== 3-Stage buffer_trace =====")
    print(buffer_trace)

    # ========= 调用分析函数 =========
    stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

    print("\n===== 统计指标 =====")

    # ---------- blocking ----------
    print("\n--- Blocking 统计 ---")
    print("总 blocking 时间:", stats["blocking"]["total_blocking_time"])
    print("各机器 blocking 时间:", stats["blocking"]["per_machine_blocking_time"])

    # ---------- 机器利用率 ----------
    print("\n--- 机器利用情况 ---")
    for m in stats["machines"]["per_machine_busy_time"]:
        print(
            f"机器 {m}: "
            f"busy={stats['machines']['per_machine_busy_time'][m]}, "
            f"idle={stats['machines']['per_machine_idle_time'][m]}, "
            f"util={stats['machines']['per_machine_utilization'][m]:.3f}"
        )

    # ---------- buffer ----------
    print("\n--- Buffer 占用情况 ---")
    for bid in stats["buffers"]["per_buffer_avg_level"]:
        print(
            f"Buffer {bid}: "
            f"avg_level={stats['buffers']['per_buffer_avg_level'][bid]:.3f}, "
            f"满载比例={stats['buffers']['per_buffer_full_ratio'][bid]:.3f}, "
            f"空载比例={stats['buffers']['per_buffer_empty_ratio'][bid]:.3f}"
        )

    # ========= 原有机制断言 =========

    # Stage0 blocking 检查
    op0_blocked = [r for r in schedule if r["op"] == 0 and r["release"] > r["end"]]
    assert len(op0_blocked) >= 1, "期望 Stage0(op0) 出现 blocking"

    # Stage2 starving 检查
    m2_ops = sorted([r for r in schedule if r["machine"] == "M2"], key=lambda x: x["start"])
    if len(m2_ops) >= 2:
        span = m2_ops[-1]["release"] - m2_ops[0]["start"]
        total_pt = sum(r["end"] - r["start"] for r in m2_ops)
        assert span > total_pt, "期望 Stage2(M2) 出现空闲等待（starving）"

def build_toy_3stage_2machines_each(n_jobs: int = 4, m2_pt: int = 8, b12_cap: int = 2, b01_cap: int = 1):
    """
    三工段串联 + 每段两台并行机（可变工件数）：
      Stage0: M0a/M0b -> B01 -> Stage1: M1a/M1b -> B12 -> Stage2: M2a/M2b

    参数：
    - n_jobs：工件数量（J1~Jn）
    - m2_pt：Stage2 两台机的加工时间（用于控制 B12 是否形成真实缓冲）
    - b12_cap：B12 容量
    - b01_cap：B01 容量
    """
    jobs = [f"J{i}" for i in range(1, n_jobs + 1)]

    operations = {}
    for j in jobs:
        operations[j] = [
            {"machines": {"M0a": 1, "M0b": 1}, "buffer_in": None,  "buffer_out": "B01"},
            {"machines": {"M1a": 5, "M1b": 4}, "buffer_in": "B01", "buffer_out": "B12"},
            {"machines": {"M2a": m2_pt, "M2b": m2_pt}, "buffer_in": "B12", "buffer_out": None},
        ]

    buffers = {
        "B01": {"capacity": b01_cap},
        "B12": {"capacity": b12_cap},
    }

    # OS 方案A：重复多轮，保证足够长
    os_seq = jobs * 50
    return operations, buffers, os_seq


def test_3stage_2machines_each():
    operations, buffers, os_seq = build_toy_3stage_2machines_each()
    sch = StageBufferWIPScheduler(operations, buffers)
    makespan, schedule, buffer_trace = sch.decode(os_seq)

    print("\n===== 3-Stage (2 machines/stage) makespan =====")
    print("makespan:", makespan)

    # 打印每台机器的加工记录数，快速看看是否“分流”了
    per_m_cnt = {}
    for r in schedule:
        per_m_cnt[r["machine"]] = per_m_cnt.get(r["machine"], 0) + 1
    print("各机器加工次数:", per_m_cnt)

    # 打印统计指标
    stats = sch.analyze(schedule, buffer_trace, makespan=makespan)
    print("\n===== 统计指标（2 machines/stage）=====")
    print("总 blocking 时间:", stats["blocking"]["total_blocking_time"])
    print("各机器 blocking:", stats["blocking"]["per_machine_blocking_time"])

    print("\n--- 机器利用情况 ---")
    for m in stats["machines"]["per_machine_busy_time"]:
        print(
            f"{m}: busy={stats['machines']['per_machine_busy_time'][m]}, "
            f"idle={stats['machines']['per_machine_idle_time'][m]}, "
            f"util={stats['machines']['per_machine_utilization'][m]:.3f}"
        )

    print("\n--- Buffer 占用情况 ---")
    for bid in stats["buffers"]["per_buffer_avg_level"]:
        print(
            f"{bid}: avg={stats['buffers']['per_buffer_avg_level'][bid]:.3f}, "
            f"full={stats['buffers']['per_buffer_full_ratio'][bid]:.3f}, "
            f"empty={stats['buffers']['per_buffer_empty_ratio'][bid]:.3f}"
        )

    # 机制层面不强行断言“必须blocking”，因为加了并行机可能显著缓解阻塞
    # 但至少要保证两个 buffer 都发生过 put/take（系统确实按缓冲逻辑运转）
    assert "B01" in buffer_trace and "B12" in buffer_trace
    b01_actions = [e[2] for e in buffer_trace["B01"]]
    b12_actions = [e[2] for e in buffer_trace["B12"]]
    assert "put" in b01_actions and "take" in b01_actions
    assert "put" in b12_actions and "take" in b12_actions

def test_3stage_2machines_make_B12_buffer():
    """
    对照实验：调慢 Stage2，让 B12 形成真实缓冲
    - 情况A：m2_pt=1（下游很快，B12 基本为空）
    - 情况B：m2_pt=6（下游变慢，B12 会积压，甚至让 Stage1 被 B12 堵住）
    """
    for m2_pt in [1, 6]:
        operations, buffers, os_seq = build_toy_3stage_2machines_each(m2_pt=m2_pt, b12_cap=1)
        sch = StageBufferWIPScheduler(operations, buffers)
        makespan, schedule, buffer_trace = sch.decode(os_seq)
        stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

        print(f"\n===== 对照：m2_pt={m2_pt} =====")
        print("makespan:", makespan)
        print("blocking per machine:", stats["blocking"]["per_machine_blocking_time"])
        print("B01:", 
              "avg", f"{stats['buffers']['per_buffer_avg_level']['B01']:.3f}",
              "full", f"{stats['buffers']['per_buffer_full_ratio']['B01']:.3f}",
              "empty", f"{stats['buffers']['per_buffer_empty_ratio']['B01']:.3f}")
        print("B12:", 
              "avg", f"{stats['buffers']['per_buffer_avg_level']['B12']:.3f}",
              "full", f"{stats['buffers']['per_buffer_full_ratio']['B12']:.3f}",
              "empty", f"{stats['buffers']['per_buffer_empty_ratio']['B12']:.3f}")

        # 核心验证：当 m2_pt=6 时，B12 不应再是“全空”
        if m2_pt == 6:
            assert stats["buffers"]["per_buffer_avg_level"]["B12"] > 0.0, "期望 m2_pt=6 时 B12 出现真实积压（avg_level>0）"

def test_b12_capacity_contrast_m2slow():
    """
    对照实验：固定 Stage2 变慢（m2_pt=6），比较 B12 容量对“缓冲行为”的影响
    - b12_cap=1：容易出现满载（但可能更“尖锐”）
    - b12_cap=2：更像真实缓冲（更平滑、更稳定的积压）
    """
    m2_pt = 6
    for b12_cap in [1, 2]:
        operations, buffers, os_seq = build_toy_3stage_2machines_each(m2_pt=m2_pt, b12_cap=b12_cap)
        sch = StageBufferWIPScheduler(operations, buffers)
        makespan, schedule, buffer_trace = sch.decode(os_seq)
        stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

        b12_avg = stats["buffers"]["per_buffer_avg_level"]["B12"]
        b12_full_ratio = stats["buffers"]["per_buffer_full_ratio"]["B12"]
        b12_empty_ratio = stats["buffers"]["per_buffer_empty_ratio"]["B12"]
        b12_full_time = stats["buffers"]["per_buffer_full_time"]["B12"]
        b12_empty_time = stats["buffers"]["per_buffer_empty_time"]["B12"]

        print(f"\n===== 对照：m2_pt={m2_pt}, B12容量={b12_cap} =====")
        print("makespan:", makespan)
        print("blocking per machine:", stats["blocking"]["per_machine_blocking_time"])
        print(
            "B12:",
            f"avg={b12_avg:.3f}",
            f"full_ratio={b12_full_ratio:.3f}",
            f"empty_ratio={b12_empty_ratio:.3f}",
            f"full_time={b12_full_time}",
            f"empty_time={b12_empty_time}",
        )

        # 机制最低要求：B12 必须出现“非全空”，否则就不叫缓冲
        assert b12_avg > 0.0, "期望 Stage2 变慢后，B12 形成真实缓冲（avg_level>0）"

    # （可选）你也可以加一个“趋势断言”：容量变大后，满载比例通常会下降或更不尖锐
    # 但 toy + OS 扫描下可能出现边界波动，所以这里先不做强断言。

def test_m2_pt_contrast_with_b12cap2():
    """
    对照实验：固定 B12 容量=2，调慢 Stage2（m2_pt）观察 B12 缓冲增强
    - m2_pt=6：轻度缓冲（你已经看到 avg≈0.167）
    - m2_pt=8：更慢的下游，理论上 B12 应更容易积压（avg_level 上升、empty_ratio 下降）

    断言趋势：
    - m2_pt=8 时 B12 平均占用 >= m2_pt=6
    - m2_pt=8 时 B12 空载比例 <= m2_pt=6
    """
    b12_cap = 2
    results = {}

    for m2_pt in [6, 8]:
        operations, buffers, os_seq = build_toy_3stage_2machines_each(m2_pt=m2_pt, b12_cap=b12_cap)
        sch = StageBufferWIPScheduler(operations, buffers)
        makespan, schedule, buffer_trace = sch.decode(os_seq)
        stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

        b12_avg = stats["buffers"]["per_buffer_avg_level"]["B12"]
        b12_empty_ratio = stats["buffers"]["per_buffer_empty_ratio"]["B12"]
        b12_full_time = stats["buffers"]["per_buffer_full_time"]["B12"]
        b12_empty_time = stats["buffers"]["per_buffer_empty_time"]["B12"]

        results[m2_pt] = {
            "makespan": makespan,
            "b12_avg": b12_avg,
            "b12_empty_ratio": b12_empty_ratio,
            "b12_full_time": b12_full_time,
            "b12_empty_time": b12_empty_time,
            "blocking": stats["blocking"]["per_machine_blocking_time"],
        }

        print(f"\n===== 对照：B12容量={b12_cap}, m2_pt={m2_pt} =====")
        print("makespan:", makespan)
        print("blocking per machine:", results[m2_pt]["blocking"])
        print(
            "B12:",
            f"avg={b12_avg:.3f}",
            f"empty_ratio={b12_empty_ratio:.3f}",
            f"full_time={b12_full_time}",
            f"empty_time={b12_empty_time}",
        )

    # --- 趋势断言（允许相等，避免 toy 案例下的边界波动导致误判）---
    assert results[8]["b12_avg"] >= results[6]["b12_avg"], "期望 m2_pt=8 时 B12 平均占用不低于 m2_pt=6"
    assert results[8]["b12_empty_ratio"] <= results[6]["b12_empty_ratio"], "期望 m2_pt=8 时 B12 空载比例不高于 m2_pt=6"

def test_njobs_contrast_buffer_stability():
    """
    对照实验：增加工件数，让缓冲统计更稳定、更像连续生产

    固定：
        m2_pt = 8
        B12 capacity = 2

    对比：
        n_jobs = 4 vs 8

    观察：
        - B12 是否出现明显积压
        - blocking 是否从下游传播到中游/上游
        - B01 是否也开始出现满载（传播链条）
    """

    m2_pt = 8
    b12_cap = 2

    results = {}

    for n_jobs in [4, 8]:

        operations, buffers, os_seq = build_toy_3stage_2machines_each(
            n_jobs=n_jobs,
            m2_pt=m2_pt,
            b12_cap=b12_cap,
            b01_cap=1
        )

        sch = StageBufferWIPScheduler(operations, buffers)

        makespan, schedule, buffer_trace = sch.decode(os_seq)

        stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

        # ---- B01 ----
        b01_avg = stats["buffers"]["per_buffer_avg_level"]["B01"]
        b01_full_time = stats["buffers"]["per_buffer_full_time"]["B01"]
        b01_empty_ratio = stats["buffers"]["per_buffer_empty_ratio"]["B01"]

        # ---- B12 ----
        b12_avg = stats["buffers"]["per_buffer_avg_level"]["B12"]
        b12_full_time = stats["buffers"]["per_buffer_full_time"]["B12"]
        b12_empty_ratio = stats["buffers"]["per_buffer_empty_ratio"]["B12"]

        results[n_jobs] = {
            "b12_full_time": b12_full_time
        }

        print(f"\n===== 对照：n_jobs={n_jobs}, m2_pt={m2_pt}, B12cap={b12_cap} =====")

        print("makespan:", makespan)

        print("blocking per machine:",
              stats["blocking"]["per_machine_blocking_time"])

        print(
            "B01:",
            f"avg={b01_avg:.3f}",
            f"empty_ratio={b01_empty_ratio:.3f}",
            f"full_time={b01_full_time}"
        )

        print(
            "B12:",
            f"avg={b12_avg:.3f}",
            f"empty_ratio={b12_empty_ratio:.3f}",
            f"full_time={b12_full_time}"
        )

    # --------- 弱断言 ---------

    # 当订单增加时，B12 应该更容易积压
    assert results[8]["b12_full_time"] >= results[4]["b12_full_time"], \
        "期望 n_jobs=8 时 B12 积压更明显"

if __name__ == "__main__":
    # print("Running 3-Stage propagation test...")
    # test_3stage_propagation()
    # print("✅ 3-Stage propagation passed")

    # print("\nRunning 3-Stage (2 machines/stage) test...")
    # test_3stage_2machines_each()
    # print("✅ 3-Stage (2 machines/stage) passed")

    # print("\nRunning B12 buffering contrast test...")
    # test_3stage_2machines_make_B12_buffer()
    # print("✅ B12 buffering contrast passed")

    # print("\nRunning B12 capacity contrast test...")
    # test_b12_capacity_contrast_m2slow()
    # print("✅ B12 capacity contrast passed")

    # print("\nRunning m2_pt contrast test (B12 cap=2)...")
    # test_m2_pt_contrast_with_b12cap2()
    # print("✅ m2_pt contrast (B12 cap=2) passed")

    # print("\nRunning n_jobs contrast test...")
    # test_njobs_contrast_buffer_stability()
    # print("✅ n_jobs contrast passed")

    print("\nRunning n_jobs contrast test...")
    test_njobs_contrast_buffer_stability()
    print("✅ n_jobs contrast passed")