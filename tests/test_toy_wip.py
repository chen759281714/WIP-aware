import os
import sys

# ===== 把项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'src' =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.solution.decoder import StageBufferWIPScheduler


def build_toy_blocking():
    """
    Toy-Blocking：
    - 2 工段：Stage0 -> Stage1
    - 1 个缓冲区 B01，容量=1
    - Stage0 很快（1），Stage1 很慢（5）
    预期：缓冲区容易满，上游出现 blocking（release > end）
    """
    operations = {
        "J1": [
            {"machines": {"M0": 1}, "buffer_in": None, "buffer_out": "B01"},
            {"machines": {"M1": 5}, "buffer_in": "B01", "buffer_out": None},
        ],
        "J2": [
            {"machines": {"M0": 1}, "buffer_in": None, "buffer_out": "B01"},
            {"machines": {"M1": 5}, "buffer_in": "B01", "buffer_out": None},
        ],
        "J3": [
            {"machines": {"M0": 1}, "buffer_in": None, "buffer_out": "B01"},
            {"machines": {"M1": 5}, "buffer_in": "B01", "buffer_out": None},
        ],
    }
    buffers = {"B01": {"capacity": 1}}
    # OS 方案A：让每个 job 多次出现，保证能覆盖到所有工序（toy 阶段更鲁棒）
    os_seq = ["J1", "J2", "J3"] * 10
    return operations, buffers, os_seq


def build_toy_starving():
    """
    Toy-Starving：
    - 2 工段：Stage0 -> Stage1
    - 1 个缓冲区 B01，容量=1
    - Stage0 很慢（5），Stage1 很快（1）
    预期：缓冲区经常为空，下游出现 starving（机器空闲等待）
    """
    operations = {
        "J1": [
            {"machines": {"M0": 5}, "buffer_in": None, "buffer_out": "B01"},
            {"machines": {"M1": 1}, "buffer_in": "B01", "buffer_out": None},
        ],
        "J2": [
            {"machines": {"M0": 5}, "buffer_in": None, "buffer_out": "B01"},
            {"machines": {"M1": 1}, "buffer_in": "B01", "buffer_out": None},
        ],
        "J3": [
            {"machines": {"M0": 5}, "buffer_in": None, "buffer_out": "B01"},
            {"machines": {"M1": 1}, "buffer_in": "B01", "buffer_out": None},
        ],
    }
    buffers = {"B01": {"capacity": 1}}
    os_seq = ["J1", "J2", "J3"] * 10
    return operations, buffers, os_seq


def test_blocking_happens():
    """验证 blocking：至少存在一条记录满足 release > end"""
    operations, buffers, os_seq = build_toy_blocking()
    sch = StageBufferWIPScheduler(operations, buffers)
    makespan, schedule, buffer_trace = sch.decode(os_seq)

    # 打印关键 schedule（只打印 op=0，最容易看到 blocking）
    print("===== Toy-Blocking schedule =====")
    print("makespan:", makespan)
    for r in schedule:
        if r["op"] == 0:
            print(f"job={r['job']}, start={r['start']}, end={r['end']}, release={r['release']}")

    blocked_ops = [r for r in schedule if r["op"] == 0 and r["release"] > r["end"]]
    assert len(blocked_ops) >= 1, "期望至少出现一次 blocking（release > end）"

    # 也顺便打印一下缓冲区轨迹（方便你人工验证）
    print("===== Toy-Blocking buffer_trace =====")
    print(buffer_trace)


def test_starving_signature():
    """
    验证 starving 的“特征”：
    - 下游机器 M1 的作业跨度（最后release-最早start）应大于其总加工时间（有空闲间隙）
    """
    operations, buffers, os_seq = build_toy_starving()
    sch = StageBufferWIPScheduler(operations, buffers)
    makespan, schedule, buffer_trace = sch.decode(os_seq)

    m1_ops = sorted([r for r in schedule if r["machine"] == "M1"], key=lambda x: x["start"])
    if len(m1_ops) >= 2:
        span = m1_ops[-1]["release"] - m1_ops[0]["start"]
        total_pt = sum(r["end"] - r["start"] for r in m1_ops)
        assert span > total_pt, "期望下游出现空闲间隙（starving 特征）"

    print("===== Toy-Starving buffer_trace =====")
    print(buffer_trace)


if __name__ == "__main__":
    print("Running Toy-Blocking test...")
    test_blocking_happens()
    print("✅ Toy-Blocking passed (blocking detected)")

    print("Running Toy-Starving test...")
    test_starving_signature()
    print("✅ Toy-Starving passed (idle gaps/starving detected)")
