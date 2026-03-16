"""
测试 OS + MS 解码是否正确

测试目标：
1. Decoder 在传入完整 ms_map 时，是否严格按指定机器解码
2. schedule 是否覆盖全部工序，且无重复、无遗漏
3. 非法 ms_map 是否会被正确拦截
"""

import os
import sys

# 把项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.solution.decoder import StageBufferWIPScheduler
from src.solution.encoder import Encoder
from src.problem.instance_generator import (
    generate_fms_wip_instance_auto_caps,
    InstanceSpec
)


def test_os_ms_decode():
    """
    主测试：
    - 生成一个小实例
    - 随机生成 OS / MS
    - 用 ms_map 解码
    - 检查每条工序是否严格按指定机器执行
    """
    print("Running OS+MS decode test...")

    # 构造一个小规模实例
    spec = InstanceSpec(
        num_stages=3,
        machines_per_stage=2,
        n_jobs=3,
        buffer_caps=[2, 2],   # 占位，auto caps 会覆盖
        pt_profile="balanced",
        seed=1,
        os_repeat=20,
    )

    spec, operations, buffers, _, _ = generate_fms_wip_instance_auto_caps(spec)

    # 初始化编码器
    encoder = Encoder(operations)

    # 随机生成 OS / MS
    os_seq = encoder.generate_random_os()
    ms_list = encoder.generate_random_ms()
    ms_map = encoder.build_ms_map(ms_list)

    # 解码
    scheduler = StageBufferWIPScheduler(operations, buffers)
    makespan, schedule, buffer_trace = scheduler.decode(
        os_seq=os_seq,
        ms_map=ms_map
    )

    print("OS:", os_seq)
    print("MS_list:", ms_list)
    print("makespan:", makespan)

    # ---------------------------------------------------
    # 1) 检查 schedule 条数是否等于总工序数
    # ---------------------------------------------------
    total_ops = encoder.get_total_operations()
    assert len(schedule) == total_ops, \
        f"schedule 条数错误：期望 {total_ops}，实际 {len(schedule)}"

    # ---------------------------------------------------
    # 2) 检查每条工序的机器是否严格等于 ms_map 指定机器
    # ---------------------------------------------------
    for rec in schedule:
        job = rec["job"]
        op = rec["op"]
        actual_machine = rec["machine"]
        expected_machine = ms_map[(job, op)]

        assert actual_machine == expected_machine, \
            f"机器分配错误：{(job, op)} 期望 {expected_machine}，实际 {actual_machine}"

    # ---------------------------------------------------
    # 3) 检查是否覆盖了所有工序，且没有重复工序
    # ---------------------------------------------------
    scheduled_ops = [(rec["job"], rec["op"]) for rec in schedule]
    expected_ops = encoder.ms_index_order

    # 先检查数量
    assert len(scheduled_ops) == len(expected_ops), \
        "schedule 中工序数量与 expected_ops 不一致"

    # 检查是否无重复
    assert len(set(scheduled_ops)) == len(scheduled_ops), \
        "schedule 中存在重复工序记录"

    # 检查是否无遗漏
    assert set(scheduled_ops) == set(expected_ops), \
        "schedule 中存在遗漏工序或非法工序"

    # ---------------------------------------------------
    # 4) 检查 makespan 基本合法
    # ---------------------------------------------------
    assert makespan >= 0, "makespan 不应为负数"

    # ---------------------------------------------------
    # 5) 打印少量 schedule 方便人工检查
    # ---------------------------------------------------
    print("\n部分调度记录：")
    for rec in schedule[:min(10, len(schedule))]:
        print(
            f"job={rec['job']}, op={rec['op']}, machine={rec['machine']}, "
            f"start={rec['start']}, end={rec['end']}, release={rec['release']}"
        )

    print("✓ OS+MS decode passed")


def test_invalid_ms_map_should_fail():
    """
    非法 ms_map 测试：
    - 故意删除一个工序的机器分配
    - decode 应该抛出异常
    """
    print("\nRunning invalid ms_map test...")

    spec = InstanceSpec(
        num_stages=3,
        machines_per_stage=2,
        n_jobs=2,
        buffer_caps=[2, 2],
        pt_profile="balanced",
        seed=2,
        os_repeat=10,
    )

    spec, operations, buffers, _, _ = generate_fms_wip_instance_auto_caps(spec)

    encoder = Encoder(operations)
    os_seq = encoder.generate_random_os()
    ms_list = encoder.generate_random_ms()
    ms_map = encoder.build_ms_map(ms_list)

    # 故意删掉一个工序映射，模拟不完整 ms_map
    first_key = encoder.ms_index_order[0]
    del ms_map[first_key]

    scheduler = StageBufferWIPScheduler(operations, buffers)

    failed = False
    try:
        scheduler.decode(os_seq=os_seq, ms_map=ms_map)
    except ValueError as e:
        failed = True
        print("捕获到预期异常：", e)

    assert failed, "非法 ms_map 本应触发异常，但没有触发"

    print("✓ invalid ms_map test passed")


if __name__ == "__main__":
    test_os_ms_decode()
    test_invalid_ms_map_should_fail()
    print("\n✅ all OS+MS decode tests passed")