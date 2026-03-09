import os
import sys

# 把项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import InstanceSpec, generate_fms_wip_instance, describe_instance
from src.solution.decoder import StageBufferWIPScheduler


def test_instance_generator_smoke():
    profiles = ["downstream_bottleneck", "mid_bottleneck", "balanced"]

    for prof in profiles:
        spec = InstanceSpec(
            num_stages=3,
            machines_per_stage=2,
            n_jobs=8,
            buffer_caps=[1, 2],      # B01=1, B12=2：你现在验证过会出现传播
            pt_profile=prof,
            pt_low=1,
            pt_high=10,
            seed=42,
            os_repeat=80,
        )

        operations, buffers, os_seq = generate_fms_wip_instance(spec)
        print("\n==============================")
        print(describe_instance(spec, operations, buffers))

        sch = StageBufferWIPScheduler(operations, buffers)
        makespan, schedule, buffer_trace = sch.decode(os_seq)
        stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

        # 基本健壮性检查
        assert makespan >= 0
        assert len(schedule) == spec.n_jobs * spec.num_stages, "schedule 记录条数应等于 N*S"
        assert "B01" in stats["buffers"]["per_buffer_avg_level"]
        assert "B12" in stats["buffers"]["per_buffer_avg_level"]

        # 打印关键观测量（用于你肉眼判断“是否像预期”）
        print("makespan:", makespan)
        print("blocking:", stats["blocking"]["per_machine_blocking_time"])
        print("B01:", "avg", f"{stats['buffers']['per_buffer_avg_level']['B01']:.3f}",
              "full_time", stats["buffers"]["per_buffer_full_time"]["B01"])
        print("B12:", "avg", f"{stats['buffers']['per_buffer_avg_level']['B12']:.3f}",
              "full_time", stats["buffers"]["per_buffer_full_time"]["B12"])


if __name__ == "__main__":
    print("Running instance generator smoke test...")
    test_instance_generator_smoke()
    print("✅ instance generator smoke test passed")