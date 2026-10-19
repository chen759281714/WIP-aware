import os
import random
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.solution.decoder import StageBufferWIPScheduler
from src.solution.encoder import Encoder


def build_two_job_operations():
    return {
        "A": [
            {"machines": {"M0": 2}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M1": 3}, "buffer_in": "B", "buffer_out": None},
        ],
        "B": [
            {"machines": {"M0": 1}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M1": 4}, "buffer_in": "B", "buffer_out": None},
        ],
    }


def test_random_os_is_explicit_complete_and_precedence_preserving():
    operations = build_two_job_operations()
    encoder = Encoder(operations, rng=random.Random(20260921))
    expected = set(encoder.ms_index_order)

    for _ in range(20):
        os_seq = encoder.generate_random_os()
        assert all(isinstance(gene, tuple) and len(gene) == 2 for gene in os_seq)
        assert len(os_seq) == len(expected)
        assert set(os_seq) == expected
        assert len(set(os_seq)) == len(os_seq)
        assert encoder.validate_os(os_seq) is True

        for job, ops in operations.items():
            positions = [os_seq.index((job, op_idx)) for op_idx in range(len(ops))]
            assert positions == sorted(positions)


def test_validate_os_rejects_invalid_chromosomes():
    encoder = Encoder(build_two_job_operations())
    valid = [("A", 0), ("B", 0), ("A", 1), ("B", 1)]
    assert encoder.validate_os(valid) is True

    invalid_sequences = [
        valid[:-1],
        [("A", 0), ("B", 0), ("A", 1), ("A", 1)],
        [("A", 0), ("X", 0), ("A", 1), ("B", 1)],
        [("A", 0), ("B", 0), ("A", 2), ("B", 1)],
        [("A", 1), ("B", 0), ("A", 0), ("B", 1)],
        [["A", 0], ("B", 0), ("A", 1), ("B", 1)],
    ]
    for os_seq in invalid_sequences:
        try:
            encoder.validate_os(os_seq)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid OS to fail validation: {os_seq!r}")


def test_build_priority_rank_matches_os_positions():
    encoder = Encoder(build_two_job_operations())
    os_seq = [("B", 0), ("A", 0), ("B", 1), ("A", 1)]
    assert encoder.build_priority_rank(os_seq) == {
        ("B", 0): 0,
        ("A", 0): 1,
        ("B", 1): 2,
        ("A", 1): 3,
    }


def test_decoder_uses_static_operation_priority():
    operations = {
        "A": [{"machines": {"M": 3}, "buffer_in": None, "buffer_out": None}],
        "B": [{"machines": {"M": 3}, "buffer_in": None, "buffer_out": None}],
    }
    ms_map = {("A", 0): "M", ("B", 0): "M"}
    scheduler = StageBufferWIPScheduler(operations, {})

    _, schedule_ab, _ = scheduler.decode([("A", 0), ("B", 0)], ms_map)
    _, schedule_ba, _ = scheduler.decode([("B", 0), ("A", 0)], ms_map)

    assert [(rec["job"], rec["start"]) for rec in schedule_ab] == [("A", 0), ("B", 3)]
    assert [(rec["job"], rec["start"]) for rec in schedule_ba] == [("B", 0), ("A", 3)]


def test_decoder_preserves_ms_exact_job_buffer_and_blocking_physics():
    operations = {
        "J0": [
            {"machines": {"M0": 1}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M2": 10}, "buffer_in": "B", "buffer_out": None},
        ],
        "J1": [
            {"machines": {"M1": 2}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M2": 1}, "buffer_in": "B", "buffer_out": None},
        ],
        "J2": [
            {"machines": {"M3": 3}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M2": 1}, "buffer_in": "B", "buffer_out": None},
        ],
    }
    buffers = {"B": {"capacity": 1, "low_wip": 1}}
    encoder = Encoder(operations)
    os_seq = [
        ("J0", 0),
        ("J1", 0),
        ("J2", 0),
        ("J0", 1),
        ("J1", 1),
        ("J2", 1),
    ]
    ms_map = encoder.build_ms_map(["M0", "M2", "M1", "M2", "M3", "M2"])

    scheduler = StageBufferWIPScheduler(operations, buffers)
    makespan, schedule, buffer_trace = scheduler.decode(os_seq, ms_map)

    assert len(schedule) == encoder.get_total_operations()
    assert len({(rec["job"], rec["op"]) for rec in schedule}) == len(schedule)
    assert all(rec["machine"] == ms_map[(rec["job"], rec["op"])] for rec in schedule)
    assert all(rec["release"] >= rec["end"] for rec in schedule)
    assert makespan == max(rec["release"] for rec in schedule)

    records = {(rec["job"], rec["op"]): rec for rec in schedule}
    for job in operations:
        assert records[(job, 1)]["start"] >= records[(job, 0)]["release"]

    blocked = records[("J2", 0)]
    assert blocked["release"] > blocked["end"]
    assert [event[3] for event in buffer_trace["B"] if event[2] == "take"] == [
        "J0",
        "J1",
        "J2",
    ]


if __name__ == "__main__":
    test_random_os_is_explicit_complete_and_precedence_preserving()
    test_validate_os_rejects_invalid_chromosomes()
    test_build_priority_rank_matches_os_positions()
    test_decoder_uses_static_operation_priority()
    test_decoder_preserves_ms_exact_job_buffer_and_blocking_physics()
    print("operation-priority encoder/decoder checks passed")
