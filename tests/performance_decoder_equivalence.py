"""State-by-state decoder comparison against a supplied source snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solution.decoder import StageBufferWIPScheduler
from src.solution.encoder import Encoder
from performance_trajectory_regression import canonical


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("perf_baseline_decoder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.StageBufferWIPScheduler


def decode_with_states(decoder_class, operations, buffers, os_seq, ms_map):
    decoder = decoder_class(operations, buffers)
    states = []
    original = decoder._select_startable

    def record_selection(**kwargs):
        before = (
            kwargs["t"],
            tuple(kwargs["job_next"].items()),
            tuple(kwargs["job_done"].items()),
            tuple(kwargs["machine_free_at"].items()),
            tuple((machine, canonical(value)) for machine, value in kwargs["blocked"].items()),
            tuple((bid, tuple(sorted(buf.content))) for bid, buf in decoder.buffers.items()),
        )
        selected = original(**kwargs)
        states.append((before, selected))
        return selected

    decoder._select_startable = record_selection
    result = decoder.decode(os_seq, ms_map, return_provenance=True)
    analysis = decoder.analyze(result[1], result[2], result[0])
    return states, canonical(result), canonical(analysis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 23])
    args = parser.parse_args()
    baseline_class = load_baseline(args.baseline)
    directory = ROOT / "data/final_benchmark/instances"
    for size in args.sizes:
        data = json.loads((directory / f"WIPHFSP_S{size}_P1_B1.json").read_text())
        operations, buffers = data["operations"], data["buffers"]
        for seed in args.seeds:
            encoder = Encoder(operations, rng=random.Random(seed))
            for index in range(3):
                os_seq = encoder.generate_random_os()
                ms_map = encoder.build_ms_map(encoder.generate_random_ms())
                old = decode_with_states(baseline_class, operations, buffers, os_seq, ms_map)
                new = decode_with_states(StageBufferWIPScheduler, operations, buffers,
                                         os_seq, ms_map)
                if old != new:
                    for part, left, right in zip(("states", "result", "analysis"), old, new):
                        if left != right:
                            raise AssertionError(f"S{size} seed={seed} sample={index}: {part} differs")
                print(f"S{size} seed={seed} sample={index}: "
                      f"{len(old[0])} startability states and full decode equal", flush=True)


if __name__ == "__main__":
    main()
