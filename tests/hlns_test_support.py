"""Small H-LNS fixture independent of historical SIS test imports."""

from src.solution.encoder import Encoder
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from src.algorithms.wip_graph_dual_population_wo_mgs import WIPGraphDualPopulationWithoutMGS
from src.algorithms.wip_graph_dual_population_wo_hlns import WIPGraphDualPopulationWithoutHLNS
from src.algorithms.wip_graph_dual_population_wo_ac import WIPGraphDualPopulationWithoutAC


# Test lookup of concrete classes, not a production mechanism-switch registry.
TEST_ALGORITHMS = (WIPGraphDualPopulation, WIPGraphDualPopulationWithoutMGS,
                   WIPGraphDualPopulationWithoutHLNS, WIPGraphDualPopulationWithoutAC)


def make_hlns_algorithm(operations, buffers, variant="HLNS_FULL", **kwargs):
    for cls in TEST_ALGORITHMS:
        if cls.algorithm_variant.fget(None) == variant:
            return cls(operations, buffers, variant=variant, **kwargs)
    raise ValueError(f"Not a current paper algorithm: {variant!r}")


def make(variant="HLNS_FULL", observe=True, seed=0, fe=18):
    ops, buffers, seq, ms = fixture()
    engine = make_hlns_algorithm(ops, buffers, variant, N=3, N_A=12, FE_max=fe,
                                 seed=seed, collect_hlns_stats=observe)
    return engine, Individual(seq, ms, operation_order=tuple(engine.encoder.ms_index_order))


def fixture():
    operations = {
        "J0": [
            {"machines": {"M0": 1, "M4": 2}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M2": 10, "M5": 11}, "buffer_in": "B", "buffer_out": None},
        ],
        "J1": [
            {"machines": {"M1": 2, "M4": 3}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M2": 1, "M5": 2}, "buffer_in": "B", "buffer_out": None},
        ],
        "J2": [
            {"machines": {"M3": 3, "M4": 4}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M2": 1, "M5": 2}, "buffer_in": "B", "buffer_out": None},
        ],
    }
    buffers = {"B": {"capacity": 1, "low_wip": 1}}
    os_seq = [("J0", 0), ("J1", 0), ("J2", 0),
              ("J0", 1), ("J1", 1), ("J2", 1)]
    ms = Encoder(operations).build_ms_map(["M0", "M2", "M1", "M2", "M3", "M2"])
    return operations, buffers, os_seq, ms


# Frozen small reference from DISVA, copied without importing its obsolete runner.
def atomic_space(engine, ind, allowed):
    """Same enumeration/order as formal basic_mutation, with only an allow filter."""
    positions = {key: i for i, key in enumerate(ind.os_seq)}
    windows = {}
    for key, pos in positions.items():
        if key not in allowed:
            continue
        lower = positions[(key[0], key[1]-1)] + 1 if key[1] else 0
        upper = (positions[(key[0], key[1]+1)] - 1
                 if key[1]+1 < len(engine.operations[key[0]]) else len(ind.os_seq)-1)
        if lower < pos or pos < upper:
            windows[key] = (lower, upper)
    ms = [key for key in engine.encoder.ms_index_order if key in allowed
          and len(engine.operations[key[0]][key[1]]["machines"]) >= 2]
    return positions, windows, ms

def restricted_basic_mutation(engine, ind, allowed, rng):
    positions, windows, ms = atomic_space(engine, ind, allowed)
    options = (["os"] if windows else []) + (["ms"] if ms else [])
    child = Individual(ind.os_seq[:], ind.ms_map.copy(), operation_order=ind.operation_order)
    if not options:
        return child, {"exhausted": True, "move_type": "none", "operation": None}
    kind = rng.choice(options)
    if kind == "ms":
        key = rng.choice(ms)
        machines = [m for m in engine.operations[key[0]][key[1]]["machines"]
                    if m != ind.ms_map[key]]
        machine = rng.choice(machines)
        child.ms_map[key] = machine
        return child, {"exhausted": False, "move_type": "MS", "operation": list(key),
                       "old_machine": ind.ms_map[key], "new_machine": machine}
    key = rng.choice(list(windows))
    lower, upper = windows[key]
    old = positions[key]
    target = rng.choice([pos for pos in range(lower, upper+1) if pos != old])
    child.os_seq = engine._insert(ind.os_seq, key, target)
    if child.os_seq is None:
        raise AssertionError("Formal atomic insertion unexpectedly unchanged")
    return child, {"exhausted": False, "move_type": "OS", "operation": list(key),
                   "old_position": old, "target_gap": target,
                   "realized_position": child.os_seq.index(key)}
