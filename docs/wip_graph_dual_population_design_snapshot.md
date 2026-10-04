# WIP Graph Dual Population: Design Snapshot

This note describes the current implementation in
`src/algorithms/wip_graph_dual_population.py`. It is a code reference, not an
experimental conclusion.

## P_M: makespan search

- The trace starts at every operation RELEASE event whose time equals `Cmax`.
- Backward tracing follows all active `processing`, `completion_release`,
  `machine`, `wip`, and `unblocking` dependencies. Unblocking and WIP edges are
  propagation links: they locate upstream constraints but create no direct move.
- An actionable machine edge (`R_u -> S_o`) can yield a precedence-feasible
  `machine_priority` OS intervention. An operation reached on the traced
  processing structure with multiple eligible machines can yield one
  `machine_reassign` action; its new machine is drawn randomly from legal
  alternatives other than the current one.
- Equivalent machine relations are deduplicated; machine reassignment is
  deduplicated by operation. Selection is uniform over the complete actionable
  set. Backtrace depth and propagation-origin tags are diagnostics only.

## P_S: shortage search

- Buffer-event traces identify low-WIP shortage intervals and their area.
  Deficit-changing event anchors start backward influence tracing through
  factual active dependencies.
- Each interval's area is divided among its reached operations to form
  influence weights `phi`; the total is checked against evaluated shortage.
- `H_S` contains the highest-weight operations needed to cover the configured
  fraction `rho` of the shortage mass. Partial relinking toward a suitable
  population or archive guide is restricted to this influence set and executes
  a fraction `eta` of the available actions. Intermediate relinking states are
  not decoded.

## Shared search rules

- The archive is shared nondominated memory used for cooperation. It is not a
  third breeding population.
- Each variation chooses specialized search or basic mutation according to
  `p_mut`; unavailable specialized actions fall back to basic mutation.
- A decoded child replaces its structurally nearest incumbent only when it
  improves that population's primary objective: makespan for `P_M`, shortage
  for `P_S`.
- Initialization evaluates `2N` individuals. Thereafter one fully decoded
  final child consumes one FE; intermediate action construction and relinking
  consume no FE.
