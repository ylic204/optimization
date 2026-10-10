# V8.5 static road reference validation — revision 5

Branch: `v85-static-bev-road-reference`.

## Executed checks

`python -m unittest test_v85_static_bev test_v85_bev -v`

17 tests passed. They cover:

- Default 720 px / 48 m static support and unchanged legacy opt-in defaults.
- No rendered text, logged route, actor history, signal or future-GT access in static export.
- Identical RGB/goal across tasks, exact preview pixels and explicit model input allowlist.
- Invalid/behind-road goals are rejected rather than replaced; deterministic forward road targets.
- Large UTM coordinates preserve sub-meter geometry before ego-frame conversion.
- Vehicle body and swept motion collision, including obstacles between sparse waypoints.
- Road boundary, inverse/sideways motion and turning-radius rejection.
- Disconnected lanes do not connect by proximity; explicit outgoing relations do.
- Lane changes require reciprocal map adjacency and forward curvature-feasible motion.
- Dijkstra gives 26 m on the controlled straight 26 m task.
- Pedestrian clearance changes the chosen path and lowers that task's objective.
- Infeasible hazard task has an empty reference and explicit status, not a zero-cost answer.
- Full NPZ export followed by independent reference geometry and cost evaluation.
- Legacy route/GT export regression and explicit rejection of static input in its processor.

A 720×720 controlled-geometry preview was generated and visually inspected: no text or margins,
visible ego heading, goal, lane direction, pedestrians and construction hazard. It is synthetic
validation geometry, **not a screenshot from nuPlan mini**.

The standalone evaluator was run on the controlled shortest-path record:
`success=true`, length/cost = 26.0, cost gap = 0.0, maximum curvature = 0.0.
The shortest task may enter a hazard (that constraint is specific to `avoid_hazards`).

## Scope of evidence

Local Python 3.12 / Shapely 2.2 were used. Code avoids Python-3.10-only annotation syntax for the
user's Python 3.9 environment; actual execution with Python 3.9 and Shapely 1.8 was not available.
nuPlan lane adapter assumptions were checked against the official `NuPlanLane` implementation:
`outgoing_edges` points to lane connectors, and `adjacent_edges` returns neighboring lanes in the
same roadblock. Tests mock these API shapes; they do not load a real map database.

Real nuPlan mini export, GPU/Qwen inference, RL training and closed-loop driving were not run.
The graph optimum is relative to the saved finite graph and sampled costs. It is not a certified
continuous optimum or a proof of driving safety. Geometry evaluation uses the configured finite
sampling resolution. Real scenario yield must be reported together with skipped-scene counts.
