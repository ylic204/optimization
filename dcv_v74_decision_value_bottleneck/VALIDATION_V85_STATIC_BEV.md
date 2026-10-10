# Revision7 resume fix

34 tests passed across `test_v85_export_resume`, `test_v85_scene_audit`, `test_v85_logged_gt`,
`test_v85_static_bev` and `test_v85_bev`. New tests verify strict parameter/content checks,
one-time compatibility validation for pre-marker r7 frames, preservation of incomplete or
incompatible existing data, atomic publication after successful export and same-run deduplication.

Exporter execution remains CPU-only. These tests do not establish real nuPlan throughput or
validate the contents of the user's existing server output directory.

---

# V8.5 object/evidence audit and quiet map rendering — revision 7

Branch: `v85-bev-evidence-audit`.

`python -m unittest test_v85_scene_audit test_v85_logged_gt test_v85_static_bev test_v85_bev -v`

30 tests passed. Six new tests cover:

- One-to-one object conversion/render accounting, out-of-view objects and bodies crossing the crop with their centers outside.
- Identical-object and ego overdraw distinguished from missing rendering; visible region membership.
- Multiple scene tags, sparse traffic retained, proximity to feasible swept reference bodies, missing-reference nulls.
- Geometric path differences separated from objective conflicts, same-path re-sampling, infeasible-reference unknowns.
- Complete mocked export, audit-only data separation, and deterministic one-row-per-scene index regeneration.
- Invalid tracked-object geometry explicitly rejected rather than silently removed.

A controlled 1440 px intersection with six vehicles, pedestrians, cones and overlapping connector lines
was rendered and visually inspected. Objects remain distinct, connectors are pale, arrows less dense,
and no labels/history/reference line enter the input. This is synthetic QA, not a real nuPlan screenshot.

No density/conflict filter or RL training change is introduced. Dataset tags and object-to-region membership
are diagnostics, not causal evidence labels. All baselines must use the same raster revision.
Real mini exports, actual source object counts, GPU token savings and training still require server runs.

Historical validation records follow; old default values apply only to their revisions.

---

# V8.5 wide BEV / logged GT boundary validation — revision 6

Branch: `v85-bev-gt-boundary`.

`python -m unittest test_v85_logged_gt test_v85_static_bev test_v85_bev -v`

24 tests passed on Python 3.12 / Shapely 2.2. New coverage:

- 1440 px / 96 m static defaults; legacy route defaults unchanged.
- First boundary intersection, side exit, later re-entry ignored, shortest-angle heading interpolation.
- Logged time retained including stationary samples; invalid time rejected, large-coordinate precision retained.
- Exact GT goal pose retained rather than snapped to a lane baseline or tangent.
- Full mocked export beyond the old 28 m goal limit, with and without numeric references.
- GT kept out of model input allowlist and input raster; audit overlay saved separately.
- Short GT skipped explicitly with no replacement goal or manufactured trajectory.

Official nuPlan `get_ego_future_trajectory` was inspected: future state timestamps are used directly;
requested horizon/sample count are not treated as proof of returned coverage. No extrapolation is done.
The default road-graph mode test still rejects any future access; GT mode intentionally reads future ego
for the public goal and held-out trajectory only. The remaining revision5 geometry tests still pass.

A 1440×1440 synthetic straight-road export was visually inspected: no text, visible start/goal,
and the green GT prefix ends at the inset boundary. This is controlled test geometry, not nuPlan data.
Its standalone `--logged-gt` shortest-task evaluation succeeded with length 67.8359995 m and
`reference_status=not_computed`, without inventing a reference cost.

Real nuPlan mini, Python 3.9, Qwen inference and RL training were not run locally. Static GT feasibility
is assessed independently: a recorded dynamic trajectory can collide with frozen actors or violate
new task constraints. It is never declared a per-task optimum. ADE/FDE computation is not added here.

The following is the historical revision5 validation record; its old default values are not revision6 defaults.

---

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
