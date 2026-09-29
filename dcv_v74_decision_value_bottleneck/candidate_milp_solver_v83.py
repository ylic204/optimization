"""Deterministic mathematical model and SciPy MILP solver for V8.3."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from optimization_spec_v83 import METRIC_NAMES


@dataclass
class SolverResult:
    selected_index: int
    objective: float
    metric_values: dict
    constraint_slacks: dict


def solve_candidate_milp(candidate_metrics, candidate_valid, task):
    """Select exactly one trajectory with a mixed-integer linear program.

    Let x_i be binary and s_j be non-negative constraint slacks:

        min  sum_i sum_k w_k m_ik x_i + rho sum_j s_j
        s.t. sum_i x_i = 1
             sum_i m_ij x_i - s_j <= limit_j
             x_i <= valid_i, x_i in {0,1}, s_j >= 0.

    With independent candidates this MILP is intentionally simple.  It keeps
    the language-to-mathematics-to-solver interface explicit; a future version
    can replace x_i by edge-flow variables without changing the task schema.
    """
    metrics = np.asarray(candidate_metrics, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=np.float64)
    weights = np.asarray(task.weight_vector, dtype=np.float64)
    constrained_names = list(task.limits)

    candidate_count = metrics.shape[0]
    slack_count = len(constrained_names)
    variable_count = candidate_count + slack_count

    # Linear objective coefficients for [candidate binaries, constraint slacks].
    c = np.concatenate(
        [metrics @ weights, np.full(slack_count, task.constraint_penalty)]
    )

    # Candidate variables are binary; slack variables are continuous.
    integrality = np.concatenate(
        [np.ones(candidate_count), np.zeros(slack_count)]
    )
    lower = np.zeros(variable_count)
    upper = np.concatenate([valid, np.full(slack_count, np.inf)])

    # First row imposes exactly one selected trajectory.
    rows = [np.concatenate([np.ones(candidate_count), np.zeros(slack_count)])]
    row_lower = [1.0]
    row_upper = [1.0]

    # Each language-generated upper bound receives one penalized slack.
    for slack_index, metric_name in enumerate(constrained_names):
        metric_index = METRIC_NAMES.index(metric_name)
        row = np.zeros(variable_count)
        row[:candidate_count] = metrics[:, metric_index]
        row[candidate_count + slack_index] = -1.0
        rows.append(row)
        row_lower.append(-np.inf)
        row_upper.append(float(task.limits[metric_name]))

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            np.stack(rows), np.asarray(row_lower), np.asarray(row_upper)
        ),
    )
    selected = int(np.argmax(result.x[:candidate_count]))
    chosen_metrics = {
        name: float(metrics[selected, index])
        for index, name in enumerate(METRIC_NAMES)
    }
    slacks = {
        name: float(result.x[candidate_count + index])
        for index, name in enumerate(constrained_names)
    }
    return SolverResult(selected, float(result.fun), chosen_metrics, slacks)
