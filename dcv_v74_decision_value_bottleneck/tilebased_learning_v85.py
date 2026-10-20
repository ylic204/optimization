"""Learning-run accounting and failure-aware, matched-task comparisons."""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


class LearningProgress:
    """Wall time starts after backbone loading. Counts separate train/eval work."""
    def __init__(self):
        self.started = time.perf_counter()
        self.values = dict(optimizer_steps=0, selector_optimizer_steps=0,
                           metric_optimizer_steps=0, training_samples=0)
        for role in ('training', 'evaluation'):
            self.values[role+'_seconds'] = 0.
            for name in ('modeling_calls', 'formal_solver_calls', 'diagnostic_solver_calls'):
                self.values[role+'_'+name] = 0

    @contextmanager
    def measure(self, env, role):
        before, started = dict(env.calls), time.perf_counter()
        try:
            yield
        finally:
            self.values[role+'_seconds'] += time.perf_counter()-started
            for name, count in env.calls.items():
                self.values[role+'_'+name] += count-before[name]

    def step(self, stage, samples):
        self.values['optimizer_steps'] += 1
        self.values['metric_optimizer_steps' if stage == 'metric' else 'selector_optimizer_steps'] += 1
        self.values['training_samples'] += samples

    def snapshot(self):
        return dict(self.values, elapsed_seconds=time.perf_counter()-self.started)


def read_outcomes(path, mode):
    return [row for line in Path(path).read_text().splitlines()
            if line.strip() and (row := json.loads(line))['mode'] == mode]


def paired_learning_report(before, after, require_same_budget=True):
    """Masks may change: this compares learning, not two prompts on one mask."""
    def index(rows):
        result = {row['sample_id']:row for row in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate sample IDs in learning comparison')
        return result
    old, new = index(before), index(after)
    if not old or old.keys() != new.keys():
        raise ValueError('learning arms must contain identical nonempty sample sets')
    differences, regrets, overlaps = [], [], []
    transitions = dict(both_success=0, before_only_success=0, after_only_success=0, both_failure=0)
    for key, a in old.items():
        b = new[key]
        for field in ('system_prompt', 'model_representation', 'total_tokens'):
            if a[field] != b[field]:
                raise ValueError(f'incompatible learning comparison: {field}')
        if require_same_budget and a['retained_tokens'] != b['retained_tokens']:
            raise ValueError('learning comparison requires the same token budget')
        differences.append(a['loss']-b['loss'])
        sa, sb = a['status'] == 'success', b['status'] == 'success'
        transitions['both_success' if sa and sb else 'before_only_success' if sa else
                    'after_only_success' if sb else 'both_failure'] += 1
        if sa and sb:
            regrets.append(a['relative_regret']-b['relative_regret'])
        left, right = set(a['selected_indices']), set(b['selected_indices'])
        overlaps.append(len(left & right)/max(1, len(left | right)))
    delta = np.asarray(differences)
    return dict(pairs=len(delta), **transitions,
        mean_task_loss_improvement=float(delta.mean()),
        task_loss_win_rate=float((delta > 1e-9).mean()),
        task_loss_tie_rate=float((np.abs(delta) <= 1e-9).mean()),
        task_loss_loss_rate=float((delta < -1e-9).mean()),
        success_rate_change=(transitions['after_only_success']-transitions['before_only_success'])/len(delta),
        mean_regret_improvement_both_success=float(np.mean(regrets)) if regrets else None,
        mean_selection_jaccard=float(np.mean(overlaps)),
        same_budget_required=require_same_budget,
        interpretation='Positive improvement favors after. Failures remain in task loss. '
                       'Success is valid modeling and solving, not optimality. '
                       'Validation selects checkpoints; correlated queries are not a significance test.')
