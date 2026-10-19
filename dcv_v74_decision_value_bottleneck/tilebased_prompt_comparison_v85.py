"""Matched prompt comparison; failures remain in all-sample task loss."""
import numpy as np


def paired_prompt_report(legacy, navigation):
    """Positive task_loss_improvement means navigation reduced penalized loss."""
    def index(rows):
        keyed = {row['sample_id']:row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError('duplicate sample IDs in paired evaluation')
        return keyed
    old,new = index(legacy),index(navigation)
    if not old or old.keys() != new.keys():
        raise ValueError('prompt arms must have identical nonempty sample sets')
    differences,regret,accuracy = [],[],[]
    transitions = dict(both_success=0,legacy_only_success=0,navigation_only_success=0,both_failure=0)
    for key,a in old.items():
        b = new[key]
        if a['selected_indices'] != b['selected_indices']:
            raise ValueError('prompt arms must retain identical visual regions')
        differences.append(a['loss']-b['loss'])
        sa,sb = a['status']=='success',b['status']=='success'
        transitions['both_success' if sa and sb else 'legacy_only_success' if sa else
                    'navigation_only_success' if sb else 'both_failure'] += 1
        if sa and sb:
            regret.append(a['relative_regret']-b['relative_regret'])
            if 'cost_cell_accuracy' in a and 'cost_cell_accuracy' in b:
                accuracy.append(b['cost_cell_accuracy']-a['cost_cell_accuracy'])
    delta = np.asarray(differences)
    return dict(pairs=len(delta),**transitions,
        mean_task_loss_improvement=float(delta.mean()),
        task_loss_win_rate=float((delta>1e-9).mean()),
        task_loss_tie_rate=float((np.abs(delta)<=1e-9).mean()),
        task_loss_loss_rate=float((delta < -1e-9).mean()),
        success_rate_change=(transitions['navigation_only_success']-transitions['legacy_only_success'])/len(delta),
        mean_regret_improvement_both_success=float(np.mean(regret)) if regret else None,
        mean_cell_accuracy_change_both_success=float(np.mean(accuracy)) if accuracy else None,
        interpretation='Positive improvement/change favors navigation. Success means valid modeling and solving, not optimality. Repeated S/G queries on one map are correlated; this report is descriptive, not a significance test.')
