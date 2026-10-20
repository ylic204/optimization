#!/usr/bin/env python3
"""Create a fixed no-water-goal task subset without changing maps or GT."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from tilebased_core_v85 import CONTRACT, coordinate

POLICY = dict(name='pkmn_no_water_goal_v1', endpoint='target',
              excluded_terrain=['lake','ocean'], excluded_costs=[6.0,10.0],
              source_filter=False, traversal_rules_changed=False)


def filter_dataset(data_root, output_root):
    root, output = Path(data_root), Path(output_root)
    config = json.loads((root/'dataset.json').read_text())
    if config.get('contract') != CONTRACT or not config.get('complete') or config.get('dataset') != 'pkmn':
        raise ValueError('requires a complete preprocessed pkmn dataset')
    if config.get('task_filter'):
        raise ValueError('input is already filtered; use the original preprocessed dataset')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty; original dataset is never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    new_config = dict(config,complete=False,task_filter=POLICY,filter_source=str(root.resolve()))
    (output/'dataset.json').write_text(json.dumps(new_config,indent=2)+'\n')
    report = dict(policy=POLICY, splits={})
    for split in ('train','val','test'):
        folder=output/split;folder.mkdir()
        cache_name=None;seen=set();counts=dict(original_tasks=0,retained_tasks=0,excluded_water_goal=0)
        with (folder/'tasks.jsonl').open('w') as kept, (folder/'excluded_tasks.jsonl').open('w') as rejected:
            for line in (root/split/'tasks.jsonl').read_text().splitlines():
                if not line.strip():continue
                record=json.loads(line);name=record['map_file']
                if Path(name).name!=name or not name.endswith('.npz'):raise ValueError('invalid map_file')
                if name!=cache_name:
                    with np.load(root/split/name,allow_pickle=False) as data:
                        costs=data['true_costs'];targets=data['targets_rc'];sources=data['sources_rc']
                    cache_name=name
                ti,si=record['target_index'],record['source_index']
                target=coordinate(record['target_rc'],costs.shape)
                source=coordinate(record['source_rc'],costs.shape)
                if tuple(targets[ti])!=target or tuple(sources[ti,si])!=source:
                    raise ValueError('manifest endpoint mismatch')
                goal_cost=float(costs[target]);counts['original_tasks']+=1
                if np.isclose(goal_cost,POLICY['excluded_costs'],atol=1e-5,rtol=0).any():
                    counts['excluded_water_goal']+=1
                    rejected.write(json.dumps(dict(record,exclusion_reason='water_goal',goal_cost=goal_cost))+'\n')
                    continue
                kept.write(line+'\n');counts['retained_tasks']+=1
                if name not in seen:
                    shutil.copy2(root/split/name,folder/name);seen.add(name)
        counts['retained_maps']=len(seen)
        report['splits'][split]=counts
        print(f'{split}: {counts}',flush=True)
    (output/'filter_report.json').write_text(json.dumps(report,indent=2)+'\n')
    new_config['complete']=True
    (output/'dataset.json').write_text(json.dumps(new_config,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();filter_dataset(a.data,a.output)


if __name__=='__main__':main()
