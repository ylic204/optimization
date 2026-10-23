#!/usr/bin/env python3
"""Build a land-only protocol and recompute offline references from original data."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from tilebased_core_v85 import CONTRACT, LAND_PROTOCOL, land_mask, shortest_path, path_cost, task_text


def prepare_land(data_root,output_root):
    root,out=Path(data_root),Path(output_root)
    config=json.loads((root/'dataset.json').read_text())
    if config.get('contract')!=CONTRACT or not config.get('complete') or config['dataset']!='pkmn':
        raise ValueError('requires complete original preprocessed pkmn data')
    if config.get('task_filter') or config.get('navigation_protocol'):
        raise ValueError('use original pkmn export, not an already filtered/rebuilt dataset')
    if out.exists() and any(out.iterdir()):raise FileExistsError('choose a new output directory')
    policy=dict(name=LAND_PROTOCOL,endpoint='both',excluded_terrain=['lake','ocean'],
                excluded_costs=[6.,10.],traversal_rules_changed=True,no_corner_cut=True,
                exclude_unreachable=True,reference='recomputed_under_land_rules')
    cfg=dict(config,complete=False,navigation_protocol=LAND_PROTOCOL,task_filter=policy,
             graph='8_neighbors_no_water_no_corner_cut_unit_steps',reference_source='land_dijkstra',
             legend=config['legend'].replace('lake=6','lake=forbidden').replace('ocean=10','ocean=forbidden'),
             original_data_root=str(root.resolve()))
    out.mkdir(parents=True,exist_ok=True)
    (out/'dataset.json').write_text(json.dumps(cfg,indent=2)+'\n')
    report=dict(protocol=policy,splits={})
    for split in ('train','val','test'):
        grouped=defaultdict(list)
        for line in (root/split/'tasks.jsonl').read_text().splitlines():
            if line.strip():
                record=json.loads(line);grouped[record['map_file']].append(record)
        folder=out/split;folder.mkdir()
        counts=dict(original_tasks=0,retained_tasks=0,water_start=0,water_goal=0,land_disconnected=0,retained_maps=0)
        with (folder/'tasks.jsonl').open('w') as stream,(folder/'excluded_tasks.jsonl').open('w') as rejected:
            for name,records in grouped.items():
                if Path(name).name!=name or not name.endswith('.npz'):raise ValueError('invalid map_file')
                with np.load(root/split/name,allow_pickle=False) as archive:
                    data={k:archive[k] for k in archive.files}
                costs=data['true_costs'];allowed=land_mask(costs)
                for key in ('gt_path_rc','gt_path_mask','gt_path_length','optimal_cost'):
                    data['original_'+key]=data[key].copy()
                data['gt_path_rc']=np.full_like(data['gt_path_rc'],-1)
                data['gt_path_mask']=np.zeros_like(data['gt_path_mask'],dtype=bool)
                data['gt_path_length']=np.zeros_like(data['gt_path_length'])
                data['optimal_cost']=np.zeros_like(data['optimal_cost'],dtype=np.float64)
                data['shortest_path_length']=np.full(data['optimal_cost'].shape,-1,dtype=np.int32)
                data['allowed_mask']=allowed
                retained=[]
                for record in records:
                    counts['original_tasks']+=1
                    ti,si=record['target_index'],record['source_index']
                    source=tuple(map(int,data['sources_rc'][ti,si]));target=tuple(map(int,data['targets_rc'][ti]))
                    if list(source)!=record['source_rc'] or list(target)!=record['target_rc']:raise ValueError('endpoint mismatch')
                    if 'map_sha256' in record and hashlib.sha256(data['image'].tobytes()).hexdigest()!=record['map_sha256']:
                        raise ValueError('image checksum mismatch')
                    reason='water_start' if not allowed[source] else 'water_goal' if not allowed[target] else None
                    route=shortest_path(costs,source,target,allowed,no_corner_cut=True) if reason is None else None
                    if reason is None and route is None:reason='land_disconnected'
                    if reason:
                        counts[reason]+=1;rejected.write(json.dumps(dict(record,exclusion_reason=reason))+'\n');continue
                    length_route=shortest_path(np.ones_like(costs),source,target,allowed,no_corner_cut=True)
                    data['gt_path_rc'][ti,si,:len(route)]=route
                    data['gt_path_mask'][ti,si,route[:,0],route[:,1]]=True
                    data['gt_path_length'][ti,si]=len(route)
                    data['optimal_cost'][ti,si]=path_cost(costs,route)
                    data['shortest_path_length'][ti,si]=len(length_route)-1
                    retained.append(dict(record,task_text=task_text('pkmn',source,target,land_only=True)))
                    counts['retained_tasks']+=1
                if retained:
                    np.savez_compressed(folder/name,**data)
                    for record in retained:stream.write(json.dumps(record)+'\n')
                    counts['retained_maps']+=1
        report['splits'][split]=counts;print(f'{split}: {counts}',flush=True)
    (out/'land_report.json').write_text(json.dumps(report,indent=2)+'\n')
    cfg['complete']=True;(out/'dataset.json').write_text(json.dumps(cfg,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare_land(args.data,args.output)


if __name__=='__main__':main()
