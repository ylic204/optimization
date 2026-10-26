"""Audit a real Full validation run before training against generated MILP feedback."""
import argparse
import hashlib
import json
from pathlib import Path

MIN_SAMPLES=32
MIN_MODEL_RATE=.95
MIN_SR=.90


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_signature():
    root=Path(__file__).resolve().parent
    files=['tilebased_algebra_v85.py','tilebased_prompts_v85.py','tilebased_solver_environment_v85.py',
           'prompts/tilebased_land_system_v85.txt']
    return {f:digest(root/f) for f in files}


def audit(run):
    run=Path(run).resolve();args=json.loads((run/'args.json').read_text())
    if args['mode']!='evaluate' or args['split']!='val' or args['model_representation']!='algebra' or args['modeling_prompt']!='navigation' or args.get('compare_prompts'):
        raise ValueError('requires an algebra/navigation evaluation on val without prompt comparison')
    rows=[json.loads(s) for s in (run/'outcomes.jsonl').read_text().splitlines() if s.strip()]
    rows=[r for r in rows if r.get('mode')=='full']
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Full samples must be nonempty and unique')
    if any(r.get('model_representation')!='algebra' or r.get('retained_tokens')!=r.get('total_tokens') or not r.get('total_tokens') for r in rows):
        raise ValueError('invalid Full observation records')
    if args.get('algebra_code_signature')!=code_signature():raise ValueError('run was produced by different/missing algebra implementation signature')
    solved=sum(r.get('solver_result',{}).get('status')=='optimal' for r in rows)
    success=sum(r['status']=='success' for r in rows)
    model_rate=solved/len(rows);sr=success/len(rows)
    return dict(passed=len(rows)>=MIN_SAMPLES and model_rate>=MIN_MODEL_RATE and sr>=MIN_SR,
                run_directory=str(run),samples=len(rows),solved_rate=model_rate,sr=sr,
                thresholds=dict(min_samples=MIN_SAMPLES,min_solved_rate=MIN_MODEL_RATE,min_sr=MIN_SR),
                args_sha256=digest(run/'args.json'),outcomes_sha256=digest(run/'outcomes.jsonl'),
                code_signature=code_signature(),dataset_signature=args['dataset_signature'])


def require_full_validation(path,args,dataset):
    if not path:raise ValueError('Run Full on val first, then validate_tilebased_full_v85.py; supply --full-validation REPORT before algebra training')
    recorded=json.loads(Path(path).read_text());current=audit(recorded['run_directory'])
    if recorded!=current or not current['passed']:raise ValueError('Full validation is failed, stale or modified; rerun Full validation')
    prior=json.loads((Path(current['run_directory'])/'args.json').read_text())
    if prior['dataset_signature']!=dataset.signature or Path(prior['data']).resolve()!=Path(args.data).resolve():raise ValueError('Full validation dataset mismatch')
    for key in ('vlm_model','load_4bit','prune_layer','modeling_max_new_tokens','modeling_prompt','model_representation'):
        if prior.get(key)!=getattr(args,key,None):raise ValueError(f'Full validation configuration mismatch: {key}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args()
    report=audit(a.run);path=Path(a.run)/'full_validation.json'
    examples={}
    for line in (Path(a.run)/'outcomes.jsonl').read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row.get('mode')!='full' or row['status']=='success':continue
        bucket=examples.setdefault(row.get('failure_category','unknown'),[])
        if len(bucket)<2:bucket.append(row)
    (Path(a.run)/'failure_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
