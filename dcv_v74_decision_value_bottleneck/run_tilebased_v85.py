"""Tilebased stages sharing the V8.5 frozen Qwen, selector, mask FM/ND and RL.

metric/distill use an explicitly labeled differentiable path-cost surrogate.
rl/evaluate ALWAYS generate JSON with frozen Qwen and call the exact solver.
experiment trains all three stages and compares untrained/trained selectors.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from gradient_flow_selector_v85 import (build_v85_model, GradientFlowSelectorV85,
    fixed_mass_mask, project_capped_simplex, endpoint_improvement_loss)
from qwen3vl_selector_v79 import topk_region_indices
from rl_posttraining_v85 import sample_region_action, policy_gradient_loss
from planning_dataset_tilebased_v85 import TilebasedDatasetV85
from tilebased_core_v85 import CONTRACT, shortest_path, path_cost
from tilebased_prompts_v85 import prompt_metadata
from tilebased_prompt_comparison_v85 import paired_prompt_report
from tilebased_learning_v85 import (LearningProgress, write_json, append_json,
    read_outcomes, paired_learning_report)
from tilebased_solver_environment_v85 import (FrozenTilebasedSolverEnvironment,
    ENVIRONMENT_CONTRACT, CORRIDOR_ENVIRONMENT_CONTRACT, environment_contract, summarize_outcomes)


class TileCostHead(nn.Module):
    """Training-only proxy. Never supplies costs to the real solver environment."""
    def __init__(self, width, hidden, grid):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                                 nn.Linear(hidden, grid*grid))

    def forward(self, fused):
        return (F.softplus(self.net(fused.float())) + .01).reshape(-1, self.grid, self.grid)


def to_device(batch, device):
    return {part: {k: v.to(device) if torch.is_tensor(v) else v for k, v in values.items()}
            for part, values in batch.items()}


def encode(model, public, budget):
    # Early frozen vision is computed once. No privileged data accepted here.
    with torch.no_grad():
        encoded = model.backbone.encode_inputs(public['image'], public['task_text'])
    budgets = public['image'].new_full((len(public['image']),), budget)
    selection, positions, valid = model.select_encoded(encoded, budgets, public['region_world_bounds'])
    return encoded, selection, positions, valid


def rollout(model, selection, positions, valid, args):
    k = max(1, min(valid.shape[1], round(args.budget*valid.shape[1])))
    mass = selection['logits'].new_full((len(valid),), float(k))
    initial = fixed_mass_mask(selection['logits'], valid, mass, args.mask_temperature)
    final, states, velocities = model.mask_flow.rollout(initial, selection['z'], selection['task_context'],
                                                       positions, valid, mass)
    scores = model.mask_flow.decision_scores(states, velocities, selection['logits'])
    return initial, final, states, velocities, scores, mass, k


def proxy_paths(public, labels, count=8):
    """Small exact-solver route set used ONLY by the differentiable warm-up loss.

    Includes a GT optimum plus uniform-cost and perturbed-cost paths. Routes,
    their true costs and this candidate set are never provided to Qwen/selector.
    """
    paths, values = [], []
    for row in range(len(labels['true_costs'])):
        costs = labels['true_costs'][row].detach().cpu().numpy()
        source = public['source_rc'][row].cpu().numpy()
        target = public['target_rc'][row].cpu().numpy()
        rng = np.random.default_rng(712)
        masks, costs_out, seen = [], [], set()
        for j in range(count*5):
            weights = costs if j == 0 else np.ones_like(costs) if j == 1 else costs*np.exp(rng.normal(0, 1.5, costs.shape))
            route = shortest_path(weights, source, target)
            key = tuple(map(tuple, route))
            if key in seen:
                continue
            seen.add(key)
            mask = np.zeros(costs.shape, np.float32)
            mask[route[:, 0], route[:, 1]] = 1.
            masks.append(mask)
            costs_out.append(path_cost(costs, route))
            if len(masks) == count:
                break
        validity = [True]*len(masks) + [False]*(count-len(masks))
        while len(masks) < count:
            masks.append(masks[0].copy()); costs_out.append(costs_out[0])
        paths.append(masks); values.append((costs_out, validity))
    device = labels['true_costs'].device
    return (torch.as_tensor(np.asarray(paths), device=device),
            torch.tensor([v[0] for v in values], device=device),
            torch.tensor([v[1] for v in values], device=device))


def proxy_values(model, encoded, mask, labels, candidates, temperature):
    predicted = model.metric_head(model.fuse_mask(encoded, mask))
    masks, true_cost, valid = candidates
    predicted_cost = (predicted[:, None]*masks).sum((-1,-2))
    scale = labels['optimal_cost'].float().clamp_min(1e-6)
    probabilities = torch.softmax((-predicted_cost/(scale[:,None]*temperature)).masked_fill(~valid, -torch.inf), -1)
    regret = (true_cost-scale[:,None]).clamp_min(0)/scale[:,None]
    task = (probabilities*regret).sum(-1)
    regression = F.smooth_l1_loss(torch.log(predicted), torch.log(labels['true_costs']), reduction='none').mean((-1,-2))
    return task, regression


def distill_loss(model, encoded, selection, positions, valid, labels, candidates, args):
    initial, final, states, velocities, scores, mass, k = rollout(model, selection, positions, valid, args)
    with torch.no_grad():
        initial_task, _ = proxy_values(model, encoded, initial.detach(), labels, candidates, args.decision_temperature)
    task, regression = proxy_values(model, encoded, final, labels, candidates, args.decision_temperature)
    matches = []
    for step, velocity in enumerate(velocities):
        state = states[step].detach().requires_grad_(True)
        teacher_task, _ = proxy_values(model, encoded, state, labels, candidates, args.decision_temperature)
        gradient = torch.autograd.grad(teacher_task.sum(), state)[0]
        target = project_capped_simplex(state.detach()-args.teacher_step_size*gradient.detach(), valid, mass)
        target_velocity = (target-state.detach())/args.teacher_step_size
        reliable = target_velocity.norm(dim=-1) > 1e-7
        cosine = 1.-F.cosine_similarity(velocity, target_velocity, dim=-1, eps=1e-7)
        direction = (cosine*reliable).sum()/reliable.sum().clamp_min(1)
        magnitude = F.smooth_l1_loss(torch.log1p(velocity.norm(dim=-1)), torch.log1p(target_velocity.norm(dim=-1)))
        proposal = project_capped_simplex(states[step]+model.mask_flow.step_size(step)*velocity, valid, mass)
        matches.append(direction+.1*magnitude+.5*F.smooth_l1_loss(proposal, target))
    gfm = torch.stack(matches).mean()
    improve = endpoint_improvement_loss(initial_task, task, args.improve_margin).mean()
    loss = task.mean()+args.lambda_gfm*gfm+args.lambda_improve*improve+args.lambda_metric*regression.mean()
    return loss, {'proxy_task_loss': task.mean(), 'gfm': gfm, 'improve': improve, 'cost_regression': regression.mean()}


def rl_loss(model, reference, env, encoded, selection, positions, valid, public, labels, args):
    _, _, _, _, scores, _, k = rollout(model, selection, positions, valid, args)
    with torch.no_grad():
        budgets = public['image'].new_full((len(valid),), args.budget)
        rs, rp, rv = reference.select_encoded(encoded, budgets, public['region_world_bounds'])
        ref_scores = rollout(reference, rs, rp, rv, args)[4]
        baseline = env.evaluate(encoded, topk_region_indices(selection['logits'], valid, k), public, labels)
        initial = scores.new_tensor([x['loss'] for x in baseline])
    rewards, logps, entropies, kls, losses = [], [], [], [], []
    for _ in range(args.rl_group_size):
        action, logp, entropy, kl = sample_region_action(scores, valid, k, args.rl_temperature, ref_scores)
        result = env.evaluate(encoded, action, public, labels)
        task = scores.new_tensor([x['loss'] for x in result])
        improvement = F.relu(task-initial+args.improve_margin)
        rewards.append(-task-args.lambda_improve*improvement)
        losses.append(task)
        logps.append(logp); entropies.append(entropy/k); kls.append(kl/k)
    loss, policy, advantage = policy_gradient_loss(torch.stack(logps), torch.stack(rewards),
        torch.stack(entropies), torch.stack(kls), args.rl_kl_weight, args.rl_entropy_weight)
    return loss, {'task_loss': torch.stack(losses).mean(), 'policy_loss': policy,
                  'advantage_abs': advantage.abs().mean(),
                  'reward_std': torch.stack(rewards).std(dim=0, unbiased=False).mean()}


@torch.no_grad()
def evaluate(model, loader, env, args, device, modes, path):
    model.eval()
    paired = isinstance(env,dict)
    environments = env if paired else {env.prompt_variant:env}
    records = {variant:{mode:[] for mode in modes} for variant in environments}
    sample_offset = 0
    generator = torch.Generator(device=device).manual_seed(args.seed+71)
    with Path(path).open('w') as stream:
        for raw in tqdm(loader, desc='Tilebased frozen modeling + solver'):
            batch = to_device(raw, device)
            public, labels = batch['public_inputs'], batch['supervision']
            encoded, selection, positions, valid = encode(model, public, args.budget)
            _, _, _, _, scores, _, k = rollout(model, selection, positions, valid, args)
            subsets = {'full': torch.arange(valid.shape[1], device=device)[None].expand(len(valid),-1),
                       'random': topk_region_indices(torch.rand(valid.shape, device=device, generator=generator),valid,k),
                       'initial': topk_region_indices(selection['logits'],valid,k),
                       'learned': topk_region_indices(scores,valid,k)}
            # Stage labels used only when the experiment holds those weights.
            subsets['untrained'] = subsets['distilled'] = subsets['learned']
            # Same encoded image, endpoints and chosen regions in both arms.
            # Alternate arm order across batches to reduce systematic timing bias.
            variants = list(environments)
            if sample_offset % 2:
                variants.reverse()
            for mode in modes:
                for variant in variants:
                    results = environments[variant].evaluate(encoded, subsets[mode], public, labels)
                    records[variant][mode].extend(results)
                    for row, result in enumerate(results):
                        length = int(labels['gt_path_length'][row])
                        stream.write(json.dumps(dict(result, mode=mode,
                            gt_path_rc=labels['gt_path_rc'][row,:length].cpu().tolist()), allow_nan=False)+'\n')
                    stream.flush()
            sample_offset += 1
    summaries = {variant:{mode:summarize_outcomes(rows) for mode,rows in group.items()}
                 for variant,group in records.items()}
    if paired:
        summaries['paired'] = {mode:paired_prompt_report(records['legacy'][mode],records['navigation'][mode])
                               for mode in modes}
        return summaries
    return next(iter(summaries.values()))


@torch.no_grad()
def validate_proxy(model, loader, args, device):
    model.eval()
    values = []
    for raw in loader:
        batch = to_device(raw, device)
        public, labels = batch['public_inputs'], batch['supervision']
        encoded, selection, positions, valid = encode(model, public, args.budget)
        if args.mode == 'metric':
            masks = torch.ones_like(selection['logits'])
        else:
            masks = rollout(model, selection, positions, valid, args)[1]
        task, regression = proxy_values(model, encoded, masks, labels, proxy_paths(public, labels), args.decision_temperature)
        values.extend((regression if args.mode == 'metric' else task).cpu().tolist())
    return {'proxy_validation_loss': float(np.mean(values)), 'not_solver_evaluation': True}


def validate_checkpoint(ckpt, dataset, args):
    if ckpt.get('contract') != CONTRACT or ckpt.get('dataset_signature') != dataset.signature:
        raise ValueError('checkpoint/data contract mismatch; legacy BEV checkpoints are not Tilebased checkpoints')
    if ckpt.get('environment') not in (ENVIRONMENT_CONTRACT,CORRIDOR_ENVIRONMENT_CONTRACT):
        raise ValueError('checkpoint solver environment mismatch')
    for name in ('hidden','latent_dim','mask_steps','prune_layer','max_text_length','mask_step_size'):
        if ckpt['args'][name] != getattr(args, name):
            raise ValueError(f'checkpoint architecture mismatch: {name}; use the saved arguments')
    required = {'distill': ('metric','distill','rl'), 'rl': ('distill','rl')}
    if args.mode in required and ckpt['stage'] not in required[args.mode]:
        raise ValueError(f'{args.mode} requires checkpoint stage {required[args.mode]}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['experiment','metric','distill','rl','evaluate'], required=True)
    p.add_argument('--data', required=True, help='preprocessed root containing dataset.json')
    p.add_argument('--vlm-model', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--load-4bit', action='store_true')
    p.add_argument('--checkpoint', default='', help='stage initialization, not exact optimizer resume')
    p.add_argument('--output', required=True, help='new output directory')
    p.add_argument('--split', choices=['train','val','test'], default='test', help='evaluate only')
    p.add_argument('--modes', nargs='+', choices=['full','random','initial','learned'], default=['full','random','initial','learned'])
    p.add_argument('--limit', type=int, default=0, help='training/evaluation tasks; 0=all')
    p.add_argument('--val-limit', type=int, default=0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--metric-epochs', type=int, default=3, help='experiment only')
    p.add_argument('--distill-epochs', type=int, default=3, help='experiment only')
    p.add_argument('--rl-epochs', type=int, default=2, help='experiment only')
    p.add_argument('--rl-limit', type=int, default=0, help='experiment: first N tasks of the training subset for expensive RL; 0=all')
    p.add_argument('--reference-modes', nargs='*', choices=['full','random'], default=['full'],
                   help='experiment references; untrained/learned comparison is always run')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--budget', type=float, default=.15)
    p.add_argument('--prune-layer', type=int, default=4)
    p.add_argument('--max-text-length', type=int, default=256)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--mask-steps', type=int, default=4)
    p.add_argument('--mask-step-size', type=float, default=.25)
    p.add_argument('--mask-temperature', type=float, default=.35)
    p.add_argument('--teacher-step-size', type=float, default=.25)
    p.add_argument('--decision-temperature', type=float, default=.1)
    p.add_argument('--lambda-gfm', type=float, default=1.)
    p.add_argument('--lambda-metric', type=float, default=.1)
    p.add_argument('--lambda-improve', type=float, default=.1)
    p.add_argument('--improve-margin', type=float, default=0.)
    p.add_argument('--rl-group-size', type=int, default=4)
    p.add_argument('--rl-temperature', type=float, default=.15)
    p.add_argument('--rl-kl-weight', type=float, default=.02)
    p.add_argument('--rl-entropy-weight', type=float, default=.001)
    p.add_argument('--failure-cost', type=float, default=10.)
    p.add_argument('--modeling-max-new-tokens', type=int, default=4096)
    p.add_argument('--model-representation', choices=['corridor','matrix'], default='corridor')
    p.add_argument('--modeling-prompt', choices=['legacy','navigation'], default='navigation')
    p.add_argument('--compare-prompts', action='store_true', help='evaluate legacy/navigation on identical selected regions in one run')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--grad-clip', type=float, default=5.)
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()
    if not 0 < a.budget <= 1 or min(a.batch,a.epochs,a.mask_steps,a.rl_group_size-1) <= 0:
        p.error('budget must be (0,1]; positive batch/epochs/mask-steps and group-size>=2 required')
    if min(a.teacher_step_size,a.decision_temperature,a.mask_step_size,a.mask_temperature,a.rl_temperature,a.lr,a.grad_clip) <= 0:
        p.error('temperatures, step sizes, learning rate and grad clip must be positive')
    if min(a.limit,a.val_limit,a.rl_limit,a.improve_margin) < 0:
        p.error('limits and improvement margin must be nonnegative')
    if min(a.metric_epochs,a.distill_epochs,a.rl_epochs) <= 0:
        p.error('experiment stage epochs must be positive')
    if len(set(a.reference_modes)) != len(a.reference_modes):
        p.error('reference modes must be unique')
    if a.mode == 'experiment' and a.checkpoint:
        p.error('experiment starts from scratch to measure untrained vs trained; use individual stages for checkpoint initialization')
    if a.compare_prompts and a.mode != 'evaluate':
        p.error('--compare-prompts is evaluation only')
    if a.compare_prompts and len(set(a.modes)) != len(a.modes):
        p.error('comparison modes must be unique')
    if a.mode in ('distill','rl') and not a.checkpoint:
        p.error('distill requires metric checkpoint; rl requires distill checkpoint')
    if a.mode == 'evaluate' and not a.checkpoint and set(a.modes)-{'full','random'}:
        p.error('without a checkpoint evaluate only --modes full random')
    return a


def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; choose a fresh run directory')
    dataset = TilebasedDatasetV85(args.data, args.split if args.mode == 'evaluate' else 'train', args.limit)
    args.region_grid = dataset.config['region_grid']
    variants = ['legacy','navigation'] if args.compare_prompts else [args.modeling_prompt]
    prompt_config = {variant:prompt_metadata(variant,dataset.config['dataset'],args.model_representation) for variant in variants}
    args.modeling_prompt_metadata = prompt_config
    args.dataset_signature = dataset.signature
    args.torch_version = str(torch.__version__)
    args.checkpoint_modeling_prompt = None
    args.checkpoint_model_representation = None
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    if checkpoint:
        validate_checkpoint(checkpoint, dataset, args)
        args.checkpoint_modeling_prompt = checkpoint['args'].get('modeling_prompt','legacy')
        args.checkpoint_model_representation = checkpoint['args'].get('model_representation','matrix')
    model = build_v85_model(args, device)
    if model.backbone.target_image_side != dataset.config['image_side']:
        raise ValueError('Qwen patch/merge geometry would resize native images; incompatible backbone')
    model.metric_head = TileCostHead(model.backbone.text_hidden_dim, args.hidden, dataset.config['grid_shape'][0]).to(device)
    if checkpoint:
        for name in ('selector','mask_flow','metric_head'):
            getattr(model, name).load_state_dict(checkpoint[name])
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    output.mkdir(parents=True, exist_ok=True)
    (output/'args.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    (output/'prompt_config.json').write_text(json.dumps(prompt_config,indent=2)+'\n')
    loader = DataLoader(dataset,batch_size=args.batch,shuffle=args.mode!='evaluate',num_workers=args.workers)
    environments = {variant:FrozenTilebasedSolverEnvironment(model.backbone,args.region_grid,
        args.modeling_max_new_tokens,args.failure_cost,prompt_variant=variant,model_representation=args.model_representation) for variant in variants}
    env = environments if args.compare_prompts else environments[args.modeling_prompt]
    if args.mode == 'evaluate':
        report = evaluate(model,loader,env,args,device,args.modes,output/'outcomes.jsonl')
        (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2)); return
    validation_data = TilebasedDatasetV85(args.data,'val',args.val_limit)
    validation_loader = DataLoader(validation_data,batch_size=args.batch,num_workers=args.workers)
    if args.mode == 'experiment':
        run_experiment(model, loader, validation_loader, env, args, device, output)
    else:
        train_stage(model, loader, validation_loader, env, args, device, output, LearningProgress())


def save_checkpoint(model, dataset, args, path, epoch, report, progress):
    saved = {name:getattr(model,name).state_dict() for name in ('selector','mask_flow','metric_head')}
    saved.update(contract=CONTRACT, environment=environment_contract(args.model_representation),
                 dataset_signature=dataset.signature, stage=args.mode, args=vars(args).copy(),
                 epoch=epoch, validation=report, progress=progress.snapshot())
    torch.save(saved, path)


def load_stage_weights(model, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for name in ('selector','mask_flow','metric_head'):
        getattr(model, name).load_state_dict(saved[name])
    return saved


def train_stage(model, loader, validation_loader, env, args, device, output, progress,
                initial_validation=None, curve_path=None):
    """Train only stage parameters; retain best AND last, return best metadata.

    In the pipeline RL starts with the distilled validation as its step-zero
    candidate. If RL does not improve it, best.pt explicitly keeps stage=distill.
    """
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'args.json', vars(args))
    model.metric_head.requires_grad_(args.mode=='metric')
    model.selector.requires_grad_(args.mode!='metric')
    model.mask_flow.requires_grad_(args.mode!='metric')
    reference = None
    if args.mode == 'rl':
        reference = GradientFlowSelectorV85(model.backbone,copy.deepcopy(model.selector),model.metric_head,copy.deepcopy(model.mask_flow))
        reference.selector.requires_grad_(False); reference.mask_flow.requires_grad_(False); reference.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=args.lr)
    best = float('inf')
    if initial_validation is not None:
        best = initial_validation['task_loss']
        initial_args = copy.copy(args)
        initial_args.mode = 'distill'
        save_checkpoint(model, loader.dataset, initial_args, output/'best.pt', 0,
                        {'learned':initial_validation}, progress)
    for epoch in range(args.epochs):
        model.eval()
        (model.metric_head if args.mode=='metric' else model.selector).train()
        if args.mode!='metric': model.mask_flow.train()
        logs = []
        for raw in tqdm(loader,desc=f'Tilebased {args.mode} {epoch+1}'):
            with progress.measure(env, 'training'):
                batch = to_device(raw,device)
                public, labels = batch['public_inputs'],batch['supervision']
                encoded,selection,positions,valid = encode(model,public,args.budget)
                if args.mode == 'metric':
                    k = max(1,round(args.budget*valid.shape[1]))
                    indices = topk_region_indices(torch.rand(valid.shape,device=device),valid,k)
                    with torch.no_grad():
                        full = model.fuse_mask(encoded,torch.ones_like(selection['logits']))
                        sparse = model.fuse_indices(encoded,indices)
                    target = torch.log(labels['true_costs'])
                    loss = .5*(F.smooth_l1_loss(torch.log(model.metric_head(full)),target)+
                               F.smooth_l1_loss(torch.log(model.metric_head(sparse)),target))
                    metrics = {'cost_regression':loss}
                elif args.mode == 'distill':
                    loss,metrics = distill_loss(model,encoded,selection,positions,valid,labels,proxy_paths(public,labels),args)
                else:
                    loss,metrics = rl_loss(model,reference,env,encoded,selection,positions,valid,public,labels,args)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite {args.mode} loss; refusing to update weights')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters,args.grad_clip,error_if_nonfinite=True)
                optimizer.step()
                progress.step(args.mode, len(public['image']))
                logs.append({k:float(v.detach()) for k,v in dict(metrics,loss=loss,grad_norm=grad_norm).items()})
            append_json(output/'updates.jsonl', dict(stage=args.mode, epoch=epoch+1,
                        metrics=logs[-1], progress=progress.snapshot()))
        with progress.measure(env, 'evaluation'):
            report = (evaluate(model,validation_loader,env,args,device,['learned'],output/f'val_{epoch+1}.jsonl')
                      if args.mode=='rl' else validate_proxy(model,validation_loader,args,device))
        score = report['learned']['task_loss'] if args.mode=='rl' else report['proxy_validation_loss']
        if not np.isfinite(score):
            raise FloatingPointError('nonfinite validation loss; refusing to select a checkpoint')
        entry = {'stage':args.mode, 'epoch':epoch+1,
                 'train':{k:float(np.mean([x[k] for x in logs])) for k in logs[0]},
                 'validation':report, 'progress':progress.snapshot(), 'is_new_best':score < best}
        append_json(output/'metrics.jsonl', entry)
        if curve_path is not None and args.mode == 'rl':
            append_json(curve_path, entry)
        print(json.dumps(entry),flush=True)
        save_checkpoint(model, loader.dataset, args, output/'last.pt', epoch+1, report, progress)
        if score < best:
            best = score
            save_checkpoint(model, loader.dataset, args, output/'best.pt', epoch+1, report, progress)
    # The next stage starts from the validation-selected checkpoint, not last.
    return load_stage_weights(model, output/'best.pt')


def run_experiment(model, loader, validation_loader, env, args, device, output):
    progress = LearningProgress()
    curve = output/'learning_curve.jsonl'
    initial_args = copy.copy(args)
    initial_args.mode = 'untrained'
    save_checkpoint(model, loader.dataset, initial_args, output/'untrained.pt', 0, {}, progress)
    with progress.measure(env, 'evaluation'):
        before = evaluate(model,validation_loader,env,args,device,
                          ['untrained']+args.reference_modes,output/'before.jsonl')
    append_json(curve, dict(stage='untrained',validation=before,progress=progress.snapshot()))
    write_json(output/'before_summary.json', before)
    for stage, epochs in (('metric',args.metric_epochs),('distill',args.distill_epochs)):
        stage_args = copy.copy(args)
        stage_args.mode, stage_args.epochs, stage_args.output = stage, epochs, str(output/stage)
        train_stage(model,loader,validation_loader,env,stage_args,device,output/stage,progress)
    with progress.measure(env, 'evaluation'):
        distilled = evaluate(model,validation_loader,env,args,device,['distilled'],output/'distilled.jsonl')
    append_json(curve,dict(stage='distilled',validation=distilled,progress=progress.snapshot()))
    write_json(output/'distilled_summary.json', distilled)
    # RL can use a smaller, predeclared prefix of TRAIN; validation never trains.
    rl_dataset = TilebasedDatasetV85(args.data, 'train',
        min(len(loader.dataset), args.rl_limit) if args.rl_limit else len(loader.dataset))
    rl_loader = DataLoader(rl_dataset,batch_size=args.batch,shuffle=True,num_workers=args.workers)
    rl_args = copy.copy(args)
    rl_args.mode, rl_args.epochs, rl_args.limit, rl_args.output = 'rl', args.rl_epochs, len(rl_dataset), str(output/'rl')
    best = train_stage(model,rl_loader,validation_loader,env,rl_args,device,output/'rl',progress,
                       initial_validation=distilled['distilled'],curve_path=curve)
    with progress.measure(env, 'evaluation'):
        after = evaluate(model,validation_loader,env,args,device,['learned'],output/'after.jsonl')
    append_json(curve,dict(stage='selected_best',validation=after,progress=progress.snapshot()))
    old_rows = read_outcomes(output/'before.jsonl','untrained')
    new_rows = read_outcomes(output/'after.jsonl','learned')
    comparisons = {'untrained_to_learned':paired_learning_report(old_rows,new_rows),
                   'distilled_to_learned':paired_learning_report(
                       read_outcomes(output/'distilled.jsonl','distilled'),new_rows)}
    for mode in args.reference_modes:
        comparisons[mode+'_to_learned'] = paired_learning_report(
            read_outcomes(output/'before.jsonl',mode),new_rows,require_same_budget=mode!='full')
    report = dict(evaluation_split='val', before=before, distilled=distilled, after=after,
        comparisons=comparisons, budget=args.budget,
        visual_token_reduction=1.-after['learned']['token_ratio'],
        selected_checkpoint='rl/best.pt', selected_stage=best['stage'], selected_epoch=best['epoch'],
        selected_checkpoint_progress=best['progress'], progress=progress.snapshot(),
        train_tasks=len(loader.dataset), rl_train_tasks=len(rl_dataset), validation_tasks=len(validation_loader.dataset),
        selection_rule='Minimum penalized validation task loss among distilled initialization and RL epoch endpoints; ties keep earlier weights.',
        note='Validation-only feasibility run, not a held-out test result. '
             'Wall time excludes model loading. Formal solver calls exclude training proxy solves; '
             'GT/uniform diagnostic solves are counted separately. Token ratio is not end-to-end speedup.')
    write_json(output/'summary.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
