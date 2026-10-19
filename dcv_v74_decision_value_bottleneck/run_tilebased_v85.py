"""Tilebased stages sharing the V8.5 frozen Qwen, selector, mask FM/ND and RL.

metric/distill use an explicitly labeled differentiable path-cost surrogate.
rl/evaluate ALWAYS generate JSON with frozen Qwen and call the exact solver.
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
from tilebased_solver_environment_v85 import (FrozenTilebasedSolverEnvironment,
    ENVIRONMENT_CONTRACT, summarize_outcomes)


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
                  'advantage_abs': advantage.abs().mean()}


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
    if ckpt.get('environment') != ENVIRONMENT_CONTRACT:
        raise ValueError('checkpoint solver environment mismatch')
    for name in ('hidden','latent_dim','mask_steps','prune_layer','max_text_length','mask_step_size'):
        if ckpt['args'][name] != getattr(args, name):
            raise ValueError(f'checkpoint architecture mismatch: {name}; use the saved arguments')
    required = {'distill': ('metric','distill','rl'), 'rl': ('distill','rl')}
    if args.mode in required and ckpt['stage'] not in required[args.mode]:
        raise ValueError(f'{args.mode} requires checkpoint stage {required[args.mode]}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['metric','distill','rl','evaluate'], required=True)
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
    if min(a.limit,a.val_limit,a.improve_margin) < 0:
        p.error('limits and improvement margin must be nonnegative')
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
    prompt_config = {variant:prompt_metadata(variant,dataset.config['dataset']) for variant in variants}
    args.modeling_prompt_metadata = prompt_config
    args.dataset_signature = dataset.signature
    args.torch_version = str(torch.__version__)
    args.checkpoint_modeling_prompt = None
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    if checkpoint:
        validate_checkpoint(checkpoint, dataset, args)
        args.checkpoint_modeling_prompt = checkpoint['args'].get('modeling_prompt','legacy')
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
        args.modeling_max_new_tokens,args.failure_cost,prompt_variant=variant) for variant in variants}
    env = environments if args.compare_prompts else environments[args.modeling_prompt]
    if args.mode == 'evaluate':
        report = evaluate(model,loader,env,args,device,args.modes,output/'outcomes.jsonl')
        (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2)); return
    validation_data = TilebasedDatasetV85(args.data,'val',args.val_limit)
    validation_loader = DataLoader(validation_data,batch_size=args.batch,num_workers=args.workers)
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
    for epoch in range(args.epochs):
        model.eval()
        (model.metric_head if args.mode=='metric' else model.selector).train()
        if args.mode!='metric': model.mask_flow.train()
        logs = []
        for raw in tqdm(loader,desc=f'Tilebased {args.mode} {epoch+1}'):
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,args.grad_clip)
            optimizer.step()
            logs.append({k:float(v.detach()) for k,v in dict(metrics,loss=loss).items()})
        report = (evaluate(model,validation_loader,env,args,device,['learned'],output/f'val_{epoch+1}.jsonl')
                  if args.mode=='rl' else validate_proxy(model,validation_loader,args,device))
        score = report['learned']['task_loss'] if args.mode=='rl' else report['proxy_validation_loss']
        entry = {'epoch':epoch+1,'train':{k:float(np.mean([x[k] for x in logs])) for k in logs[0]},'validation':report}
        with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
        print(json.dumps(entry),flush=True)
        if score < best:
            best = score
            saved = {name:getattr(model,name).state_dict() for name in ('selector','mask_flow','metric_head')}
            saved.update(contract=CONTRACT,environment=ENVIRONMENT_CONTRACT,dataset_signature=dataset.signature,
                         stage=args.mode,args=vars(args),epoch=epoch+1,validation=report)
            torch.save(saved,output/'best.pt')

if __name__ == '__main__':
    main()
