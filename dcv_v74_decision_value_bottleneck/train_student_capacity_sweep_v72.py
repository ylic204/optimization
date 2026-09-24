import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import CFG
from dataset import DCVDataset
from models import DualResolutionPerception
from decision import (
    move_batch,
    causal_state_features,
    legal_mask,
    teacher_gradient_direction,
    semantic_hard_decision,
)
from student_v72 import GradientStudentV72, gradient_distillation_loss


def load_perception(path, device):
    p = DualResolutionPerception(CFG.feat_dim).to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt.get('perception', ckpt)
    p.load_state_dict(state, strict=False)
    p.eval()
    for q in p.parameters():
        q.requires_grad_(False)
    return p


def privileged_onehot(batch):
    return F.one_hot(batch['patch_state'].long(), num_classes=4).float()


def action_from_score(score, eligible):
    return score.masked_fill(~eligible, -1e9).argmax(-1)


def mean(xs):
    return float(np.mean(xs)) if xs else float('nan')


def parse_archs(spec):
    out = []
    # format: 128x2x4,256x4x8,512x4x8
    for item in spec.split(','):
        h, l, a = item.strip().lower().split('x')
        out.append((int(h), int(l), int(a)))
    return out


def evaluate_teacher_states(model, perception, loader, budget, device):
    model.eval()
    cosines, top1 = [], []
    per_step = None

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            B = batch['image'].shape[0]
            T = batch['traj_grad'].shape[1]
            if per_step is None:
                per_step = [[] for _ in range(T)]

            hf = perception.encode_all_high(batch['image'])
            pf = perception.encode_preview(batch['image'])
            priv = privileged_onehot(batch)
            bf = torch.full((B,), float(budget), device=device)

            for t in range(T):
                w = batch['traj_w'][:, t]
                target = batch['traj_grad'][:, t]
                eligible = legal_mask(batch, w)
                sf = causal_state_features(pf, hf, w)
                outer = torch.full((B,), t / max(T - 1, 1), device=device)
                pred, _ = model(sf, batch['patch_graph_feat'], w, outer, bf, eligible, priv)

                pp = pred * eligible.float()
                tt = target * eligible.float()
                c = F.cosine_similarity(pp, tt, dim=-1, eps=1e-8)
                pa = action_from_score(pred, eligible)
                ta = action_from_score(target, eligible)

                cosines.extend(c.cpu().tolist())
                top1.extend((pa == ta).float().cpu().tolist())
                per_step[t].extend(c.cpu().tolist())

    return {
        'teacher_state_cosine': mean(cosines),
        'teacher_top1': mean(top1),
        'per_step_cosine': [mean(x) for x in per_step],
    }


def evaluate_onpolicy(model, perception, loader, budget, device):
    model.eval()
    K = CFG.visual_budget_k(budget)
    cosines, top1 = [], []
    regrets, optimal = [], []
    teacher_regrets, teacher_optimal = [], []

    for batch in loader:
        batch = move_batch(batch, device)
        B = batch['image'].shape[0]
        M = CFG.n_patches
        rows = torch.arange(B, device=device)

        with torch.no_grad():
            hf = perception.encode_all_high(batch['image'])
            pf = perception.encode_preview(batch['image'])
        priv = privileged_onehot(batch)
        bf = torch.full((B,), float(budget), device=device)

        ws = torch.zeros(B, M, device=device)
        wt = torch.zeros_like(ws)

        for t in range(K):
            es = legal_mask(batch, ws)
            sf = causal_state_features(pf, hf, ws)
            outer = torch.full((B,), t / max(K - 1, 1), device=device)
            with torch.no_grad():
                pred, _ = model(sf, batch['patch_graph_feat'], ws, outer, bf, es, priv)
            gt = teacher_gradient_direction(ws, batch)

            c = F.cosine_similarity(pred * es.float(), gt * es.float(), dim=-1, eps=1e-8)
            a = action_from_score(pred, es)
            at_here = action_from_score(gt, es)
            cosines.extend(c.detach().cpu().tolist())
            top1.extend((a == at_here).float().cpu().tolist())
            ws = ws.clone(); ws[rows, a] = 1.0

            et = legal_mask(batch, wt)
            gt_ref = teacher_gradient_direction(wt, batch)
            at = action_from_score(gt_ref, et)
            wt = wt.clone(); wt[rows, at] = 1.0

        os = semantic_hard_decision(ws, batch)
        ot = semantic_hard_decision(wt, batch)
        regrets.extend(os['hard_regret'].detach().cpu().tolist())
        optimal.extend(os['optimal'].float().detach().cpu().tolist())
        teacher_regrets.extend(ot['hard_regret'].detach().cpu().tolist())
        teacher_optimal.extend(ot['optimal'].float().detach().cpu().tolist())

    return {
        'onpolicy_cosine': mean(cosines),
        'onpolicy_top1': mean(top1),
        'hard_regret': mean(regrets),
        'optimal_path_rate': mean(optimal),
        'teacher_hard_regret': mean(teacher_regrets),
        'teacher_optimal_path_rate': mean(teacher_optimal),
    }


def train_one_arch(args, arch, head_mode, dataset, perception, device, run_dir):
    hidden, layers, heads = arch
    subset = Subset(dataset, list(range(min(args.num_samples, len(dataset)))))
    train_loader = DataLoader(subset, batch_size=min(args.batch, len(subset)), shuffle=True, num_workers=0)
    eval_loader = DataLoader(subset, batch_size=min(args.batch, len(subset)), shuffle=False, num_workers=0)

    model = GradientStudentV72(
        feat_dim=CFG.feat_dim,
        graph_dim=4,
        hidden=hidden,
        layers=layers,
        heads=heads,
        privileged_dim=4,
        head_mode=head_mode,
        head_temperature=args.head_temperature,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    best_cos = -1.0
    best_state = None

    for ep in range(args.epochs):
        model.train()
        epoch_cos, epoch_loss = [], []

        for batch in train_loader:
            batch = move_batch(batch, device)
            B = batch['image'].shape[0]
            T = batch['traj_grad'].shape[1]

            with torch.no_grad():
                hf = perception.encode_all_high(batch['image'])
                pf = perception.encode_preview(batch['image'])
            priv = privileged_onehot(batch)
            bf = torch.full((B,), float(args.budget), device=device)

            total = model.head.weight.sum() * 0.0
            batch_cos = []

            for t in range(T):
                w = batch['traj_w'][:, t]
                target = batch['traj_grad'][:, t]
                eligible = legal_mask(batch, w)
                sf = causal_state_features(pf, hf, w)
                outer = torch.full((B,), t / max(T - 1, 1), device=device)
                pred, logits = model(sf, batch['patch_graph_feat'], w, outer, bf, eligible, priv)

                ld = gradient_distillation_loss(
                    pred, logits, target, eligible,
                    lambda_cos=args.lambda_cos,
                    lambda_kl=args.lambda_kl,
                    lambda_rank=args.lambda_rank,
                    rank_margin=args.rank_margin,
                )
                total = total + ld['loss']
                batch_cos.append(ld['cosine'])

            total = total / T
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

            epoch_loss.append(float(total.detach().cpu()))
            epoch_cos.append(float(torch.stack(batch_cos).mean().detach().cpu()))

        train_cos = mean(epoch_cos)
        if train_cos > best_cos:
            best_cos = train_cos
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

        if ep < 5 or (ep + 1) % args.print_every == 0:
            print(
                f'[{hidden}x{layers}x{heads} {head_mode}] epoch={ep:03d} '
                f'loss={mean(epoch_loss):.5f} train_cos={train_cos:.4f}'
            )

        if train_cos >= args.stop_cos and ep >= 20:
            break

    model.load_state_dict(best_state)
    teacher_state = evaluate_teacher_states(model, perception, eval_loader, args.budget, device)
    onpolicy = evaluate_onpolicy(model, perception, eval_loader, args.budget, device)

    result = {
        'hidden': hidden,
        'layers': layers,
        'heads': heads,
        'head_mode': head_mode,
        'best_train_cosine': best_cos,
        **teacher_state,
        **onpolicy,
    }

    ckpt = run_dir / f'student_h{hidden}_l{layers}_a{heads}_{head_mode}.pt'
    torch.save({'student': model.state_dict(), 'config': result}, ckpt)
    result['checkpoint'] = str(ckpt)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--perception', required=True)
    ap.add_argument('--outdir', default='capacity_sweep_v72')
    ap.add_argument('--num-samples', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--budget', type=float, default=0.15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--archs', default='128x2x4,256x4x8,512x4x8,512x6x8')
    ap.add_argument('--heads', default='relu_l2,softmax')
    ap.add_argument('--head-temperature', type=float, default=1.0)
    ap.add_argument('--lambda-cos', type=float, default=CFG.gd_lambda_cos)
    ap.add_argument('--lambda-kl', type=float, default=CFG.gd_lambda_kl)
    ap.add_argument('--lambda-rank', type=float, default=CFG.gd_lambda_rank)
    ap.add_argument('--rank-margin', type=float, default=CFG.gd_rank_margin)
    ap.add_argument('--stop-cos', type=float, default=0.98)
    ap.add_argument('--print-every', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = DCVDataset(args.data, args.teacher)
    if len(ds) == 0:
        raise RuntimeError(f'empty dataset: data={args.data}, teacher={args.teacher}')

    perception = load_perception(args.perception, device)
    run_dir = Path(args.outdir); run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for arch in parse_archs(args.archs):
        for head_mode in [x.strip() for x in args.heads.split(',') if x.strip()]:
            print('\n' + '=' * 80)
            print('RUN', arch, head_mode)
            print('=' * 80)
            results.append(train_one_arch(args, arch, head_mode, ds, perception, device, run_dir))

    results = sorted(results, key=lambda x: x['teacher_state_cosine'], reverse=True)
    (run_dir / 'capacity_sweep_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    print('\n=== V7.2 CAPACITY SWEEP SUMMARY ===')
    print('arch/head'.ljust(28), 'same_cos  top1   onpol_cos  opt_rate  regret')
    for r in results:
        name = f"{r['hidden']}x{r['layers']}x{r['heads']}/{r['head_mode']}"
        print(
            name.ljust(28),
            f"{r['teacher_state_cosine']:.4f}  "
            f"{100*r['teacher_top1']:.1f}%  "
            f"{r['onpolicy_cosine']:.4f}    "
            f"{100*r['optimal_path_rate']:.1f}%   "
            f"{r['hard_regret']:.4f}"
        )

    best = results[0]
    print('\nBest by same-subset Teacher-state cosine:')
    print(json.dumps(best, indent=2))
    if best['teacher_state_cosine'] >= 0.95:
        print('CAPACITY PASS: direct gradient distillation is representable on the small fixed subset.')
    elif best['teacher_state_cosine'] >= 0.85:
        print('PARTIAL: capacity helps, but representation/target structure still limits fitting.')
    else:
        print('CAPACITY FAIL: scaling alone does not solve the Teacher-to-Student mapping.')


if __name__ == '__main__':
    main()
