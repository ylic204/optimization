import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import DCVDataset
from models import DualResolutionPerception, GradientFlowNet, FixedStepNeurodynamic
from decision import (
    move_batch,
    causal_state_features,
    hard_decision_from_state_feat,
    exact_counterfactual_gains,
    legal_mask,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_v7/test')
    ap.add_argument('--ckpt', default='checkpoints/v7_outcome_rl_15.pt')
    ap.add_argument('--budget', type=float, default=CFG.budget_frac)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(DCVDataset(args.data), batch_size=args.batch)

    c = torch.load(args.ckpt, map_location=device)
    perc = DualResolutionPerception(CFG.feat_dim).to(device)
    perc.load_state_dict(c['perception']); perc.eval()
    flow = GradientFlowNet(CFG.feat_dim, 4, CFG.hidden_dim).to(device)
    flow.load_state_dict(c['flow']); flow.eval()
    nd = FixedStepNeurodynamic(CFG.nd_steps).to(device)
    nd.load_state_dict(c['nd']); nd.eval()

    regrets, optimal_rates, selected_gain_ratios = [], [], []
    chosen_costs, optimal_costs = [], []
    K = CFG.visual_budget_k(args.budget)

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            B = batch['image'].shape[0]
            hf = perc.encode_all_high(batch['image'])
            pf = perc.encode_preview(batch['image'])
            w = torch.zeros(B, CFG.n_patches, device=device)
            z = torch.zeros_like(w)
            budget = torch.full((B,), float(args.budget), device=device)
            rows = torch.arange(B, device=device)

            for t in range(K):
                sf = causal_state_features(pf, hf, w)
                outer = torch.full((B,), t / max(K - 1, 1), device=device)
                z = nd(flow, sf, batch['patch_graph_feat'], z, w, outer, budget)
                elig = legal_mask(batch, w)
                a = z.masked_fill(~elig, -1e9).argmax(-1)

                gains, _ = exact_counterfactual_gains(perc, pf, hf, w, batch)
                best_gain = gains.max(-1).values.clamp_min(0.0)
                chosen_gain = gains.gather(1, a[:, None]).squeeze(1).clamp_min(0.0)
                ratio = torch.where(best_gain > 1e-8, chosen_gain / best_gain, torch.ones_like(best_gain))
                selected_gain_ratios.extend(ratio.cpu().tolist())

                w = w.clone(); w[rows, a] = 1.0

            sf = causal_state_features(pf, hf, w)
            out = hard_decision_from_state_feat(perc, sf, w, batch)
            regrets.extend(out['hard_regret'].cpu().tolist())
            optimal_rates.extend(out['optimal'].float().cpu().tolist())
            chosen_costs.extend(out['chosen_true_cost'].cpu().tolist())
            optimal_costs.extend(out['optimal_true_cost'].cpu().tolist())

    saving = 1.0 - K / CFG.n_patches
    print('\n=== V7.2 DECISION RESULTS ===')
    print(f'Hard Decision Regret (primary)   = {np.mean(regrets):.6f}')
    print(f'Median Hard Decision Regret      = {np.median(regrets):.6f}')
    print(f'Optimal-Path Rate (cost-based)   = {100*np.mean(optimal_rates):.2f}%')
    print(f'Mean chosen true task cost       = {np.mean(chosen_costs):.4f}')
    print(f'Mean full-info optimal task cost = {np.mean(optimal_costs):.4f}')
    print(f'One-step GainRatio (diagnostic)  = {np.mean(selected_gain_ratios):.4f}')
    print(f'Selected patches                 = {K} / {CFG.n_patches}')
    print(f'High-res tokens                  = {K*CFG.high_tokens_per_crop} / {CFG.full_high_tokens}')
    print(f'Visual Saving                    = {100*saving:.2f}%')
    print('\nDefinition: different path IDs count as optimal whenever their TRUE task cost equals C* within tolerance.')


if __name__ == '__main__':
    main()
