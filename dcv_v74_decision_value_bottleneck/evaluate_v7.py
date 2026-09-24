import argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from config import CFG
from dataset import DCVDataset
from models import DualResolutionPerception, GradientFlowNet, FixedStepNeurodynamic
from decision import move_batch, causal_state_features, exact_outcome_from_state_feat, exact_counterfactual_gains, legal_mask


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v7/test'); ap.add_argument('--ckpt',default='checkpoints/v7_outcome_rl_15.pt'); ap.add_argument('--budget',type=float,default=CFG.budget_frac); ap.add_argument('--batch',type=int,default=64); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); loader=DataLoader(DCVDataset(args.data),batch_size=args.batch)
    c=torch.load(args.ckpt,map_location=device); perc=DualResolutionPerception(CFG.feat_dim).to(device); perc.load_state_dict(c['perception']); perc.eval(); flow=GradientFlowNet(CFG.feat_dim,4,CFG.hidden_dim).to(device); flow.load_state_dict(c['flow']); flow.eval(); nd=FixedStepNeurodynamic(CFG.nd_steps).to(device); nd.load_state_dict(c['nd']); nd.eval()
    path_acc=[]; regrets=[]; sel_hits=[]; N=0; K=CFG.visual_budget_k(args.budget)
    with torch.no_grad():
        for b in loader:
            b=move_batch(b,device); B=b['image'].shape[0]; N+=B
            # all-high cache below is EVALUATION ONLY, used to define exact importance oracle.
            hf=perc.encode_all_high(b['image']); pf=perc.encode_preview(b['image']); w=torch.zeros(B,CFG.n_patches,device=device); z=torch.zeros_like(w); budget=torch.full((B,),float(args.budget),device=device)
            for t in range(K):
                sf=causal_state_features(pf,hf,w); outer=torch.full((B,),t/max(K-1,1),device=device); z=nd(flow,sf,b['patch_graph_feat'],z,w,outer,budget); elig=legal_mask(b,w); a=z.masked_fill(~elig,-1e9).argmax(-1)
                gains,_=exact_counterfactual_gains(perc,pf,hf,w,b); best=gains.max(-1).values; chosen=gains.gather(1,a[:,None]).squeeze(1); sel_hits.extend((chosen>=best-CFG.selection_tie_tol).float().cpu().tolist())
                rows=torch.arange(B,device=device); w=w.clone(); w[rows,a]=1.0
            sf=causal_state_features(pf,hf,w); r,p=exact_outcome_from_state_feat(perc,sf,w,b); regrets.extend(r.cpu().tolist()); path_acc.extend((r<=CFG.path_tol).float().cpu().tolist())
    saving=1-K/CFG.n_patches; print('\n=== V7 FINAL RESULTS ==='); print(f'Important-region SelectionAcc@1 = {100*np.mean(sel_hits):.2f}%'); print(f'Shortest-path PathAcc            = {100*np.mean(path_acc):.2f}%'); print(f'Path Regret                      = {np.mean(regrets):.4f}'); print(f'Selected patches                 = {K} / {CFG.n_patches}'); print(f'High-res tokens                  = {K*CFG.high_tokens_per_crop} / {CFG.full_high_tokens}'); print(f'Visual Saving                    = {100*saving:.2f}%')

if __name__=='__main__': main()
