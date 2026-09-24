import argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CFG
from dataset import DCVDataset
from decision import move_batch, legal_mask, teacher_gradient_direction, semantic_exact_regret


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v7/train'); ap.add_argument('--out',default='teacher_v7/train_15'); ap.add_argument('--budget',type=float,default=CFG.budget_frac); ap.add_argument('--batch',type=int,default=CFG.batch_size); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); loader=DataLoader(DCVDataset(args.data),batch_size=args.batch,shuffle=False); od=Path(args.out); od.mkdir(parents=True,exist_ok=True)
    K=CFG.visual_budget_k(args.budget); initR=[]; finalR=[]; zero_dirs=0; total=0
    for batch in tqdm(loader,desc='build decision-gradient teacher'):
        names=batch['name']; batch=move_batch(batch,device); B=batch['image'].shape[0]; M=CFG.n_patches; w=torch.zeros(B,M,device=device)
        traj_w=[w.clone()]; traj_g=[]; traj_action=[]; traj_reg=[semantic_exact_regret(w,batch)]
        for t in range(K):
            g=teacher_gradient_direction(w,batch); elig=legal_mask(batch,w); zero_dirs+=int((g.sum(-1)<=CFG.grad_eps).sum().item()); total+=B
            action=g.masked_fill(~elig,-1e9).argmax(-1); rows=torch.arange(B,device=device); traj_g.append(g); traj_action.append(action); w=w.clone(); w[rows,action]=1.0; traj_w.append(w.clone()); traj_reg.append(semantic_exact_regret(w,batch))
        initR.extend(traj_reg[0].detach().cpu().tolist()); finalR.extend(traj_reg[-1].detach().cpu().tolist())
        A=dict(traj_w=torch.stack(traj_w,1).cpu().numpy().astype(np.float32), traj_grad=torch.stack(traj_g,1).cpu().numpy().astype(np.float32), traj_action=torch.stack(traj_action,1).cpu().numpy().astype(np.int64), traj_regret=torch.stack(traj_reg,1).cpu().numpy().astype(np.float32))
        for i,n in enumerate(names): np.savez_compressed(od/f'{n}.npz',**{k:v[i] for k,v in A.items()})
    print('\n=== V7 gradient teacher ==='); print(f'K={K} initial_regret={np.mean(initR):.4f} final_regret={np.mean(finalR):.4f} zero_direction_fraction={zero_dirs/max(total,1):.4f}')

if __name__=='__main__': main()
