import argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CFG
from dataset import DCVDataset
from models import DualResolutionPerception, GradientFlowNet, FixedStepNeurodynamic
from decision import move_batch
from policy import rollout_cached


def eval_greedy(flow,nd,perc,loader,budget,device):
    regs=[]; acc=[]
    with torch.no_grad():
        for b in loader:
            b=move_batch(b,device); hf=perc.encode_all_high(b['image']); pf=perc.encode_preview(b['image']); o=rollout_cached(flow,nd,perc,b,pf,hf,budget,sample=False)
            regs.extend(o['regret'].cpu().tolist()); acc.extend((o['regret']<=CFG.path_tol).float().cpu().tolist())
    return float(np.mean(regs)),float(np.mean(acc))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v7/train'); ap.add_argument('--val',default='data_v7/val'); ap.add_argument('--stage1',default='checkpoints/v7_gdfm_15.pt'); ap.add_argument('--out',default='checkpoints/v7_outcome_rl_15.pt'); ap.add_argument('--budget',type=float,default=CFG.budget_frac); ap.add_argument('--epochs',type=int,default=CFG.epochs_rl); ap.add_argument('--batch',type=int,default=CFG.batch_size); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tr=DataLoader(DCVDataset(args.data),batch_size=args.batch,shuffle=True); val_loader=DataLoader(DCVDataset(args.val),batch_size=args.batch)
    c=torch.load(args.stage1,map_location=device); perc=DualResolutionPerception(CFG.feat_dim).to(device); perc.load_state_dict(c['perception']); perc.eval(); [p.requires_grad_(False) for p in perc.parameters()]
    flow=GradientFlowNet(CFG.feat_dim,4,CFG.hidden_dim).to(device); flow.load_state_dict(c['flow']); nd=FixedStepNeurodynamic(CFG.nd_steps).to(device); nd.load_state_dict(c['nd'])
    params=list(flow.parameters())+list(nd.parameters()); opt=torch.optim.AdamW(params,lr=CFG.lr_rl,weight_decay=CFG.wd); Path(args.out).parent.mkdir(parents=True,exist_ok=True); best=(-1.,1e9)
    for ep in range(args.epochs):
        flow.train(); nd.train(); losses=[]; rewards=[]; ents=[]; train_acc=[]
        for b in tqdm(tr,desc=f'Outcome-RL {ep}'):
            b=move_batch(b,device)
            with torch.no_grad(): hf=perc.encode_all_high(b['image']); pf=perc.encode_preview(b['image'])
            o=rollout_cached(flow,nd,perc,b,pf,hf,args.budget,sample=True)
            reward=-o['regret'].detach(); adv=(reward-reward.mean())/(reward.std(unbiased=False)+1e-6)
            loss_pg=-(adv*o['sum_logp']).mean(); entropy=o['entropy'].mean(); loss=loss_pg-CFG.entropy_coef*entropy
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step()
            losses.append(loss.item()); rewards.append(reward.mean().item()); ents.append(entropy.item()); train_acc.append((o['regret']<=CFG.path_tol).float().mean().item())
        vr,va=eval_greedy(flow,nd,perc,val_loader,args.budget,device)
        print(f'epoch={ep} outcome_loss={np.mean(losses):.4f} reward={np.mean(rewards):.4f} entropy={np.mean(ents):.3f} train_path_acc={np.mean(train_acc):.3f} | val_path_acc={va:.3f} val_regret={vr:.4f}')
        key=(va,-vr)
        if key>(best[0],-best[1]):
            best=(va,vr); torch.save({'flow':flow.state_dict(),'nd':nd.state_dict(),'perception':perc.state_dict(),'budget':args.budget,'val_path_acc':va,'val_regret':vr,'version':'v7_outcome_visual_rl'},args.out)
    print('saved',args.out,'best_path_acc',best[0],'best_regret',best[1])

if __name__=='__main__': main()
