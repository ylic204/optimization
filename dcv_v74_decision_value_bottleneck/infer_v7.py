import argparse, torch
from config import CFG
from dataset import DCVDataset
from models import DualResolutionPerception, GradientFlowNet, FixedStepNeurodynamic
from decision import move_batch
from lazy_inference_v7 import lazy_rollout


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v7/test'); ap.add_argument('--index',type=int,default=0); ap.add_argument('--ckpt',default='checkpoints/v7_outcome_rl_15.pt'); ap.add_argument('--budget',type=float,default=CFG.budget_frac); args=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample=DCVDataset(args.data)[args.index]; b={k:(v[None].to(device) if torch.is_tensor(v) else [v]) for k,v in sample.items() if k!='name'}; c=torch.load(args.ckpt,map_location=device)
    p=DualResolutionPerception(CFG.feat_dim).to(device); p.load_state_dict(c['perception']); p.eval(); f=GradientFlowNet(CFG.feat_dim,4,CFG.hidden_dim).to(device); f.load_state_dict(c['flow']); f.eval(); nd=FixedStepNeurodynamic(CFG.nd_steps).to(device); nd.load_state_dict(c['nd']); nd.eval(); o=lazy_rollout(f,nd,p,b,args.budget)
    print('sample',sample['name']); print('selected patches',o['actions'][0].cpu().tolist()); print('path index',int(o['path_idx'][0])); print('regret',float(o['regret'][0])); print('saving',f"{100*(1-o['K']/CFG.n_patches):.2f}%")

if __name__=='__main__': main()
