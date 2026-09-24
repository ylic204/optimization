import argparse
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CFG
from dataset import DCVDataset
from models import DualResolutionPerception, GradientFlowNet, FixedStepNeurodynamic
from decision import move_batch, causal_state_features, legal_mask, soft_decision_regret_from_state_feat, masked_cosine


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v7/train'); ap.add_argument('--val',default='data_v7/val'); ap.add_argument('--teacher',default='teacher_v7/train_15'); ap.add_argument('--teacher-val',default='teacher_v7/val_15'); ap.add_argument('--perception',default='checkpoints/perception_v7.pt'); ap.add_argument('--out',default='checkpoints/v7_gdfm_15.pt'); ap.add_argument('--budget',type=float,default=CFG.budget_frac); ap.add_argument('--epochs',type=int,default=CFG.epochs_flow); ap.add_argument('--batch',type=int,default=CFG.batch_size); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tr=DataLoader(DCVDataset(args.data,args.teacher),batch_size=args.batch,shuffle=True); va=DataLoader(DCVDataset(args.val,args.teacher_val),batch_size=args.batch)
    perc=DualResolutionPerception(CFG.feat_dim).to(device); perc.load_state_dict(torch.load(args.perception,map_location=device)['model'],strict=False)
    for p in list(perc.high_encoder.parameters())+list(perc.high_head.parameters()): p.requires_grad_(False)
    for p in list(perc.preview_encoder.parameters())+list(perc.preview_cost_head.parameters()): p.requires_grad_(True)
    flow=GradientFlowNet(CFG.feat_dim,4,CFG.hidden_dim).to(device); nd=FixedStepNeurodynamic(CFG.nd_steps).to(device)
    params=list(flow.parameters())+list(nd.parameters())+list(perc.preview_encoder.parameters())+list(perc.preview_cost_head.parameters())
    opt=torch.optim.AdamW([{'params':list(flow.parameters())+list(nd.parameters()),'lr':CFG.lr_flow},{'params':list(perc.preview_encoder.parameters())+list(perc.preview_cost_head.parameters()),'lr':CFG.lr_preview}],weight_decay=CFG.wd)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); best=1e9

    def epoch(loader,train):
        flow.train(train); nd.train(train); perc.preview_encoder.train(train); perc.preview_cost_head.train(train); perc.high_encoder.eval(); perc.high_head.eval(); sums=dict(total=0.,fm=0.,gd=0.,task=0.,cos=0.); nb=0
        for batch in tqdm(loader,desc='GDFM train' if train else 'GDFM val'):
            batch=move_batch(batch,device); B=batch['image'].shape[0]; T=batch['traj_grad'].shape[1]; rows=torch.arange(B,device=device); t_idx=torch.randint(0,T,(B,),device=device)
            with torch.no_grad(): hf=perc.encode_all_high(batch['image'])
            pf=perc.encode_preview(batch['image']) if train else perc.encode_preview(batch['image']).detach()
            w=batch['traj_w'][rows,t_idx]; target=batch['traj_grad'][rows,t_idx]; prev=torch.zeros_like(target); hasprev=t_idx>0
            if hasprev.any(): prev[hasprev]=batch['traj_grad'][rows[hasprev],t_idx[hasprev]-1]
            sf=causal_state_features(pf,hf,w); elig=legal_mask(batch,w); outer=t_idx.float()/max(T-1,1); budget=torch.full((B,),float(args.budget),device=device)
            tau=torch.rand(B,device=device); z_tau=(1-tau[:,None])*prev+tau[:,None]*target; target_v=target-prev
            v=flow(sf,batch['patch_graph_feat'],z_tau,w,tau,outer,budget)
            denom=elig.float().sum().clamp_min(1.0); loss_fm=(((v-target_v)**2)*elig.float()).sum()/denom
            zK=nd(flow,sf,batch['patch_graph_feat'],prev,w,outer,budget); cos=masked_cosine(zK,target,elig); mag=((((zK-target)**2)*elig.float()).sum()/denom)
            loss_gd=(1-cos).mean()+CFG.gd_magnitude_weight*mag
            # Preview learns only from downstream task loss on hard causal states, never labels.
            loss_soft_regret=soft_decision_regret_from_state_feat(perc,sf,w,batch).mean()
            loss=CFG.lambda_fm*loss_fm+CFG.lambda_gd*loss_gd+CFG.lambda_task*loss_soft_regret
            if train:
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step()
            for k,x in [('total',loss),('fm',loss_fm),('gd',loss_gd),('task',loss_soft_regret)]: sums[k]+=float(x.detach().cpu())
            sums['cos']+=float(cos.mean().detach().cpu()); nb+=1
        return {k:v/max(nb,1) for k,v in sums.items()}

    for ep in range(args.epochs):
        a=epoch(tr,True); b=epoch(va,False); al,be=nd.coefficients(); print(f"epoch={ep} train={a['total']:.4f} val={b['total']:.4f} | FM={b['fm']:.4f} GD={b['gd']:.4f} task={b['task']:.4f} grad_cos={b['cos']:.3f} alpha={al.detach().cpu().numpy().round(3)} beta={be.detach().cpu().numpy().round(3)}")
        if b['total']<best:
            best=b['total']; torch.save({'flow':flow.state_dict(),'nd':nd.state_dict(),'perception':perc.state_dict(),'val_loss':best,'version':'v7_gradient_distilled_flow_nd'},args.out)
    print('saved',args.out,'best_val',best)

if __name__=='__main__': main()
