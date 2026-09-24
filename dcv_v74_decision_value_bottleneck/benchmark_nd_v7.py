import argparse, time
import numpy as np, torch
from torch.utils.data import DataLoader
from config import CFG
from dataset import DCVDataset
from models import DualResolutionPerception, GradientFlowNet, FixedStepNeurodynamic
from decision import move_batch, causal_state_features, legal_mask, masked_cosine


def euler(flow,sf,gf,z0,w,outer,budget,steps):
    z=z0
    for k in range(steps):
        it=torch.full((z.shape[0],),(k+0.5)/steps,device=z.device); v=flow(sf,gf,z,w,it,outer,budget); z=z+v/steps
    return z


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v7/val'); ap.add_argument('--teacher',default='teacher_v7/val_15'); ap.add_argument('--ckpt',default='checkpoints/v7_gdfm_15.pt'); ap.add_argument('--budget',type=float,default=CFG.budget_frac); ap.add_argument('--batch',type=int,default=64); args=ap.parse_args()
    torch.manual_seed(CFG.seed+700); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); loader=DataLoader(DCVDataset(args.data,args.teacher),batch_size=args.batch,shuffle=False)
    c=torch.load(args.ckpt,map_location=device); perc=DualResolutionPerception(CFG.feat_dim).to(device); perc.load_state_dict(c['perception']); perc.eval(); flow=GradientFlowNet(CFG.feat_dim,4,CFG.hidden_dim).to(device); flow.load_state_dict(c['flow']); flow.eval(); nd=FixedStepNeurodynamic(CFG.nd_steps).to(device); nd.load_state_dict(c['nd']); nd.eval()
    steps_list=[1,2,4,8,16,32]; cos_nd=[]; cos_e={s:[] for s in steps_list}; t_nd=0.; t_e={s:0. for s in steps_list}; n=0
    with torch.no_grad():
        for b in loader:
            b=move_batch(b,device); B=b['image'].shape[0]; T=b['traj_grad'].shape[1]; hf=perc.encode_all_high(b['image']); pf=perc.encode_preview(b['image']); t_idx=torch.randint(0,T,(B,),device=device); rows=torch.arange(B,device=device); w=b['traj_w'][rows,t_idx]; target=b['traj_grad'][rows,t_idx]; prev=torch.zeros_like(target); hp=t_idx>0
            if hp.any(): prev[hp]=b['traj_grad'][rows[hp],t_idx[hp]-1]
            sf=causal_state_features(pf,hf,w); elig=legal_mask(b,w); outer=t_idx.float()/max(T-1,1); budget=torch.full((B,),float(args.budget),device=device)
            if device.type=='cuda': torch.cuda.synchronize()
            st=time.perf_counter(); z=nd(flow,sf,b['patch_graph_feat'],prev,w,outer,budget); 
            if device.type=='cuda': torch.cuda.synchronize()
            t_nd+=time.perf_counter()-st; cos_nd.extend(masked_cosine(z,target,elig).cpu().tolist())
            for s in steps_list:
                if device.type=='cuda': torch.cuda.synchronize()
                st=time.perf_counter(); ze=euler(flow,sf,b['patch_graph_feat'],prev,w,outer,budget,s)
                if device.type=='cuda': torch.cuda.synchronize()
                t_e[s]+=time.perf_counter()-st; cos_e[s].extend(masked_cosine(ze,target,elig).cpu().tolist())
            n+=B
    cnd=float(np.mean(cos_nd)); matched=None
    for s in steps_list:
        if np.mean(cos_e[s])>=cnd-0.002: matched=s; break
    al,be=nd.coefficients(); print('\n=== V7 ND FIXED-STEP ACCELERATION ==='); print(f'Learned Flow-ND: K={CFG.nd_steps} gradient_cos={cnd:.4f} latency={1000*t_nd/n:.4f} ms/sample');
    for s in steps_list: print(f'Plain Flow-Euler: steps={s:2d} gradient_cos={np.mean(cos_e[s]):.4f} latency={1000*t_e[s]/n:.4f} ms/sample')
    same_k=float(np.mean(cos_e.get(CFG.nd_steps, cos_e[steps_list[0]])))
    print(f'Same-K gain: {cnd-same_k:+.4f} cosine at K={CFG.nd_steps}')
    if cnd < 0.70:
        print('ND acceleration: NOT ESTABLISHED (gradient direction has not reached the minimum accuracy gate 0.70).')
    elif matched is None:
        print(f'ND step speedup: > {steps_list[-1]/CFG.nd_steps:.2f}x (Euler did not match learned-ND accuracy within {steps_list[-1]} steps)')
    else:
        print(f'ND step speedup: {matched/CFG.nd_steps:.2f}x to matched gradient accuracy')
    print('learned alpha=',np.round(al.detach().cpu().numpy(),4),' beta=',np.round(be.detach().cpu().numpy(),4))

if __name__=='__main__': main()
