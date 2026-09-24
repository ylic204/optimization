import argparse, random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CFG
from dataset import DCVDataset
from models import DualResolutionPerception
from decision import gather_edge_features


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v7/train'); ap.add_argument('--val',default='data_v7/val'); ap.add_argument('--out',default='checkpoints/perception_v7.pt'); ap.add_argument('--epochs',type=int,default=CFG.epochs_perception); ap.add_argument('--batch',type=int,default=CFG.batch_size); args=ap.parse_args()
    random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr=DataLoader(DCVDataset(args.data),batch_size=args.batch,shuffle=True); va=DataLoader(DCVDataset(args.val),batch_size=args.batch)
    m=DualResolutionPerception(CFG.feat_dim).to(device)
    for p in list(m.preview_encoder.parameters())+list(m.preview_cost_head.parameters()): p.requires_grad_(False)
    params=list(m.high_encoder.parameters())+list(m.high_head.parameters()); opt=torch.optim.AdamW(params,lr=CFG.lr_perception,weight_decay=CFG.wd)
    cw=torch.tensor([1.,1.4,1.8,4.],device=device); best=1e9; Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    for ep in range(args.epochs):
        m.train(); tl=[]
        for b in tqdm(tr,desc=f'high perception {ep}'):
            img=b['image'].to(device); edge_patch=b['edge_patch'].to(device); y=b['edge_state'].to(device)
            hf=m.encode_all_high(img); logits=m.classify_high_features(gather_edge_features(hf,edge_patch)); loss=F.cross_entropy(logits.reshape(-1,4),y.reshape(-1),weight=cw)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,2.0); opt.step(); tl.append(loss.item())
        m.eval(); vl=[]; acc=[]
        with torch.no_grad():
            for b in va:
                img=b['image'].to(device); epat=b['edge_patch'].to(device); y=b['edge_state'].to(device); hf=m.encode_all_high(img); logits=m.classify_high_features(gather_edge_features(hf,epat)); loss=F.cross_entropy(logits.reshape(-1,4),y.reshape(-1),weight=cw)
                vl.append(loss.item()); acc.append((logits.argmax(-1)==y).float().mean().item())
        v=float(np.mean(vl)); a=float(np.mean(acc)); print(f'epoch={ep} train_high_ce={np.mean(tl):.4f} val_high_ce={v:.4f} high_4way_acc={a:.4f}')
        if v<best: best=v; torch.save({'model':m.state_dict(),'val_loss':best,'version':'v7_high_only'},args.out)

if __name__=='__main__': main()
