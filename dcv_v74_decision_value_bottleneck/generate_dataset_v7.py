import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from config import CFG
from graph_utils import fixed_layered_graph, edge_graph_features, costs_from_states, choose_path
from render_utils_v7 import render_scene


def sample_states(rng,cfg):
    states=np.zeros(cfg.n_edges,dtype=np.int64)
    abnormal=rng.random(cfg.n_edges)<cfg.edge_abnormal_prob
    probs=np.asarray([cfg.rough_given_abnormal,cfg.hazard_given_abnormal,cfg.blocked_given_abnormal],dtype=np.float64)
    probs/=probs.sum()
    sev=rng.choice(np.asarray([1,2,3],dtype=np.int64),size=cfg.n_edges,p=probs)
    states[abnormal]=sev[abnormal]
    return states


def coarse_oracle_regret(base_cost,states,path_mask,true_edge_cost,opt_cost,cfg):
    coarse=base_cost+(states!=0).astype(np.float32)*cfg.preview_abnormal_penalty
    idx,_,_=choose_path(path_mask,coarse)
    true_pc=path_mask@true_edge_cost
    regret=max((float(true_pc[idx])-float(opt_cost))/max(float(opt_cost),1e-6),0.0)
    return idx,regret


def build_one(rng,cfg):
    edges,base_cost,path_mask=fixed_layered_graph(rng)
    require_critical=rng.random()<cfg.critical_scene_fraction
    last=None
    for _ in range(cfg.max_generation_attempts):
        states=sample_states(rng,cfg)
        true_edge_cost,true_trav=costs_from_states(base_cost,states,cfg)
        opt_idx,opt_cost,_=choose_path(path_mask,true_edge_cost)
        coarse_idx,coarse_reg=coarse_oracle_regret(base_cost,states,path_mask,true_edge_cost,opt_cost,cfg)
        last=(states,true_edge_cost,true_trav,opt_idx,opt_cost,coarse_idx,coarse_reg)
        if (not require_critical) or coarse_reg>=cfg.critical_min_coarse_regret:
            break
    states,true_edge_cost,true_trav,opt_idx,opt_cost,coarse_idx,coarse_reg=last

    # Generalized placement: graph edges occupy random patches, not hard-coded first 33.
    edge_patch=rng.choice(cfg.n_patches,size=cfg.n_edges,replace=False).astype(np.int64)
    patch_state=np.zeros(cfg.n_patches,dtype=np.int64)
    patch_state[edge_patch]=states
    distractors=np.setdiff1d(np.arange(cfg.n_patches),edge_patch)
    if len(distractors):
        active=rng.random(len(distractors))<0.35
        probs=np.asarray([cfg.rough_given_abnormal,cfg.hazard_given_abnormal,cfg.blocked_given_abnormal],dtype=np.float64); probs/=probs.sum()
        ds=rng.choice(np.asarray([1,2,3]),size=len(distractors),p=probs)
        patch_state[distractors[active]]=ds[active]

    image=render_scene(patch_state,cfg,rng)
    ef=edge_graph_features(base_cost,path_mask)
    patch_graph_feat=np.zeros((cfg.n_patches,4),dtype=np.float32)
    for e,j in enumerate(edge_patch):
        patch_graph_feat[j,0]=1.0
        patch_graph_feat[j,1:]=ef[e]

    return dict(
        image=image.astype(np.uint8), edges=edges.astype(np.int64), base_cost=base_cost.astype(np.float32),
        edge_patch=edge_patch, edge_state=states.astype(np.int64), patch_state=patch_state,
        path_mask=path_mask.astype(np.float32), true_edge_cost=true_edge_cost.astype(np.float32),
        true_traversable=true_trav.astype(np.float32), optimal_path_idx=np.int64(opt_idx), optimal_cost=np.float32(opt_cost),
        optimal_edge_mask=path_mask[opt_idx].astype(np.float32), patch_graph_feat=patch_graph_feat,
        coarse_oracle_path_idx=np.int64(coarse_idx), coarse_oracle_regret=np.float32(coarse_reg),
        critical_scene=np.int64(coarse_reg>=cfg.critical_min_coarse_regret),
    )


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data_v7'); ap.add_argument('--split',required=True,choices=['train','val','test']); ap.add_argument('--n',type=int,required=True); ap.add_argument('--seed',type=int,default=7); args=ap.parse_args()
    rng=np.random.default_rng(args.seed); od=Path(args.out)/args.split; od.mkdir(parents=True,exist_ok=True)
    for p in od.glob('*.npz'): p.unlink()
    for i in tqdm(range(args.n),desc=f'generate {args.split} GRID={CFG.grid}'):
        np.savez_compressed(od/f'{i:06d}.npz',**build_one(rng,CFG))
    print(f'saved {args.n} scenes to {od}; patches={CFG.n_patches}; image={CFG.image_size}x{CFG.image_size}')

if __name__=='__main__': main()
