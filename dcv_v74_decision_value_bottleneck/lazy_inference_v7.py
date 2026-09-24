import time, torch
from config import CFG
from decision import legal_mask, exact_outcome_from_state_feat


@torch.no_grad()
def lazy_rollout(flow,nd,perception,batch,budget_frac):
    image=batch['image']; B=image.shape[0]; M=CFG.n_patches; K=CFG.visual_budget_k(budget_frac); device=image.device
    t=time.perf_counter(); preview=perception.encode_preview(image); preview_ms=(time.perf_counter()-t)*1000
    state_feat=preview.clone(); w=torch.zeros(B,M,device=device); z=torch.zeros(B,M,device=device); selected=[]; high_ms=policy_ms=0.0; budget=torch.full((B,),float(budget_frac),device=device)
    for step in range(K):
        outer=torch.full((B,),step/max(K-1,1),device=device); tp=time.perf_counter(); z=nd(flow,state_feat,batch['patch_graph_feat'],z,w,outer,budget); elig=legal_mask(batch,w); a=z.masked_fill(~elig,-1e9).argmax(-1); policy_ms+=(time.perf_counter()-tp)*1000
        th=time.perf_counter(); h=perception.encode_selected_high(image,a); high_ms+=(time.perf_counter()-th)*1000; rows=torch.arange(B,device=device); state_feat[rows,a]=h; w[rows,a]=1.0; selected.append(a)
    regret,path=exact_outcome_from_state_feat(perception,state_feat,w,batch)
    return {'w':w,'z':z,'actions':torch.stack(selected,1),'regret':regret,'path_idx':path,'preview_ms':preview_ms,'high_ms':high_ms,'policy_ms':policy_ms,'K':K}
