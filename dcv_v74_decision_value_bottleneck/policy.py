import torch
from config import CFG
from decision import causal_state_features, legal_mask, exact_outcome_from_state_feat


def policy_scores(flow,nd,state_feat,batch,w,z,outer_t,budget_frac):
    return nd(flow,state_feat,batch['patch_graph_feat'],z,w,outer_t,budget_frac)


def masked_distribution(scores,eligible,temp=None):
    temp=CFG.policy_temperature if temp is None else temp
    logits=scores/temp; logits=logits.masked_fill(~eligible,-1e9); return torch.distributions.Categorical(logits=logits)


def rollout_cached(flow,nd,perception,batch,preview_feat,high_feat,budget_frac,sample=True,detach_state=True):
    B=preview_feat.shape[0]; M=CFG.n_patches; K=CFG.visual_budget_k(budget_frac); device=preview_feat.device
    w=torch.zeros(B,M,device=device); z=torch.zeros(B,M,device=device); logps=[]; ents=[]; actions=[]
    budget=torch.full((B,),float(budget_frac),device=device)
    for t in range(K):
        sf=causal_state_features(preview_feat,high_feat,w); outer=torch.full((B,),t/max(K-1,1),device=device)
        z_new=policy_scores(flow,nd,sf,batch,w,z,outer,budget); elig=legal_mask(batch,w); dist=masked_distribution(z_new,elig)
        a=dist.sample() if sample else dist.logits.argmax(-1); logps.append(dist.log_prob(a)); ents.append(dist.entropy()); actions.append(a)
        rows=torch.arange(B,device=device); w=w.clone(); w[rows,a]=1.0; z=z_new.detach() if detach_state else z_new
    sf=causal_state_features(preview_feat,high_feat,w); regret,path=exact_outcome_from_state_feat(perception,sf,w,batch)
    return {'w':w,'z':z,'actions':torch.stack(actions,1),'sum_logp':torch.stack(logps,1).sum(1),'entropy':torch.stack(ents,1).mean(1),'regret':regret,'path_idx':path}
