import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


def split_patches(x,grid):
    B,C,H,W=x.shape; h=H//grid; w=W//grid
    p=x.unfold(2,h,h).unfold(3,w,w).permute(0,2,3,1,4,5).contiguous()
    return p.view(B,grid*grid,C,h,w)


class CheapPreviewEncoder(nn.Module):
    def __init__(self,feat_dim=64):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,feat_dim),nn.LayerNorm(feat_dim))
    def forward(self,image):
        x=F.interpolate(image,size=(CFG.preview_size,CFG.preview_size),mode='bilinear',align_corners=False)
        p=split_patches(x,CFG.grid); B,M,C,H,W=p.shape
        return self.cnn(p.reshape(B*M,C,H,W)).view(B,M,-1)


class HighResTokenEncoder(nn.Module):
    def __init__(self,feat_dim=64):
        super().__init__(); ps=CFG.high_patch_size; crop=CFG.high_crop_size
        self.tokens_per_crop=(crop//ps)**2
        self.stem=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU())
        self.patch_embed=nn.Conv2d(32,feat_dim,kernel_size=ps,stride=ps)
        self.cls=nn.Parameter(torch.zeros(1,1,feat_dim)); self.pos=nn.Parameter(torch.zeros(1,self.tokens_per_crop+1,feat_dim))
        layer=nn.TransformerEncoderLayer(d_model=feat_dim,nhead=4,dim_feedforward=feat_dim*2,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(layer,num_layers=2); self.norm=nn.LayerNorm(feat_dim)
    def forward(self,crops):
        x=self.patch_embed(self.stem(crops)).flatten(2).transpose(1,2)
        cls=self.cls.expand(x.shape[0],-1,-1); x=torch.cat([cls,x],1); x=x+self.pos[:,:x.shape[1]]
        return self.norm(self.enc(x))[:,0]


class DualResolutionPerception(nn.Module):
    def __init__(self,feat_dim=64):
        super().__init__(); self.preview_encoder=CheapPreviewEncoder(feat_dim); self.high_encoder=HighResTokenEncoder(feat_dim)
        self.preview_cost_head=nn.Linear(feat_dim,1); self.high_head=nn.Linear(feat_dim,4)
    def encode_preview(self,image): return self.preview_encoder(image)
    def _crops(self,image): return split_patches(image,CFG.grid)
    def encode_all_high(self,image):
        p=self._crops(image); B,M,C,H,W=p.shape
        return self.high_encoder(p.reshape(B*M,C,H,W)).view(B,M,-1)
    def encode_selected_high(self,image,patch_index):
        p=self._crops(image); rows=torch.arange(image.shape[0],device=image.device)
        return self.high_encoder(p[rows,patch_index])
    def predict_preview_penalty(self,feat): return F.softplus(self.preview_cost_head(feat).squeeze(-1))
    def classify_high_features(self,feat): return self.high_head(feat)


class GradientFlowNet(nn.Module):
    """Gradient-conditioned visual flow field. No STOP head and no preference head."""
    def __init__(self,feat_dim=64,graph_dim=4,hidden=128):
        super().__init__()
        # state feature + graph feature + z + acquired mask + inner time + outer time + budget
        self.in_proj=nn.Linear(feat_dim+graph_dim+5,hidden)
        layer=nn.TransformerEncoderLayer(d_model=hidden,nhead=CFG.transformer_heads,dim_feedforward=hidden*2,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=CFG.transformer_layers)
        self.velocity=nn.Linear(hidden,1)
    def forward(self,state_feat,graph_feat,z,w_binary,inner_t,outer_t,budget_frac):
        B,M,_=state_feat.shape
        def ex(v):
            if v.ndim==1: return v[:,None,None].expand(B,M,1)
            if v.ndim==2: return v[...,None]
            return v
        x=torch.cat([state_feat,graph_feat,ex(z),ex(w_binary),ex(inner_t),ex(outer_t),ex(budget_frac)],dim=-1)
        h=self.encoder(self.in_proj(x))
        return self.velocity(h).squeeze(-1)


class FixedStepNeurodynamic(nn.Module):
    """Learnable fixed-step Euler/inertial integrator around the Flow velocity field."""
    def __init__(self,steps=None):
        super().__init__(); self.steps=CFG.nd_steps if steps is None else int(steps)
        # softplus(-2)~0.127, suitable stable initial step; beta starts near 0.
        self.raw_alpha=nn.Parameter(torch.full((self.steps,),-1.25))
        self.raw_beta=nn.Parameter(torch.full((self.steps,),-4.0))
    def coefficients(self):
        alpha=F.softplus(self.raw_alpha)
        beta=torch.sigmoid(self.raw_beta)*CFG.nd_beta_max
        return alpha,beta
    def forward(self,flow,state_feat,graph_feat,z0,w_binary,outer_t,budget_frac,return_trace=False):
        z=z0; z_prev=z0
        alpha,beta=self.coefficients(); trace=[z]
        for k in range(self.steps):
            inner_t=torch.full((z.shape[0],), (k+0.5)/self.steps, device=z.device, dtype=z.dtype)
            v=flow(state_feat,graph_feat,z,w_binary,inner_t,outer_t,budget_frac)
            z_next=z+alpha[k]*v+beta[k]*(z-z_prev)
            z_prev,z=z,z_next
            trace.append(z)
        return (z,trace) if return_trace else z
