import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from rope_utils import init_t_xy, init_random_2d_freqs, compute_axial_freqs, compute_mixed_freqs, pair

# Performer implementation adapted and modified from: https://github.com/lucidrains/performer-pytorch

@torch.jit.script
def fused_rope_kernel(q, k, freqs_cis):
    b, h, n, d = q.shape
    
    q_reshaped = q.reshape(b, h, n, -1, 2)
    k_reshaped = k.reshape(b, h, n, -1, 2)
    
    # Convert to complex numbers
    q_complex = torch.view_as_complex(q_reshaped.float())
    k_complex = torch.view_as_complex(k_reshaped.float())
    print(q_complex.shape)
    print(k_complex.shape)

    q_rot = torch.view_as_real(q_complex * freqs_cis)
    k_rot = torch.view_as_real(k_complex * freqs_cis) 
    
    q_out = q_rot.reshape(b, h, n, d)
    k_out = k_rot.reshape(b, h, n, d)
    
    return q_out.type_as(q), k_out.type_as(k)

@torch.jit.script
def adaptive_kernel_feature_map(x, projection, seq_len, normalize=True):

    # Calculate feature dimension
    feat_dim = projection.shape[0]
    
    # Efficient normalization
    if normalize:
        x = x * (x.shape[-1] ** -0.25)
    
    projection = projection[:feat_dim]
    
    # ReLU kernel computation
    x_proj = torch.einsum('bhnd,jd->bhnj', x, projection)
    x_proj = F.relu(x_proj) + 1e-6

    #print(x_proj.shape)

    return x_proj

@torch.jit.script
def efficient_linear_attention(q, k, v):
    # Compute key aggregation
    k_cumsum = k.sum(dim=-2)
    #print(k_cumsum.shape)
    #print(q.shape)
    
    # Compute context matrices
    context = torch.einsum('...nd,...ne->...de', k, v)
    #print(context.shape)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    #print(D_inv.shape)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

class PerformerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        causal: bool = False,
        rope_theta: float = 100.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        self.theta = rope_theta
        
        # Efficient linear projections with a single matrix
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        # Initialize random projection matrix
        self.num_features = min(dim_head, int(math.log(16*16 + 1) * dim_head))  # typical sequence length for 16 x 16 patches
        projection = torch.randn(self.num_features, dim_head) * (2 / dim_head) ** 0.5
        self.register_buffer('projection', projection)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        b, n, _, h = *x.shape, self.heads
        
        # Efficient QKV computation
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        
        if freqs_cis is not None:
            q_cls, q_content = q[:, :, :1], q[:, :, 1:]
            k_cls, k_content = k[:, :, :1], k[:, :, 1:]
            
            # Apply fused RoPE operation
            #print(q_content.shape)
            #print(k_content.shape)
            q_content, k_content = fused_rope_kernel(q_content, k_content, freqs_cis)
            
            # Recombine with CLS token
            q = torch.cat([q_cls, q_content], dim=2)
            k = torch.cat([k_cls, k_content], dim=2)
        
        # Apply adaptive kernel computation
        q = adaptive_kernel_feature_map(q, self.projection, n)
        k = adaptive_kernel_feature_map(k, self.projection, n)
        
        #print(q.shape)
        #print(k.shape)
        # Efficient linear attention
        out = efficient_linear_attention(q, k, v)
        
        # Final projection with merged operations
        out = rearrange(out, 'b h n d -> b n (h d)')
        #print(out.shape)
        return self.dropout(self.to_out(out))

class RopePerformerViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        rope_type: str = 'axial',
        rope_theta: float = 100.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width
        
        # Efficient patch embedding using fused operations
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim, eps=1e-6),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim, eps=1e-6),
        )
        
        # Initialize RoPE frequencies
        if rope_type == 'axial':
            freqs_cis = compute_axial_freqs(
                dim=dim_head,
                end_x=image_width // patch_width,
                end_y=image_height // patch_height,
                theta=rope_theta
            )
            self.register_buffer('freqs_cis', freqs_cis)
        else:
            freqs = init_random_2d_freqs(dim=dim_head, num_heads=heads, theta=rope_theta)
            print(freqs.shape)
            self.register_buffer('freqs', freqs)

            t_x, t_y = init_t_xy(image_width // patch_width, image_height // patch_height)
            #print(t_x.shape)
            #print(t_y.shape)
            self.register_buffer('t_x', t_x)
            self.register_buffer('t_y', t_y)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerformerAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, rope_theta=rope_theta),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.rope_type = rope_type

    @torch.jit.ignore
    def update_rope_freqs(self, x):
        """
        Updates RoPE frequencies based on input size.
        """
        if self.rope_type == 'mixed':
            freqs_cis = compute_mixed_freqs(self.freqs, self.t_x, self.t_y, self.layers[0][0].heads)
        else:
            freqs_cis = self.freqs_cis
        print(freqs_cis.shape)
        return freqs_cis

    def forward(self, img):
        # Efficient patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        
        # Get RoPE frequencies
        freqs_cis = self.update_rope_freqs(x)
        #print(freqs_cis.shape)
        
        # Apply transformer layers
        for attn, ff in self.layers:
            x = attn(x, freqs_cis) + x
            x = ff(x) + x
        
        # Classification
        return self.mlp_head(x[:, 0])