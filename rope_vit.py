import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from rope_utils import init_t_xy, init_random_2d_freqs, compute_axial_freqs, compute_mixed_freqs, apply_rotary_pos_emb, pair

# vit implementation adapted and modified from https://github.com/lucidrains/vit-pytorch

class RoPEAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim) # True if heads > 1 or dim_head != dim

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, freqs_cis=None):
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # debug
        #print(q.shape)
        #print(k.shape)
        #print(v.shape)

        if freqs_cis is not None:

            q_cls, q_content = q[:, :, :1], q[:, :, 1:]
            k_cls, k_content = k[:, :, :1], k[:, :, 1:]
            
            # Apply RoPE to content tokens
            q_content_rope, k_content_rope = apply_rotary_pos_emb(q_content, k_content, freqs_cis)
            
            # Concatenate back with CLS token
            q = torch.cat([q_cls, q_content_rope], dim=2)
            k = torch.cat([k_cls, k_content_rope], dim=2)
        
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class RoPEViT(nn.Module):
    def __init__(
        self, 
        image_size=224, 
        patch_size=16, 
        num_classes=1000, 
        dim=768, 
        depth=12, 
        heads=12, 
        mlp_dim=3072, 
        pool='cls', 
        channels=3, 
        dim_head=64, 
        dropout=0., 
        emb_dropout=0.,
        rope_type='axial',  # 'axial' or 'mixed'
        rope_theta=100.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        patch_dim = channels * patch_height * patch_width
        
        self.patch_size = patch_size
        self.rope_type = rope_type
        self.num_heads = heads
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            )

        # Initialize positional embeddings based on RoPE type
        if rope_type == 'mixed':

            # Initialize a single freq tensor with depth dimension
            freqs = init_random_2d_freqs(dim=dim_head, num_heads=heads, depth=depth, theta=rope_theta)

            #print(freqs.shape)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            # Register position indices as buffers
            t_x, t_y = init_t_xy(image_width // patch_width, image_height // patch_height)

            # debug
            #print(t_x.shape)
            #print(t_y.shape)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)

        else:  # axial rope

            freqs_cis = compute_axial_freqs(dim=dim_head, end_x=image_width // patch_width, end_y=image_height // patch_height, theta=rope_theta)
            #print(freqs_cis.shape)
            self.register_buffer('freqs_cis', freqs_cis)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer blocks with RoPE attention
        self.transformer = nn.ModuleList([])
        for _ in range(depth):

            self.transformer.append(nn.ModuleList([
                RoPEAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    # nn.Linear(mlp_dim, mlp_dim),
                    # nn.GELU(),
                    # nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                    )
                ]))

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        #print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        # Compute RoPE frequencies
        if self.rope_type == 'mixed':

            if self.freqs_t_x.shape[0] != n:

                t_x, t_y = init_t_xy(end_x=img.shape[-1] // self.patch_size, end_y=img.shape[-2] // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)

            else:

                t_x, t_y = self.freqs_t_x, self.freqs_t_y

            freqs_cis = compute_mixed_freqs(self.freqs, t_x, t_y, self.num_heads)
            #print(freqs_cis.shape)

        else:  # axial
            if self.freqs_cis.shape[0] != n:

                freqs_cis = compute_axial_freqs(dim=x.shape[-1] // self.num_heads, end_x=img.shape[-1] // self.patch_size, end_y=img.shape[-2] // self.patch_size)
                freqs_cis = freqs_cis.to(x.device)
                #print(freqs_cis.shape)

            else:

                freqs_cis = self.freqs_cis
                #print(freqs_cis.shape)

        # Apply transformer blocks
        for i, (attn, ff) in enumerate(self.transformer):
            if self.rope_type == 'mixed':

                x = attn(x, freqs_cis[i]) + x
                #print(x.shape)

            else:

                x = attn(x, freqs_cis) + x
                #print(x.shape)

            x = ff(x) + x

        # Pool and classify
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x