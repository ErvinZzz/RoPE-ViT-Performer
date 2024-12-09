import torch

# RoPE implementation referenced Heo, Byeongho, et al. "Rotary position embedding for vision transformer." European Conference on Computer Vision. Springer, Cham, 2025.
# https://github.com/naver-ai/rope-vit

# helpers
def pair(t):
    
    return t if isinstance(t, tuple) else (t, t)

# RoPE
def init_t_xy(end_x, end_y):

    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()

    return t_x, t_y

def compute_axial_freqs(dim, end_x, end_y, theta = 100.0):

    # Generate base frequencies using geometric sequence
    freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    
    t_x, t_y = init_t_xy(end_x, end_y)
    # Compute outer product for position-frequency interaction
    freqs_x = torch.outer(t_x, freqs_base)
    freqs_y = torch.outer(t_y, freqs_base)
    
    # Convert to complex numbers for rotation
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    freqs = torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)
    return freqs

def init_random_2d_freqs(dim, num_heads, depth = 1, theta = 10.0, rotate = True):

    # Generate magnitudes using geometric sequence
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    
    freqs_x = []
    freqs_y = []
    
    for i in range(num_heads):
        # Random rotation angle per head if rotate=True
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        
        # Generate orthogonal frequency components
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    
    #print(freqs_x)
    #print(freqs_y)

    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    
    if depth > 1:
        # For ViT: expand for multiple layers
        freqs = freqs.unsqueeze(1).expand(-1, depth, -1, -1)  # [2, depth, num_heads, dim//4]
    
    return freqs

def compute_mixed_freqs(freqs, t_x, t_y, num_heads):

    N = t_x.shape[0]
    is_vit = freqs.dim() == 4
    
    with torch.cuda.amp.autocast(enabled=False):
        if is_vit:
            depth = freqs.shape[1]
            # Matrix multiply for position-frequency interaction
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))  # [N, depth, num_heads, dim//4]
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))

            #print(freqs_x.shape)
            #print(freqs_y.shape)
            
            # Reshape and permute to [depth, heads, N, dim//4]
            freqs_x = freqs_x.view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
            freqs_y = freqs_y.view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
            #print(freqs_x.shape)
            #print(freqs_y.shape)

        else:
            # For Performer: direct einsum computation
            freqs_x = torch.einsum('n,hd->hnd', t_x, freqs[0])  # [heads, N, dim//4]
            freqs_y = torch.einsum('n,hd->hnd', t_y, freqs[1])

            #print(freqs_x.shape)
            #print(freqs_y.shape)
            
            # Add batch dimension: [1, heads, N, dim//4]
            freqs_x = freqs_x.unsqueeze(0)
            freqs_y = freqs_y.unsqueeze(0)

            #print(freqs_x.shape)
            #print(freqs_y.shape)
        
        # Create complex rotations
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
        print(freqs_cis.shape)
    
    return freqs_cis

def apply_rotary_pos_emb(q, k, freqs_cis):

    # For ViT: take first layer if multi-layer frequencies provided
    if freqs_cis.dim() == 4 and freqs_cis.shape[0] > 1:
        freqs_cis = freqs_cis[0]
    
    # Reshape queries and keys for complex multiplication
    q_reshape = q.reshape(*q.shape[:-1], -1, 2)  # [batch, heads, seq_len, dim//2, 2]
    k_reshape = k.reshape(*k.shape[:-1], -1, 2)
    
    print(q_reshape.shape)
    print(k_reshape.shape)

    # Convert to complex numbers
    q_complex = torch.view_as_complex(q_reshape.float())  # [batch, heads, seq_len, dim//2]
    k_complex = torch.view_as_complex(k_reshape.float())
    q_out = torch.view_as_real(q_complex * freqs_cis).flatten(3)  # [batch, heads, seq_len, dim]
    k_out = torch.view_as_real(k_complex * freqs_cis).flatten(3)

    #print(q_out.shape)
    #print(k_out.shape)
    
    return q_out.type_as(q), k_out.type_as(k)
