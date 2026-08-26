import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding3D(nn.Module):

    def __init__(self, num_freqs=6):
        super().__init__()

        self.num_freqs = num_freqs

    def forward(self, x):

        # x: [N, 3]

        features = [x]

        for i in range(self.num_freqs):

            freq = 2.0 ** i

            features.append(torch.sin(freq * x))
            features.append(torch.cos(freq * x))

        return torch.cat(features, dim=-1)

class SuperGaussianSelfAttention(nn.Module):

    def __init__(
        self,
        remaining_dim,
        pos_dim,
        d_model=128,
        num_heads=4,
        num_freqs=6
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.pos_encoder = PositionalEncoding3D(
            num_freqs=num_freqs
        )

        self.input_norm = nn.LayerNorm(
            remaining_dim + pos_dim
        )

        self.input_proj = nn.Linear(
            remaining_dim + pos_dim,
            d_model
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.output_proj = nn.Linear(
            d_model,
            d_model
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        group_xyz,
        group_remaining
    ):

        pos = self.pos_encoder(group_xyz)

        x = torch.cat(
            [pos, group_remaining],
            dim=-1
        )

        x = self.input_norm(x)

        u = self.input_proj(x)

        Q = self.q_proj(u)
        K = self.k_proj(u)
        V = self.v_proj(u)

        G = u.shape[0]

        Q = Q.view(
            G,
            self.num_heads,
            self.head_dim
        ).transpose(0, 1)

        K = K.view(
            G,
            self.num_heads,
            self.head_dim
        ).transpose(0, 1)

        V = V.view(
            G,
            self.num_heads,
            self.head_dim
        ).transpose(0, 1)

        attention = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        attention = F.softmax(
            attention,
            dim=-1
        )

        out = torch.matmul(
            attention,
            V
        )

        out = out.transpose(0, 1).contiguous()

        out = out.view(
            G,
            self.d_model
        )

        out = self.output_proj(out)

        out = self.output_norm(
            out + u
        )

        return out, attention


# ============================================================
# SPARSE LOCAL SELF-ATTENTION
# ============================================================

class SparseLocalAttention(nn.Module):

    def __init__(
        self,
        input_dim=49,
        embed_dim=128,
        num_heads=4
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_proj = nn.Linear(
            input_dim,
            embed_dim
        )

        self.q_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.k_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.v_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, neighbor_indices):

        N = x.shape[0]
        K_neighbors = neighbor_indices.shape[1]

        # ----------------------------------------------------
        # Project Gaussian features
        # ----------------------------------------------------

        x = self.input_proj(x)
        x = self.norm(x)

        # ----------------------------------------------------
        # Q for every Gaussian
        # ----------------------------------------------------

        Q = self.q_proj(x)

        # ----------------------------------------------------
        # K and V for 10 neighbors
        # ----------------------------------------------------

        neighbor_x = x[neighbor_indices]

        K = self.k_proj(neighbor_x)
        V = self.v_proj(neighbor_x)

        # Q:
        # [N, D]
        #
        # K,V:
        # [N, 10, D]

        # ----------------------------------------------------
        # Split attention heads
        # ----------------------------------------------------

        Q = Q.view(
            N,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            N,
            K_neighbors,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            N,
            K_neighbors,
            self.num_heads,
            self.head_dim
        )

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = torch.einsum(
            "nhd,nkhd->nhk",
            Q,
            K
        )

        scores = scores / (self.head_dim ** 0.5)

        attention = F.softmax(
            scores,
            dim=-1
        )

        # ----------------------------------------------------
        # Weighted aggregation
        # ----------------------------------------------------

        local = torch.einsum(
            "nhk,nkhd->nhd",
            attention,
            V
        )

        # [N, heads, head_dim]
        # -> [N, embed_dim]

        local = local.reshape(
            N,
            self.embed_dim
        )

        local = self.out_proj(local)

        return local, attention