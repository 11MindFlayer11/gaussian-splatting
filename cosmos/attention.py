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

        freqs = 2.0 ** torch.arange(
            self.num_freqs,
            device=x.device,
            dtype=x.dtype
        )

        scaled_x = x.unsqueeze(1) * freqs.view(1, -1, 1)

        encoded = torch.stack(
            [
                torch.sin(scaled_x),
                torch.cos(scaled_x)
            ],
            dim=2
        )

        encoded = encoded.flatten(start_dim=1)

        return torch.cat(
            [x, encoded],
            dim=-1
        )
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

    def forward(self, x, neighbor_indices, chunk_size=50000):
        N = x.shape[0]
        x = self.input_proj(x)
        x = self.norm(x)

        Q_full = self.q_proj(x)
        outputs = []
        attn_chunks = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            Q = Q_full[start:end].view(end - start, self.num_heads, self.head_dim)

            neighbor_x = x[neighbor_indices[start:end]]
            K = self.k_proj(neighbor_x).view(end - start, -1, self.num_heads, self.head_dim)
            V = self.v_proj(neighbor_x).view(end - start, -1, self.num_heads, self.head_dim)

            scores = torch.einsum("nhd,nkhd->nhk", Q, K) / (self.head_dim ** 0.5)
            attention = F.softmax(scores, dim=-1)
            local = torch.einsum("nhk,nkhd->nhd", attention, V)

            outputs.append(local.reshape(end - start, self.embed_dim))
            attn_chunks.append(attention)

        local = torch.cat(outputs, dim=0)
        attention = torch.cat(attn_chunks, dim=0)

        local = self.out_proj(local)
        return local, attention