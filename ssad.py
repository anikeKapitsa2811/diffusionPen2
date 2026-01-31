import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_steps=600):
        super().__init__()
        pe = torch.zeros(max_steps, dim)
        position = torch.arange(0, max_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t):
        return self.pe[t]  # [B, dim]


# ---------- Self-Attention Block ----------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


# ---------- Content Encoder with selectable mode ----------
class ContentEncoder(nn.Module):
    def __init__(self, d_s=1280, d_x=320, seq_len=10, max_steps=600, mode="TP", n_crops=5):
        """
        mode: one of ["TP", "TPL", "CP", "TA", "CA", "TS"]
        d_s:  style feature dimension (here 1280)
        n_crops: number of crops per writer (here 5)
        """
        super().__init__()
        self.mode = mode.upper()
        self.n_crops = n_crops

        # Projection from 1280 → 320 (Eq. 25)
        self.Ws = nn.Linear(d_s, d_x)

        # Attention and positional encoders
        self.self_attn = SelfAttentionBlock(d_x)
        self.pe_t = PositionalEncoding(d_x, max_steps)
        self.LPE = nn.Parameter(torch.randn(1, 1, d_x))  # used for TPL variant

        # timestep embedding (Eq. 32)
        self.linear1_t = nn.Linear(d_x, d_x)
        self.linear2_t = nn.Linear(d_x, d_x)

    def forward(self, y, s_w, t):
        """
        y   : [B, L, d_x] text embeddings
        s_w : [5*B, d_s]  style embeddings (5 crops per sample)
        t   : [B]         timestep indices
        """
        B, L, _ = y.shape

        # ---- aggregate 5 crops → one per batch item ----
        s_w = s_w.view(B, self.n_crops, -1)          # [B, 5, 1280]
        s_w = s_w.mean(dim=1)                        # [B, 1280] (average style vector)

        # ---- Eq. (25): project style embedding ----
        
        if self.mode in ["TPL","TA"]:
            s_hat = self.Ws(s_w)                         # [B, 320]
            s_tok = s_hat.unsqueeze(1)                   # [B, 1, 320]

        # ---- Mode-specific inclusion ----
        if self.mode == "TP":
            y_cat = torch.cat([y, s_tok], dim=1)
            c = self.self_attn(y_cat)

        elif self.mode == "TPL":
            s_tok_plus = s_tok + self.LPE
            y_cat = torch.cat([y, s_tok_plus], dim=1)
            c = self.self_attn(y_cat)

        elif self.mode == "CP":
            s_broadcast = s_hat.unsqueeze(1).expand(-1, L, -1)
            y_mod = y + s_broadcast
            c = self.self_attn(y_mod)

        elif self.mode == "TA":
            y_att = self.self_attn(y)
            c = torch.cat([y_att, s_tok], dim=1)

        elif self.mode == "CA":
            y_att = self.self_attn(y)
            s_broadcast = s_hat.unsqueeze(1).expand(-1, L, -1)
            c = y_att + s_broadcast

        elif self.mode == "TS":
            y_att = self.self_attn(y)
            c = y_att  # style used only in timestep branch
            t_emb =  t + s_w #s_hat if self.mode in ["TS", "TP", "TPL"] else 0

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # ---- Eq. (32): timestep embedding ----
        """
        t_emb = self.linear2_t(
                    F.silu(self.linear1_t(self.pe_t(t)))
                 ) + (s_hat if self.mode in ["TS", "TP", "TPL"] else 0)
        """



        return c, t_emb


# ---------- Dummy calling script ----------
if __name__ == "__main__":
    B, L = 2, 40        # batch size, sequence length
    n_crops = 6
    d_s, d_x = 1280, 768

    # Dummy tensors
    y = torch.randn(B, L, d_x)             # text embeddings
    s_w = torch.randn(B * n_crops, d_s)    # style embeddings [5*B, 1280]
    t = torch.randint(0, 600, (B,))        # timestep indices

    for mode in ["TP", "TPL", "CP", "TA", "CA", "TS"]:
        encoder = ContentEncoder(d_s=d_s, d_x=d_x, seq_len=L, mode=mode, n_crops=n_crops)
        C, t_emb = encoder(y, s_w, t)
        print(f"Mode: {mode:>3} | C shape: {tuple(C.shape)} | t_emb shape: {tuple(t_emb.shape)}")
