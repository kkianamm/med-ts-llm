"""
BiomedCoOp adapted to ECG time-series classification (PTB-XL).

BiomedCoOp (Koleilat et al., arXiv:2411.15232) is a prompt-learning method for
CLIP-style vision-language models. The ideas ported here:

  * LLM prompt ensembling: 50 textual descriptions per class are embedded with a
    frozen text encoder and averaged into a class prototype  P_g  (Eq. 3).
  * Cosine-similarity classification between a sample embedding and the class
    prototypes, with a learnable temperature (Eq. 1).
  * SCCM: class prototypes are made *learnable* (init = P_g) and pulled back
    toward the LLM ensemble with an MSE term (Eq. 9).
  * KDSP: a *teacher* prototype set P_s is built per batch by pruning outlier
    prompts via a Median-Absolute-Deviation modified z-score (Eq. 5-7), then the
    student (learnable prototypes) is distilled toward the teacher with a KL term
    (Eq. 10).

Key difference from the paper: there is no pretrained ECG<->text CLIP, so we LEARN
the alignment. A time-series encoder produces an ECG embedding which is projected
into the frozen text-prototype space; the prototypes live in the text encoder's
space and are anchored there by SCCM. Total loss = CE + λ1·SCCM + λ2·KDSP, with the
SCCM+KDSP terms exposed to the task via `self.aux_loss`.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .PatchTST import PatchEmbedding
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer


DEFAULT_CLASS_CODES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ----------------------------------------------------------------------------- #
# Helpers (module-level so they can be monkeypatched / reused)
# ----------------------------------------------------------------------------- #
def load_class_prompts(path, class_codes):
    """Return a list[list[str]] of prompts ordered by `class_codes`.

    Accepts: a combined JSON dict {code: [prompts]}, a JSON list of objects with
    {class_code, prompts}, or a directory of per-class JSON files each shaped like
    {"class_code": "...", "prompts": [...]}.
    """
    path = Path(path)
    by_code = {}

    def ingest(obj):
        if isinstance(obj, dict) and "class_code" in obj and "prompts" in obj:
            by_code[obj["class_code"]] = list(obj["prompts"])
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict) and "prompts" in v:
                    by_code[k] = list(v["prompts"])
                elif isinstance(v, list):
                    by_code[k] = list(v)
        elif isinstance(obj, list):
            for item in obj:
                ingest(item)

    if path.is_dir():
        for fp in sorted(path.glob("*.json")):
            ingest(json.loads(fp.read_text()))
    else:
        ingest(json.loads(path.read_text()))

    missing = [c for c in class_codes if c not in by_code]
    if missing:
        raise ValueError(f"No prompts found for classes {missing} in {path}")
    return [by_code[c] for c in class_codes]


def encode_texts(texts, model_name, device, batch_size=64, max_length=64):
    """Frozen text encoder with mean pooling -> L2-normalized [N, D] embeddings."""
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            hs = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(out, dim=0)


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #
class BiomedCoOpTS(nn.Module):

    supported_tasks = ["classification"]
    supported_modes = ["multivariate", "univariate"]

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.mc = config.models.biomedcoop_ts
        self.task = config.task
        assert self.task == "classification"

        self.enc_in = dataset.n_features
        self.num_class = dataset.n_classes
        self.seq_len = config.history_len
        assert config.pred_len == self.seq_len

        # ---- loss weights / KDSP hyperparameters ----
        self.lambda_sccm = self.mc.get("lambda_sccm", 1.0)
        self.lambda_kdsp = self.mc.get("lambda_kdsp", 1.0)
        self.z_threshold = self.mc.get("z_threshold", 1.25)  # ζs in the paper
        self.learnable_prototypes = self.mc.get("learnable_prototypes", True)

        # ---- text prototypes from LLM-generated prompts ----
        class_codes = self.mc.get("class_codes", DEFAULT_CLASS_CODES)
        assert len(class_codes) == self.num_class, \
            f"class_codes ({len(class_codes)}) must match n_classes ({self.num_class})"
        prompts = load_class_prompts(self.mc.prompts_path, class_codes)   # list[C][Ni]
        n_per = min(len(p) for p in prompts)
        flat = [t for cls in prompts for t in cls[:n_per]]
        emb = encode_texts(flat, self.mc.text_model, device="cpu")        # [C*n, Dt]
        Dt = emb.shape[1]
        Tg = emb.view(self.num_class, n_per, Dt)                          # [C, n, Dt]
        Pg = F.normalize(Tg.mean(dim=1), dim=-1)                          # [C, Dt]

        self.register_buffer("Tg", F.normalize(Tg, dim=-1))
        self.register_buffer("Pg", Pg)
        self.text_dim = Dt
        if self.learnable_prototypes:
            self.prototypes = nn.Parameter(Pg.clone())
        else:
            self.register_buffer("prototypes", Pg.clone())

        self.logit_scale = nn.Parameter(torch.tensor(2.6593))  # log(1/0.07)

        # ---- time-series encoder (PatchTST-style) -> projects to text space ----
        d_model = self.mc.d_model
        n_heads = self.mc.n_heads
        d_ff = self.mc.d_ff
        e_layers = self.mc.e_layers
        patch_len = self.mc.patching.patch_len
        stride = self.mc.patching.stride
        dropout = config.training.dropout

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, stride, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 3, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation="gelu")
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, Dt))

        self.aux_loss = torch.zeros(())

    # -- encode an ECG window into the text-prototype space --
    def encode_ts(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = x.permute(0, 2, 1)                              # [B, M, L]
        enc_out, n_vars = self.patch_embedding(x)           # [B*M, P, d_model]
        enc_out, _ = self.encoder(enc_out)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        pooled = enc_out.mean(dim=(1, 2))                   # [B, d_model]
        V = self.proj(pooled)                               # [B, Dt]
        return F.normalize(V, dim=-1)

    def _kdsp_teacher(self, V, scale):
        """Build outlier-pruned teacher prototypes P_s and return teacher logits."""
        C, n, Dt = self.Tg.shape
        with torch.no_grad():
            sim = scale * (V.detach() @ self.Tg.reshape(C * n, Dt).T)     # [B, C*n]
            sim = sim.reshape(V.shape[0], C, n)
            S = sim.mean(dim=0)                                           # [C, n] per-prompt score
            Ms = S.median(dim=1, keepdim=True).values
            D = (S - Ms).abs().median(dim=1, keepdim=True).values
            z = (S - Ms) / (D + 1e-6)
            keep = (z.abs() < self.z_threshold).float().unsqueeze(-1)     # [C, n, 1]
            w = keep / keep.sum(dim=1, keepdim=True).clamp(min=1.0)
            Ps = (self.Tg * w).sum(dim=1)                                 # [C, Dt]
            # classes with no inliers -> fall back to full ensemble
            empty = (keep.sum(dim=1).squeeze(-1) == 0)
            if empty.any():
                Ps[empty] = self.Pg[empty]
            Ps = F.normalize(Ps, dim=-1)
        return scale * (V.detach() @ Ps.T)                               # [B, C]

    def forward(self, inputs):
        x_enc = inputs["x_enc"]
        V = self.encode_ts(x_enc)                                        # [B, Dt]

        protos = F.normalize(self.prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (V @ protos.T)                                  # [B, C]

        if self.training:
            aux = logits.new_zeros(())
            if self.learnable_prototypes and self.lambda_sccm > 0:
                sccm = ((self.prototypes - self.Pg) ** 2).sum(dim=-1).mean()
                aux = aux + self.lambda_sccm * sccm
            if self.lambda_kdsp > 0:
                teacher = self._kdsp_teacher(V, scale)
                teacher_p = F.softmax(teacher, dim=-1).detach()
                kdsp = F.kl_div(F.log_softmax(logits, dim=-1), teacher_p,
                                reduction="batchmean")
                aux = aux + self.lambda_kdsp * kdsp
            self.aux_loss = aux
        else:
            self.aux_loss = logits.new_zeros(())

        return logits
