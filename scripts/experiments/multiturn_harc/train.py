"""Multi-turn HARC training for Qwen 2.5 7B on Apple Silicon (MPS).

Key differences from stock HARC:
1. Multi-turn conversations — harm measured at last-user-token position where
   full conversation context is encoded (same position logic, richer context)
2. MPS device instead of CUDA
3. Reduced batch sizes for 64 GB unified memory
4. Direction extraction uses single-turn data (captures fundamental geometry),
   training uses multi-turn data (teaches the model to recognize escalation)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    out_dir: str = "runs/multiturn_harc"
    seed: int = 42
    data_dir: str = "data"

    # batch — smaller for MPS 64 GB (two model copies)
    micro_batch_harmful: int = 2
    micro_batch_benign: int = 2
    grad_accum: int = 4

    # optim
    lr: float = 5e-5
    warmup: int = 100
    max_steps: int = 2000

    # lora
    lora_r: int = 16
    lora_alpha: int = 32

    # losses
    coupling_margin: float = 0.5
    lambda_couple: float = 1.0
    lambda_couple_resp: float = 0.0  # response coupling weight (0 = off)
    lambda_kl: float = 10.0
    lambda_ce: float = 1.0
    response_n_tokens: int = 32      # how many response tokens to couple

    # directions
    n_extract: int = 200
    extract_batch_size: int = 4
    extract_max_len: int = 256
    recompute_every: int = 500
    beta_ema: float = 0.3

    # layers
    active_layers: list = field(default_factory=lambda: [21, 22, 23, 24])

    # sequence
    max_context_len: int = 1024
    max_response_len: int = 256


# ── Data ────────────────────────────────────────────────────────────────

@dataclass
class MultiTurnSample:
    category: str          # "harmful" or "benign"
    messages: list[dict]   # conversation context (ends with user)
    response: str          # final assistant response


def load_training_data(data_dir: Path) -> dict[str, list[MultiTurnSample]]:
    out = {"harmful": [], "benign": []}
    for cat in ("harmful", "benign"):
        path = data_dir / f"train_{cat}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} — run prepare_data.py first")
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                out[cat].append(MultiTurnSample(
                    category=cat,
                    messages=rec["messages"],
                    response=rec["response"],
                ))
    print(f"[data] Loaded harmful={len(out['harmful'])}, benign={len(out['benign'])}")
    return out


def load_extract_data(data_dir: Path) -> tuple[list[str], list[str]]:
    path = data_dir / "extract_directions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run prepare_data.py first")
    with path.open() as f:
        d = json.load(f)
    return d["harmful"], d["harmless"]


def infinite_shuffle(items: list, seed: int) -> Iterator:
    rng = random.Random(seed)
    pool = list(items)
    while True:
        rng.shuffle(pool)
        yield from pool


# ── Collation ───────────────────────────────────────────────────────────

@dataclass
class MTBatch:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    response_mask: Tensor
    t_inst: Tensor        # last user-content token position
    t_post: Tensor        # assistant header position
    is_harmful: Tensor
    categories: list[str]


def format_multiturn(tokenizer, messages: list[dict]) -> str:
    """Format multi-turn messages with chat template, adding generation prompt."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def post_inst_offset(tokenizer) -> int:
    """Number of template tokens between last user content and generation start."""
    dummy = "XPLACEHOLDERX7890"
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": dummy}],
        tokenize=False, add_generation_prompt=True,
    )
    full_ids = tokenizer(formatted, add_special_tokens=False).input_ids
    dummy_ids = tokenizer(dummy, add_special_tokens=False).input_ids
    for start in range(len(full_ids) - len(dummy_ids) + 1):
        if full_ids[start:start + len(dummy_ids)] == dummy_ids:
            return len(full_ids) - (start + len(dummy_ids))
    return 3  # fallback for Qwen-style templates


def collate_multiturn(
    batch: list[MultiTurnSample],
    tokenizer,
    max_context_len: int = 1024,
    max_response_len: int = 256,
) -> MTBatch:
    P = post_inst_offset(tokenizer)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    encoded = []
    for s in batch:
        # Tokenize multi-turn context
        context_str = format_multiturn(tokenizer, s.messages)
        saved_side = tokenizer.truncation_side
        try:
            tokenizer.truncation_side = "left"
            context_ids = tokenizer(
                context_str, add_special_tokens=False,
                truncation=True, max_length=max_context_len,
            ).input_ids
        finally:
            tokenizer.truncation_side = saved_side

        # Tokenize response
        resp_ids = tokenizer(
            s.response, add_special_tokens=False,
            truncation=True, max_length=max_response_len,
        ).input_ids + [tokenizer.eos_token_id]

        encoded.append((context_ids, resp_ids, s.category))

    T = max(len(c) + len(r) for c, r, _ in encoded)
    B = len(encoded)

    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    response_mask = torch.zeros((B, T), dtype=torch.long)
    t_inst = torch.zeros(B, dtype=torch.long)
    t_post = torch.zeros(B, dtype=torch.long)
    is_harmful = torch.zeros(B, dtype=torch.bool)
    categories = []

    for i, (ctx_ids, r_ids, cat) in enumerate(encoded):
        clen = len(ctx_ids)
        rlen = len(r_ids)
        seq = ctx_ids + r_ids
        L_seq = clen + rlen
        input_ids[i, :L_seq] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :L_seq] = 1
        response_mask[i, clen:clen + rlen] = 1

        # For harmful: train refusal CE on response tokens
        if cat == "harmful":
            labels[i, clen:clen + rlen] = torch.tensor(r_ids, dtype=torch.long)

        # t_post = last context token (assistant header end)
        # t_inst = last user content token (P tokens before t_post)
        t_post[i] = clen - 1
        t_inst[i] = max(0, clen - 1 - P)
        is_harmful[i] = (cat == "harmful")
        categories.append(cat)

    return MTBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        response_mask=response_mask,
        t_inst=t_inst,
        t_post=t_post,
        is_harmful=is_harmful,
        categories=categories,
    )


# ── Direction extraction (single-turn, same as HARC) ───────────────────

@dataclass
class Directions:
    v_ref: Tensor   # (L+1, H) refusal direction per layer
    v_harm: Tensor  # (L+1, H) harm direction per layer


def format_single(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )


@torch.no_grad()
def extract_directions(
    model, tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    batch_size: int = 4,
    max_length: int = 256,
    device: torch.device = None,
) -> Directions:
    """Difference-of-means direction extraction at t_inst and t_post."""
    if device is None:
        device = next(model.parameters()).device

    is_peft = type(model).__name__.startswith("Peft")
    blocks = (model.base_model.model.model.layers if is_peft
              else model.model.layers)
    n_layers = len(blocks)
    P = post_inst_offset(tokenizer)

    def collect(prompts):
        all_inst = [[] for _ in range(n_layers)]
        all_post = [[] for _ in range(n_layers)]

        # Process one prompt at a time — batched left-padding
        # causes NaN on MPS in deeper layers.
        for prompt in prompts:
            text = format_single(tokenizer, prompt)
            ids = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length,
                add_special_tokens=False,
            ).input_ids.to(device)

            store = {}
            handles = []
            for L in range(n_layers):
                def make_hook(layer_idx):
                    def hook(module, inputs):
                        x = inputs[0]
                        # Single sample, no padding — last token is real
                        store[layer_idx] = (
                            x[:, -1-P, :].detach().float().cpu(),
                            x[:, -1, :].detach().float().cpu(),
                        )
                    return hook
                handles.append(blocks[L].register_forward_pre_hook(make_hook(L)))

            try:
                model(input_ids=ids, use_cache=False)
            finally:
                for h in handles:
                    h.remove()

            for L in range(n_layers):
                all_inst[L].append(store[L][0])
                all_post[L].append(store[L][1])

        inst = [torch.cat(chunks, dim=0) for chunks in all_inst]
        post = [torch.cat(chunks, dim=0) for chunks in all_post]
        return inst, post

    print(f"[extract] Collecting activations for {len(harmful_prompts)} harmful prompts...")
    h_inst, h_post = collect(harmful_prompts)
    print(f"[extract] Collecting activations for {len(harmless_prompts)} harmless prompts...")
    s_inst, s_post = collect(harmless_prompts)

    v_harm_layers = []
    v_ref_layers = []
    for L in range(n_layers):
        diff_inst = h_inst[L].mean(0) - s_inst[L].mean(0)
        diff_post = h_post[L].mean(0) - s_post[L].mean(0)
        # Normalize in float32 to avoid float16 precision issues
        v_harm_layers.append(F.normalize(diff_inst.float(), dim=-1))
        v_ref_layers.append(F.normalize(diff_post.float(), dim=-1))

    return Directions(
        v_ref=torch.stack(v_ref_layers),
        v_harm=torch.stack(v_harm_layers),
    )


# ── Losses (from HARC, simplified) ─────────────────────────────────────

def cos_sim(x: Tensor, v: Tensor) -> Tensor:
    return F.cosine_similarity(x, v.unsqueeze(0).expand_as(x), dim=-1)


def coupling_loss(
    h_inst: dict[int, Tensor],
    h_post: dict[int, Tensor],
    v_ref: Tensor,
    v_harm: Tensor,
    is_harmful: Tensor,
    margin: float,
) -> dict[str, Tensor]:
    """Push harmful above margin on v_ref/v_harm, retain below zero."""
    per_layer = []
    proj_data = {}

    for L in sorted(h_inst.keys()):
        proj_ref = cos_sim(h_post[L], v_ref[L].detach())
        proj_harm = cos_sim(h_inst[L], v_harm[L].detach())

        l_h = torch.relu(margin - proj_ref) + torch.relu(margin - proj_harm)
        l_b = torch.relu(proj_ref) + torch.relu(proj_harm)
        per_sample = torch.where(is_harmful, l_h, l_b)
        per_layer.append(per_sample.mean())

        if is_harmful.any():
            proj_data[f"proj_ref_h_L{L}"] = proj_ref[is_harmful].mean().detach()
            proj_data[f"proj_harm_h_L{L}"] = proj_harm[is_harmful].mean().detach()
        if (~is_harmful).any():
            proj_data[f"proj_ref_b_L{L}"] = proj_ref[~is_harmful].mean().detach()
            proj_data[f"proj_harm_b_L{L}"] = proj_harm[~is_harmful].mean().detach()

    return {"total": torch.stack(per_layer).mean(), **proj_data}


def kl_retain_loss(lora_logits: Tensor, base_logits: Tensor, mask: Tensor) -> Tensor:
    base_logp = F.log_softmax(base_logits.detach().float(), dim=-1)
    lora_logp = F.log_softmax(lora_logits.float(), dim=-1)
    kl = (base_logp.exp() * (base_logp - lora_logp)).sum(-1)
    denom = mask.float().sum().clamp_min(1.0)
    return (kl * mask.float()).sum() / denom


def ce_refusal_loss(logits: Tensor, labels: Tensor) -> Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def response_coupling_loss(
    h_resp: dict[int, Tensor],
    v_ref: Tensor,
    v_harm: Tensor,
    is_harmful: Tensor,
    margin: float,
) -> dict[str, Tensor]:
    """Coupling loss on mean-pooled response-window residuals."""
    per_layer = []
    proj_data = {}

    for L in sorted(h_resp.keys()):
        proj_ref = cos_sim(h_resp[L], v_ref[L].detach())
        proj_harm = cos_sim(h_resp[L], v_harm[L].detach())

        l_h = torch.relu(margin - proj_ref) + torch.relu(margin - proj_harm)
        l_b = torch.relu(proj_ref) + torch.relu(proj_harm)
        per_sample = torch.where(is_harmful, l_h, l_b)
        per_layer.append(per_sample.mean())

        if is_harmful.any():
            proj_data[f"proj_ref_resp_h_L{L}"] = proj_ref[is_harmful].mean().detach()
            proj_data[f"proj_harm_resp_h_L{L}"] = proj_harm[is_harmful].mean().detach()
        if (~is_harmful).any():
            proj_data[f"proj_ref_resp_b_L{L}"] = proj_ref[~is_harmful].mean().detach()
            proj_data[f"proj_harm_resp_b_L{L}"] = proj_harm[~is_harmful].mean().detach()

    return {"total": torch.stack(per_layer).mean(), **proj_data}


# ── Hooked forward ─────────────────────────────────────────────────────

class HookedForward:
    def __init__(self, model, is_peft: bool):
        self.model = model
        self.blocks = (model.base_model.model.model.layers if is_peft
                       else model.model.layers)

    def forward(self, input_ids, attention_mask, capture_layers, t_inst, t_post,
                with_grad, resp_start=None, resp_n=None):
        """Forward with hooks. If resp_start/resp_n given, also captures
        mean-pooled response window activations in store_resp[L]."""
        store = {}
        store_resp = {}
        handles = []
        # Precompute per-sample sequence lengths from attention_mask
        seq_lengths = attention_mask.sum(dim=1) if resp_start is not None else None
        if capture_layers:
            for L in capture_layers:
                def make_hook(layer_idx):
                    def hook(module, inputs):
                        x = inputs[0]
                        B = x.size(0)
                        arange = torch.arange(B, device=x.device)
                        store[layer_idx] = (
                            x[arange, t_inst.to(x.device), :],
                            x[arange, t_post.to(x.device), :],
                        )
                        # Response window: mean-pool first resp_n tokens after resp_start
                        if resp_start is not None and resp_n is not None and resp_n > 0:
                            resp_acts = []
                            for b in range(B):
                                s = resp_start[b].item()
                                # Bound by actual sequence length, not tensor width
                                real_end = seq_lengths[b].item()
                                e = min(s + resp_n, real_end)
                                if e > s:
                                    resp_acts.append(x[b, s:e, :].mean(dim=0))
                                else:
                                    resp_acts.append(x[b, s, :])
                            store_resp[layer_idx] = torch.stack(resp_acts)
                    return hook
                handles.append(self.blocks[L].register_forward_pre_hook(make_hook(L)))

        try:
            ctx = torch.enable_grad() if with_grad else torch.no_grad()
            with ctx:
                out = self.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, output_hidden_states=False,
                )
        finally:
            for h in handles:
                h.remove()
        return out, store, store_resp


# ── EMA blend ───────────────────────────────────────────────────────────

def ema_blend(v_base: Tensor, v_fresh: Tensor, beta: float) -> Tensor:
    blended = (1 - beta) * v_base + beta * v_fresh
    return F.normalize(blended, dim=-1)


# ── Training loop ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="5-step smoke test")
    args = ap.parse_args()

    cfg = Config()
    if args.config:
        with open(args.config) as f:
            user_cfg = yaml.safe_load(f)
        for k, v in user_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.smoke:
        cfg.max_steps = 5
        cfg.recompute_every = 3
        cfg.n_extract = 16

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16  # MPS float16 is more reliable than bf16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"[device] {device}, dtype={dtype}")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── Load tokenizer ──
    print(f"[load] tokenizer: {cfg.model_id}")
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Load data ──
    data_dir = Path(cfg.data_dir)
    train_data = load_training_data(data_dir)
    extract_harmful, extract_harmless = load_extract_data(data_dir)
    extract_harmful = extract_harmful[:cfg.n_extract]
    extract_harmless = extract_harmless[:cfg.n_extract]

    # ── Load single model with LoRA (saves ~14 GB vs two copies) ──
    print(f"[load] model: {cfg.model_id}")
    M_inner = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=dtype,
        device_map={"": device},
    )
    M_inner.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    M_lora = get_peft_model(M_inner, lora_cfg)
    M_lora.print_trainable_parameters()

    # ── Extract initial directions (with adapters disabled = base model) ──
    print("[extract] Initial direction extraction (adapters disabled = base)...")
    M_lora.disable_adapter_layers()
    M_lora.eval()
    M_lora.gradient_checkpointing_disable()
    base_dirs = extract_directions(
        M_lora, tok, extract_harmful, extract_harmless,
        batch_size=cfg.extract_batch_size,
        max_length=cfg.extract_max_len, device=device,
    )
    M_lora.enable_adapter_layers()
    M_lora.gradient_checkpointing_enable()
    M_lora.train()
    cur_dirs = base_dirs

    # Log cos(v_ref, v_harm) at active layers
    cos_rh = (base_dirs.v_ref * base_dirs.v_harm).sum(-1)
    for L in cfg.active_layers:
        print(f"  L{L}: cos(v_ref, v_harm) = {cos_rh[L].item():+.3f}")

    torch.save({
        "v_ref": base_dirs.v_ref.cpu(),
        "v_harm": base_dirs.v_harm.cpu(),
    }, out / "directions_base.pt")

    M_hooked = HookedForward(M_lora, is_peft=True)

    # ── Iterators ──
    iter_harmful = infinite_shuffle(train_data["harmful"], cfg.seed + 1)
    iter_benign = infinite_shuffle(train_data["benign"], cfg.seed + 2)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in M_lora.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    def lr_at(step):
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        return cfg.lr

    log_f = (out / "train_log.jsonl").open("w")
    optimizer.zero_grad()
    accum = 0
    t0 = time.time()

    print(f"\n[train] Starting {cfg.max_steps} steps, batch={cfg.micro_batch_harmful}h+{cfg.micro_batch_benign}b, accum={cfg.grad_accum}")
    print(f"[train] Active layers: {cfg.active_layers}")

    for step in range(cfg.max_steps):
        # ── Recompute directions periodically ──
        if step > 0 and step % cfg.recompute_every == 0:
            print(f"[step {step}] Recomputing directions (EMA β={cfg.beta_ema})...")
            M_lora.eval()
            M_lora.gradient_checkpointing_disable()
            # Extract with LoRA active (current model state)
            fresh_dirs = extract_directions(
                M_lora, tok, extract_harmful, extract_harmless,
                batch_size=cfg.extract_batch_size,
                max_length=cfg.extract_max_len, device=device,
            )
            M_lora.gradient_checkpointing_enable()
            M_lora.train()
            cur_dirs = Directions(
                v_ref=ema_blend(base_dirs.v_ref, fresh_dirs.v_ref, cfg.beta_ema),
                v_harm=ema_blend(base_dirs.v_harm, fresh_dirs.v_harm, cfg.beta_ema),
            )
            cos_drift_ref = (base_dirs.v_ref * cur_dirs.v_ref).sum(-1)
            cos_drift_harm = (base_dirs.v_harm * cur_dirs.v_harm).sum(-1)
            for L in cfg.active_layers:
                print(f"  L{L}: drift ref={cos_drift_ref[L].item():.3f}, "
                      f"harm={cos_drift_harm[L].item():.3f}")
            torch.save({
                "v_ref": cur_dirs.v_ref.cpu(),
                "v_harm": cur_dirs.v_harm.cpu(),
                "step": step,
            }, out / f"directions_step_{step:06d}.pt")

        # ── Sample batch ──
        samples = []
        for _ in range(cfg.micro_batch_harmful):
            samples.append(next(iter_harmful))
        for _ in range(cfg.micro_batch_benign):
            samples.append(next(iter_benign))

        batch = collate_multiturn(
            samples, tok,
            max_context_len=cfg.max_context_len,
            max_response_len=cfg.max_response_len,
        )

        ids = batch.input_ids.to(device)
        mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)
        resp_mask = batch.response_mask.to(device)
        t_inst = batch.t_inst
        t_post = batch.t_post
        is_h = batch.is_harmful.to(device)

        # ── LoRA forward with hooks ──
        resp_start = t_post + 1  # response starts after last context token
        out_lora, store, store_resp = M_hooked.forward(
            ids, mask,
            capture_layers=cfg.active_layers,
            t_inst=t_inst, t_post=t_post,
            with_grad=True,
            resp_start=resp_start,
            resp_n=cfg.response_n_tokens if cfg.lambda_couple_resp > 0 else None,
        )

        # ── Coupling loss ──
        h_inst_dict = {L: store[L][0] for L in cfg.active_layers}
        h_post_dict = {L: store[L][1] for L in cfg.active_layers}
        couple = coupling_loss(
            h_inst_dict, h_post_dict,
            cur_dirs.v_ref.to(device), cur_dirs.v_harm.to(device),
            is_h, cfg.coupling_margin,
        )
        L_couple = couple["total"]

        # ── Response coupling loss (optional) ──
        L_couple_resp = torch.tensor(0.0, device=device)
        if cfg.lambda_couple_resp > 0 and store_resp:
            resp_couple = response_coupling_loss(
                store_resp,
                cur_dirs.v_ref.to(device), cur_dirs.v_harm.to(device),
                is_h, cfg.coupling_margin,
            )
            L_couple_resp = resp_couple["total"]
            # Merge resp projection diagnostics (skip "total" to avoid collision)
            for k, v in resp_couple.items():
                if k != "total":
                    couple[k] = v

        # ── KL retention on benign (disable adapters = base model) ──
        benign_idx = (~is_h).nonzero(as_tuple=True)[0]
        if benign_idx.numel() > 0:
            M_lora.disable_adapter_layers()
            with torch.no_grad():
                out_base = M_lora(
                    input_ids=ids[benign_idx],
                    attention_mask=mask[benign_idx],
                    use_cache=False,
                )
            M_lora.enable_adapter_layers()
            L_kl = kl_retain_loss(
                out_lora.logits[benign_idx],
                out_base.logits,
                resp_mask[benign_idx],
            )
        else:
            L_kl = torch.tensor(0.0, device=device)

        # ── CE refusal on harmful ──
        harmful_idx = is_h.nonzero(as_tuple=True)[0]
        if harmful_idx.numel() > 0:
            L_ce = ce_refusal_loss(out_lora.logits[harmful_idx], labels[harmful_idx])
        else:
            L_ce = torch.tensor(0.0, device=device)

        # ── Total loss ──
        L_total = (cfg.lambda_couple * L_couple
                   + cfg.lambda_couple_resp * L_couple_resp
                   + cfg.lambda_kl * L_kl
                   + cfg.lambda_ce * L_ce)

        (L_total / cfg.grad_accum).backward()
        accum += 1

        if accum == cfg.grad_accum:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step)
            optimizer.step()
            optimizer.zero_grad()
            accum = 0

        # ── Logging ──
        if step % 10 == 0:
            elapsed = time.time() - t0
            rec = {
                "step": step,
                "elapsed": round(elapsed, 1),
                "L_total": round(float(L_total.item()), 4),
                "L_couple": round(float(L_couple.item()), 4),
                "L_couple_resp": round(float(L_couple_resp.item()), 4),
                "L_kl": round(float(L_kl.item()), 4),
                "L_ce": round(float(L_ce.item()), 4),
                "seq_len": int(ids.shape[1]),
            }
            # Add projection diagnostics
            for k, v in couple.items():
                if k != "total" and isinstance(v, Tensor):
                    rec[k] = round(float(v.item()), 4)
            print(json.dumps(rec))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        # ── Checkpoints ──
        if step > 0 and step % 500 == 0:
            ckpt_dir = out / f"checkpoint_{step:06d}"
            M_lora.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            print(f"[step {step}] Saved checkpoint to {ckpt_dir}")

    # ── Final save ──
    log_f.close()
    M_lora.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")
    print(f"\n[done] Trained {cfg.max_steps} steps in {time.time()-t0:.0f}s")
    print(f"[done] Final model saved to {out / 'final'}")


if __name__ == "__main__":
    main()
