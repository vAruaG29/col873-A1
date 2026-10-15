from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

if not os.environ.get("CUDA_VISIBLE_DEVICES") and os.environ.get("GPU", "") != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_parent(path)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str, obj: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


PII_LABELS: Tuple[str, ...] = (
    "BOD", "BUILDING", "CARDISSUER", "CITY", "COUNTRY", "DATE", "DRIVERLICENSE",
    "EMAIL", "GEOCOORD", "GIVENNAME1", "GIVENNAME2", "IDCARD", "IP", "LASTNAME1",
    "LASTNAME2", "LASTNAME3", "PASS", "PASSPORT", "POSTCODE", "SECADDRESS",
    "SEX", "SOCIALNUMBER", "STATE", "STREET", "TEL", "TIME", "TITLE", "USERNAME",
)

LABEL_GLOSS: Dict[str, str] = {
    "BOD": "date of birth",
    "BUILDING": "building name or number",
    "CARDISSUER": "payment card issuer / brand",
    "CITY": "city",
    "COUNTRY": "country",
    "DATE": "any other date",
    "DRIVERLICENSE": "driver licence number",
    "EMAIL": "email address",
    "GEOCOORD": "geographic coordinates",
    "GIVENNAME1": "first given name mentioned",
    "GIVENNAME2": "second given name mentioned",
    "IDCARD": "identification / card number",
    "IP": "IPv4 or IPv6 address",
    "LASTNAME1": "first surname mentioned",
    "LASTNAME2": "second surname mentioned",
    "LASTNAME3": "third surname mentioned",
    "PASS": "password or passcode",
    "PASSPORT": "passport number",
    "POSTCODE": "postal / ZIP code",
    "SECADDRESS": "secondary address (apt, suite, unit)",
    "SEX": "sex or gender",
    "SOCIALNUMBER": "social security / national insurance number",
    "STATE": "state, province or region",
    "STREET": "street name",
    "TEL": "telephone number",
    "TIME": "time of day",
    "TITLE": "personal title (Mr., Dr., ...)",
    "USERNAME": "username or handle",
}

PLACEHOLDER_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")


SANITIZER_SYSTEM_PROMPT = (
    "You are a PII sanitizer. Rewrite the user's text, replacing every span of "
    "personally identifiable information with a placeholder of the form [LABEL], "
    "where LABEL is the PII type. Copy every other character exactly, including "
    "whitespace, punctuation and line breaks. Output only the rewritten text."
)

def zeroshot_system_prompt() -> str:
    lines = [
        "You are a PII sanitizer. Rewrite the user's text, replacing every span of "
        "personally identifiable information with a placeholder of the form [LABEL].",
        "",
        "Use exactly these labels:",
    ]
    for lab in PII_LABELS:
        lines.append(f"- [{lab}]: {LABEL_GLOSS[lab]}")
    lines += [
        "",
        "Rules:",
        "1. Copy every non-PII character exactly, including whitespace, punctuation, "
        "line breaks and JSON/XML structure.",
        "2. Replace only the PII value itself, never the surrounding field name or label.",
        "3. Do not add explanations, comments or code fences.",
        "4. Output only the rewritten text.",
    ]
    return "\n".join(lines)


def build_sanitizer_messages(
    source_text: str,
    system_prompt: str = SANITIZER_SYSTEM_PROMPT,
    few_shot: Sequence[Tuple[str, str]] = (),
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for src, tgt in few_shot:
        msgs.append({"role": "user", "content": src})
        msgs.append({"role": "assistant", "content": tgt})
    msgs.append({"role": "user", "content": source_text})
    return msgs


def select_few_shot(
    train_rows: Sequence[Dict[str, Any]], n_shot: int, seed: int = 42
) -> List[Tuple[str, str]]:
    if n_shot <= 0:
        return []
    cands = [
        r
        for r in train_rows
        if 2 <= len(r.get("privacy_mask", [])) <= 5 and len(r["source_text"]) <= 260
    ]
    cands.sort(key=lambda r: (len(r["source_text"]), r["id"]))
    pool = cands[: max(n_shot * 8, n_shot)]
    rng = random.Random(seed)
    picked = rng.sample(pool, k=min(n_shot, len(pool)))
    picked.sort(key=lambda r: r["id"])
    return [(r["source_text"], r["target_text"]) for r in picked]


def assistant_eos(tok) -> str:
    if "<|im_end|>" in tok.get_vocab():
        return "<|im_end|>"
    return tok.eos_token or ""


def pick_dtype(requested: str = "auto"):
    import torch

    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if not torch.cuda.is_available():
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def load_tokenizer(model_name: str, padding_side: str = "right"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_ADAPTERS_FILE = os.path.join(_REPO_ROOT, "hf_adapters.json")


def hub_adapter_config() -> Dict[str, Any]:
    """Contents of hf_adapters.json, or {} if it is missing or still a placeholder."""
    try:
        with open(HF_ADAPTERS_FILE, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return {}
    repo = str(cfg.get("repo_id", ""))
    if repo.count("/") != 1 or "<" in repo:
        return {}
    return cfg


def resolve_adapter(path: str) -> Tuple[str, Dict[str, Any]]:
    """Map an ``--adapter_path`` to ``(model_id, hub_kwargs)`` for PEFT.

    ``adapters/`` is git-ignored, so a fresh clone has no adapter folders. The
    published copies live in one Hugging Face repo (named in hf_adapters.json)
    with one subfolder per adapter, mirroring ``adapters/``. Resolution order:

      1. a local folder always wins, so a machine that trained its own
         adapters keeps using them;
      2. ``adapters/<name>`` that is not on disk becomes ``<repo_id>`` with
         ``subfolder=<name>``, and PEFT downloads it on first use, so the
         handout's commands (``--adapter_path adapters/best_sanitizer``) work
         unchanged on a fresh clone with no manual download step;
      3. anything else, e.g. an explicit ``user/repo``, goes to PEFT unchanged.
    """
    if os.path.isdir(path):
        return path, {}
    norm = os.path.normpath(path)
    if os.path.basename(os.path.dirname(norm)) != "adapters":
        return path, {}
    cfg = hub_adapter_config()
    if not cfg:
        raise FileNotFoundError(
            f"adapter '{path}' is not on disk and hf_adapters.json names no Hub "
            f"repo to fetch it from ({HF_ADAPTERS_FILE})"
        )
    name = os.path.basename(norm)
    kwargs: Dict[str, Any] = {"subfolder": name}
    if cfg.get("revision"):
        kwargs["revision"] = cfg["revision"]
    print(f"[adapter] {path} not on disk -> {cfg['repo_id']} (subfolder {name}) "
          f"from the Hugging Face Hub", flush=True)
    return cfg["repo_id"], kwargs


def load_causal_lm(
    model_name: str,
    quant: str = "none",
    dtype: str = "auto",
    device: str = "cuda",
    adapter_path: Optional[str] = None,
    merge_adapter: bool = False,
    for_training: bool = False,
    gradient_checkpointing: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = pick_dtype(dtype)
    kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}

    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        kwargs["device_map"] = {"": 0} if device.startswith("cuda") else {"": device}

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if quant != "4bit":
        model = model.to(device)

    if quant == "4bit" and for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

    if adapter_path:
        from peft import PeftModel

        model_id, hub_kwargs = resolve_adapter(adapter_path)
        model = PeftModel.from_pretrained(model, model_id, **hub_kwargs)
        if merge_adapter and quant != "4bit":
            model = model.merge_and_unload()

    if not for_training:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    if gradient_checkpointing and for_training:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    disable_generation_defaults(model)
    return model, torch_dtype


def get_model_device(model):
    dev = getattr(model, "device", None)
    if dev is not None:
        try:
            if hasattr(dev, "type"):
                return dev
        except Exception:
            pass
    for p in model.parameters():
        return p.device
    import torch

    return torch.device("cpu")


def disable_generation_defaults(model) -> None:
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    gc.do_sample = False
    gc.temperature = 1.0
    gc.top_p = 1.0
    gc.top_k = 0
    if hasattr(gc, "typical_p"):
        gc.typical_p = 1.0
    gc.repetition_penalty = 1.0
    gc.no_repeat_ngram_size = 0


def generate_safe_kwargs(
    tok,
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_new_tokens: Optional[int] = None,
    num_return_sequences: int = 1,
    eos_token_id=None,
    **extra,
) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        pad_token_id=tok.pad_token_id,
        repetition_penalty=1.0,
        use_cache=True,
        num_return_sequences=int(num_return_sequences),
    )
    eos = eos_token_id if eos_token_id is not None else tok.eos_token_id
    if eos is not None:
        kw["eos_token_id"] = eos
    if min_new_tokens is not None:
        kw["min_new_tokens"] = int(min_new_tokens)
    if do_sample:
        kw["temperature"] = float(temperature)
        kw["top_p"] = float(top_p)
        if top_k:
            kw["top_k"] = int(top_k)
    kw.update(extra)
    return kw


def attach_lora(
    model,
    target_modules: Sequence[str],
    r: int,
    alpha: int,
    dropout: float = 0.05,
):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    model = get_peft_model(model, cfg)
    disable_generation_defaults(model)
    model.print_trainable_parameters()
    try:
        trainable, total = model.get_nb_trainable_parameters()
    except Exception:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    stats = {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }
    return model, stats


def cuda_peak_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def cuda_reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


@dataclass
class GenOutput:
    texts: List[str]
    latencies_ms: List[float]
    n_input_tokens: int = 0
    n_new_tokens: int = 0
    wallclock_s: float = 0.0


def _sorted_batches(lengths: Sequence[int], batch_size: int) -> List[List[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def batched_generate(
    model,
    tok,
    prompts: Sequence[str],
    batch_size: int = 16,
    max_new_tokens: Optional[int] = None,
    len_multiplier: float = 1.6,
    len_pad: int = 48,
    max_new_cap: int = 768,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    stop_at_eos: bool = True,
    progress: bool = True,
    desc: str = "generate",
) -> GenOutput:
    import torch

    prev_side = tok.padding_side
    tok.padding_side = "left"
    enc_all = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    lengths = [len(x) for x in enc_all]
    batches = _sorted_batches(lengths, batch_size)

    texts: List[List[str]] = [[] for _ in prompts]
    lat = [0.0 for _ in prompts]
    n_in = 0
    n_new = 0
    t_start = time.time()

    iterator = batches
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(batches, desc=desc, unit="batch")
        except Exception:
            pass

    eos_ids = None
    if stop_at_eos:
        eos_ids = getattr(model.generation_config, "eos_token_id", None) or tok.eos_token_id

    for batch in iterator:
        blen = max(lengths[i] for i in batch)
        budget = max_new_tokens or min(max_new_cap, int(len_multiplier * blen) + len_pad)
        enc = tok.pad(
            {"input_ids": [enc_all[i] for i in batch]},
            padding=True,
            return_tensors="pt",
        )
        device = get_model_device(model)
        enc = {k: v.to(device) for k, v in enc.items()}
        t0 = time.time()
        with torch.no_grad():
            gen_kwargs = generate_safe_kwargs(
                tok,
                max_new_tokens=budget,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                eos_token_id=eos_ids,
            )
            out = model.generate(**enc, **gen_kwargs)
        dt = (time.time() - t0) * 1000.0
        in_len = enc["input_ids"].shape[1]
        new_tok = out[:, in_len:]
        n_in += int(enc["attention_mask"].sum().item())
        n_new += int((new_tok != tok.pad_token_id).sum().item())
        decoded = tok.batch_decode(new_tok, skip_special_tokens=True)
        for k, idx in enumerate(batch):
            chunk = decoded[k * num_return_sequences : (k + 1) * num_return_sequences]
            texts[idx] = list(chunk)
            lat[idx] = dt / max(len(batch), 1)

    tok.padding_side = prev_side
    flat = [t[0] if t else "" for t in texts] if num_return_sequences == 1 else texts
    return GenOutput(
        texts=flat,
        latencies_ms=lat,
        n_input_tokens=n_in,
        n_new_tokens=n_new,
        wallclock_s=time.time() - t_start,
    )


def render_prompts(
    tok,
    sources: Sequence[str],
    system_prompt: str = SANITIZER_SYSTEM_PROMPT,
    few_shot: Sequence[Tuple[str, str]] = (),
) -> List[str]:
    return [
        tok.apply_chat_template(
            build_sanitizer_messages(s, system_prompt, few_shot),
            tokenize=False,
            add_generation_prompt=True,
        )
        for s in sources
    ]


def clean_generation(text: str) -> str:
    t = text
    fence = re.match(r"^\s*```[a-zA-Z0-9_+-]*\n(.*?)\n?```\s*$", t, flags=re.S)
    if fence:
        t = fence.group(1)
    return t


def token_logprobs(logits, targets, chunk: int = 8):
    import torch

    out = torch.empty(targets.shape, dtype=torch.float32, device=targets.device)
    for i in range(0, logits.shape[0], max(1, chunk)):
        lg = logits[i : i + chunk].float()
        lp = torch.log_softmax(lg, dim=-1)
        out[i : i + chunk] = lp.gather(-1, targets[i : i + chunk].unsqueeze(-1)).squeeze(-1)
        del lg, lp
    return out


def score_texts(
    model,
    tok,
    texts: Sequence[str],
    batch_size: int = 16,
    max_len: int = 1024,
    add_eos: bool = False,
    progress: bool = False,
    desc: str = "score",
    logit_chunk: int = 8,
) -> Tuple[List[float], List[int]]:
    import torch

    prev_side = tok.padding_side
    tok.padding_side = "right"
    ids = []
    for t in texts:
        x = tok(t, add_special_tokens=False)["input_ids"][:max_len]
        if add_eos and tok.eos_token_id is not None:
            x = (x + [tok.eos_token_id])[:max_len]
        if len(x) < 2:
            x = x + [tok.eos_token_id if tok.eos_token_id is not None else 0]
        ids.append(x)

    lengths = [len(x) for x in ids]
    batches = _sorted_batches(lengths, batch_size)
    sums = [0.0] * len(texts)
    counts = [0] * len(texts)

    iterator = batches
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(batches, desc=desc, unit="batch")
        except Exception:
            pass

    for batch in iterator:
        enc = tok.pad({"input_ids": [ids[i] for i in batch]}, padding=True, return_tensors="pt")
        device = get_model_device(model)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            tgt = input_ids[:, 1:]
            tok_lp = token_logprobs(logits[:, :-1, :], tgt, chunk=logit_chunk)
            del logits
        mask = attn[:, 1:].to(tok_lp.dtype)
        s = (tok_lp * mask).sum(dim=1)
        c = mask.sum(dim=1)
        for k, i in enumerate(batch):
            sums[i] = float(s[k].item())
            counts[i] = int(c[k].item())

    tok.padding_side = prev_side
    return sums, counts


def corpus_perplexity(
    model, tok, texts: Sequence[str], batch_size: int = 16, max_len: int = 1024,
    add_eos: bool = True, progress: bool = True,
) -> Dict[str, float]:
    sums, counts = score_texts(
        model, tok, texts, batch_size=batch_size, max_len=max_len,
        add_eos=add_eos, progress=progress, desc="ppl",
    )
    tot_lp = sum(sums)
    tot_n = sum(counts)
    mean_nll = -tot_lp / max(tot_n, 1)
    per_doc = [
        -s / max(c, 1) for s, c in zip(sums, counts)
    ]
    return {
        "ppl": float(math.exp(min(mean_nll, 50.0))),
        "mean_nll": float(mean_nll),
        "n_docs": len(texts),
        "n_tokens": int(tot_n),
        "mean_doc_ppl": float(sum(math.exp(min(x, 50.0)) for x in per_doc) / max(len(per_doc), 1)),
    }


def log_perplexity_from_score(sum_logprob: float, n_tokens: int) -> float:
    return -sum_logprob / max(n_tokens, 1)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def git_rev() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def append_csv_row(path: str, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    import csv
    import fcntl

    ensure_parent(path)
    with open(path, "a+", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0, os.SEEK_END)
            new_file = fh.tell() == 0
            w = csv.DictWriter(fh, fieldnames=list(fieldnames))
            if new_file:
                w.writeheader()
            w.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
