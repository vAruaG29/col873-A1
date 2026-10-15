from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
from common import (
    append_csv_row,
    attach_lora,
    corpus_perplexity,
    cuda_peak_gb,
    cuda_reset_peak,
    dtype_name,
    ensure_parent,
    get_model_device,
    gpu_name,
    load_causal_lm,
    load_tokenizer,
    read_jsonl,
    set_seed,
    write_json,
)

VICTIM_CSV_FIELDS = [
    "victim_name", "corpus", "n_rows", "epochs", "lora_r", "lora_alpha",
    "target_modules", "lora_dropout", "quant", "trainable_params", "trainable_pct",
    "peak_gpu_mem_gb", "train_wallclock_s", "final_train_loss",
    "heldout_ppl_base", "heldout_ppl_victim", "train_ppl_victim", "memorization_gap",
]


def build_examples(rows, tok, max_seq_len: int) -> Tuple[List[List[int]], Dict[str, Any]]:
    eos = tok.eos_token_id
    out: List[List[int]] = []
    n_trunc = 0
    lens: List[int] = []
    for r in rows:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids = ids + [eos]
        lens.append(len(ids))
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            n_trunc += 1
        if len(ids) < 2:
            continue
        out.append(ids)
    lens.sort()
    return out, {
        "n_examples": len(out),
        "n_truncated": n_trunc,
        "tok_len_p50": lens[len(lens) // 2] if lens else 0,
        "tok_len_p95": lens[int(0.95 * (len(lens) - 1))] if lens else 0,
        "tok_len_max": lens[-1] if lens else 0,
        "total_tokens": sum(lens),
    }


def collate(batch: Sequence[List[int]], pad_id: int):
    import torch

    L = max(len(x) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids in batch:
        pad = L - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(list(ids) + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def make_batches(examples, batch_size: int, seed: int) -> List[List[int]]:
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    window = batch_size * 32
    batches: List[List[int]] = []
    for w0 in range(0, len(idx), window):
        chunk = idx[w0 : w0 + window]
        chunk.sort(key=lambda i: len(examples[i]))
        for b0 in range(0, len(chunk), batch_size):
            batches.append(chunk[b0 : b0 + batch_size])
    random.Random(seed + 1).shuffle(batches)
    return batches


def train(model, tok, examples, args, torch_dtype, heldout_texts, eval_fn) -> Dict[str, Any]:
    import torch
    from transformers import get_cosine_schedule_with_warmup

    device = get_model_device(model)
    pad_id = tok.pad_token_id
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        if p.dtype != torch.float32:
            p.data = p.data.float()

    opt = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
    )
    per_epoch = math.ceil(len(examples) / args.batch_size)
    updates_per_epoch = math.ceil(per_epoch / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    sched = get_cosine_schedule_with_warmup(
        opt, int(args.warmup_ratio * total_updates), total_updates
    )
    use_amp = torch_dtype in (torch.bfloat16, torch.float16) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=(torch_dtype == torch.float16 and use_amp))

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    ensure_parent(log_path)
    log_fh = open(log_path, "w", encoding="utf-8")

    n_tokens = 0
    step = 0
    all_losses: List[float] = []
    epoch_ppl: List[Dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        batches = make_batches(examples, args.batch_size, args.seed + epoch)
        opt.zero_grad(set_to_none=True)
        for bi, bidx in enumerate(batches):
            input_ids, labels, attn = collate([examples[i] for i in bidx], pad_id)
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
            n_tokens += int(attn.sum().item())
            with torch.autocast("cuda", dtype=torch_dtype, enabled=use_amp):
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss ep{epoch} batch{bi}; skipping")
                opt.zero_grad(set_to_none=True)
                continue
            scaled = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            all_losses.append(float(loss.item()))

            if ((bi + 1) % args.grad_accum == 0) or bi == len(batches) - 1:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    el = time.time() - t0
                    win = all_losses[-args.log_every * args.grad_accum :]
                    rec = {
                        "epoch": epoch, "step": step, "total_steps": total_updates,
                        "loss": sum(win) / max(len(win), 1),
                        "lr": sched.get_last_lr()[0],
                        "tokens_per_s": n_tokens / max(el, 1e-6), "elapsed_s": el,
                    }
                    log_fh.write(json.dumps(rec) + "\n")
                    log_fh.flush()
                    print(
                        f"[{args.victim_name}] ep{epoch} step {step}/{total_updates} "
                        f"loss {rec['loss']:.4f} {rec['tokens_per_s']:.0f} tok/s {el/60:.1f}min",
                        flush=True,
                    )

        if args.eval_every_epoch and heldout_texts:
            ppl = eval_fn(model)
            epoch_ppl.append({"epoch": epoch, **ppl})
            print(f"[{args.victim_name}] end of epoch {epoch}: heldout ppl {ppl['ppl']:.3f}", flush=True)
            model.train()

    wallclock = time.time() - t0
    log_fh.close()
    tail = all_losses[-(50 * args.grad_accum) :] or [float("nan")]
    return {
        "train_wallclock_s": wallclock,
        "train_tokens": n_tokens,
        "tokens_per_s": n_tokens / max(wallclock, 1e-6),
        "final_train_loss": sum(tail) / len(tail),
        "optimizer_steps": step,
        "epoch_heldout_ppl": epoch_ppl,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Part C.2: train a victim model")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="MUST be a base LM, not -Instruct")
    p.add_argument("--corpus", default="data/partC_corpus_raw.jsonl")
    p.add_argument("--heldout", default="data/partC_heldout.jsonl")
    p.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="adapters/victim_raw")
    p.add_argument("--victim_name", default="", help="defaults to basename(output_dir)")
    p.add_argument("--report_file", default="", help="defaults to results/partC_victim_<name>.json")
    p.add_argument("--results_csv", default="results/partC_victims.csv")

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--gradient_checkpointing", action="store_true")

    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--eval_every_epoch", action="store_true", default=True)
    p.add_argument("--no_eval_every_epoch", dest="eval_every_epoch", action="store_false")
    p.add_argument("--train_ppl_sample", type=int, default=600,
                   help="training docs sampled for the train-perplexity / memorization gap")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.limit, args.epochs, args.log_every, args.train_ppl_sample = 128, 1, 1, 32
    args.victim_name = args.victim_name or os.path.basename(args.output_dir.rstrip("/"))
    args.report_file = args.report_file or f"results/partC_victim_{args.victim_name}.json"

    import torch

    set_seed(args.seed)
    cuda_reset_peak()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    if "instruct" in args.model.lower():
        print("[warn] the victim must be a BASE LM; got a model whose name says instruct",
              flush=True)

    print(f"=== victim {args.victim_name}: {args.model} on {args.corpus} ===", flush=True)
    tok = load_tokenizer(args.model, padding_side="right")
    model, torch_dtype = load_causal_lm(
        args.model, quant=args.quant, dtype=args.dtype, device=device,
        for_training=True, gradient_checkpointing=args.gradient_checkpointing,
    )

    corpus = read_jsonl(args.corpus, limit=args.limit or None)
    heldout = read_jsonl(args.heldout) if args.heldout else []
    heldout_texts = [r["text"] for r in heldout]
    print(f"    corpus rows={len(corpus)}  heldout={len(heldout_texts)}", flush=True)

    def eval_heldout(m):
        m.eval()
        with torch.no_grad():
            return corpus_perplexity(
                m, tok, heldout_texts, batch_size=args.eval_batch_size,
                max_len=args.max_seq_len, add_eos=True, progress=False,
            )

    base_ppl: Dict[str, Any] = {}
    if heldout_texts:
        base_ppl = eval_heldout(model)
        print(f"    pre-trained held-out ppl = {base_ppl['ppl']:.3f}", flush=True)

    model, pstats = attach_lora(
        model, target_modules, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    print(f"    dtype={dtype_name(torch_dtype)} trainable={pstats['trainable_params']:,} "
          f"({pstats['trainable_pct']:.4f}%)", flush=True)

    examples, dstats = build_examples(corpus, tok, args.max_seq_len)
    print(f"    examples={dstats['n_examples']} tokens={dstats['total_tokens']:,} "
          f"len p50={dstats['tok_len_p50']} max={dstats['tok_len_max']} "
          f"truncated={dstats['n_truncated']}", flush=True)

    cuda_reset_peak()
    tr = train(model, tok, examples, args, torch_dtype, heldout_texts, eval_heldout)
    peak = cuda_peak_gb()
    print(f"    trained in {tr['train_wallclock_s']/60:.1f} min, peak {peak:.2f} GiB", flush=True)

    model.save_pretrained(args.output_dir)
    card = os.path.join(args.output_dir, "README.md")
    if os.path.exists(card):
        os.remove(card)
    print(f"    adapter -> {args.output_dir}", flush=True)

    victim_ppl = eval_heldout(model) if heldout_texts else {}

    train_ppl: Dict[str, Any] = {}
    by_reps: Dict[str, Any] = {}
    if args.train_ppl_sample > 0:
        rng = random.Random(args.seed)
        seen: Dict[str, Dict[str, Any]] = {}
        for r in corpus:
            seen.setdefault(r["doc_id"], r)
        docs = list(seen.values())
        sample = rng.sample(docs, k=min(args.train_ppl_sample, len(docs)))
        train_ppl = corpus_perplexity(
            model, tok, [r["text"] for r in sample], batch_size=args.eval_batch_size,
            max_len=args.max_seq_len, add_eos=True, progress=False,
        )
        groups: Dict[int, List[str]] = collections.defaultdict(list)
        for r in sample:
            groups[int(r["n_reps"])].append(r["text"])
        for k in sorted(groups):
            by_reps[str(k)] = corpus_perplexity(
                model, tok, groups[k], batch_size=args.eval_batch_size,
                max_len=args.max_seq_len, add_eos=True, progress=False,
            )
        print(f"    train-sample ppl = {train_ppl['ppl']:.3f} "
              f"(held-out {victim_ppl.get('ppl', float('nan')):.3f})", flush=True)
        for k, v in by_reps.items():
            print(f"      n_reps={k:>3s}: train ppl {v['ppl']:.3f} (n={v['n_docs']})", flush=True)

    mem_gap = (
        victim_ppl.get("ppl", float("nan")) - train_ppl.get("ppl", float("nan"))
        if train_ppl and victim_ppl else float("nan")
    )

    report = {
        "victim": args.victim_name,
        "model": args.model,
        "corpus": args.corpus,
        "n_rows": len(corpus),
        "timestamp": common.now_stamp(),
        "gpu": gpu_name(),
        "dtype": dtype_name(torch_dtype),
        "args": vars(args),
        "param_stats": pstats,
        "data_stats": dstats,
        "train_stats": tr,
        "peak_gpu_mem_gb": peak,
        "heldout_ppl_base": base_ppl,
        "heldout_ppl_victim": victim_ppl,
        "train_ppl_victim": train_ppl,
        "train_ppl_by_n_reps": by_reps,
        "memorization_gap_heldout_minus_train": mem_gap,
    }
    write_json(args.report_file, report)
    print(f"    report -> {args.report_file}", flush=True)

    row = {
        "victim_name": args.victim_name,
        "corpus": args.corpus,
        "n_rows": len(corpus),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": "+".join(target_modules),
        "lora_dropout": args.lora_dropout,
        "quant": args.quant,
        "trainable_params": pstats["trainable_params"],
        "trainable_pct": round(pstats["trainable_pct"], 6),
        "peak_gpu_mem_gb": round(peak, 4),
        "train_wallclock_s": round(tr["train_wallclock_s"], 2),
        "final_train_loss": round(tr["final_train_loss"], 6),
        "heldout_ppl_base": round(base_ppl.get("ppl", float("nan")), 4),
        "heldout_ppl_victim": round(victim_ppl.get("ppl", float("nan")), 4),
        "train_ppl_victim": round(train_ppl.get("ppl", float("nan")), 4) if train_ppl else "",
        "memorization_gap": round(mem_gap, 4) if mem_gap == mem_gap else "",
    }
    append_csv_row(args.results_csv, VICTIM_CSV_FIELDS, row)
    print(f"=== victim {args.victim_name} done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
