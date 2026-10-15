from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
from common import (
    append_csv_row,
    SANITIZER_SYSTEM_PROMPT,
    assistant_eos,
    attach_lora,
    batched_generate,
    build_sanitizer_messages,
    clean_generation,
    cuda_peak_gb,
    cuda_reset_peak,
    disable_generation_defaults,
    dtype_name,
    ensure_parent,
    get_model_device,
    gpu_name,
    load_causal_lm,
    load_tokenizer,
    token_logprobs,
    read_jsonl,
    render_prompts,
    set_seed,
    write_json,
    write_jsonl,
)
from metrics import evaluate_predictions
from spans import recover_spans

CSV_FIELDS = [
    "run_id", "method", "target_modules", "lora_r", "lora_alpha", "quant",
    "trainable_params", "total_params", "trainable_pct", "peak_gpu_mem_gb",
    "train_wallclock_s", "tokens_per_s", "final_train_loss", "val_loss",
    "val_entity_f1", "val_leak_rate",
]


def build_examples(
    rows: Sequence[Dict[str, Any]], tok, max_seq_len: int, system_prompt: str
) -> Tuple[List[Tuple[List[int], List[int]]], Dict[str, Any]]:
    eos = assistant_eos(tok)
    out: List[Tuple[List[int], List[int]]] = []
    n_trunc = 0
    n_no_target = 0
    lens: List[int] = []
    for r in rows:
        prompt = tok.apply_chat_template(
            build_sanitizer_messages(r["source_text"], system_prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["target_text"] + eos, add_special_tokens=False)["input_ids"]
        ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + list(c_ids)
        lens.append(len(ids))
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            labels = labels[:max_seq_len]
            n_trunc += 1
        if all(l == -100 for l in labels):
            n_no_target += 1
            continue
        out.append((ids, labels))
    lens.sort()
    stats = {
        "n_examples": len(out),
        "n_truncated": n_trunc,
        "n_dropped_no_target": n_no_target,
        "tok_len_p50": lens[len(lens) // 2] if lens else 0,
        "tok_len_p95": lens[int(0.95 * (len(lens) - 1))] if lens else 0,
        "tok_len_max": lens[-1] if lens else 0,
    }
    return out, stats


def collate(
    batch: Sequence[Tuple[List[int], List[int]]], pad_id: int
):
    import torch

    L = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = L - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def length_sorted_batches(examples, batch_size: int, shuffle_seed: Optional[int] = None):
    import random

    idx = list(range(len(examples)))
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(idx)
    window = batch_size * 32
    batches: List[List[int]] = []
    for w0 in range(0, len(idx), window):
        chunk = idx[w0 : w0 + window]
        chunk.sort(key=lambda i: len(examples[i][0]))
        for b0 in range(0, len(chunk), batch_size):
            batches.append(chunk[b0 : b0 + batch_size])
    if shuffle_seed is not None:
        random.Random(shuffle_seed + 1).shuffle(batches)
    return batches


def train(
    model,
    tok,
    examples,
    args,
    torch_dtype,
) -> Dict[str, Any]:
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
    per_epoch_batches = math.ceil(len(examples) / args.batch_size)
    updates_per_epoch = math.ceil(per_epoch_batches / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    sched = get_cosine_schedule_with_warmup(
        opt, int(args.warmup_ratio * total_updates), total_updates
    )

    use_amp = torch_dtype in (torch.bfloat16, torch.float16) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=(torch_dtype == torch.float16 and use_amp))

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    ensure_parent(log_path)
    log_fh = open(log_path, "w", encoding="utf-8")

    model.train()
    n_tokens = 0
    step = 0
    recent: List[float] = []
    all_losses: List[float] = []
    t0 = time.time()

    for epoch in range(args.epochs):
        batches = length_sorted_batches(examples, args.batch_size, shuffle_seed=args.seed + epoch)
        opt.zero_grad(set_to_none=True)
        for bi, batch_idx in enumerate(batches):
            input_ids, labels, attn = collate([examples[i] for i in batch_idx], pad_id)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = attn.to(device)
            n_tokens += int(attn.sum().item())

            with torch.autocast("cuda", dtype=torch_dtype, enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at epoch {epoch} batch {bi}; skipping")
                opt.zero_grad(set_to_none=True)
                continue

            scaled = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            recent.append(float(loss.item()))
            all_losses.append(float(loss.item()))

            is_last = bi == len(batches) - 1
            if ((bi + 1) % args.grad_accum == 0) or is_last:
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
                    win = recent[-args.log_every * args.grad_accum :]
                    rec = {
                        "epoch": epoch,
                        "step": step,
                        "total_steps": total_updates,
                        "loss": sum(win) / max(len(win), 1),
                        "lr": sched.get_last_lr()[0],
                        "tokens": n_tokens,
                        "tokens_per_s": n_tokens / max(el, 1e-6),
                        "elapsed_s": el,
                    }
                    log_fh.write(json.dumps(rec) + "\n")
                    log_fh.flush()
                    print(
                        f"[{args.run_id}] ep{epoch} step {step}/{total_updates} "
                        f"loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                        f"{rec['tokens_per_s']:.0f} tok/s  {el/60:.1f}min",
                        flush=True,
                    )

    wallclock = time.time() - t0
    log_fh.close()
    tail = all_losses[-(50 * args.grad_accum) :] or [float("nan")]
    return {
        "train_wallclock_s": wallclock,
        "train_tokens": n_tokens,
        "tokens_per_s": n_tokens / max(wallclock, 1e-6),
        "final_train_loss": sum(tail) / len(tail),
        "train_loss_last_batch": all_losses[-1] if all_losses else float("nan"),
        "optimizer_steps": step,
        "total_updates_planned": total_updates,
    }


def eval_loss(model, tok, examples, batch_size: int, torch_dtype) -> Dict[str, float]:
    import torch

    model.eval()
    pad_id = tok.pad_token_id
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    tot_loss = 0.0
    tot_tok = 0
    use_amp = torch_dtype in (torch.bfloat16, torch.float16) and torch.cuda.is_available()
    for b0 in range(0, len(order), batch_size):
        batch_idx = order[b0 : b0 + batch_size]
        input_ids, labels, attn = collate([examples[i] for i in batch_idx], pad_id)
        device = get_model_device(model)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attn = attn.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch_dtype, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            tl = labels[:, 1:]
            mask = tl != -100
            if mask.any():
                lp = token_logprobs(logits[:, :-1, :], tl.clamp(min=0), chunk=2)
                tot_loss += float((-lp * mask).sum().item())
                tot_tok += int(mask.sum().item())
            del logits
    return {
        "val_loss": tot_loss / max(tot_tok, 1),
        "val_ppl": math.exp(min(tot_loss / max(tot_tok, 1), 50.0)),
        "val_tokens": tot_tok,
    }


def eval_generation(
    model, tok, val_rows, args, system_prompt: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    subset = val_rows[: args.val_gen_n] if args.val_gen_n > 0 else val_rows
    prompts = render_prompts(tok, [r["source_text"] for r in subset], system_prompt)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    disable_generation_defaults(model)
    go = batched_generate(
        model, tok, prompts, batch_size=args.gen_batch_size,
        max_new_tokens=None, do_sample=False, desc=f"{args.run_id}:val-gen",
    )
    recs: List[Dict[str, Any]] = []
    for r, txt, lat in zip(subset, go.texts, go.latencies_ms):
        pt = clean_generation(txt)
        al = recover_spans(r["source_text"], pt)
        recs.append(
            {
                "id": r["id"],
                "source_text": r["source_text"],
                "predicted_text": pt,
                "predicted_spans": al.spans,
                "gold_spans": r["privacy_mask"],
                "changed_chars": al.changed_chars,
                "removals": al.removals,
                "diag": al.to_diag(),
                "latency_ms": lat,
            }
        )
    m = evaluate_predictions(recs)
    m["n_val_generated"] = len(subset)
    m["gen_wallclock_s"] = go.wallclock_s
    return m, recs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Part A: LoRA/QLoRA sweep for the PII sanitizer")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train_file", default="data/partA_train.jsonl")
    p.add_argument("--val_file", default="data/partA_validation.jsonl")
    p.add_argument(
        "--target_modules",
        default="q_proj,v_proj",
        help="comma-separated module suffixes, e.g. q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_id", default="A0")
    p.add_argument("--output_dir", default="adapters/A0")
    p.add_argument("--results_csv", default="results/partA_sweep.csv")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8, help="per-device micro-batch")
    p.add_argument("--grad_accum", type=int, default=4, help="effective batch = batch_size*grad_accum")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_seq_len", type=int, default=768)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="off by default so peak-memory numbers are comparable across runs")

    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--val_gen_n", type=int, default=300,
                   help="fixed prefix of the validation file used for generative scoring")
    p.add_argument("--skip_val_gen", action="store_true")
    p.add_argument("--save_val_predictions", default="",
                   help="optional path for the validation generations")

    p.add_argument("--limit_train", type=int, default=0, help="debug: cap training rows")
    p.add_argument("--limit_val", type=int, default=0, help="debug: cap validation rows")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run: 64 train / 32 val / 8 generations")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.limit_train, args.limit_val, args.val_gen_n = 64, 32, 8
        args.log_every = 1
    if args.quant == "4bit" and not args.gradient_checkpointing:
        print("[note] enabling gradient checkpointing for --quant 4bit "
              "(required to fit QLoRA on 40GB at this batch/seq length)", flush=True)
        args.gradient_checkpointing = True

    import torch
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    set_seed(args.seed)
    cuda_reset_peak()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    method = "QLoRA" if args.quant == "4bit" else "LoRA"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== {args.run_id}: {method} r={args.lora_r} alpha={args.lora_alpha} "
          f"modules={target_modules} quant={args.quant} ===", flush=True)
    print(f"    gpu={gpu_name()} seed={args.seed}", flush=True)

    tok = load_tokenizer(args.model, padding_side="right")
    model, torch_dtype = load_causal_lm(
        args.model, quant=args.quant, dtype=args.dtype, device=device,
        for_training=True, gradient_checkpointing=args.gradient_checkpointing,
    )
    model, pstats = attach_lora(
        model, target_modules, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    print(f"    dtype={dtype_name(torch_dtype)} trainable={pstats['trainable_params']:,} "
          f"({pstats['trainable_pct']:.4f}% of {pstats['total_params']:,})", flush=True)
    if args.gradient_checkpointing:
        print("    gradient_checkpointing=ON (QLoRA / 4-bit)", flush=True)

    train_rows = read_jsonl(args.train_file, limit=args.limit_train or None)
    val_rows = read_jsonl(args.val_file, limit=args.limit_val or None)
    print(f"    train={len(train_rows)} val={len(val_rows)}", flush=True)

    train_ex, tstats = build_examples(train_rows, tok, args.max_seq_len, SANITIZER_SYSTEM_PROMPT)
    val_ex, vstats = build_examples(val_rows, tok, args.max_seq_len, SANITIZER_SYSTEM_PROMPT)
    print(f"    token lens: p50={tstats['tok_len_p50']} p95={tstats['tok_len_p95']} "
          f"max={tstats['tok_len_max']} truncated={tstats['n_truncated']}", flush=True)

    tr = train(model, tok, train_ex, args, torch_dtype)
    peak_gb = cuda_peak_gb()
    print(f"    trained in {tr['train_wallclock_s']/60:.1f} min, "
          f"{tr['tokens_per_s']:.0f} tok/s, peak {peak_gb:.2f} GiB", flush=True)

    model.save_pretrained(args.output_dir)
    card = os.path.join(args.output_dir, "README.md")
    if os.path.exists(card):
        os.remove(card)
    print(f"    adapter saved to {args.output_dir}", flush=True)

    vl = eval_loss(model, tok, val_ex, args.eval_batch_size, torch_dtype)
    print(f"    val_loss={vl['val_loss']:.4f} (ppl {vl['val_ppl']:.2f})", flush=True)

    gen_metrics: Dict[str, Any] = {}
    if not args.skip_val_gen:
        gen_metrics, recs = eval_generation(model, tok, val_rows, args, SANITIZER_SYSTEM_PROMPT)
        print(f"    val_entity_f1(IoU>=.5,label)={gen_metrics['overlap_iou50']['micro_f1']:.4f} "
              f"exact_f1={gen_metrics['exact']['micro_f1']:.4f} "
              f"leak_rate={gen_metrics['leak_rate']:.4f} "
              f"over_mask={gen_metrics['over_masking_rate']:.4f} "
              f"well_formed={gen_metrics['well_formed_rate']:.4f}", flush=True)
        if args.save_val_predictions:
            write_jsonl(
                args.save_val_predictions,
                [
                    {
                        "id": r["id"], "source_text": r["source_text"],
                        "predicted_text": r["predicted_text"],
                        "predicted_spans": r["predicted_spans"],
                        "gold_spans": r["gold_spans"],
                        "diag": r["diag"], "latency_ms": r["latency_ms"],
                    }
                    for r in recs
                ],
            )

    row = {
        "run_id": args.run_id,
        "method": method,
        "target_modules": "+".join(target_modules),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quant": args.quant,
        "trainable_params": pstats["trainable_params"],
        "total_params": pstats["total_params"],
        "trainable_pct": round(pstats["trainable_pct"], 6),
        "peak_gpu_mem_gb": round(peak_gb, 4),
        "train_wallclock_s": round(tr["train_wallclock_s"], 2),
        "tokens_per_s": round(tr["tokens_per_s"], 2),
        "final_train_loss": round(tr["final_train_loss"], 6),
        "val_loss": round(vl["val_loss"], 6),
        "val_entity_f1": round(gen_metrics.get("overlap_iou50", {}).get("micro_f1", float("nan")), 6)
        if gen_metrics else "",
        "val_leak_rate": round(gen_metrics.get("leak_rate", float("nan")), 6) if gen_metrics else "",
    }
    append_csv_row(args.results_csv, CSV_FIELDS, row)
    print(f"    appended row to {args.results_csv}", flush=True)

    meta = {
        "run_id": args.run_id,
        "timestamp": common.now_stamp(),
        "args": vars(args),
        "gpu": gpu_name(),
        "dtype": dtype_name(torch_dtype),
        "param_stats": pstats,
        "train_stats": tr,
        "data_stats": {"train": tstats, "val": vstats},
        "peak_gpu_mem_gb": peak_gb,
        "peak_gpu_mem_reserved_gb": (
            torch.cuda.max_memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else 0.0
        ),
        "val_loss": vl,
        "val_generation_metrics": gen_metrics,
        "csv_row": row,
        "system_prompt": SANITIZER_SYSTEM_PROMPT,
    }
    write_json(os.path.join(args.output_dir, "run_meta.json"), meta)
    print(f"=== {args.run_id} done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
