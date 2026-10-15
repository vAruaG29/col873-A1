from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
from common import (
    SANITIZER_SYSTEM_PROMPT,
    PLACEHOLDER_RE,
    batched_generate,
    clean_generation,
    dtype_name,
    ensure_parent,
    gpu_name,
    load_causal_lm,
    load_tokenizer,
    normalize_ws,
    read_jsonl,
    render_prompts,
    select_few_shot,
    set_seed,
    write_json,
    write_jsonl,
    zeroshot_system_prompt,
)
from metrics import evaluate_predictions, find_leaks
from spans import recover_spans, spans_to_masked_text

GENERATIVE = {"finetuned", "zeroshot", "fewshot"}


def run_generative(texts: Sequence[str], args) -> Tuple[List[str], List[float], Dict[str, Any]]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.base_model, padding_side="left")
    adapter = args.adapter_path if args.system == "finetuned" else None
    if args.system == "finetuned" and not adapter:
        raise SystemExit("--system finetuned requires --adapter_path")
    model, torch_dtype = load_causal_lm(
        args.base_model, quant=args.quant, dtype=args.dtype, device=device,
        adapter_path=adapter, merge_adapter=args.merge_adapter, for_training=False,
    )

    if args.system == "finetuned":
        system_prompt = SANITIZER_SYSTEM_PROMPT
        few_shot: List[Tuple[str, str]] = []
    else:
        system_prompt = zeroshot_system_prompt()
        few_shot = []
        if args.n_shot > 0:
            train_rows = read_jsonl(args.few_shot_file)
            few_shot = select_few_shot(train_rows, args.n_shot, seed=args.seed)

    prompts = render_prompts(tok, texts, system_prompt, few_shot)
    go = batched_generate(
        model, tok, prompts, batch_size=args.gen_batch_size,
        max_new_tokens=args.max_new_tokens or None, do_sample=False,
        desc=f"{args.system}:generate",
    )
    info = {
        "dtype": dtype_name(torch_dtype),
        "gpu": gpu_name(),
        "gen_wallclock_s": go.wallclock_s,
        "n_input_tokens": go.n_input_tokens,
        "n_new_tokens": go.n_new_tokens,
        "system_prompt": system_prompt,
        "n_shot": args.n_shot if args.system != "finetuned" else 0,
        "merge_adapter": args.merge_adapter,
    }
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return [clean_generation(t) for t in go.texts], list(go.latencies_ms), info


def run_rule_based(texts: Sequence[str], args) -> Tuple[List[str], List[float], Dict[str, Any]]:
    from baselines import PresidioRunner, regex_spans

    runner = None
    if args.system == "presidio":
        runner = PresidioRunner(spacy_model=args.spacy_model)

    outs: List[str] = []
    lats: List[float] = []
    t_all = time.time()
    try:
        from tqdm.auto import tqdm

        it = tqdm(texts, desc=f"{args.system}:spans", unit="doc")
    except Exception:
        it = texts
    for t in it:
        t0 = time.time()
        if runner is not None:
            sp = runner.spans(t, score_threshold=args.presidio_threshold)
        else:
            sp = regex_spans(t, use_url=not args.regex_no_url, use_names=not args.regex_no_names)
        out = spans_to_masked_text(t, sp)
        lats.append((time.time() - t0) * 1000.0)
        outs.append(out)
    info = {"wallclock_s": time.time() - t_all}
    if args.system == "presidio":
        info["spacy_model"] = args.spacy_model
        info["presidio_threshold"] = args.presidio_threshold
    return outs, lats, info


def sanitize_texts(texts: Sequence[str], args) -> Tuple[List[str], List[float], Dict[str, Any]]:
    if args.system == "oracle":
        raise RuntimeError("oracle is handled by the caller (it needs target_text)")
    if args.system in GENERATIVE:
        return run_generative(texts, args)
    return run_rule_based(texts, args)


def build_records(rows, predicted, latencies) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for r, pt, lat in zip(rows, predicted, latencies):
        al = recover_spans(r["source_text"], pt)
        leaks = find_leaks(pt, r["privacy_mask"], min_len=1)
        recs.append(
            {
                "id": r["id"],
                "language": r.get("language", "en"),
                "source_text": r["source_text"],
                "predicted_text": pt,
                "predicted_spans": al.spans,
                "gold_spans": r["privacy_mask"],
                "leaked_values": leaks,
                "latency_ms": round(float(lat), 3),
                "changed_chars": al.changed_chars,
                "removals": al.removals,
                "diag": al.to_diag(),
            }
        )
    return recs


def to_prediction_line(rec: Dict[str, Any], redact: bool) -> Dict[str, Any]:
    leaked = [l["value"] for l in rec["leaked_values"]]
    if redact:
        leaked = [
            (v[:2] + "*" * max(0, len(v) - 3) + v[-1:]) if len(v) > 3 else "*" * len(v)
            for v in leaked
        ]
    def _span(s: Dict[str, Any]) -> Dict[str, Any]:
        return {"start": int(s["start"]), "end": int(s["end"]), "label": s["label"]}

    return {
        "id": rec["id"],
        "language": rec["language"],
        "source_text": rec["source_text"],
        "predicted_text": rec["predicted_text"],
        "predicted_spans": [_span(s) for s in rec["predicted_spans"]],
        "gold_spans": [_span(s) for s in rec["gold_spans"]],
        "leaked_values": leaked,
        "latency_ms": rec["latency_ms"],
        "diag": rec["diag"],
    }


def sanitize_corpus(args) -> int:
    rows = read_jsonl(args.sanitize_corpus, limit=args.limit or None)
    uniq: Dict[str, int] = {}
    order: List[str] = []
    for r in rows:
        t = r["text"]
        if t not in uniq:
            uniq[t] = len(order)
            order.append(t)
    print(f"[sanitize_corpus] {len(rows)} rows, {len(order)} distinct texts", flush=True)

    outs, lats, info = sanitize_texts(order, args)
    mapping = {t: o for t, o in zip(order, outs)}

    out_rows: List[Dict[str, Any]] = []
    n_changed = 0
    chars_in = 0
    chars_out = 0
    for r in rows:
        new = dict(r)
        san = mapping[r["text"]]
        new["text"] = san
        new["text_raw_len"] = len(r["text"])
        if san != r["text"]:
            n_changed += 1
        chars_in += len(r["text"])
        chars_out += len(san)
        out_rows.append(new)
    n_written = write_jsonl(args.output_corpus, out_rows)
    print(f"[sanitize_corpus] wrote {n_written} rows -> {args.output_corpus}", flush=True)

    report: Dict[str, Any] = {
        "system": args.system,
        "adapter": args.adapter_path,
        "source_corpus": args.sanitize_corpus,
        "output_corpus": args.output_corpus,
        "n_rows": len(rows),
        "n_distinct_texts": len(order),
        "n_rows_changed": n_changed,
        "frac_rows_changed": n_changed / max(len(rows), 1),
        "chars_in": chars_in,
        "chars_out": chars_out,
        "mean_placeholders_per_row": sum(len(PLACEHOLDER_RE.findall(mapping[r["text"]])) for r in rows)
        / max(len(rows), 1),
        "runtime": info,
    }

    if args.canaries and os.path.exists(args.canaries):
        cans = read_jsonl(args.canaries)
        by_id = {c["canary_id"]: c for c in cans}
        per_canary: List[Dict[str, Any]] = []
        secret_survived = 0
        text_changed = 0
        seen: set = set()
        for r in rows:
            if not r.get("is_canary") or r.get("canary_id") in seen:
                continue
            cid = r["canary_id"]
            seen.add(cid)
            meta = by_id.get(cid, {})
            san = mapping[r["text"]]
            surv = normalize_ws(str(meta.get("secret", "\x00"))) in normalize_ws(san)
            changed = san != r["text"]
            secret_survived += int(surv)
            text_changed += int(changed)
            per_canary.append(
                {
                    "canary_id": cid,
                    "n_reps": meta.get("n_reps"),
                    "text_changed": changed,
                    "secret_survived_verbatim": surv,
                    "placeholders": PLACEHOLDER_RE.findall(san),
                    "sanitized_text": san,
                }
            )
        report["canary_sanitization"] = {
            "n_canaries": len(per_canary),
            "n_text_changed": text_changed,
            "n_secret_survived_verbatim": secret_survived,
            "secret_survival_rate": secret_survived / max(len(per_canary), 1),
            "per_canary": per_canary,
        }
        print(
            f"[sanitize_corpus] canaries: {text_changed}/{len(per_canary)} texts altered, "
            f"{secret_survived}/{len(per_canary)} secrets survived verbatim",
            flush=True,
        )

    if args.targeted_file and os.path.exists(args.targeted_file):
        tg = read_jsonl(args.targeted_file)
        san_by_doc: Dict[str, str] = {}
        for r in rows:
            san_by_doc.setdefault(r["doc_id"], mapping[r["text"]])
        surv: List[Dict[str, Any]] = []
        n_member = 0
        n_surv = 0
        for t in tg:
            if t["split"] != "member":
                continue
            n_member += 1
            san = san_by_doc.get(t["doc_id"], "")
            alive = normalize_ws(t["gold_value"]) in normalize_ws(san)
            n_surv += int(alive)
            if alive:
                surv.append({"id": t["id"], "label": t["label"], "doc_id": t["doc_id"]})
        report["targeted_value_survival"] = {
            "n_member_prompts": n_member,
            "n_gold_values_surviving": n_surv,
            "survival_rate": n_surv / max(n_member, 1),
            "surviving": surv[:200],
        }
        print(
            f"[sanitize_corpus] targeted gold values surviving sanitization: "
            f"{n_surv}/{n_member}",
            flush=True,
        )

    if args.report_file:
        write_json(args.report_file, report)
        print(f"[sanitize_corpus] report -> {args.report_file}", flush=True)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Part B: sanitizer evaluation + Part C.2 corpus sanitization")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter_path", default="")
    p.add_argument("--test_file", default="data/partB_test.jsonl")
    p.add_argument(
        "--system",
        choices=["finetuned", "regex", "presidio", "zeroshot", "fewshot", "oracle"],
        default="finetuned",
    )
    p.add_argument("--output_predictions", default="results/partB_predictions.jsonl")
    p.add_argument("--report_file", default="results/partB_metrics.json")

    p.add_argument("--test_set_name", default="", help="label written into the report")
    p.add_argument("--limit", type=int, default=0, help="debug: evaluate only the first N rows")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=0, help="0 = adaptive per batch")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--merge_adapter", action="store_true", default=True,
                   help="fold LoRA into the base weights for faster generation")
    p.add_argument("--no_merge_adapter", dest="merge_adapter", action="store_false")
    p.add_argument("--n_shot", type=int, default=0, help="demos for the zeroshot/fewshot baseline")
    p.add_argument("--few_shot_file", default="data/partA_train.jsonl",
                   help="demos are drawn from TRAIN, never from a test split")

    p.add_argument("--spacy_model", default="en_core_web_lg")
    p.add_argument("--presidio_threshold", type=float, default=0.35)
    p.add_argument("--regex_no_url", action="store_true")
    p.add_argument("--regex_no_names", action="store_true")

    p.add_argument("--redact_predictions", action="store_true",
                   help="mask leaked values in the predictions file")
    p.add_argument("--per_class_top", type=int, default=10)

    p.add_argument("--sanitize_corpus", default="", help="raw corpus JSONL to sanitize")
    p.add_argument("--output_corpus", default="data/partC_corpus_sanitized.jsonl")
    p.add_argument("--canaries", default="data/partC_canaries.jsonl")
    p.add_argument("--targeted_file", default="data/partC_targeted.jsonl")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    if args.system == "fewshot" and args.n_shot == 0:
        args.n_shot = 5

    if args.sanitize_corpus:
        return sanitize_corpus(args)

    rows = read_jsonl(args.test_file, limit=args.limit or None)
    print(f"[{args.system}] {len(rows)} rows from {args.test_file}", flush=True)

    t0 = time.time()
    if args.system == "oracle":
        predicted = [r["target_text"] for r in rows]
        lats = [0.0] * len(rows)
        info = {"note": "gold masked text scored through the same alignment pipeline"}
    else:
        predicted, lats, info = sanitize_texts([r["source_text"] for r in rows], args)
    total_s = time.time() - t0

    recs = build_records(rows, predicted, lats)
    m = evaluate_predictions(recs, per_class_top=args.per_class_top)

    report: Dict[str, Any] = {
        "system": args.system,
        "adapter": args.adapter_path or None,
        "base_model": args.base_model,
        "test_file": args.test_file,
        "test_set": args.test_set_name or os.path.splitext(os.path.basename(args.test_file))[0],
        "n": m["n"],
        "timestamp": common.now_stamp(),
        "total_wallclock_s": total_s,
        "docs_per_s": len(rows) / max(total_s, 1e-9),
        "runtime": info,
        "seed": args.seed,
    }
    report.update(m)

    if args.output_predictions:
        write_jsonl(
            args.output_predictions,
            [to_prediction_line(r, args.redact_predictions) for r in recs],
        )
        print(f"[{args.system}] predictions -> {args.output_predictions}", flush=True)
    if args.report_file:
        write_json(args.report_file, report)
        print(f"[{args.system}] metrics -> {args.report_file}", flush=True)

    ex = m["exact"]
    ov = m["overlap_iou50"]
    ag = m["overlap_iou50_label_agnostic"]
    print(
        f"\n=== {args.system} on {report['test_set']} (n={m['n']}) ===\n"
        f"  exact        P={ex['micro_p']:.4f} R={ex['micro_r']:.4f} F1={ex['micro_f1']:.4f} macroF1={ex['macro_f1']:.4f}\n"
        f"  IoU>=0.5     P={ov['micro_p']:.4f} R={ov['micro_r']:.4f} F1={ov['micro_f1']:.4f} macroF1={ov['macro_f1']:.4f}\n"
        f"  label-free   P={ag['micro_p']:.4f} R={ag['micro_r']:.4f} F1={ag['micro_f1']:.4f}\n"
        f"  leak_rate={m['leak_rate']:.4f}  (len>=3: {m['leak_rate_len_ge_3']:.4f}, doc-level: {m['doc_level_leak_rate']:.4f})\n"
        f"  over_masking={m['over_masking_rate']:.4f}  (non-space {m['over_masking_rate_nonspace']:.4f})\n"
        f"  well_formed={m['well_formed_rate']:.4f}   {total_s:.1f}s total\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
