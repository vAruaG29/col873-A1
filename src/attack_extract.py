from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import os
import random
import re
import sys
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
from common import (
    corpus_perplexity,
    dtype_name,
    generate_safe_kwargs,
    get_model_device,
    gpu_name,
    load_causal_lm,
    load_tokenizer,
    log_perplexity_from_score,
    normalize_ws,
    read_jsonl,
    score_texts,
    set_seed,
    write_json,
    write_jsonl,
)


def _progress(it, desc: str, total: Optional[int] = None):
    try:
        from tqdm.auto import tqdm

        return tqdm(it, desc=desc, total=total, unit="batch")
    except Exception:
        return it


def edit_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def zlib_entropy(text: str) -> float:
    return float(len(zlib.compress(text.encode("utf-8"))))


def load_pii_index(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return out
    for row in read_jsonl(path):
        raw = row.get("value", row.get("pii", row.get("text", row.get("gold_value", ""))))
        key = normalize_ws(str(raw))
        if key:
            out[key] = row
    return out


def build_fallback_pii_index(
    corpus: Sequence[Dict[str, Any]],
    targeted: Sequence[Dict[str, Any]],
    canaries: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for t in targeted:
        if t.get("split") != "member":
            continue
        key = normalize_ws(str(t.get("gold_value", "")))
        if key:
            out[key] = {"value": t["gold_value"], "label": t.get("label"), "doc_id": t.get("doc_id"), "source": "targeted"}
    for c in canaries:
        key = normalize_ws(str(c.get("secret", "")))
        if key:
            out[key] = {"value": str(c["secret"]), "label": "CANARY", "doc_id": c.get("canary_id"), "source": "canary"}
    seen_text: Set[str] = set()
    for r in corpus:
        text = r.get("text", "")
        if text in seen_text:
            continue
        seen_text.add(text)
        for lab, val in extract_pii_strings(text):
            key = normalize_ws(val)
            out.setdefault(key, {"value": val, "label": lab, "doc_id": r.get("doc_id"), "source": "pattern"})
    return out


class CorpusIndex:
    def __init__(self, docs: Sequence[Tuple[str, str]], shingle: int = 25, stride: int = 10):
        self.shingle = shingle
        self.docs: Dict[str, str] = {}
        self.index: Dict[str, List[str]] = collections.defaultdict(list)
        seen_text: Dict[str, str] = {}
        for doc_id, text in docs:
            if text in seen_text:
                continue
            seen_text[text] = doc_id
            self.docs[doc_id] = text
            for i in range(0, max(0, len(text) - shingle + 1), stride):
                self.index[text[i : i + shingle]].append(doc_id)

    def longest_match(self, gen: str) -> Tuple[int, Optional[str]]:
        if len(gen) < self.shingle:
            return 0, None
        cands: Set[str] = set()
        for i in range(0, len(gen) - self.shingle + 1):
            hit = self.index.get(gen[i : i + self.shingle])
            if hit:
                cands.update(hit)
        best = 0
        best_doc: Optional[str] = None
        for d in cands:
            t = self.docs[d]
            sm = difflib.SequenceMatcher(None, gen, t, autojunk=False)
            m = sm.find_longest_match(0, len(gen), 0, len(t))
            if m.size > best:
                best = m.size
                best_doc = d
        return best, best_doc


def greedy_continue(
    model, tok, prefixes: Sequence[List[int]], max_new_tokens: int, batch_size: int, desc: str
) -> List[List[int]]:
    import torch

    out: List[List[int]] = [[] for _ in prefixes]
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    pad = tok.pad_token_id
    for batch in _progress(batches, desc):
        L = max(len(prefixes[i]) for i in batch)
        input_ids = torch.tensor(
            [[pad] * (L - len(prefixes[i])) + prefixes[i] for i in batch], dtype=torch.long
        ).to(get_model_device(model))
        attn = torch.tensor(
            [[0] * (L - len(prefixes[i])) + [1] * len(prefixes[i]) for i in batch], dtype=torch.long
        ).to(get_model_device(model))
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                **generate_safe_kwargs(
                    tok,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                ),
            )
        new = gen[:, L:].tolist()
        for k, i in enumerate(batch):
            out[i] = new[k]
    return out


def sample_documents(
    model, tok, n: int, max_new_tokens: int, batch_size: int, temperature: float,
    top_k: int, top_p: float, min_new_tokens: int, seed: int, desc: str = "untargeted:sample",
) -> List[str]:
    import torch

    torch.manual_seed(seed)
    start = tok.eos_token_id if tok.eos_token_id is not None else 0
    outs: List[str] = []
    todo = n
    pbar_total = math.ceil(n / batch_size)
    batches = []
    while todo > 0:
        b = min(batch_size, todo)
        batches.append(b)
        todo -= b
    for b in _progress(batches, desc, total=pbar_total):
        input_ids = torch.full((b, 1), start, dtype=torch.long, device=get_model_device(model))
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                **generate_safe_kwargs(
                    tok,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ),
            )
        outs.extend(tok.batch_decode(gen[:, 1:], skip_special_tokens=True))
    return outs[:n]


def attack_discoverable(model, tok, corpus, args) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    docs: Dict[str, Dict[str, Any]] = {}
    for r in corpus:
        docs.setdefault(r["doc_id"], r)

    groups: Dict[Tuple[bool, int], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in docs.values():
        groups[(bool(r.get("is_canary")), int(r["n_reps"]))].append(r)

    selected: List[Dict[str, Any]] = []
    plan: Dict[Tuple[bool, int], int] = {}
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        pool = sorted(groups[key], key=lambda r: r["doc_id"])
        take = pool if args.disc_per_group <= 0 or len(pool) <= args.disc_per_group else rng.sample(
            pool, args.disc_per_group
        )
        plan[key] = len(take)
        selected.extend(take)
    print(f"[discoverable] {len(selected)} documents "
          f"({args.prefix_tokens}-token prefix -> {args.suffix_tokens}-token continuation)", flush=True)

    items: List[Dict[str, Any]] = []
    n_excluded_short = 0
    for r in selected:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        n = len(ids)
        if n < 4:
            n_excluded_short += 1
            continue
        if n < args.prefix_tokens + args.min_suffix_tokens:
            pre_n = max(1, n // 2)
            pre = ids[:pre_n]
            suf = ids[pre_n:]
        else:
            pre = ids[: args.prefix_tokens]
            suf = ids[args.prefix_tokens : args.prefix_tokens + args.suffix_tokens]
        items.append(
            {
                "doc_id": r["doc_id"],
                "is_canary": bool(r.get("is_canary")),
                "n_reps": int(r["n_reps"]),
                "prefix_ids": pre,
                "gold_ids": suf,
                "gold_text": tok.decode(suf, skip_special_tokens=True),
            }
        )
    gens = greedy_continue(
        model, tok, [it["prefix_ids"] for it in items], args.suffix_tokens,
        args.gen_batch_size, "discoverable"
    )

    per_group: Dict[str, Dict[str, Any]] = {}
    canary_group: Dict[str, Dict[str, Any]] = {}
    recs: List[Dict[str, Any]] = []
    all_exact: List[int] = []
    all_sim: List[float] = []
    for it, g in zip(items, gens):
        gen_text = tok.decode(g[: len(it["gold_ids"])], skip_special_tokens=True)
        exact = int(gen_text.strip() == it["gold_text"].strip())
        tok_exact = int(g[: len(it["gold_ids"])] == it["gold_ids"])
        sim = edit_sim(gen_text, it["gold_text"])
        all_exact.append(exact)
        all_sim.append(sim)
        bucket = canary_group if it["is_canary"] else per_group
        key = str(it["n_reps"])
        b = bucket.setdefault(key, {"n": 0, "exact": 0, "tok_exact": 0, "sim": 0.0})
        b["n"] += 1
        b["exact"] += exact
        b["tok_exact"] += tok_exact
        b["sim"] += sim
        recs.append(
            {
                "doc_id": it["doc_id"], "n_reps": it["n_reps"], "is_canary": it["is_canary"],
                "exact": exact, "edit_sim": sim,
                "gold_text": it["gold_text"], "generated": gen_text,
            }
        )

    def finish(d: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in sorted(d, key=lambda x: int(x)):
            v = d[k]
            out[k] = {
                "exact_match_rate": v["exact"] / max(v["n"], 1),
                "token_exact_match_rate": v["tok_exact"] / max(v["n"], 1),
                "mean_edit_sim": v["sim"] / max(v["n"], 1),
                "n": v["n"],
            }
        return out

    return {
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "n_documents": len(items),
        "n_excluded_short": n_excluded_short,
        "exact_match_rate": sum(all_exact) / max(len(all_exact), 1),
        "mean_edit_sim": sum(all_sim) / max(len(all_sim), 1),
        "by_n_reps": finish(per_group),
        "canary_by_n_reps": finish(canary_group),
        "_records": recs,
    }


PII_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("EMAIL", re.compile(r"[^\s@<>\"',;:()\[\]]+@[^\s@<>\"',;:()\[\]]+\.[A-Za-z]{2,}")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("IPV6", re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b")),
    ("TEL", re.compile(r"\+\d[\d\s().\-]{5,20}\d")),
    ("SSN", re.compile(r"\b\d{3}[-.]\d{2,3}[-.]\d{4}\b")),
    ("LONGID", re.compile(r"\b\d{9,19}\b")),
    ("IDCARD", re.compile(r"\b[A-Z]{2}\d{5}[A-Z]{2}\b")),
]


def extract_pii_strings(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for lab, rx in PII_PATTERNS:
        for m in rx.finditer(text):
            v = m.group(0).strip()
            if len(v) >= 6:
                out.append((lab, v))
    return out


def attack_untargeted(model, tok, ref_model, corpus, args, extras: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    samples = sample_documents(
        model, tok, args.num_samples, args.sample_max_new_tokens, args.sample_batch_size,
        args.temperature, args.top_k, args.top_p, args.sample_min_new_tokens, args.seed,
    )
    uniq: List[str] = []
    seen: Set[str] = set()
    for s in samples:
        key = normalize_ws(s)
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    print(f"[untargeted] {len(samples)} samples, {len(uniq)} unique "
          f"({time.time()-t0:.0f}s)", flush=True)

    v_sum, v_cnt = score_texts(model, tok, uniq, batch_size=args.score_batch_size,
                               max_len=args.max_seq_len, progress=True, desc="score:victim")
    lower = [s.lower() for s in uniq]
    l_sum, l_cnt = score_texts(model, tok, lower, batch_size=args.score_batch_size,
                               max_len=args.max_seq_len, progress=True, desc="score:lower")
    if ref_model is not None:
        r_sum, r_cnt = score_texts(ref_model, tok, uniq, batch_size=args.score_batch_size,
                                   max_len=args.max_seq_len, progress=True, desc="score:ref")
    else:
        r_sum, r_cnt = v_sum, v_cnt

    scores: List[Dict[str, Any]] = []
    for i, s in enumerate(uniq):
        lp_v = log_perplexity_from_score(v_sum[i], v_cnt[i])
        lp_l = log_perplexity_from_score(l_sum[i], l_cnt[i])
        lp_r = log_perplexity_from_score(r_sum[i], r_cnt[i])
        z = zlib_entropy(s)
        scores.append(
            {
                "idx": i,
                "ppl": lp_v,
                "ratio_ref": lp_v / lp_r if lp_r > 0 else float("inf"),
                "ratio_lower": lp_v / lp_l if lp_l > 0 else float("inf"),
                "ratio_zlib": lp_v / z if z > 0 else float("inf"),
                "n_tokens": v_cnt[i],
            }
        )

    index: CorpusIndex = extras["index"]
    verified: List[Tuple[int, Optional[str]]] = []
    for s in _progress(uniq, "untargeted:verify"):
        verified.append(index.longest_match(s))
    is_mem = [int(v[0] >= args.mem_min_chars) for v in verified]

    prec: Dict[str, float] = {}
    top_lists: Dict[str, List[int]] = {}
    for metric in ("ppl", "ratio_ref", "ratio_lower", "ratio_zlib"):
        ranked = sorted(scores, key=lambda d: (d[metric], d["idx"]))
        top = [d["idx"] for d in ranked[: args.precision_at]]
        top_lists[metric] = top
        prec[metric] = sum(is_mem[i] for i in top) / max(len(top), 1)

    docs_recovered = {v[1] for v, m in zip(verified, is_mem) if m and v[1]}

    corpus_norm: str = extras["corpus_norm"]
    pii_found: Dict[str, Set[str]] = collections.defaultdict(set)
    for s in uniq:
        for lab, v in extract_pii_strings(s):
            if normalize_ws(v) in corpus_norm:
                pii_found[lab].add(v)
    n_pii = sum(len(v) for v in pii_found.values())

    pii_index: Dict[str, Dict[str, Any]] = extras.get("pii_index") or {}
    recovered_index: Set[str] = set()
    if pii_index:
        for s in uniq:
            hay = normalize_ws(s)
            for key in pii_index:
                if key and len(key) >= 3 and key in hay:
                    recovered_index.add(key)
    distinct_pii = len(recovered_index) if pii_index else n_pii
    pii_den = len(pii_index) if pii_index else 0

    targeted_vals: Dict[str, str] = extras.get("targeted_values", {})
    hit_targeted = [
        tid for tid, val in targeted_vals.items()
        if any(normalize_ws(val) in normalize_ws(s) for s in uniq)
    ] if targeted_vals else []
    canary_secrets: Dict[str, str] = extras.get("canary_secrets", {})
    hit_canary = [
        cid for cid, sec in canary_secrets.items()
        if any(sec in s for s in uniq)
    ] if canary_secrets else []

    out = {
        "n_generated": len(samples),
        "n_unique": len(uniq),
        "sampling": {
            "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
            "max_new_tokens": args.sample_max_new_tokens,
        },
        "verification": {
            "criterion": f"longest common substring >= {args.mem_min_chars} chars",
            "n_verified_memorized": int(sum(is_mem)),
            "verified_rate": sum(is_mem) / max(len(uniq), 1),
            "mean_longest_match_chars": sum(v[0] for v in verified) / max(len(verified), 1),
            "max_longest_match_chars": max([v[0] for v in verified] or [0]),
        },
        "precision_at_100": prec,
        "precision_at_k": args.precision_at,
        "distinct_docs_recovered": len(docs_recovered),
        "doc_recovery_rate": len(docs_recovered) / max(len(index.docs), 1),
        "distinct_pii_values_recovered": distinct_pii,
        "pii_recovery_rate": (distinct_pii / pii_den) if pii_den else 0.0,
        "pii_index_size": len(pii_index),
        "verified_pii_strings": n_pii,
        "verified_pii_by_type": {k: len(v) for k, v in sorted(pii_found.items())},
        "pii_recovery": {
            "targeted_gold_values_recovered": len(hit_targeted),
            "targeted_gold_values_total": len(targeted_vals),
            "targeted_recovery_rate": len(hit_targeted) / max(len(targeted_vals), 1),
            "canary_secrets_recovered": len(hit_canary),
            "canary_secrets_total": len(canary_secrets),
        },
        "wallclock_s": time.time() - t0,
    }
    if args.dump_generations:
        out["_generations"] = [
            {
                "text": uniq[d["idx"]],
                "ppl": d["ppl"], "ratio_ref": d["ratio_ref"],
                "ratio_lower": d["ratio_lower"], "ratio_zlib": d["ratio_zlib"],
                "longest_match_chars": verified[d["idx"]][0],
                "matched_doc": verified[d["idx"]][1],
                "verified_memorized": bool(is_mem[d["idx"]]),
            }
            for d in sorted(scores, key=lambda x: (x["ppl"], x["idx"]))[: args.precision_at]
        ]
    return out


def _canary_digit_plan(tok, c):
    if "{secret}" not in c.get("template", ""):
        return None
    secret = str(c["secret"])
    head = c["template"].split("{secret}")[0].format(project=c.get("project", ""))
    full = c["template"].format(project=c.get("project", ""), secret=secret)
    hid = tok(head, add_special_tokens=False)["input_ids"]
    fid = tok(full, add_special_tokens=False)["input_ids"]
    dig = [tok(str(d), add_special_tokens=False)["input_ids"] for d in range(10)]
    if any(len(v) != 1 for v in dig):
        return None
    dig = [v[0] for v in dig]
    w = len(secret)
    if fid[: len(hid)] != hid:
        return None
    if fid[len(hid): len(hid) + w] != [dig[int(ch)] for ch in secret]:
        return None
    return hid, dig, w


def attack_canary_exact(model, tok, canaries, args) -> Optional[Dict[str, Any]]:
    import torch

    plans = [_canary_digit_plan(tok, c) for c in canaries]
    if any(p is None for p in plans):
        print("[canary] exact enumeration unavailable (tokenization check failed); "
              "using the sampled estimator", flush=True)
        return None

    device = get_model_device(model)
    B = args.canary_exact_batch
    per_canary: List[Dict[str, Any]] = []
    t0 = time.time()

    for c, plan in zip(canaries, plans):
        hid, dig, w = plan
        R = 10 ** w
        n_prefix = 10 ** (w - 1)
        L = len(hid)
        base = torch.tensor(hid, device=device)
        DT = torch.tensor(dig, device=device)
        nll = torch.empty(R, device=device, dtype=torch.float32)

        with torch.no_grad():
            for s0 in range(0, n_prefix, B):
                n = min(B, n_prefix - s0)
                idx = torch.arange(s0, s0 + n, device=device)
                d = torch.stack(
                    [(idx // 10 ** (w - 2 - i)) % 10 for i in range(w - 1)], 1
                )
                seq = torch.cat([base.unsqueeze(0).expand(n, -1), DT[d]], 1)
                lg = model(seq).logits[:, L - 1: L + w - 1, :].float()
                lp = lg.index_select(-1, DT) - torch.logsumexp(lg, -1, keepdim=True)
                chain = lp[:, : w - 1, :].gather(2, d.unsqueeze(-1)).squeeze(-1).sum(1)
                nll[s0 * 10: (s0 + n) * 10] = -(chain.unsqueeze(1) + lp[:, w - 1, :]).reshape(-1)

        t_idx = int(str(c["secret"]))
        true_nll = float(nll[t_idx].item())
        rank = int((nll < nll[t_idx]).sum().item()) + 1
        exposure = math.log2(R) - math.log2(rank)
        alt = nll.clone()
        alt[t_idx] = float("inf")
        best_alt = float(alt.min().item())
        per_canary.append(
            {
                "canary_id": c["canary_id"],
                "n_reps": int(c["n_reps"]),
                "rank": rank,
                "exposure": exposure,
                "secret_nll": true_nll,
                "best_alt_nll": best_alt,
                "margin_bits": (best_alt - true_nll) / math.log(2),
                "mean_alt_nll": float(nll.mean().item()),
            }
        )
        print(f"[canary-exact] {c['canary_id']} n_reps={c['n_reps']:>2} "
              f"rank={rank}/{R} exposure={exposure:.2f} margin={per_canary[-1]['margin_bits']:.1f}b",
              flush=True)
        del nll, alt

    by_reps: Dict[str, Dict[str, Any]] = {}
    groups: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for pc in per_canary:
        groups[pc["n_reps"]].append(pc)
    for k in sorted(groups):
        g = groups[k]
        exps = [x["exposure"] for x in g]
        by_reps[str(k)] = {
            "mean_exposure": sum(exps) / len(exps),
            "min_exposure": min(exps),
            "max_exposure": max(exps),
            "mean_rank": sum(x["rank"] for x in g) / len(g),
            "mean_margin_bits": sum(x["margin_bits"] for x in g) / len(g),
            "mean_secret_nll": sum(x["secret_nll"] for x in g) / len(g),
            "n_rank1": sum(int(x["rank"] == 1) for x in g),
            "n": len(g),
        }
    exposures = [x["exposure"] for x in per_canary]
    R = 10 ** plans[0][2]
    return {
        "method": "exact_enumeration",
        "n_canaries": len(per_canary),
        "mean_exposure": sum(exposures) / len(exposures),
        "space_size": R,
        "max_exposure": math.log2(R),
        "n_rank1": sum(int(x["rank"] == 1) for x in per_canary),
        "by_n_reps": by_reps,
        "per_canary": per_canary,
        "wallclock_s": time.time() - t0,
    }


def attack_canary(model, tok, canaries, args) -> Dict[str, Any]:
    R = args.canary_space
    n_alt = args.canary_num_alternatives
    width = len(str(R - 1))
    per_canary: List[Dict[str, Any]] = []
    t0 = time.time()

    for c in canaries:
        secret = str(c["secret"])
        rng = random.Random(f"{args.seed}:{c['canary_id']}")
        alts: Set[str] = set()
        while len(alts) < n_alt:
            cand = str(rng.randrange(R)).zfill(width)
            if cand != secret:
                alts.add(cand)
        cands = [secret] + sorted(alts)
        texts = [c["template"].format(project=c["project"], secret=s) for s in cands]
        sums, cnts = score_texts(
            model, tok, texts, batch_size=args.canary_batch_size,
            max_len=args.max_seq_len, progress=True, desc=f"canary:{c['canary_id']}",
        )
        logppl = [-s for s in sums]
        true_score = logppl[0]
        n_better = sum(1 for x in logppl[1:] if x < true_score)
        rank_in_sample = 1 + n_better
        rank_hat = (rank_in_sample / (n_alt + 1)) * R
        exposure = math.log2(R) - math.log2(max(rank_hat, 1e-12))
        per_canary.append(
            {
                "canary_id": c["canary_id"],
                "n_reps": int(c["n_reps"]),
                "rank_in_sample": rank_in_sample,
                "rank_hat": rank_hat,
                "exposure": exposure,
                "true_log_ppl": true_score,
                "mean_alt_log_ppl": sum(logppl[1:]) / max(len(logppl) - 1, 1),
                "saturated": rank_in_sample == 1,
            }
        )
        print(f"[canary] {c['canary_id']} n_reps={c['n_reps']:>2} "
              f"rank={rank_in_sample}/{n_alt+1} exposure={exposure:.2f}", flush=True)

    by_reps: Dict[str, Dict[str, Any]] = {}
    groups: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for pc in per_canary:
        groups[pc["n_reps"]].append(pc)
    for k in sorted(groups):
        g = groups[k]
        exps = [x["exposure"] for x in g]
        by_reps[str(k)] = {
            "mean_exposure": sum(exps) / len(exps),
            "min_exposure": min(exps),
            "max_exposure": max(exps),
            "mean_rank_in_sample": sum(x["rank_in_sample"] for x in g) / len(g),
            "n_saturated": sum(int(x["saturated"]) for x in g),
            "n": len(g),
        }

    max_measurable = math.log2(R) - math.log2(R / (n_alt + 1))
    exposures = [x["exposure"] for x in per_canary]
    return {
        "n_canaries": len(per_canary),
        "mean_exposure": (sum(exposures) / len(exposures)) if exposures else 0.0,
        "space_size": R,
        "n_alternatives": n_alt,
        "max_measurable_exposure": max_measurable,
        "estimator_note": (
            f"With {n_alt} sampled alternatives the smallest observable rank is 1, "
            f"so rank_hat bottoms out at {R/(n_alt+1):.0f} and exposure saturates at "
            f"{max_measurable:.2f} bits (the true ceiling is log2(|R|) = {math.log2(R):.2f}). "
            "Any canary reported at the ceiling is therefore a lower bound: the "
            "estimator cannot distinguish 'best of 10k' from 'best of 1M'."
        ),
        "by_n_reps": by_reps,
        "per_canary": per_canary,
        "wallclock_s": time.time() - t0,
    }


def attack_targeted(model, tok, targeted, args) -> Dict[str, Any]:
    import torch

    t0 = time.time()
    items: List[Dict[str, Any]] = []
    for r in targeted:
        pre = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        gold_tok = tok(r["gold_value"], add_special_tokens=False)["input_ids"]
        budget = min(args.targeted_max_new_tokens, len(gold_tok) + args.targeted_extra_tokens)
        items.append({**r, "prefix_ids": pre or [tok.eos_token_id], "budget": max(budget, 4)})

    budgets: Dict[int, List[int]] = collections.defaultdict(list)
    for i, it in enumerate(items):
        budgets[it["budget"]].append(i)

    greedy_out: List[str] = [""] * len(items)
    for b, idxs in sorted(budgets.items()):
        gens = greedy_continue(
            model, tok, [items[i]["prefix_ids"] for i in idxs], b,
            args.gen_batch_size, f"targeted:greedy(+{b})",
        )
        for i, g in zip(idxs, gens):
            greedy_out[i] = tok.decode(g, skip_special_tokens=True)

    sampled_out: List[List[str]] = [[] for _ in items]
    if args.targeted_k > 0:
        torch.manual_seed(args.seed)
        pad = tok.pad_token_id
        for b, idxs in sorted(budgets.items()):
            order = sorted(idxs, key=lambda i: len(items[i]["prefix_ids"]))
            bs = max(1, args.gen_batch_size // max(1, args.targeted_k // 2))
            batches = [order[i : i + bs] for i in range(0, len(order), bs)]
            for batch in _progress(batches, f"targeted:sample(+{b})"):
                L = max(len(items[i]["prefix_ids"]) for i in batch)
                input_ids = torch.tensor(
                    [[pad] * (L - len(items[i]["prefix_ids"])) + items[i]["prefix_ids"] for i in batch],
                    dtype=torch.long,
                ).to(get_model_device(model))
                attn = torch.tensor(
                    [[0] * (L - len(items[i]["prefix_ids"])) + [1] * len(items[i]["prefix_ids"])
                     for i in batch],
                    dtype=torch.long,
                ).to(get_model_device(model))
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids, attention_mask=attn,
                        **generate_safe_kwargs(
                            tok,
                            max_new_tokens=b,
                            min_new_tokens=b,
                            do_sample=True,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            num_return_sequences=args.targeted_k,
                        ),
                    )
                new = gen[:, L:]
                dec = tok.batch_decode(new, skip_special_tokens=True)
                for k, i in enumerate(batch):
                    sampled_out[i] = dec[k * args.targeted_k : (k + 1) * args.targeted_k]

    per_split: Dict[str, Dict[str, Any]] = {}
    by_label: Dict[str, Dict[str, Any]] = collections.defaultdict(
        lambda: {"n": 0, "greedy": 0, "sampled": 0}
    )
    recs: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        gold = normalize_ws(it["gold_value"])
        g_hit = int(bool(gold) and gold in normalize_ws(greedy_out[i]))
        s_hit = int(any(bool(gold) and gold in normalize_ws(s) for s in sampled_out[i]))
        any_hit = int(bool(g_hit or s_hit))
        sp = per_split.setdefault(
            it["split"], {"n": 0, "greedy": 0, "sampled_any": 0, "either": 0}
        )
        sp["n"] += 1
        sp["greedy"] += g_hit
        sp["sampled_any"] += s_hit
        sp["either"] += any_hit
        bl = by_label[f"{it['split']}:{it['label']}"]
        bl["n"] += 1
        bl["greedy"] += g_hit
        bl["sampled"] += s_hit
        recs.append(
            {
                "id": it["id"], "split": it["split"], "label": it["label"],
                "doc_id": it["doc_id"], "greedy_hit": bool(g_hit), "sampled_hit": bool(s_hit),
                "gold_value": it["gold_value"], "greedy_continuation": greedy_out[i],
            }
        )

    def rates(d: Dict[str, Any]) -> Dict[str, Any]:
        n = max(d["n"], 1)
        greedy = d["greedy"] / n
        sampled = d["sampled_any"] / n
        return {
            "n": d["n"],
            "greedy_leak_rate": greedy,
            "sampled_any10_leak_rate": sampled,
            "top1_leak_rate": greedy,
            "sampled_leak_rate": sampled,
            "any_leak_rate": d["either"] / n,
        }

    member = per_split.get("member", {"n": 0, "greedy": 0, "sampled_any": 0, "either": 0})
    control = per_split.get("control", {"n": 0, "greedy": 0, "sampled_any": 0, "either": 0})
    mr = rates(member)
    cr = rates(control)
    return {
        "n_member": member["n"],
        "n_control": control["n"],
        "n_prompts": member["n"],
        "k_samples": args.targeted_k,
        "decoding": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p},
        "top1_leak_rate": mr["greedy_leak_rate"],
        "top10_leak_rate": mr["sampled_any10_leak_rate"],
        "control_top1_leak_rate": cr["greedy_leak_rate"],
        "control_top10_leak_rate": cr["sampled_any10_leak_rate"],
        "member": mr,
        "control": cr,
        "member_minus_control_top1": mr["top1_leak_rate"] - cr["top1_leak_rate"],
        "member_minus_control_sampled": mr["sampled_leak_rate"] - cr["sampled_leak_rate"],
        "by_split_label": {
            k: {
                "n": v["n"],
                "greedy_rate": v["greedy"] / max(v["n"], 1),
                "sampled_rate": v["sampled"] / max(v["n"], 1),
            }
            for k, v in sorted(by_label.items())
        },
        "wallclock_s": time.time() - t0,
        "_records": recs,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Part C.3: extraction attacks")
    p.add_argument("--victim_model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--adapter_path", default="", help="omit for the un-fine-tuned control")
    p.add_argument("--ref_model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--corpus", default="data/partC_corpus_raw.jsonl",
                   help="corpus the victim was trained on (verification + prefixes)")
    p.add_argument("--canaries", default="data/partC_canaries.jsonl")
    p.add_argument("--targeted_file", default="data/partC_targeted.jsonl")
    p.add_argument("--heldout", default="data/partC_heldout.jsonl")
    p.add_argument(
        "--pii_index",
        default="data/partC_pii_index.jsonl",
        help="JSONL of known training-corpus PII values (built by src/build_pii_index.py)",
    )
    p.add_argument("--attack", default="all",
                   help="'all', one attack, or a comma-separated subset of "
                        "{discoverable,untargeted,canary,targeted}")
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_file", default="results/partC_attacks.json")
    p.add_argument("--victim_name", default="", help="label written into the report")
    p.add_argument("--records_dir", default="", help="optional dir for per-item attack records")

    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--merge_adapter", action="store_true", default=True)
    p.add_argument("--no_merge_adapter", dest="merge_adapter", action="store_false")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--gen_batch_size", type=int, default=32)
    p.add_argument("--score_batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--no_ref_model", action="store_true",
                   help="skip the reference model (ratio_ref becomes 1.0)")

    p.add_argument("--prefix_tokens", type=int, default=32)
    p.add_argument("--suffix_tokens", type=int, default=32)
    p.add_argument("--min_suffix_tokens", type=int, default=8)
    p.add_argument("--disc_per_group", type=int, default=400,
                   help="documents sampled per repetition bucket (0 = all)")

    p.add_argument("--sample_max_new_tokens", type=int, default=128)
    p.add_argument("--sample_min_new_tokens", type=int, default=24)
    p.add_argument("--sample_batch_size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--mem_min_chars", type=int, default=50)
    p.add_argument("--precision_at", type=int, default=100)
    p.add_argument("--dump_generations", action="store_true", default=True,
                   help="keep the top-ranked generations in the report")
    p.add_argument("--no_dump_generations", dest="dump_generations", action="store_false")

    p.add_argument("--canary_num_alternatives", type=int, default=9999)
    p.add_argument("--no_canary_exact", action="store_true",
                   help="skip exact 10^6 enumeration and report only the sampled estimator")
    p.add_argument("--canary_exact_batch", type=int, default=1024,
                   help="sequences per forward pass during exact canary enumeration")
    p.add_argument("--canary_space", type=int, default=10 ** 6)
    p.add_argument("--canary_batch_size", type=int, default=128,
                   help="200k short sequences are scored here; logits dominate memory")

    p.add_argument("--targeted_k", type=int, default=10)
    p.add_argument("--targeted_extra_tokens", type=int, default=8)
    p.add_argument("--targeted_max_new_tokens", type=int, default=32)

    p.add_argument("--limit", type=int, default=0, help="debug: cap corpus rows")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.num_samples = 32
        args.disc_per_group = 4
        args.canary_num_alternatives = 49
        args.no_canary_exact = True
        args.targeted_k = 2
        args.limit = 400
    args.victim_name = args.victim_name or (
        os.path.basename(args.adapter_path.rstrip("/")) if args.adapter_path else "control_pretrained"
    )

    import torch

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ALL_ATTACKS = ["discoverable", "untargeted", "canary", "targeted"]
    if args.attack == "all":
        which = list(ALL_ATTACKS)
    else:
        requested = [a.strip() for a in args.attack.split(",") if a.strip()]
        bad = [a for a in requested if a not in ALL_ATTACKS]
        if bad:
            raise SystemExit(f"unknown attack(s): {bad}; choose from {ALL_ATTACKS} or 'all'")
        which = [a for a in ALL_ATTACKS if a in requested]

    print(f"=== attacks {which} against '{args.victim_name}' ===", flush=True)
    tok = load_tokenizer(args.victim_model, padding_side="left")
    victim, torch_dtype = load_causal_lm(
        args.victim_model, quant=args.quant, dtype=args.dtype, device=device,
        adapter_path=args.adapter_path or None, merge_adapter=args.merge_adapter,
        for_training=False,
    )
    print(f"    victim={args.victim_model} adapter={args.adapter_path or '(none)'} "
          f"dtype={dtype_name(torch_dtype)} gpu={gpu_name()}", flush=True)

    ref_model = None
    if "untargeted" in which and not args.no_ref_model:
        if args.adapter_path:
            ref_model, _ = load_causal_lm(
                args.ref_model, quant=args.quant, dtype=args.dtype, device=device,
                for_training=False,
            )
            print(f"    reference={args.ref_model}", flush=True)
        else:
            print("    [note] victim IS the pre-trained control; ratio_ref == 1 by definition",
                  flush=True)

    corpus = read_jsonl(args.corpus, limit=args.limit or None)
    canaries = read_jsonl(args.canaries) if args.canaries and os.path.exists(args.canaries) else []
    targeted = (
        read_jsonl(args.targeted_file)
        if args.targeted_file and os.path.exists(args.targeted_file)
        else []
    )
    heldout = read_jsonl(args.heldout) if args.heldout and os.path.exists(args.heldout) else []

    report: Dict[str, Any] = {
        "victim": args.victim_name,
        "victim_model": args.victim_model,
        "adapter": args.adapter_path or None,
        "ref_model": args.ref_model if ref_model is not None else None,
        "corpus": os.path.basename(args.corpus),
        "corpus_rows": len(corpus),
        "attacks_run": which,
        "seed": args.seed,
        "timestamp": common.now_stamp(),
        "gpu": gpu_name(),
        "dtype": dtype_name(torch_dtype),
        "args": vars(args),
    }

    if heldout:
        ppl = corpus_perplexity(
            victim, tok, [r["text"] for r in heldout], batch_size=args.eval_batch_size,
            max_len=args.max_seq_len, add_eos=True, progress=False,
        )
        report["heldout_ppl"] = ppl["ppl"]
        report["heldout_ppl_detail"] = ppl
        print(f"    held-out ppl = {ppl['ppl']:.3f}", flush=True)

    records: Dict[str, Any] = {}

    if "discoverable" in which:
        d = attack_discoverable(victim, tok, corpus, args)
        records["discoverable"] = d.pop("_records", [])
        report["discoverable"] = d
        print(f"[discoverable] overall exact={d['exact_match_rate']:.4f} "
              f"mean_edit_sim={d['mean_edit_sim']:.4f}", flush=True)
        for k, v in d["by_n_reps"].items():
            print(f"    n_reps={k:>3s}: exact={v['exact_match_rate']:.4f} "
                  f"sim={v['mean_edit_sim']:.3f} (n={v['n']})", flush=True)
        for k, v in d["canary_by_n_reps"].items():
            print(f"    canary n_reps={k:>3s}: exact={v['exact_match_rate']:.4f} (n={v['n']})",
                  flush=True)

    if "untargeted" in which:
        pii_index = load_pii_index(args.pii_index)
        if not pii_index:
            pii_index = build_fallback_pii_index(corpus, targeted, canaries)
            print(f"    [note] pii_index missing or empty; built fallback with {len(pii_index)} values",
                  flush=True)
        extras = {
            "index": CorpusIndex([(r["doc_id"], r["text"]) for r in corpus]),
            "corpus_norm": normalize_ws(" \n ".join(r["text"] for r in corpus)),
            "targeted_values": {
                t["id"]: t["gold_value"] for t in targeted if t["split"] == "member"
            },
            "canary_secrets": {c["canary_id"]: str(c["secret"]) for c in canaries},
            "pii_index": pii_index,
        }
        u = attack_untargeted(victim, tok, ref_model, corpus, args, extras)
        records["untargeted"] = u.pop("_generations", [])
        report["untargeted"] = u
        print(f"[untargeted] unique={u['n_unique']} verified={u['verification']['n_verified_memorized']} "
              f"docs={u['distinct_docs_recovered']} pii={u['verified_pii_strings']}", flush=True)
        print(f"    precision@{args.precision_at}: "
              + ", ".join(f"{k}={v:.3f}" for k, v in u["precision_at_100"].items()), flush=True)

    if "canary" in which and canaries:
        sampled = attack_canary(victim, tok, canaries, args)
        for k, v in sampled["by_n_reps"].items():
            print(f"[canary-sampled] n_reps={k:>3s}: mean_exposure={v['mean_exposure']:.3f} "
                  f"saturated={v['n_saturated']}/{v['n']}", flush=True)
        report["canary"] = sampled
        if not args.no_canary_exact:
            exact = attack_canary_exact(victim, tok, canaries, args)
            if exact is not None:
                report["canary"]["exact"] = exact
                for k, v in exact["by_n_reps"].items():
                    print(f"[canary-exact] n_reps={k:>3s}: mean_exposure={v['mean_exposure']:.3f} "
                          f"rank1={v['n_rank1']}/{v['n']} margin={v['mean_margin_bits']:.1f}b", flush=True)

    if "targeted" in which and targeted:
        t = attack_targeted(victim, tok, targeted, args)
        records["targeted"] = t.pop("_records", [])
        report["targeted"] = t
        print(f"[targeted] member top1={t['top1_leak_rate']:.4f} sampled={t['top10_leak_rate']:.4f} | "
              f"control top1={t['control_top1_leak_rate']:.4f} "
              f"sampled={t['control_top10_leak_rate']:.4f}", flush=True)

    write_json(args.report_file, report)
    print(f"    report -> {args.report_file}", flush=True)

    if args.records_dir:
        os.makedirs(args.records_dir, exist_ok=True)
        for name, rows in records.items():
            if rows:
                path = os.path.join(args.records_dir, f"{args.victim_name}_{name}.jsonl")
                write_jsonl(path, rows)
                print(f"    records -> {path}", flush=True)
    print(f"=== attacks on {args.victim_name} done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
