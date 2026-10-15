from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import ensure_parent, write_json

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None

PALETTE = {
    "victim_raw": "#c1462f",
    "victim_san": "#2f6fc1",
    "control_pretrained": "#6b6b6b",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _num(x, default=float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def load_sweep(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        by_id[r.get("run_id", "?")] = r
    return [by_id[k] for k in sorted(by_id)]


def fig_part_a(rows: Sequence[Dict[str, Any]], out_base: str) -> Optional[str]:
    if not HAS_MPL:
        print("[partA_fig] matplotlib not installed; skipped (tables still written)")
        return None
    if not rows:
        print("[partA_fig] no sweep rows; skipped")
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    def annotate(ax, xs, ys, labels):
        for x, y, lab in zip(xs, ys, labels):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)

    ids = [r.get("run_id", "?") for r in rows]
    f1 = [_num(r.get("val_entity_f1")) for r in rows]
    pct = [_num(r.get("trainable_pct")) for r in rows]
    mem = [_num(r.get("peak_gpu_mem_gb")) for r in rows]
    wall = [_num(r.get("train_wallclock_s")) / 60.0 for r in rows]
    quant = [(r.get("quant") or "none") for r in rows]
    colors = ["#c1462f" if q == "4bit" else "#2f6fc1" for q in quant]

    ax = axes[0]
    ax.scatter(pct, f1, c=colors, s=90, zorder=3, edgecolor="white", linewidth=0.8)
    annotate(ax, pct, f1, ids)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (% of total, log scale)")
    ax.set_ylabel("val entity F1  (IoU$\\geq$0.5 + label)")
    ax.set_title("A. Accuracy vs adapter size")
    ax.grid(alpha=0.3, zorder=0)

    ax = axes[1]
    ax.scatter(mem, f1, c=colors, s=90, zorder=3, edgecolor="white", linewidth=0.8)
    annotate(ax, mem, f1, ids)
    ax.set_xlabel("peak GPU memory (GiB)")
    ax.set_ylabel("val entity F1")
    ax.set_title("B. Accuracy vs peak memory")
    ax.grid(alpha=0.3, zorder=0)

    ax = axes[2]
    ax.scatter(wall, f1, c=colors, s=90, zorder=3, edgecolor="white", linewidth=0.8)
    annotate(ax, wall, f1, ids)
    ax.set_xlabel("training wall-clock (min)")
    ax.set_ylabel("val entity F1")
    ax.set_title("C. Accuracy vs training time")
    ax.grid(alpha=0.3, zorder=0)

    handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#2f6fc1", markersize=10, label="LoRA (bf16 base)"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#c1462f", markersize=10, label="QLoRA (4-bit base)"),
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=9, frameon=True)
    fig.suptitle("Part A -- PEFT sweep: what each axis buys", fontsize=13, y=1.02)
    fig.tight_layout()
    saved = []
    for ext in ("png", "pdf"):
        p = f"{out_base}.{ext}"
        ensure_parent(p)
        fig.savefig(p, dpi=170, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    print(f"[partA_fig] -> {', '.join(saved)}")
    return saved[0]


def fig_exposure(reports: Dict[str, Dict[str, Any]], out_base: str) -> Optional[str]:
    if not HAS_MPL:
        return None
    styles = {"control_pretrained": ("base", "#6f6f6f", "o"),
              "victim_raw": ("V-raw", "#c0392b", "s"),
              "victim_san": ("V-san", "#2a6fbb", "^")}
    series = []
    for name, (label, color, marker) in styles.items():
        can = (reports.get(name) or {}).get("canary") or {}
        by = can.get("by_n_reps") or {}
        if not by:
            continue
        ks = sorted(by, key=int)
        ex = (can.get("exact") or {}).get("by_n_reps") or {}
        series.append((label, color, marker, [int(k) for k in ks],
                       [by[k]["mean_exposure"] for k in ks],
                       [ex[k]["mean_exposure"] for k in ks] if ex else None,
                       [ex[k].get("mean_margin_bits") for k in ks] if ex else None, can))
    if not series:
        return None
    has_exact = any(x[5] for x in series)
    fig, axes = plt.subplots(1, 2 if has_exact else 1, figsize=(11, 4.2) if has_exact else (6.5, 4.2))
    ax = axes[0] if has_exact else axes
    for label, color, marker, xs, samp, exact, _m, _c in series:
        ax.plot(xs, samp, marker=marker, color=color, lw=1.8, label=f"{label}, sampled estimator")
        if exact:
            ax.plot(xs, exact, marker=marker, color=color, lw=1.2, ls="--", mfc="none",
                    label=f"{label}, exact rank")
    can = series[0][7]
    ceiling = can.get("max_measurable_exposure")
    if ceiling:
        ax.axhline(ceiling, color="#999999", lw=0.9, ls=":")
        ax.text(1.05, ceiling + 0.3, f"sampled-estimator ceiling ({ceiling:.2f} bits)", fontsize=8, color="#666666")
    if has_exact:
        top = math.log2(can.get("space_size", 10 ** 6))
        ax.axhline(top, color="#333333", lw=0.8)
        ax.text(1.05, top - 1.0, f"log2|R| = {top:.2f} bits", fontsize=8, color="#333333")
    ax.set_title("Mean canary exposure")
    ax.set_ylabel("exposure (bits)")
    ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.36))
    axes_list = list(axes) if has_exact else [axes]
    if has_exact:
        ax2 = axes[1]
        for label, color, marker, xs, _s, _e, margin, _c in series:
            if margin and all(v is not None for v in margin):
                ax2.plot(xs, margin, marker=marker, color=color, lw=1.8, label=label)
        ax2.axhline(0, color="#333333", lw=0.7)
        ax2.set_title("Log-likelihood margin of the secret over the best alternative")
        ax2.set_ylabel("margin (bits)")
        ax2.legend(fontsize=8)
    for a in axes_list:
        a.set_xscale("log", base=2)
        a.set_xticks([1, 4, 16, 64])
        a.set_xticklabels(["1", "4", "16", "64"])
        a.set_xlabel("canary repetitions in the training corpus")
        a.grid(alpha=0.3)
    fig.tight_layout()
    saved = []
    for ext in ("png", "pdf"):
        path = f"{out_base}.{ext}"
        ensure_parent(path)
        fig.savefig(path, dpi=170, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    print(f"[partC_exposure] -> {', '.join(saved)}")
    return saved[0]


PARTB_COLS = [
    "system",
    "test_set",
    "adapter",
    "n",
    "n_gold_spans",
    "n_pred_spans",
    "exact_micro_p",
    "exact_micro_r",
    "exact_micro_f1",
    "exact_macro_f1",
    "iou50_micro_p",
    "iou50_micro_r",
    "iou50_micro_f1",
    "iou50_macro_f1",
    "iou50_labelfree_f1",
    "iou50_mapped_f1",
    "leak_rate",
    "leak_rate_len_ge_3",
    "doc_level_leak_rate",
    "over_masking_rate",
    "over_masking_rate_nonspace",
    "well_formed_rate",
    "mean_latency_ms",
    "total_wallclock_s",
]


def table_part_b(paths: Sequence[str], out_csv: str) -> int:
    rows: List[Dict[str, Any]] = []
    for p in sorted(paths):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as exc:
            print(f"[partB_summary] skip {p}: {exc}")
            continue
        if "exact" not in d:
            continue
        rows.append(
            {
                "system": d.get("system"),
                "test_set": d.get("test_set"),
                "adapter": d.get("adapter"),
                "n": d.get("n"),
                "n_gold_spans": d.get("n_gold_spans"),
                "n_pred_spans": d.get("n_pred_spans"),
                "exact_micro_p": d["exact"]["micro_p"],
                "exact_micro_r": d["exact"]["micro_r"],
                "exact_micro_f1": d["exact"]["micro_f1"],
                "exact_macro_f1": d["exact"]["macro_f1"],
                "iou50_micro_p": d["overlap_iou50"]["micro_p"],
                "iou50_micro_r": d["overlap_iou50"]["micro_r"],
                "iou50_micro_f1": d["overlap_iou50"]["micro_f1"],
                "iou50_macro_f1": d["overlap_iou50"]["macro_f1"],
                "iou50_labelfree_f1": d.get("overlap_iou50_label_agnostic", {}).get("micro_f1"),
                "iou50_mapped_f1": d.get("overlap_iou50_mapped", {}).get("micro_f1"),
                "leak_rate": d.get("leak_rate"),
                "leak_rate_len_ge_3": d.get("leak_rate_len_ge_3"),
                "doc_level_leak_rate": d.get("doc_level_leak_rate"),
                "over_masking_rate": d.get("over_masking_rate"),
                "over_masking_rate_nonspace": d.get("over_masking_rate_nonspace"),
                "well_formed_rate": d.get("well_formed_rate"),
                "mean_latency_ms": (d.get("latency") or {}).get("mean_ms"),
                "total_wallclock_s": d.get("total_wallclock_s"),
            }
        )
    if not rows:
        print("[partB_summary] nothing to write")
        return 0
    ensure_parent(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PARTB_COLS)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (str(x["test_set"]), str(x["system"]))):
            w.writerow(r)
    print(f"[partB_summary] {len(rows)} systems -> {out_csv}")
    return len(rows)


PARTC_COLS = [
    "victim", "corpus", "heldout_ppl", "disc_exact_overall", "disc_mean_edit_sim",
    "disc_reps1", "disc_reps2", "disc_reps8", "disc_reps32",
    "canary_exp_1", "canary_exp_4", "canary_exp_16", "canary_exp_64",
    "canary_exact_1", "canary_exact_4", "canary_exact_16", "canary_exact_64",
    "canary_margin_1", "canary_margin_4", "canary_margin_16", "canary_margin_64",
    "unt_n_unique", "unt_verified", "unt_docs_recovered", "unt_pii_strings",
    "p100_ppl", "p100_ratio_ref", "p100_ratio_lower", "p100_ratio_zlib",
    "tgt_member_top1", "tgt_member_sampled", "tgt_control_top1", "tgt_control_sampled",
]


def table_part_c(reports: Dict[str, Dict[str, Any]], out_csv: str, out_json: str) -> int:
    rows: List[Dict[str, Any]] = []
    for name, d in sorted(reports.items()):
        disc = d.get("discoverable", {}) or {}
        by = disc.get("by_n_reps", {}) or {}
        can = (d.get("canary", {}) or {}).get("by_n_reps", {}) or {}
        ex = ((d.get("canary", {}) or {}).get("exact", {}) or {}).get("by_n_reps", {}) or {}
        unt = d.get("untargeted", {}) or {}
        p100 = unt.get("precision_at_100", {}) or {}
        tgt = d.get("targeted", {}) or {}
        rows.append(
            {
                "victim": name,
                "corpus": d.get("corpus"),
                "heldout_ppl": d.get("heldout_ppl"),
                "disc_exact_overall": disc.get("exact_match_rate"),
                "disc_mean_edit_sim": disc.get("mean_edit_sim"),
                **{f"disc_reps{k}": (by.get(k, {}) or {}).get("exact_match_rate") for k in ("1", "2", "8", "32")},
                **{f"canary_exp_{k}": (can.get(k, {}) or {}).get("mean_exposure") for k in ("1", "4", "16", "64")},
                **{f"canary_exact_{k}": (ex.get(k, {}) or {}).get("mean_exposure") for k in ("1", "4", "16", "64")},
                **{f"canary_margin_{k}": (ex.get(k, {}) or {}).get("mean_margin_bits") for k in ("1", "4", "16", "64")},
                "unt_n_unique": unt.get("n_unique"),
                "unt_verified": (unt.get("verification", {}) or {}).get("n_verified_memorized"),
                "unt_docs_recovered": unt.get("distinct_docs_recovered"),
                "unt_pii_strings": unt.get("verified_pii_strings"),
                "p100_ppl": p100.get("ppl"),
                "p100_ratio_ref": p100.get("ratio_ref"),
                "p100_ratio_lower": p100.get("ratio_lower"),
                "p100_ratio_zlib": p100.get("ratio_zlib"),
                "tgt_member_top1": tgt.get("top1_leak_rate", (tgt.get("member") or {}).get("greedy_leak_rate")),
                "tgt_member_sampled": tgt.get("top10_leak_rate", (tgt.get("member") or {}).get("sampled_any10_leak_rate")),
                "tgt_control_top1": tgt.get("control_top1_leak_rate", (tgt.get("control") or {}).get("greedy_leak_rate")),
                "tgt_control_sampled": tgt.get("control_top10_leak_rate", (tgt.get("control") or {}).get("sampled_any10_leak_rate")),
            }
        )
    if not rows:
        print("[partC_comparison] nothing to write")
        return 0
    ensure_parent(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PARTC_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    write_json(out_json, {"victims": rows})
    print(f"[partC_comparison] {len(rows)} victims -> {out_csv}, {out_json}")
    return len(rows)


def export_partA_val_metrics(adapters_dir: str, out_csv: str) -> int:
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(adapters_dir, "*", "run_meta.json"))):
        try:
            meta = json.load(open(meta_path, encoding="utf-8"))
        except Exception:
            continue
        g = meta.get("val_generation_metrics") or {}
        if not g or not meta.get("run_id") or os.path.basename(os.path.dirname(meta_path)) != meta["run_id"]:
            continue
        rows.append({
            "run_id": meta["run_id"], "n": g.get("n"),
            "exact_micro_p": g["exact"]["micro_p"], "exact_micro_r": g["exact"]["micro_r"],
            "exact_micro_f1": g["exact"]["micro_f1"], "exact_macro_f1": g["exact"]["macro_f1"],
            "iou50_micro_p": g["overlap_iou50"]["micro_p"], "iou50_micro_r": g["overlap_iou50"]["micro_r"],
            "iou50_micro_f1": g["overlap_iou50"]["micro_f1"], "iou50_macro_f1": g["overlap_iou50"]["macro_f1"],
            "leak_rate": g.get("leak_rate"), "over_masking_rate": g.get("over_masking_rate"),
            "well_formed_rate": g.get("well_formed_rate"),
        })
    if not rows:
        return 0
    rows.sort(key=lambda r: r["run_id"])
    ensure_parent(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[partA_val_metrics] {len(rows)} runs -> {out_csv}")
    return len(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the deliverable figures and summary tables")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--sweep_csv", default="")
    p.add_argument("--partA_fig", default="")
    p.add_argument("--exposure_fig", default="")
    p.add_argument("--adapters_dir", default="adapters")
    p.add_argument("--partB_glob", default="", help="default: <results_dir>/partB_metrics*.json")
    p.add_argument("--attack_glob", default="", help="default: <results_dir>/partC_attacks*.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rd = args.results_dir
    sweep_csv = args.sweep_csv or os.path.join(rd, "partA_sweep.csv")
    partA_fig = args.partA_fig or os.path.join(rd, "partA_fig")
    exposure_fig = args.exposure_fig or os.path.join(rd, "partC_exposure")
    partB_glob = args.partB_glob or os.path.join(rd, "partB_metrics*.json")
    attack_glob = args.attack_glob or os.path.join(rd, "partC_attacks*.json")

    partA_fig = partA_fig[:-4] if partA_fig.endswith((".png", ".pdf")) else partA_fig
    exposure_fig = exposure_fig[:-4] if exposure_fig.endswith((".png", ".pdf")) else exposure_fig

    fig_part_a(load_sweep(sweep_csv), partA_fig)

    reports: Dict[str, Dict[str, Any]] = {}
    for p in sorted(glob.glob(attack_glob)):
        if os.path.basename(p).startswith("partC_comparison"):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as exc:
            print(f"[attacks] skip {p}: {exc}")
            continue
        name = d.get("victim") or os.path.splitext(os.path.basename(p))[0]
        reports[name] = d
    if reports:
        print(f"[attacks] loaded {len(reports)} report(s): {sorted(reports)}")
    fig_exposure(reports, exposure_fig)
    table_part_c(reports, os.path.join(rd, "partC_comparison.csv"),
                 os.path.join(rd, "partC_comparison.json"))
    table_part_b(glob.glob(partB_glob), os.path.join(rd, "partB_summary.csv"))
    export_partA_val_metrics(args.adapters_dir, os.path.join(rd, "partA_val_metrics.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
