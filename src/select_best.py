from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import write_json


def _f(x, default=float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        by_id[r["run_id"]] = r
    return [by_id[k] for k in sorted(by_id)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Select the best Part A adapter for Part B")
    p.add_argument("--sweep_csv", default="results/partA_sweep.csv")
    p.add_argument("--adapters_dir", default="adapters")
    p.add_argument("--dest", default="adapters/best_sanitizer")
    p.add_argument("--report_file", default="results/partA_selection.json")
    p.add_argument("--f1_tolerance", type=float, default=0.01)
    p.add_argument("--force_run_id", default="", help="override the rule and pick this run")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)

    if not os.path.exists(args.sweep_csv):
        raise SystemExit(f"no sweep results at {args.sweep_csv}; run the Part A sweep first")
    rows = load_rows(args.sweep_csv)
    usable = [r for r in rows if _f(r.get("val_entity_f1")) == _f(r.get("val_entity_f1"))]
    usable = [r for r in usable if not str(r.get("run_id", "")).endswith("_ckpt")]
    if not usable:
        raise SystemExit("no rows with a val_entity_f1 value; re-run the sweep without --skip_val_gen")

    best_f1 = max(_f(r["val_entity_f1"]) for r in usable)
    shortlist = [r for r in usable if _f(r["val_entity_f1"]) >= best_f1 - args.f1_tolerance]
    shortlist.sort(
        key=lambda r: (
            _f(r.get("val_leak_rate"), 1.0),
            _f(r.get("trainable_params"), float("inf")),
            _f(r.get("peak_gpu_mem_gb"), float("inf")),
            _f(r.get("train_wallclock_s"), float("inf")),
        )
    )
    chosen = shortlist[0]
    if args.force_run_id:
        forced = [r for r in rows if r["run_id"] == args.force_run_id]
        if not forced:
            raise SystemExit(f"--force_run_id {args.force_run_id} is not in {args.sweep_csv}")
        chosen = forced[0]

    print(f"{'run':<6}{'method':<8}{'r':>4}{'alpha':>6}{'train%':>9}{'memGB':>8}"
          f"{'min':>7}{'F1':>8}{'leak':>8}")
    for r in sorted(rows, key=lambda x: -_f(x.get("val_entity_f1"), -1)):
        mark = " <-- selected" if r["run_id"] == chosen["run_id"] else (
            "  (shortlist)" if any(s["run_id"] == r["run_id"] for s in shortlist) else "")
        print(f"{r['run_id']:<6}{r.get('method',''):<8}{r.get('lora_r',''):>4}"
              f"{r.get('lora_alpha',''):>6}{_f(r.get('trainable_pct')):>9.4f}"
              f"{_f(r.get('peak_gpu_mem_gb')):>8.2f}{_f(r.get('train_wallclock_s'))/60:>7.1f}"
              f"{_f(r.get('val_entity_f1')):>8.4f}{_f(r.get('val_leak_rate')):>8.4f}{mark}")

    src = os.path.join(args.adapters_dir, chosen["run_id"])
    if not os.path.isdir(src):
        raise SystemExit(f"adapter directory {src} not found (expected adapters/<run_id>)")

    justification = (
        f"Selected {chosen['run_id']} ({chosen.get('method')}, r={chosen.get('lora_r')}, "
        f"alpha={chosen.get('lora_alpha')}). Best F1 in the sweep was {best_f1:.4f}; "
        f"{len(shortlist)} run(s) fell within {args.f1_tolerance:.3f} of it, and among those "
        f"{chosen['run_id']} had the lowest validation leak rate "
        f"({_f(chosen.get('val_leak_rate')):.4f}) at {_f(chosen.get('trainable_pct')):.4f}% "
        f"trainable parameters and {_f(chosen.get('peak_gpu_mem_gb')):.2f} GiB peak memory. "
        "Leak rate is the privacy-relevant error for a sanitizer, so it breaks near-ties on F1."
    )
    print("\n" + justification)

    record = {
        "sweep_csv": args.sweep_csv,
        "rule": {
            "step1": f"shortlist runs with val_entity_f1 >= best - {args.f1_tolerance}",
            "step2": "minimise val_leak_rate",
            "step3": "then fewer trainable params, then less memory, then faster",
            "forced": args.force_run_id or None,
        },
        "best_f1_observed": best_f1,
        "shortlist": [r["run_id"] for r in shortlist],
        "selected": chosen,
        "selected_adapter_source": src,
        "destination": args.dest,
        "justification": justification,
        "all_runs": rows,
    }

    if args.dry_run:
        print("[dry run] adapter not copied")
    else:
        if os.path.islink(args.dest) or os.path.isfile(args.dest):
            os.remove(args.dest)
        elif os.path.isdir(args.dest):
            shutil.rmtree(args.dest)
        shutil.copytree(src, args.dest)
        print(f"copied {src} -> {args.dest}")
    write_json(args.report_file, record)
    print(f"decision -> {args.report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
