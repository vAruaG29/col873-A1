from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attack_extract import extract_pii_strings
from common import normalize_ws, read_jsonl, write_jsonl


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the Part C PII index")
    p.add_argument("--corpus", default="data/partC_corpus_raw.jsonl")
    p.add_argument("--canaries", default="data/partC_canaries.jsonl")
    p.add_argument("--targeted_file", default="data/partC_targeted.jsonl")
    p.add_argument("--output", default="data/partC_pii_index.jsonl")
    args = p.parse_args(argv)

    corpus = read_jsonl(args.corpus)
    canaries = read_jsonl(args.canaries) if os.path.exists(args.canaries) else []
    targeted = read_jsonl(args.targeted_file) if os.path.exists(args.targeted_file) else []

    by_key: Dict[str, Dict[str, Any]] = {}

    def add(value: str, label: str, doc_id: Any, source: str, min_len: int = 3) -> None:
        key = normalize_ws(str(value))
        if not key or len(key) < min_len:
            return
        by_key.setdefault(
            key,
            {"value": str(value), "label": label, "doc_id": doc_id, "source": source},
        )

    for t in targeted:
        if t.get("split") == "member":
            add(t["gold_value"], t.get("label", ""), t.get("doc_id"), "targeted", min_len=1)

    for c in canaries:
        add(str(c["secret"]), "CANARY", c.get("canary_id"), "canary", min_len=1)

    seen_text = set()
    for r in corpus:
        text = r.get("text", "")
        if text in seen_text:
            continue
        seen_text.add(text)
        for lab, val in extract_pii_strings(text):
            add(val, lab, r.get("doc_id"), "pattern")

    rows: List[Dict[str, Any]] = sorted(by_key.values(), key=lambda x: (x["source"], x["label"], x["value"]))
    n = write_jsonl(args.output, rows)
    src_counts: Dict[str, int] = {}
    for r in rows:
        src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
    print(f"[pii_index] {n} distinct values -> {args.output}  ({src_counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
