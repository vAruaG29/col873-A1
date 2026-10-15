from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import read_jsonl

EXPECTED_ROWS = {
    "partA_train.jsonl": 20000,
    "partA_validation.jsonl": 2000,
    "partB_test.jsonl": 2000,
    "partB_ood_test.jsonl": 1000,
    "partC_corpus_raw.jsonl": 8007,
    "partC_heldout.jsonl": 500,
    "partC_canaries.jsonl": 20,
    "partC_targeted.jsonl": 400,
}


def check_manifest(data_dir: str) -> List[str]:
    problems: List[str] = []
    man = os.path.join(data_dir, "manifest.jsonl")
    if not os.path.exists(man):
        return [f"missing manifest: {man}"]
    for line in open(man, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        path = os.path.join(data_dir, item["file"])
        if not os.path.exists(path):
            problems.append(f"{item['file']}: MISSING")
            continue
        payload = open(path, "rb").read()
        digest = hashlib.sha256(payload).hexdigest()
        rows = len(payload.splitlines())
        ok = True
        if len(payload) != item["bytes"]:
            problems.append(f"{item['file']}: bytes {len(payload)} != {item['bytes']}")
            ok = False
        if rows != item["rows"]:
            problems.append(f"{item['file']}: rows {rows} != {item['rows']}")
            ok = False
        if digest != item["sha256"]:
            problems.append(f"{item['file']}: sha256 mismatch")
            ok = False
        exp = EXPECTED_ROWS.get(item["file"])
        if exp is not None and item["rows"] != exp:
            problems.append(f"{item['file']}: manifest rows {item['rows']} != assignment's {exp}")
        print(f"  {'OK  ' if ok else 'FAIL'} {item['file']:28s} rows={rows:<6d} bytes={len(payload):<10d} sha256={digest[:16]}...")
    return problems


def check_ab(data_dir: str, name: str) -> List[str]:
    problems: List[str] = []
    rows = read_jsonl(os.path.join(data_dir, name))
    bad_off = 0
    bad_target = 0
    labels: collections.Counter = collections.Counter()
    for r in rows:
        src = r["source_text"]
        for m in r["privacy_mask"]:
            labels[m["label"]] += 1
            if src[m["start"] : m["end"]] != m["value"]:
                bad_off += 1
        parts: List[str] = []
        prev = 0
        for m in sorted(r["privacy_mask"], key=lambda x: x["start"]):
            parts.append(src[prev : m["start"]])
            parts.append("[" + m["label"] + "]")
            prev = m["end"]
        parts.append(src[prev:])
        if "".join(parts) != r["target_text"]:
            bad_target += 1
    if bad_off:
        problems.append(f"{name}: {bad_off} spans where source_text[start:end] != value")
    if bad_target:
        problems.append(f"{name}: {bad_target} rows where target_text != span substitution")
    print(f"  {'OK  ' if not (bad_off or bad_target) else 'FAIL'} {name:28s} "
          f"rows={len(rows)} spans={sum(labels.values())} labels={len(labels)} "
          f"offset_errors={bad_off} target_mismatch={bad_target}")
    return problems


def check_c(data_dir: str) -> List[str]:
    problems: List[str] = []
    corpus = read_jsonl(os.path.join(data_dir, "partC_corpus_raw.jsonl"))
    heldout = read_jsonl(os.path.join(data_dir, "partC_heldout.jsonl"))
    canaries = read_jsonl(os.path.join(data_dir, "partC_canaries.jsonl"))
    targeted = read_jsonl(os.path.join(data_dir, "partC_targeted.jsonl"))

    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in corpus:
        groups[r["doc_id"]].append(r)
    for doc_id, rs in groups.items():
        n_reps = int(rs[0]["n_reps"])
        if len(rs) != n_reps:
            problems.append(f"{doc_id}: {len(rs)} rows but n_reps={n_reps}")
        if sorted(int(x["repetition_index"]) for x in rs) != list(range(n_reps)):
            problems.append(f"{doc_id}: repetition_index is not 0..n_reps-1")
        if len({x["text"] for x in rs}) != 1:
            problems.append(f"{doc_id}: repeated rows differ in text")
    reps = collections.Counter(int(v[0]["n_reps"]) for v in groups.values() if not v[0]["is_canary"])
    creps = collections.Counter(int(c["n_reps"]) for c in canaries)
    print(f"  OK   partC corpus: {len(corpus)} rows, {len(groups)} doc_ids, "
          f"non-canary n_reps={dict(sorted(reps.items()))}")
    print(f"  OK   partC canaries: {len(canaries)}, n_reps={dict(sorted(creps.items()))}")

    ctexts = {r["text"] for r in corpus}
    overlap = sum(1 for r in heldout if r["text"] in ctexts)
    if overlap:
        problems.append(f"partC_heldout: {overlap} documents also occur in the training corpus")
    print(f"  {'OK  ' if not overlap else 'FAIL'} partC heldout: {len(heldout)} docs, "
          f"corpus overlap={overlap}")

    by_doc = {d: v[0]["text"] for d, v in groups.items()}
    miss = 0
    for t in targeted:
        if t["split"] != "member":
            continue
        txt = by_doc.get(t["doc_id"])
        if txt is None or (t["prompt"] + t["gold_value"]) not in txt:
            miss += 1
    if miss:
        problems.append(f"partC_targeted: {miss} member prompts are not verbatim in their document")
    splits = collections.Counter(t["split"] for t in targeted)
    print(f"  {'OK  ' if not miss else 'FAIL'} partC targeted: {dict(splits)}, "
          f"member prompt+gold not found={miss}")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify the frozen dataset bundle")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--skip_content", action="store_true", help="digests only")
    args = p.parse_args(argv)

    print(f"== manifest check ({args.data_dir}) ==")
    problems = check_manifest(args.data_dir)
    if not args.skip_content:
        print("== record invariants ==")
        for name in ("partA_train.jsonl", "partA_validation.jsonl", "partB_test.jsonl",
                     "partB_ood_test.jsonl"):
            problems += check_ab(args.data_dir, name)
        problems += check_c(args.data_dir)

    print()
    if problems:
        print(f"FAILED with {len(problems)} problem(s):")
        for x in problems:
            print(f"  - {x}")
        return 1
    print("dataset bundle OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
