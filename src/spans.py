from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from common import PLACEHOLDER_RE


def _split_placeholders(text: str) -> Tuple[List[str], List[str]]:
    literals: List[str] = []
    labels: List[str] = []
    pos = 0
    for m in PLACEHOLDER_RE.finditer(text):
        literals.append(text[pos : m.start()])
        labels.append(m.group(1))
        pos = m.end()
    literals.append(text[pos:])
    return literals, labels


def _runs(sorted_idx: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i in sorted_idx:
        if out and out[-1][1] == i:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return [(s, e) for s, e in out]


@dataclass
class Alignment:
    spans: List[Dict[str, Any]] = field(default_factory=list)
    removals: List[Tuple[int, int]] = field(default_factory=list)
    changed_chars: Set[int] = field(default_factory=set)
    ws_removed_chars: int = 0
    inserted_chars: int = 0
    inserted_nonspace: int = 0
    ambiguous_adjacent: int = 0
    inserted_placeholders: int = 0
    native_brackets: int = 0
    n_placeholders: int = 0
    truncated: bool = False
    empty_output: bool = False
    repaired: bool = False

    @property
    def removed_chars(self) -> int:
        return sum(e - s for s, e in self.removals)

    @property
    def well_formed(self) -> bool:
        return (
            not self.empty_output
            and self.removed_chars == 0
            and self.ambiguous_adjacent == 0
            and self.inserted_nonspace == 0
        )

    def to_diag(self) -> Dict[str, Any]:
        return {
            "n_placeholders": self.n_placeholders,
            "ambiguous_adjacent": self.ambiguous_adjacent,
            "inserted_placeholders": self.inserted_placeholders,
            "native_brackets_copied": self.native_brackets,
            "unlabeled_removals": len(self.removals),
            "unlabeled_removed_chars": self.removed_chars,
            "ws_removed_chars": self.ws_removed_chars,
            "inserted_chars": self.inserted_chars,
            "inserted_nonspace": self.inserted_nonspace,
            "truncated_before_align": self.truncated,
            "boundary_repaired": self.repaired,
            "empty_output": self.empty_output,
            "well_formed": self.well_formed,
        }


def _trim_span(source: str, start: int, end: int, label: str) -> Dict[str, Any]:
    s = max(0, min(int(start), len(source)))
    e = max(0, min(int(end), len(source)))
    if e < s:
        s, e = e, s
    while s < e and source[s].isspace():
        s += 1
    while e > s and source[e - 1].isspace():
        e -= 1
    return {"start": s, "end": e, "label": label}


def _anchor_difflib(source, lit_concat, owner, k):
    matched_src = bytearray(len(source))
    matched_lit = bytearray(len(lit_concat))
    first_src = [None] * (k + 1)
    last_src = [None] * (k + 1)
    n_matched = 0
    if lit_concat:
        sm = difflib.SequenceMatcher(None, source, lit_concat, autojunk=False)
        for a, b, size in sm.get_matching_blocks():
            if size == 0:
                continue
            n_matched += size
            for t in range(size):
                si, li = a + t, b + t
                o = owner[li]
                matched_src[si] = 1
                matched_lit[li] = 1
                if first_src[o] is None:
                    first_src[o] = si
                last_src[o] = si
    return matched_src, matched_lit, first_src, last_src, n_matched


def _anchor_verbatim(source, literals, lit_concat, k):
    anchors = [None] * (k + 1)
    hi_limit = len(source)
    for i in range(k, -1, -1):
        lit = literals[i]
        if lit == "":
            continue
        q = source.rfind(lit, 0, hi_limit)
        if q < 0:
            return None
        anchors[i] = (q, q + len(lit))
        hi_limit = q

    matched_src = bytearray(len(source))
    first_src = [None] * (k + 1)
    last_src = [None] * (k + 1)
    n_matched = 0
    for i, a in enumerate(anchors):
        if a is None:
            continue
        s, e = a
        for si in range(s, e):
            matched_src[si] = 1
        first_src[i], last_src[i] = s, e - 1
        n_matched += e - s
    matched_lit = bytearray(b"\x01" * len(lit_concat))
    return matched_src, matched_lit, first_src, last_src, n_matched


def _assemble(source, literals, labels, lit_concat, anchors, truncated):
    matched_src, matched_lit, first_src, last_src, n_matched = anchors
    k = len(labels)
    al = Alignment()
    al.truncated = truncated
    al.n_placeholders = k

    al.inserted_chars = len(lit_concat) - n_matched
    if al.inserted_chars:
        al.inserted_nonspace = sum(
            1 for i, ch in enumerate(lit_concat) if not matched_lit[i] and not ch.isspace()
        )

    raw = []
    cursor = 0
    j = 1
    while j <= k:
        pe = last_src[j - 1]
        lo = cursor if pe is None else max(cursor, pe + 1)
        m = None
        for t in range(j, k + 1):
            if first_src[t] is not None:
                m = t
                break
        hi = len(source) if m is None else int(first_src[m])
        if hi < lo:
            hi = lo
        run_last = k if m is None else m

        raw.append((lo, hi, labels[j - 1]))
        for t in range(j + 1, run_last + 1):
            raw.append((hi, hi, labels[t - 1]))
            al.ambiguous_adjacent += 1

        cursor = hi if m is None else int(last_src[m]) + 1
        j = run_last + 1

    kept_raw = []
    for lo, hi, lab in raw:
        if source[lo:hi].strip() == "[" + lab + "]":
            al.native_brackets += 1
            for i in range(lo, hi):
                matched_src[i] = 1
            continue
        kept_raw.append((lo, hi, lab))
    raw = kept_raw

    unmatched = [i for i in range(len(source)) if not matched_src[i]]
    al.changed_chars = set(unmatched)
    covered = set()
    for s, e, _ in raw:
        covered.update(range(s, e))
    leftover = sorted(set(unmatched) - covered)
    for s, e in _runs(leftover):
        if source[s:e].strip() == "":
            al.ws_removed_chars += e - s
        else:
            al.removals.append((s, e))

    al.inserted_placeholders = sum(1 for lo, hi, _ in raw if lo == hi)
    spans = [_trim_span(source, s, e, lab) for s, e, lab in raw]
    spans.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    dedup, seen = [], set()
    for sp in spans:
        key = (sp["start"], sp["end"], sp["label"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(sp)
    al.spans = dedup
    return al


def recover_spans(source: str, predicted: str, max_pred_ratio: float = 4.0) -> Alignment:
    source = source or ""
    predicted = predicted or ""

    truncated = False
    limit = int(max(64, max_pred_ratio * max(len(source), 1)))
    if len(predicted) > limit:
        predicted = predicted[:limit]
        truncated = True

    literals, labels = _split_placeholders(predicted)
    k = len(labels)

    if not predicted.strip():
        al = Alignment()
        al.truncated = truncated
        al.n_placeholders = k
        al.empty_output = True
        if source:
            al.changed_chars.update(range(len(source)))
            if source.strip():
                al.removals.append((0, len(source)))
            else:
                al.ws_removed_chars += len(source)
        return al

    lit_concat = "".join(literals)
    owner = []
    for i, lit in enumerate(literals):
        owner.extend([i] * len(lit))

    al = _assemble(
        source, literals, labels, lit_concat,
        _anchor_difflib(source, lit_concat, owner, k), truncated,
    )

    if al.removed_chars > 0 and k:
        alt_anchors = _anchor_verbatim(source, literals, lit_concat, k)
        if alt_anchors is not None:
            alt = _assemble(
                source, literals, labels, lit_concat, alt_anchors, truncated
            )
            if alt.removed_chars < al.removed_chars and len(alt.spans) == len(al.spans):
                alt.repaired = True
                al = alt
    return al


def spans_to_masked_text(source: str, spans: Sequence[Dict[str, Any]]) -> str:
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    out: List[str] = []
    prev = 0
    for sp in ordered:
        s, e = int(sp["start"]), int(sp["end"])
        if s < prev:
            continue
        out.append(source[prev:s])
        out.append("[" + str(sp["label"]) + "]")
        prev = e
    out.append(source[prev:])
    return "".join(out)


def resolve_overlaps(
    spans: Sequence[Dict[str, Any]], priority: Optional[Dict[str, int]] = None
) -> List[Dict[str, Any]]:
    priority = priority or {}

    def key(sp: Dict[str, Any]):
        return (-priority.get(sp["label"], 0), -(int(sp["end"]) - int(sp["start"])), int(sp["start"]))

    kept: List[Dict[str, Any]] = []
    for sp in sorted(spans, key=key):
        if any(
            not (int(sp["end"]) <= int(kp["start"]) or int(sp["start"]) >= int(kp["end"]))
            for kp in kept
        ):
            continue
        kept.append(sp)
    kept.sort(key=lambda s: int(s["start"]))
    return kept


def char_set(spans: Sequence[Dict[str, Any]]) -> Set[int]:
    out: Set[int] = set()
    for sp in spans:
        out.update(range(int(sp["start"]), int(sp["end"])))
    return out
