from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from common import normalize_ws

PROTECTED_NUMERIC = {"IPV4", "IPV6"}

LABEL_ALIASES: Dict[str, str] = {
    "FIRSTNAME": "GIVENNAME",
    "MIDDLENAME": "GIVENNAME",
    "PREFIX": "TITLE",
    "SUFFIX": "TITLE",
    "PHONENUMBER": "TEL",
    "PASSWORD": "PASS",
    "PIN": "PASS",
    "SSN": "SOCIALNUMBER",
    "ACCOUNTNUMBER": "IDCARD",
    "CREDITCARDNUMBER": "IDCARD",
    "CREDITCARDCVV": "IDCARD",
    "MASKEDNUMBER": "IDCARD",
    "IBAN": "IDCARD",
    "BIC": "IDCARD",
    "PHONEIMEI": "IDCARD",
    "BITCOINADDRESS": "IDCARD",
    "ETHEREUMADDRESS": "IDCARD",
    "LITECOINADDRESS": "IDCARD",
    "VEHICLEVIN": "IDCARD",
    "VEHICLEVRM": "IDCARD",
    "IPV4": "IP",
    "IPV6": "IP",
    "MAC": "IP",
    "DOB": "BOD",
    "ZIPCODE": "POSTCODE",
    "COUNTY": "STATE",
    "BUILDINGNUMBER": "BUILDING",
    "SECONDARYADDRESS": "SECADDRESS",
    "NEARBYGPSCOORDINATE": "GEOCOORD",
    "GENDER": "SEX",
}

CRITERIA = ("exact", "iou50")
LABEL_MODES = ("normalized", "strict", "agnostic", "mapped")

METRIC_BLOCKS: Tuple[Tuple[str, str, str], ...] = (
    ("exact", "normalized", "exact"),
    ("iou50", "normalized", "overlap_iou50"),
    ("exact", "strict", "exact_strict_label"),
    ("iou50", "strict", "overlap_iou50_strict_label"),
    ("exact", "agnostic", "exact_label_agnostic"),
    ("iou50", "agnostic", "overlap_iou50_label_agnostic"),
    ("exact", "mapped", "exact_mapped"),
    ("iou50", "mapped", "overlap_iou50_mapped"),
)

_TRAILING_DIGITS = re.compile(r"\d+$")


def normalize_label(label: str) -> str:
    lab = (label or "").strip().upper()
    if lab in PROTECTED_NUMERIC:
        return lab
    stem = _TRAILING_DIGITS.sub("", lab)
    return stem if len(stem) >= 3 else lab


def label_key(label: str, mode: str) -> str:
    if mode == "agnostic":
        return ""
    if mode == "strict":
        return (label or "").strip().upper()
    norm = normalize_label(label)
    if mode == "mapped":
        return LABEL_ALIASES.get(norm, norm)
    return norm


def span_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    a1, a2 = int(a["start"]), int(a["end"])
    b1, b2 = int(b["start"]), int(b["end"])
    if a1 == a2 and b1 == b2:
        return 1.0 if a1 == b1 else 0.0
    inter = max(0, min(a2, b2) - max(a1, b1))
    union = (a2 - a1) + (b2 - b1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_spans(
    gold: Sequence[Dict[str, Any]],
    pred: Sequence[Dict[str, Any]],
    criterion: str = "iou50",
    mode: str = "normalized",
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    gk = [label_key(g["label"], mode) for g in gold]
    pk = [label_key(p["label"], mode) for p in pred]
    pairs: List[Tuple[int, int]] = []
    used_g: Set[int] = set()
    used_p: Set[int] = set()

    if criterion == "exact":
        buckets: Dict[Tuple[int, int, str], List[int]] = collections.defaultdict(list)
        for pi, p in enumerate(pred):
            buckets[(int(p["start"]), int(p["end"]), pk[pi])].append(pi)
        for gi, g in enumerate(gold):
            key = (int(g["start"]), int(g["end"]), gk[gi])
            cand = buckets.get(key)
            while cand:
                pi = cand.pop(0)
                if pi in used_p:
                    continue
                pairs.append((gi, pi))
                used_g.add(gi)
                used_p.add(pi)
                break
    else:
        cands: List[Tuple[float, int, int]] = []
        for gi, g in enumerate(gold):
            for pi, p in enumerate(pred):
                if gk[gi] != pk[pi]:
                    continue
                v = span_iou(g, p)
                if v >= 0.5:
                    cands.append((v, gi, pi))
        cands.sort(key=lambda t: (-t[0], t[1], t[2]))
        for _v, gi, pi in cands:
            if gi in used_g or pi in used_p:
                continue
            pairs.append((gi, pi))
            used_g.add(gi)
            used_p.add(pi)

    unmatched_g = [i for i in range(len(gold)) if i not in used_g]
    unmatched_p = [i for i in range(len(pred)) if i not in used_p]
    return pairs, unmatched_g, unmatched_p


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


class EntityScorer:
    def __init__(self, criterion: str, mode: str):
        self.criterion = criterion
        self.mode = mode
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.cls_tp: Dict[str, int] = collections.Counter()
        self.cls_fp: Dict[str, int] = collections.Counter()
        self.cls_fn: Dict[str, int] = collections.Counter()
        self.support: Dict[str, int] = collections.Counter()

    def add(self, gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]]):
        pairs, ug, up = match_spans(gold, pred, self.criterion, self.mode)
        for gi, _pi in pairs:
            c = normalize_label(gold[gi]["label"])
            self.tp += 1
            self.cls_tp[c] += 1
            self.support[c] += 1
        for gi in ug:
            c = normalize_label(gold[gi]["label"])
            self.fn += 1
            self.cls_fn[c] += 1
            self.support[c] += 1
        for pi in up:
            c = normalize_label(pred[pi]["label"])
            self.fp += 1
            self.cls_fp[c] += 1
        return pairs, ug, up

    def summary(self) -> Dict[str, float]:
        p, r, f = _prf(self.tp, self.fp, self.fn)
        f1s = []
        for c, sup in self.support.items():
            if sup <= 0:
                continue
            _, _, cf = _prf(self.cls_tp[c], self.cls_fp[c], self.cls_fn[c])
            f1s.append(cf)
        return {
            "micro_p": p,
            "micro_r": r,
            "micro_f1": f,
            "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }

    def per_class(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        classes = set(self.support) | set(self.cls_fp)
        for c in sorted(classes, key=lambda x: (-self.support.get(x, 0), x)):
            p, r, f = _prf(self.cls_tp[c], self.cls_fp[c], self.cls_fn[c])
            out[c] = {
                "p": p,
                "r": r,
                "f1": f,
                "support": int(self.support.get(c, 0)),
                "tp": int(self.cls_tp[c]),
                "fp": int(self.cls_fp[c]),
                "fn": int(self.cls_fn[c]),
            }
        return out


def find_leaks(
    predicted_text: str, gold_mask: Sequence[Dict[str, Any]], min_len: int = 1
) -> List[Dict[str, Any]]:
    hay = normalize_ws(predicted_text or "")
    out: List[Dict[str, Any]] = []
    for m in gold_mask:
        val = normalize_ws(m.get("value", ""))
        if len(val) < max(1, min_len):
            continue
        if val and val in hay:
            out.append({"value": m.get("value", ""), "label": m.get("label", ""), "len": len(val)})
    return out


class RateAccumulator:
    def __init__(self):
        self.num = 0
        self.den = 0

    def add(self, num: int, den: int):
        self.num += num
        self.den += den

    @property
    def rate(self) -> float:
        return self.num / self.den if self.den else 0.0


def over_masking_counts(
    source: str, gold_mask: Sequence[Dict[str, Any]], changed_chars: Set[int]
) -> Tuple[int, int, int, int]:
    pii: Set[int] = set()
    for m in gold_mask:
        pii.update(range(int(m["start"]), int(m["end"])))
    den = 0
    num = 0
    den_ns = 0
    num_ns = 0
    for i, ch in enumerate(source):
        if i in pii:
            continue
        den += 1
        is_ws = ch.isspace()
        if not is_ws:
            den_ns += 1
        if i in changed_chars:
            num += 1
            if not is_ws:
                num_ns += 1
    return num, den, num_ns, den_ns


@dataclass
class ErrorTaxonomy:
    missed_no_overlap: collections.Counter = field(default_factory=collections.Counter)
    boundary_error: collections.Counter = field(default_factory=collections.Counter)
    wrong_label: collections.Counter = field(default_factory=collections.Counter)
    boundary_and_label: collections.Counter = field(default_factory=collections.Counter)
    covered_by_unlabeled_removal: collections.Counter = field(default_factory=collections.Counter)
    spurious: collections.Counter = field(default_factory=collections.Counter)
    label_confusion: collections.Counter = field(default_factory=collections.Counter)
    leaked_by_class: collections.Counter = field(default_factory=collections.Counter)
    missed_and_leaked: collections.Counter = field(default_factory=collections.Counter)

    def add_doc(
        self,
        gold: Sequence[Dict[str, Any]],
        pred: Sequence[Dict[str, Any]],
        removals: Sequence[Tuple[int, int]],
        leaked_values: Sequence[Dict[str, Any]],
    ) -> None:
        pairs, ug, up = match_spans(gold, pred, "iou50", "normalized")
        leaked_norm = {normalize_ws(l["value"]) for l in leaked_values}
        removal_chars: Set[int] = set()
        for s, e in removals:
            removal_chars.update(range(s, e))

        for gi in ug:
            g = gold[gi]
            gc = normalize_label(g["label"])
            gs, ge = int(g["start"]), int(g["end"])
            best_iou = 0.0
            best_label: Optional[str] = None
            any_overlap = False
            for p in pred:
                v = span_iou(g, p)
                if max(0, min(ge, int(p["end"])) - max(gs, int(p["start"]))) > 0:
                    any_overlap = True
                if v > best_iou:
                    best_iou = v
                    best_label = normalize_label(p["label"])
            if not any_overlap:
                if removal_chars & set(range(gs, ge)):
                    self.covered_by_unlabeled_removal[gc] += 1
                else:
                    self.missed_no_overlap[gc] += 1
            elif best_label == gc:
                self.boundary_error[gc] += 1
            elif best_iou >= 0.5:
                self.wrong_label[gc] += 1
                self.label_confusion[f"{gc}->{best_label}"] += 1
            else:
                self.boundary_and_label[gc] += 1
                if best_label:
                    self.label_confusion[f"{gc}->{best_label}"] += 1
            if normalize_ws(g.get("value", "")) in leaked_norm:
                self.missed_and_leaked[gc] += 1

        for pi in up:
            self.spurious[normalize_label(pred[pi]["label"])] += 1

        for l in leaked_values:
            self.leaked_by_class[normalize_label(l["label"])] += 1

    def to_dict(self, top: int = 25) -> Dict[str, Any]:
        def d(c: collections.Counter) -> Dict[str, int]:
            return {k: int(v) for k, v in c.most_common(top)}

        return {
            "totals": {
                "missed_no_overlap": int(sum(self.missed_no_overlap.values())),
                "covered_by_unlabeled_removal": int(sum(self.covered_by_unlabeled_removal.values())),
                "boundary_error_correct_label": int(sum(self.boundary_error.values())),
                "wrong_label_good_span": int(sum(self.wrong_label.values())),
                "boundary_and_label": int(sum(self.boundary_and_label.values())),
                "spurious_predictions": int(sum(self.spurious.values())),
                "missed_and_leaked": int(sum(self.missed_and_leaked.values())),
            },
            "missed_no_overlap_by_class": d(self.missed_no_overlap),
            "covered_by_unlabeled_removal_by_class": d(self.covered_by_unlabeled_removal),
            "boundary_error_by_class": d(self.boundary_error),
            "wrong_label_by_class": d(self.wrong_label),
            "boundary_and_label_by_class": d(self.boundary_and_label),
            "spurious_by_class": d(self.spurious),
            "top_label_confusions": d(self.label_confusion),
            "leaked_by_class": d(self.leaked_by_class),
            "missed_and_leaked_by_class": d(self.missed_and_leaked),
        }


def evaluate_predictions(
    records: Sequence[Dict[str, Any]],
    per_class_top: int = 10,
) -> Dict[str, Any]:
    scorers = {key: EntityScorer(crit, mode) for crit, mode, key in METRIC_BLOCKS}
    tax = ErrorTaxonomy()
    leak = RateAccumulator()
    leak3 = RateAccumulator()
    om = RateAccumulator()
    om_ns = RateAccumulator()
    doc_leak = RateAccumulator()
    wf = RateAccumulator()
    lat: List[float] = []
    diag_tot: collections.Counter = collections.Counter()
    n_gold = 0
    n_pred = 0

    for rec in records:
        gold = rec["gold_spans"]
        pred = rec["predicted_spans"]
        src = rec["source_text"]
        n_gold += len(gold)
        n_pred += len(pred)
        for s in scorers.values():
            s.add(gold, pred)

        leaks = rec.get("leaked_values")
        if leaks is None:
            leaks = find_leaks(rec["predicted_text"], gold, min_len=1)
        leak.add(len(leaks), len(gold))
        leak3.add(
            len(find_leaks(rec["predicted_text"], gold, min_len=3)),
            sum(1 for m in gold if len(normalize_ws(m.get("value", ""))) >= 3),
        )
        doc_leak.add(1 if leaks else 0, 1)

        changed = rec.get("changed_chars") or set()
        if not isinstance(changed, set):
            changed = set(changed)
        num, den, num_ns, den_ns = over_masking_counts(src, gold, changed)
        om.add(num, den)
        om_ns.add(num_ns, den_ns)

        diag = rec.get("diag") or {}
        wf.add(1 if diag.get("well_formed") else 0, 1)
        for k, v in diag.items():
            if isinstance(v, bool):
                diag_tot[k] += int(v)
            elif isinstance(v, (int, float)):
                diag_tot[k] += v
        tax.add_doc(gold, pred, rec.get("removals") or [], leaks)
        if rec.get("latency_ms") is not None:
            lat.append(float(rec["latency_ms"]))

    out: Dict[str, Any] = {"n": len(records), "n_gold_spans": n_gold, "n_pred_spans": n_pred}
    for _crit, _mode, key in METRIC_BLOCKS:
        out[key] = scorers[key].summary()

    pc_primary = scorers["overlap_iou50"].per_class()
    pc_exact = scorers["exact"].per_class()
    top_classes = [c for c in list(pc_primary.keys()) if pc_primary[c]["support"] > 0][:per_class_top]
    out["per_class"] = {c: pc_primary[c] for c in top_classes}
    out["per_class_all"] = pc_primary
    out["per_class_exact"] = pc_exact
    out["per_class_top10"] = top_classes
    out["leak_rate"] = leak.rate
    out["leak_rate_len_ge_3"] = leak3.rate
    out["leaked_values_total"] = leak.num
    out["doc_level_leak_rate"] = doc_leak.rate
    out["over_masking_rate"] = om.rate
    out["over_masking_rate_nonspace"] = om_ns.rate
    out["over_masked_chars"] = om.num
    out["non_pii_chars"] = om.den
    out["well_formed_rate"] = wf.rate
    out["alignment_diagnostics"] = {k: (int(v) if float(v).is_integer() else v) for k, v in diag_tot.items()}
    out["error_taxonomy"] = tax.to_dict()
    if lat:
        lat_sorted = sorted(lat)
        out["latency"] = {
            "mean_ms": sum(lat) / len(lat),
            "p50_ms": lat_sorted[len(lat_sorted) // 2],
            "p95_ms": lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))],
        }
    return out
