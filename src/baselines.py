from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


P_PATTERN = 30
P_ID = 25
P_DATETIME = 20
P_CUE = 10
P_NAME = 5


_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

PATTERN_RULES: List[Tuple[str, int, "re.Pattern[str]", int, Optional[str]]] = [
    ("EMAIL", P_PATTERN, re.compile(r"[^\s@<>\"',;:()\[\]]+@[^\s@<>\"',;:()\[\]]+\.[A-Za-z]{2,}"), 0, None),
    ("URL", P_PATTERN, re.compile(r"(?:https?://|www\.)[^\s<>\"'\]),]+"), 0, None),
    ("IP", P_PATTERN, re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"), 0, None),
    ("IP", P_PATTERN, re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b"), 0, None),
    ("GEOCOORD", P_PATTERN, re.compile(r"\[\s*-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+\s*\]"), 0, None),
    ("TEL", P_PATTERN, re.compile(r"\+\d[\d\s().\-]{5,20}\d"), 0, None),
    ("IDCARD", P_ID, re.compile(r"\b\d{13,19}\b"), 0, "luhn_or_long"),
    ("IDCARD", P_ID, re.compile(r"\b[A-Z]{2}\d{5}[A-Z]{2}\b"), 0, None),
    ("SOCIALNUMBER", P_ID, re.compile(r"\b\d{3}[-.]\d{2}[-.]\d{4}\b"), 0, None),
    ("SOCIALNUMBER", P_ID, re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b"), 0, None),
    ("POSTCODE", P_ID, re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s\d[A-Z]{2}\b"), 0, None),
    ("DATE", P_DATETIME, re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?\b"), 0, None),
    ("DATE", P_DATETIME, re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), 0, None),
    ("DATE", P_DATETIME, re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH + r"\.?\s+\d{4}\b"), 0, None),
    ("DATE", P_DATETIME, re.compile(_MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b"), 0, None),
    ("BOD", P_DATETIME, re.compile(_MONTH + r"\s*/\s*\d{2}\b"), 0, None),
    ("TIME", P_DATETIME, re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap]\.?[Mm]\.?)?"), 0, None),
    ("TIME", P_DATETIME, re.compile(r"\b\d{1,2}\s*o'clock\b"), 0, None),
    ("TIME", P_DATETIME, re.compile(r"\b\d{1,2}\s*[APap]\.?[Mm]\.?\b"), 0, None),
    ("SEX", P_NAME, re.compile(r"\b(?:Male|Female|Non-binary|Nonbinary|Masculine|Feminine|Intersex)\b"), 0, None),
    ("TITLE", P_NAME, re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Rev|Sir|Lady|Lord|Major|Captain|Colonel|Prelate|Solicitor|Baron|Baroness|Count|Countess)\b\.?"), 0, None),
]


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _validate(kind: Optional[str], text: str) -> bool:
    if kind is None:
        return True
    if kind == "luhn_or_long":
        return _luhn(text) or len(text) >= 11
    return True


CUE_MAP: List[Tuple[str, str]] = [
    ("social security number", "SOCIALNUMBER"),
    ("social number", "SOCIALNUMBER"),
    ("socialnumber", "SOCIALNUMBER"),
    ("ssn", "SOCIALNUMBER"),
    ("national insurance number", "SOCIALNUMBER"),
    ("identification card number", "IDCARD"),
    ("id card number", "IDCARD"),
    ("id card", "IDCARD"),
    ("idcard", "IDCARD"),
    ("identity card", "IDCARD"),
    ("card number", "IDCARD"),
    ("driver's license", "DRIVERLICENSE"),
    ("drivers license", "DRIVERLICENSE"),
    ("driver license", "DRIVERLICENSE"),
    ("driving licence", "DRIVERLICENSE"),
    ("driver licence", "DRIVERLICENSE"),
    ("driverlicense", "DRIVERLICENSE"),
    ("passport", "PASSPORT"),
    ("password", "PASS"),
    ("passcode", "PASS"),
    ("date of birth", "BOD"),
    ("bodily origin date", "BOD"),
    ("birthdate", "BOD"),
    ("dateofbirth", "BOD"),
    ("birth date", "BOD"),
    ("dob", "BOD"),
    ("telephone number", "TEL"),
    ("telephone", "TEL"),
    ("phone number", "TEL"),
    ("contact number", "TEL"),
    ("mobile number", "TEL"),
    ("phone", "TEL"),
    ("ip address", "IP"),
    ("ipaddress", "IP"),
    ("email address", "EMAIL"),
    ("emailaddress", "EMAIL"),
    ("email", "EMAIL"),
    ("username", "USERNAME"),
    ("user name", "USERNAME"),
    ("given name", "GIVENNAME1"),
    ("givenname", "GIVENNAME1"),
    ("first name", "GIVENNAME1"),
    ("firstname", "GIVENNAME1"),
    ("middle name", "GIVENNAME2"),
    ("middlename", "GIVENNAME2"),
    ("last name", "LASTNAME1"),
    ("lastname", "LASTNAME1"),
    ("surname", "LASTNAME1"),
    ("secondary address", "SECADDRESS"),
    ("secondaryaddress", "SECADDRESS"),
    ("second address", "SECADDRESS"),
    ("secaddress", "SECADDRESS"),
    ("postal code", "POSTCODE"),
    ("postcode", "POSTCODE"),
    ("zip code", "POSTCODE"),
    ("zipcode", "POSTCODE"),
    ("building number", "BUILDING"),
    ("building", "BUILDING"),
    ("street", "STREET"),
    ("city", "CITY"),
    ("country", "COUNTRY"),
    ("state", "STATE"),
    ("province", "STATE"),
    ("geographical coordinates", "GEOCOORD"),
    ("geocoordinates", "GEOCOORD"),
    ("coordinates", "GEOCOORD"),
    ("card issuer", "CARDISSUER"),
    ("cardissuer", "CARDISSUER"),
    ("timestamp", "TIME"),
    ("time", "TIME"),
    ("date", "DATE"),
    ("gender", "SEX"),
    ("sex", "SEX"),
    ("title", "TITLE"),
]

_CUE_ALT = "|".join(re.escape(c) for c, _ in sorted(CUE_MAP, key=lambda kv: -len(kv[0])))
_CUE_TO_LABEL = {c: l for c, l in CUE_MAP}

CUE_RE = re.compile(
    r"(?i)\b(?P<cue>" + _CUE_ALT + r")"
    r"(?P<mid>[A-Za-z_ ]{0,18}?)"
    r"[\"']?\s*(?:number|no\.?|#|id)?\s*[\"']?\s*"
    r"(?::|=|>)\s*"
    r"[\"']?\s*"
    r"(?P<val>[^\n\r\"'<,;}\]]{1,48})"
)

_VALUE_TRIM = " \t .,;:!?)('\"-_/\\"

_CUE_MAX_TOKENS = {"TITLE": 2, "SEX": 3}
_CUE_MAX_CHARS = {"TITLE": 20, "SEX": 22}

_SALUTATION_STOP = {
    "members", "member", "students", "student", "owners", "owner", "team",
    "teams", "guardian", "guardians", "parents", "parent", "colleagues",
    "colleague", "customers", "customer", "clients", "client", "sir", "madam",
    "all", "friends", "friend", "forum", "pet", "pets", "valued", "users",
    "user", "residents", "resident", "participants", "participant", "patient",
    "patients", "staff", "everyone", "applicant", "applicants", "subscriber",
    "subscribers", "reader", "readers", "attendee", "attendees", "neighbour",
    "neighbours", "neighbor", "neighbors", "shareholders", "employees",
    "employee", "families", "family", "supporters", "volunteers", "hiring",
    "manager", "committee", "department", "community", "professor", "doctor",
}

_CUE_STOPWORDS = {
    "", "n/a", "na", "none", "null", "no data", "unknown", "not provided",
    "not available", "-", "--", "tbd", "nil",
}


def _cue_spans(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in CUE_RE.finditer(text):
        cue = m.group("cue").lower()
        label = _CUE_TO_LABEL.get(cue)
        if label is None:
            continue
        raw = m.group("val")
        start = m.start("val")
        stripped = raw.strip(_VALUE_TRIM)
        if not stripped or stripped.lower() in _CUE_STOPWORDS:
            continue
        if label in _CUE_MAX_TOKENS:
            if len(stripped.split()) > _CUE_MAX_TOKENS[label]:
                continue
            if len(stripped) > _CUE_MAX_CHARS.get(label, 48):
                continue
        off = raw.find(stripped)
        s = start + (off if off >= 0 else 0)
        e = s + len(stripped)
        out.append({"start": s, "end": e, "label": label, "_prio": P_CUE})
    return out


def _dear_name_spans(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rx = re.compile(r"\b(?:Dear|Hello|Hi)\s+((?:[A-Z][\w'\-]+)(?:\s+[A-Z][\w'\-]+){0,2})")
    for m in rx.finditer(text):
        toks: List[Tuple[int, int]] = []
        for tm in re.finditer(r"[A-Z][\w'\-]+", m.group(1)):
            if tm.group(0).lower() in _SALUTATION_STOP:
                continue
            toks.append((m.start(1) + tm.start(), m.start(1) + tm.end()))
        if not toks:
            continue
        if len(toks) == 1:
            out.append({"start": toks[0][0], "end": toks[0][1], "label": "GIVENNAME1", "_prio": P_NAME})
        else:
            out.append({"start": toks[0][0], "end": toks[0][1], "label": "GIVENNAME1", "_prio": P_NAME})
            out.append({"start": toks[-1][0], "end": toks[-1][1], "label": "LASTNAME1", "_prio": P_NAME})
            for s, e in toks[1:-1]:
                out.append({"start": s, "end": e, "label": "GIVENNAME2", "_prio": P_NAME})
    return out


def regex_spans(
    text: str, use_url: bool = True, use_names: bool = True, use_cues: bool = True
) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    for label, prio, rx, grp, val in PATTERN_RULES:
        if label == "URL" and not use_url:
            continue
        if label in ("SEX", "TITLE") and not use_names:
            continue
        for m in rx.finditer(text):
            s, e = m.span(grp)
            frag = text[s:e]
            if not _validate(val, frag):
                continue
            frag2 = frag.strip(_VALUE_TRIM)
            if not frag2:
                continue
            off = frag.find(frag2)
            cands.append({"start": s + off, "end": s + off + len(frag2), "label": label, "_prio": prio})
    if use_cues:
        cands.extend(_cue_spans(text))
    if use_names:
        cands.extend(_dear_name_spans(text))

    kept = _resolve_with_prio(cands)
    for k in kept:
        k.pop("_prio", None)
    return kept


def _resolve_with_prio(cands: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for sp in sorted(cands, key=lambda c: (-c["_prio"], -(c["end"] - c["start"]), c["start"], c["label"])):
        if sp["end"] <= sp["start"]:
            continue
        if any(not (sp["end"] <= k["start"] or sp["start"] >= k["end"]) for k in kept):
            continue
        kept.append(dict(sp))
    kept.sort(key=lambda s: s["start"])
    return kept


PRESIDIO_MAP: Dict[str, str] = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "TEL",
    "IP_ADDRESS": "IP",
    "URL": "URL",
    "CREDIT_CARD": "IDCARD",
    "IBAN_CODE": "IDCARD",
    "US_BANK_NUMBER": "IDCARD",
    "US_ITIN": "IDCARD",
    "UK_NHS": "IDCARD",
    "MEDICAL_LICENSE": "IDCARD",
    "CRYPTO": "IDCARD",
    "US_SSN": "SOCIALNUMBER",
    "US_PASSPORT": "PASSPORT",
    "US_DRIVER_LICENSE": "DRIVERLICENSE",
    "DATE_TIME": "DATE",
    "LOCATION": "CITY",
    "NRP": "COUNTRY",
    "PERSON": "PERSON",
}


def _ensure_spacy_model(name: str) -> None:
    """Install the spaCy pipeline Presidio needs on first use.

    The course requires every model to be fetched by the code itself, with no
    manual ``python -m spacy download`` step. ``spacy.cli.download`` picks the
    pipeline version compatible with the installed spaCy, so it is safer than
    pinning a wheel URL. A failure here is not fatal: create_engine() below
    still reports a missing model with the manual command as a fallback.
    """
    try:
        import spacy
        if spacy.util.is_package(name):
            return
        print(f"[presidio] spaCy model '{name}' not installed; downloading it", flush=True)
        spacy.cli.download(name)
        # make the freshly pip-installed package importable in this process
        import importlib
        import site
        importlib.reload(site)
        importlib.invalidate_caches()
    except SystemExit:
        pass  # spacy.cli.download exits non-zero on failure; fall through
    except Exception as exc:
        print(f"[presidio] automatic download of '{name}' failed: {exc}", flush=True)


class PresidioRunner:
    def __init__(self, language: str = "en", spacy_model: str = "en_core_web_lg"):
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except Exception as exc:
            raise SystemExit(
                "Presidio is not installed. Either\n"
                "  pip install presidio-analyzer presidio-anonymizer\n"
                f"  python -m spacy download {spacy_model}\n"
                "or run the comparison with --system zeroshot instead.\n"
                f"(import error: {exc})"
            )
        _ensure_spacy_model(spacy_model)
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": spacy_model}],
            }
        )
        try:
            engine = provider.create_engine()
        except Exception as exc:
            raise SystemExit(
                f"spaCy model '{spacy_model}' is unavailable: {exc}\n"
                f"Run: python -m spacy download {spacy_model}\n"
                "(or pass --spacy_model en_core_web_sm)"
            )
        self.analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=[language])
        self.language = language

    def spans(self, text: str, score_threshold: float = 0.35) -> List[Dict[str, Any]]:
        results = self.analyzer.analyze(
            text=text, language=self.language, score_threshold=score_threshold
        )
        cands: List[Dict[str, Any]] = []
        for r in results:
            mapped = PRESIDIO_MAP.get(r.entity_type)
            if mapped is None:
                continue
            if mapped == "PERSON":
                toks = [
                    (r.start + m.start(), r.start + m.end())
                    for m in re.finditer(r"\S+", text[r.start : r.end])
                ]
                if not toks:
                    continue
                if len(toks) == 1:
                    cands.append({"start": toks[0][0], "end": toks[0][1], "label": "GIVENNAME1", "_prio": P_NAME})
                else:
                    cands.append({"start": toks[0][0], "end": toks[0][1], "label": "GIVENNAME1", "_prio": P_NAME})
                    cands.append({"start": toks[-1][0], "end": toks[-1][1], "label": "LASTNAME1", "_prio": P_NAME})
                    for s, e in toks[1:-1]:
                        cands.append({"start": s, "end": e, "label": "GIVENNAME2", "_prio": P_NAME})
                continue
            cands.append(
                {"start": r.start, "end": r.end, "label": mapped, "_prio": int(100 * float(r.score))}
            )
        kept = _resolve_with_prio(cands)
        for k in kept:
            k.pop("_prio", None)
        return kept
