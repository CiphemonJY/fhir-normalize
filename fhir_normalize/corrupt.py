#!/usr/bin/env python3
"""
Data Corruption Harness — turn clean Synthea FHIR into messy, real-world clinical data.

Synthea emits *perfect* FHIR (coded SNOMED-CT/LOINC, fully nested). Real clinical feeds
(legacy billing claims, discharge summaries, HL7 dumps) are nothing like that — they're
free-text, abbreviated, mis-coded, and flattened into ad-hoc CSVs. To make the PoC
meaningful, we deliberately degrade the clean data so the Edge Node's semantic-validation
layer has something real to reconstruct.

Three degradations (each independently toggleable):
  1. strip_codes    — drop SNOMED-CT/LOINC codes, keep only (mangled) display text.
  2. free_text      — replace standardized terms with clinician shorthand
                      ("Essential hypertension" -> "pt c/o HTN").
  3. flatten_csv    — collapse the nested FHIR Bundle into a wide, poorly-typed CSV row
                      that mimics a legacy billing/discharge feed.

Usage:
    python poc/corruption_harness.py --in synthea/output/fhir --out poc/corrupted \
        --modes strip_codes,free_text,flatten_csv

Each input *.json Bundle yields:
    <out>/freetext/<id>.txt   (corrupted free-text note — the validator's input)
    <out>/claims.csv          (appended flattened row — legacy-feed mimic)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Clinical shorthand: standardized term -> messy free-text the validator must decode ──
ABBREVIATIONS = {
    "essential hypertension": "pt c/o HTN",
    "hypertension": "HTN",
    "type 2 diabetes mellitus": "T2DM, poorly controlled",
    "diabetes": "DM",
    "hyperlipidemia": "HLD",
    "myocardial infarction": "h/o MI",
    "chronic obstructive pulmonary disease": "COPD exac",
    "asthma": "asthma, mild intermittent",
    "atrial fibrillation": "afib w/ RVR",
    "congestive heart failure": "CHF, EF reduced",
    "acute bronchitis": "acute bronchitis, likely viral",
    "chronic kidney disease": "CKD stage III",
    "body weight": "wt",
    "body height": "ht",
    "heart rate": "HR",
    "respiratory rate": "RR",
    "systolic blood pressure": "SBP",
    "diastolic blood pressure": "DBP",
    "hemoglobin a1c": "A1c",
}
# Random clinician noise sprinkled into the free text.
NOISE = ["", " - stable", ", cont current mgmt", " (per pt report)", " w/u pending",
         ", f/u 2wks", " - see flowsheet", ", no acute distress"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _mess_term(display: str, rng: random.Random) -> str:
    """Map a clean clinical display string to messy shorthand + noise."""
    key = _norm(display)
    text = ABBREVIATIONS.get(key)
    if text is None:
        # Unknown term: keep but abbreviate words and lowercase (mimic sloppy entry).
        text = re.sub(r"\b(\w{8,})\b", lambda m: m.group(1)[:4] + ".", key)
    return text + rng.choice(NOISE)


# ── FHIR extraction (only the bits a messy feed would carry) ──────────────────
def _entries(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e.get("resource", {}) for e in bundle.get("entry", []) if e.get("resource")]


def _patient(resources: List[Dict[str, Any]]) -> Dict[str, Any]:
    return next((r for r in resources if r.get("resourceType") == "Patient"), {})


def _display(concept: Dict[str, Any]) -> str:
    if concept.get("text"):
        return concept["text"]
    for c in concept.get("coding", []):
        if c.get("display"):
            return c["display"]
    return ""


def extract_clinical_facts(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the patient + a flat list of conditions/observations from a FHIR Bundle."""
    res = _entries(bundle)
    pat = _patient(res)
    conditions, observations = [], []
    for r in res:
        rt = r.get("resourceType")
        if rt == "Condition":
            conditions.append({
                "display": _display(r.get("code", {})),
                "snomed": next((c.get("code") for c in r.get("code", {}).get("coding", [])
                                if "snomed" in c.get("system", "")), None),
            })
        elif rt == "Observation":
            val = r.get("valueQuantity", {})
            observations.append({
                "display": _display(r.get("code", {})),
                "loinc": next((c.get("code") for c in r.get("code", {}).get("coding", [])
                               if "loinc" in c.get("system", "")), None),
                "value": val.get("value"), "unit": val.get("unit"),
            })
    return {"patient": pat, "conditions": conditions, "observations": observations}


# ── Corruption stages ─────────────────────────────────────────────────────────
def to_free_text(facts: Dict[str, Any], strip_codes: bool, free_text: bool,
                 rng: random.Random) -> str:
    """Render the facts as a messy clinical note (the validator's reconstruction target)."""
    lines = []
    p = facts["patient"]
    age = p.get("birthDate", "")[:4]
    sex = (p.get("gender", "") or "?")[0].upper()
    lines.append(f"{sex} pt, DOB yr {age}. Hx/problems:")
    for c in facts["conditions"]:
        term = _mess_term(c["display"], rng) if free_text else c["display"]
        code = "" if strip_codes else f" [SNOMED {c['snomed']}]"
        lines.append(f"  - {term}{code}")
    if facts["observations"]:
        lines.append("Vitals/labs:")
        for o in facts["observations"][:8]:
            term = _mess_term(o["display"], rng) if free_text else o["display"]
            code = "" if strip_codes else f" [LOINC {o['loinc']}]"
            v = f" {o['value']} {o['unit']}".rstrip() if o.get("value") is not None else ""
            lines.append(f"  {term}:{v}{code}")
    return "\n".join(lines)


def flatten_to_csv_row(facts: Dict[str, Any]) -> Dict[str, str]:
    """Collapse the nested bundle into one wide, poorly-typed row (legacy-feed mimic)."""
    p = facts["patient"]
    name = ""
    if p.get("name"):
        n = p["name"][0]
        name = " ".join(n.get("given", []) + [n.get("family", "")]).strip()
    # Semicolon-jammed multi-values, no codes, inconsistent casing — like real claims feeds.
    dx = ";".join(_norm(c["display"]) for c in facts["conditions"])
    return {
        "PATIENT_NM": name.upper(),
        "DOB": p.get("birthDate", ""),
        "SEX": (p.get("gender", "") or "")[:1].upper(),
        "DX_FREETEXT": dx,                      # no SNOMED, jammed together
        "PROBLEM_CT": str(len(facts["conditions"])),
        "LAST_A1C": next((str(o.get("value", "")) for o in facts["observations"]
                          if "a1c" in _norm(o["display"])), ""),
    }


# ── Driver ────────────────────────────────────────────────────────────────────
def corrupt_bundle(bundle: Dict[str, Any], modes: set, rng: random.Random) -> Dict[str, Any]:
    facts = extract_clinical_facts(bundle)
    out = {}
    if "free_text" in modes or "strip_codes" in modes:
        out["free_text"] = to_free_text(
            facts, strip_codes=("strip_codes" in modes),
            free_text=("free_text" in modes), rng=rng)
    if "flatten_csv" in modes:
        out["csv_row"] = flatten_to_csv_row(facts)
    out["_facts"] = facts
    return out


def run(in_dir: str, out_dir: str, modes: set, seed: int = 42) -> Dict[str, int]:
    rng = random.Random(seed)
    inp, outp = Path(in_dir), Path(out_dir)
    (outp / "freetext").mkdir(parents=True, exist_ok=True)
    csv_rows: List[Dict[str, str]] = []
    n = 0
    for f in sorted(inp.glob("*.json")):
        try:
            bundle = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if bundle.get("resourceType") != "Bundle":
            continue
        corrupted = corrupt_bundle(bundle, modes, rng)
        stem = f.stem
        if "free_text" in corrupted:
            (outp / "freetext" / f"{stem}.txt").write_text(corrupted["free_text"], encoding="utf-8")
        if "csv_row" in corrupted:
            csv_rows.append(corrupted["csv_row"])
        n += 1
    if csv_rows:
        cols = list(csv_rows[0].keys())
        with open(outp / "claims.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(csv_rows)
    return {"bundles": n, "csv_rows": len(csv_rows)}


def main():
    ap = argparse.ArgumentParser(description="Corrupt clean Synthea FHIR into messy clinical data")
    ap.add_argument("--in", dest="in_dir", required=True, help="dir of Synthea FHIR *.json bundles")
    ap.add_argument("--out", dest="out_dir", default="poc/corrupted")
    ap.add_argument("--modes", default="strip_codes,free_text,flatten_csv",
                    help="comma list: strip_codes,free_text,flatten_csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    stats = run(a.in_dir, a.out_dir, set(a.modes.split(",")), a.seed)
    print(f"Corrupted {stats['bundles']} bundles → {a.out_dir} "
          f"(freetext notes + {stats['csv_rows']} claims.csv rows)")


if __name__ == "__main__":
    main()
