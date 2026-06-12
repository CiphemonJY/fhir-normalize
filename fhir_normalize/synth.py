#!/usr/bin/env python3
"""
Pure-Python synthetic FHIR generator — a tiny stand-in for Synthea so the whole PoC runs
with no Java/network. Emits FHIR R4 Bundles with REAL SNOMED-CT/LOINC codes drawn from a
known concept pool, so downstream we can measure how well the validator recovers them
(ground-truth uniformity scoring). For scale/realism use Synthea (poc/synthea/generate.sh);
for a fast, deterministic demo/test use this.

    python poc/sample_fhir.py --n 200 --out poc/synthea/output/fhir
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# Concepts whose codes the validator's seed ontology knows (so recovery is measurable).
CONDITIONS = [
    ("59621000", "Essential hypertension"),
    ("44054006", "Type 2 diabetes mellitus"),
    ("55822004", "Hyperlipidemia"),
    ("22298006", "Myocardial infarction"),
    ("13645005", "Chronic obstructive pulmonary disease"),
    ("195967001", "Asthma"),
    ("49436004", "Atrial fibrillation"),
    ("42343007", "Congestive heart failure"),
    ("431855005", "Chronic kidney disease stage 3"),
]
OBSERVATIONS = [
    ("4548-4", "Hemoglobin A1c", "%", (5.0, 11.0)),
    ("8867-4", "Heart rate", "/min", (55, 105)),
    ("8480-6", "Systolic blood pressure", "mm[Hg]", (110, 175)),
    ("29463-7", "Body weight", "kg", (55, 120)),
]
GIVEN = ["John", "Mary", "Robert", "Linda", "James", "Patricia", "David", "Jennifer"]
FAMILY = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
SNOMED_SYS, LOINC_SYS = "http://snomed.info/sct", "http://loinc.org"


def make_bundle(idx: int, rng: random.Random) -> Dict:
    pid = f"patient-{idx:05d}"
    entries: List[Dict] = [{"resource": {
        "resourceType": "Patient", "id": pid,
        "gender": rng.choice(["male", "female"]),
        "birthDate": f"{rng.randint(1945, 2005)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "name": [{"given": [rng.choice(GIVEN)], "family": rng.choice(FAMILY)}],
    }}]
    for code, display in rng.sample(CONDITIONS, rng.randint(1, 3)):
        entries.append({"resource": {
            "resourceType": "Condition",
            "code": {"text": display,
                     "coding": [{"system": SNOMED_SYS, "code": code, "display": display}]},
            "clinicalStatus": {"coding": [{"code": "active"}]},
        }})
    for code, display, unit, (lo, hi) in rng.sample(OBSERVATIONS, rng.randint(1, len(OBSERVATIONS))):
        entries.append({"resource": {
            "resourceType": "Observation",
            "code": {"text": display,
                     "coding": [{"system": LOINC_SYS, "code": code, "display": display}]},
            "valueQuantity": {"value": round(rng.uniform(lo, hi), 1), "unit": unit},
        }})
    return {"resourceType": "Bundle", "type": "transaction", "id": f"bundle-{idx:05d}", "entry": entries}


def run(n: int, out_dir: str, seed: int = 7) -> int:
    rng = random.Random(seed)
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (outp / f"patient-{i:05d}.json").write_text(
            json.dumps(make_bundle(i, rng), indent=2), encoding="utf-8")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="poc/synthea/output/fhir")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    n = run(a.n, a.out, a.seed)
    print(f"Wrote {n} synthetic FHIR bundles -> {a.out}")


if __name__ == "__main__":
    main()
