"""
End-to-end benchmark: generate synthetic FHIR → corrupt it into messy free-text →
recover canonical codes with the deterministic RAG normalizer → score against
ground truth.

The headline number is `code_recovery_rate`: the fraction of each patient's
original SNOMED/LOINC codes the normalizer recovered from corrupted free-text.
That is the quantitative claim — "messy clinical dialects can be mapped back to
canonical, uniform codes" — measured, not asserted.

    python -m fhir_normalize.benchmark --n 200
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from . import corrupt as ch
from . import synth
from . import validate as V
from .ontology import build


def _codes_of(bundle: dict) -> set:
    """The set of SNOMED/LOINC codes in a bundle's coded entries.

    Compares on the code value (not system), so a recovered code counts even
    though the normalizer labels the system 'SNOMED-CT' while source FHIR uses
    the canonical URI 'http://snomed.info/sct' — the code itself is the ground
    truth being recovered.
    """
    return {
        c.get("code")
        for e in bundle.get("entry", [])
        for c in e.get("resource", {}).get("code", {}).get("coding", [])
        if c.get("code")
    }


def run(n: int, use_llm: bool = False, seed: int = 7) -> dict:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        d = Path(d)
        fhir, corrupted, dbp, clean = d / "fhir", d / "corrupt", d / "db", d / "clean"

        # 1. Synthetic FHIR + ground-truth codes per patient.
        synth.run(n, str(fhir), seed)
        truth = {f.stem: _codes_of(json.loads(f.read_text())) for f in fhir.glob("*.json")}

        # 2. Corrupt: strip codes + clinician shorthand + flatten to a legacy CSV.
        ch.run(str(fhir), str(corrupted), {"strip_codes", "free_text", "flatten_csv"}, seed)

        # 3. Ontology + normalizer (deterministic RAG; LLM optional).
        build(str(dbp), None)
        if not use_llm:
            V.call_llm = None
        V.run(str(corrupted), str(clean), str(dbp))

        # 4. Score recovery vs ground truth.
        scored, total_recall, uniform = 0, 0.0, 0
        for f in clean.glob("*.json"):
            bundle = json.loads(f.read_text())
            got = _codes_of(bundle)
            want = truth.get(f.stem, set())
            if not want:
                continue
            scored += 1
            total_recall += len(got & want) / len(want)
            if V.validate_bundle(bundle):
                uniform += 1
        recall = (total_recall / scored) if scored else 0.0
        return {
            "patients": n,
            "scored": scored,
            "code_recovery_rate": round(recall, 3),
            "schema_uniform_pct": round(100 * uniform / max(scored, 1), 1),
        }


def main():
    ap = argparse.ArgumentParser(description="Clinical code-recovery benchmark")
    ap.add_argument("--n", type=int, default=200, help="number of synthetic patients")
    ap.add_argument("--llm", action="store_true", help="use an LLM normalizer (default: deterministic RAG)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    m = run(a.n, a.llm, a.seed)
    print(f"  patients scored    : {m['scored']}")
    print(f"  code recovery rate : {m['code_recovery_rate'] * 100:.1f}%  "
          f"(fraction of original SNOMED/LOINC codes recovered from messy text)")
    print(f"  schema uniform     : {m['schema_uniform_pct']:.1f}%  "
          f"(bundles that re-validate as canonical FHIR)")


if __name__ == "__main__":
    main()
