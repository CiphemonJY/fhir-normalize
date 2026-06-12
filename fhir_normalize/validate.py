#!/usr/bin/env python3
"""
LLM Semantic Validator — reconstruct valid, coded FHIR from messy clinical free-text.

This is the layer that guarantees DATA UNIFORMITY before federated training. Every node,
no matter how garbled its source feed, emits FHIR conforming to the SAME schema and the
SAME terminology (SNOMED-CT/LOINC). Without it, each hospital would train on differently-
shaped data and the aggregated model would learn noise.

Pipeline per record:
  1. Read corrupted free-text (e.g. "pt c/o HTN, T2DM poorly controlled").
  2. RAG: for each candidate term, retrieve the closest standardized concept(s) from the
     local Ontology DB (vector search over SNOMED-CT/LOINC).
  3. LLM: a local quantized model (GGUF via llama.cpp, or any OpenAI-compatible HTTP endpoint)
     is prompted with the messy text + retrieved candidates and asked to emit a strict FHIR
     Bundle, choosing codes ONLY from the retrieved candidates (RAG-grounded, no hallucinated codes).
  4. Validate: structural + terminology checks; reject/repair anything off-schema.

Output: one clean FHIR Bundle JSON per input. These feed the federated client.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from .ontology import OntologyDB

# LLM is optional. The deterministic RAG path (crosswalk lookup + retrieval) is the
# default and needs no model; set call_llm to a callable to enable an LLM normalizer.
call_llm = None  # type: ignore

_SYSTEM = """\
You are a clinical terminology normalizer. Convert messy clinician free-text into a STRICT
FHIR R4 Bundle (JSON only, no prose). Rules:
- Output a Bundle with one Patient and one Condition/Observation per clinical fact.
- For every code you assign, you MUST pick from the CANDIDATE CODES provided. Never invent a code.
- If no candidate fits a fact, omit the code but keep the text. Do not hallucinate values.
Output ONLY the JSON object."""


def _candidate_block(db: OntologyDB, text: str, k: int = 4) -> str:
    """Build the RAG context: retrieved standardized concepts for the lines in `text`."""
    seen, lines = set(), []
    for raw in re.split(r"[\n,;]", text):
        term = raw.strip(" -:").strip()
        if len(term) < 3:
            continue
        for hit in db.query(term, k=k):
            key = (hit["system"], hit["code"])
            if key not in seen:
                seen.add(key)
                lines.append(f'  - "{hit["display"]}"  ({hit["system"]} {hit["code"]})')
    return "CANDIDATE CODES (choose only from these):\n" + "\n".join(lines[:24])


def _best(db: OntologyDB, term: str, system_kw: str):
    """RAG top hit for `term`, restricted to a code system (snomed/loinc)."""
    for h in db.query(term, k=4):
        if system_kw in (h.get("system", "")).lower():
            return h
    return {}


def _stub_bundle(text: str, db: OntologyDB) -> Dict[str, Any]:
    """
    Deterministic, section-aware reconstruction (no-LLM fallback) using RAG:
      - "Hx/problems" lines  -> Condition (SNOMED)
      - "Vitals/labs" lines  -> Observation (LOINC) with parsed value
    Keeps the PoC fully runnable and now recovers lab/observation codes too.
    """
    entries = [{"resource": {"resourceType": "Patient", "gender": "unknown"}}]
    in_obs = False
    for raw in text.split("\n"):
        line = raw.strip()
        low = line.lower()
        if low.startswith(("vitals", "labs", "vitals/labs")):
            in_obs = True
            continue
        if (not line) or low.startswith(("hx", "problems")) or "pt, dob" in low:
            continue
        if in_obs:
            m = re.match(r"[-\s]*(.+?)\s*:\s*([\d.]+)?\s*(\S+)?", line)
            term = (m.group(1) if m else line).strip(" -:")
            hit = _best(db, term, "loinc")
            if hit.get("code"):
                res = {"resourceType": "Observation",
                       "code": {"coding": [{"system": hit["system"], "code": hit["code"],
                                            "display": hit["display"]}], "text": term}}
                if m and m.group(2):
                    try:
                        res["valueQuantity"] = {"value": float(m.group(2)),
                                                "unit": (m.group(3) or "").strip("[]")}
                    except ValueError:
                        pass
                entries.append({"resource": res})
        else:
            term = line.strip(" -:")
            hit = _best(db, term, "snomed") or (db.query(term, k=1) or [{}])[0]
            if hit.get("code"):
                entries.append({"resource": {
                    "resourceType": "Condition",
                    "code": {"coding": [{"system": hit["system"], "code": hit["code"],
                                         "display": hit["display"]}], "text": term}}})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def _llm(prompt: str, system: str):
    """Optional LLM normalizer via an HTTP endpoint (env FHIR_NORMALIZE_LLM_ENDPOINT,
    llama.cpp / OpenAI-compatible). Returns None → deterministic RAG fallback."""
    import os
    endpoint = os.environ.get("FHIR_NORMALIZE_LLM_ENDPOINT")
    if endpoint:
        try:
            import json as _j, urllib.request as _u
            body = _j.dumps({"prompt": f"{system}\n\n{prompt}", "temperature": 0.1,
                             "max_tokens": 2048, "stream": False}).encode()
            req = _u.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
            with _u.urlopen(req, timeout=120) as r:
                resp = _j.loads(r.read().decode())
            # Accept llama.cpp {"content":...} / OpenAI shape.
            return (resp.get("content") or resp.get("response")
                    or resp.get("choices", [{}])[0].get("text")
                    or resp.get("choices", [{}])[0].get("message", {}).get("content"))
        except Exception:
            pass
    if call_llm is not None:
        return call_llm(prompt, system)
    return None


def reconstruct(text: str, db: OntologyDB) -> Dict[str, Any]:
    candidates = _candidate_block(db, text)
    prompt = f"{candidates}\n\nMESSY NOTE:\n{text}\n\nReturn the FHIR R4 Bundle JSON:"
    raw = _llm(prompt, _SYSTEM)
    if raw is None:
        return _stub_bundle(text, db)
    raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
    try:
        bundle = json.loads(raw)
        if bundle.get("resourceType") == "Bundle":
            return _enforce_candidates(bundle, db, text)
    except Exception:
        pass
    return _stub_bundle(text, db)  # repair: fall back if the model went off-schema


def _enforce_candidates(bundle: Dict[str, Any], db: OntologyDB, text: str) -> Dict[str, Any]:
    """Terminology guard: drop any code the model used that isn't a real ontology concept."""
    valid = {(h["system"], h["code"]) for line in text.splitlines()
             for h in db.query(line, k=5)}
    for e in bundle.get("entry", []):
        coding = e.get("resource", {}).get("code", {}).get("coding", [])
        e["resource"].setdefault("code", {})["coding"] = [
            c for c in coding if (c.get("system"), c.get("code")) in valid]
    return bundle


def validate_bundle(bundle: Dict[str, Any]) -> bool:
    """Minimal structural + uniformity check (every node's output must pass the same gate)."""
    if bundle.get("resourceType") != "Bundle" or not isinstance(bundle.get("entry"), list):
        return False
    return any(e.get("resource", {}).get("resourceType") == "Patient"
               for e in bundle["entry"])


def run(in_dir: str, out_dir: str, db_path: str) -> Dict[str, int]:
    db = OntologyDB(db_path)
    inp, outp = Path(in_dir) / "freetext", Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for f in sorted(inp.glob("*.txt")):
        bundle = reconstruct(f.read_text(encoding="utf-8"), db)
        if validate_bundle(bundle):
            (outp / f"{f.stem}.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
            ok += 1
        else:
            bad += 1
    print(f"Validator: reconstructed {ok} clean FHIR bundles, rejected {bad}.")
    return {"ok": ok, "rejected": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="/data/corrupted")
    ap.add_argument("--out", dest="out_dir", default="/data/clean_fhir")
    ap.add_argument("--db", default="/data/ontology_db")
    a = ap.parse_args()
    run(a.in_dir, a.out_dir, a.db)


if __name__ == "__main__":
    main()
