"""End-to-end + unit tests. All synthetic, no network, no GPU, no heavy deps."""
import json
import tempfile
from pathlib import Path

from fhir_normalize import build, reconstruct, run_benchmark, validate_bundle
from fhir_normalize.ontology import OntologyDB


def test_recovery_rate_on_synthetic():
    """The deterministic RAG normalizer recovers the large majority of codes."""
    m = run_benchmark(n=80, seed=7)
    assert m["scored"] == 80
    assert m["code_recovery_rate"] >= 0.85, m
    assert m["schema_uniform_pct"] == 100.0, m


def test_recovery_is_deterministic():
    """Same seed → same number (reproducible benchmark)."""
    assert run_benchmark(n=40, seed=1) == run_benchmark(n=40, seed=1)


def test_ontology_maps_shorthand_to_canonical():
    """The clinical-alias index resolves messy shorthand to the right concept."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        build(d, None)
        db = OntologyDB(d)
        # "HTN" / "T2DM" / "A1c" are aliases folded into the embedding text.
        top = db.query("pt c/o HTN", k=1)[0]
        assert top["display"] == "Essential hypertension"
        assert db.query("T2DM", k=1)[0]["code"] == "44054006"
        assert db.query("HbA1c", k=3)  # LOINC A1c retrievable


def test_reconstruct_emits_valid_bundle():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        build(d, None)
        db = OntologyDB(d)
        bundle = reconstruct("pt c/o HTN, poorly controlled DM. A1c elevated.", db)
        assert validate_bundle(bundle)
        codes = {c["code"] for e in bundle.get("entry", [])
                 for c in e.get("resource", {}).get("code", {}).get("coding", []) if c.get("code")}
        assert "59621000" in codes  # hypertension recovered from "HTN"
