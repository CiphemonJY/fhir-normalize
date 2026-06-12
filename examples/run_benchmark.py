"""Minimal end-to-end demo. Run: python examples/run_benchmark.py"""
import tempfile

from fhir_normalize import build, OntologyDB, reconstruct, run_benchmark

print("Benchmark on 200 synthetic patients:")
print(" ", run_benchmark(n=200, seed=7))

# A messy, shorthand clinical note in the line/section format real feeds produce.
NOTE = "\n".join([
    "Hx/problems",
    "- pt c/o HTN",
    "- poorly controlled DM",
    "- afib w/ RVR",
    "- h/o MI",
    "Vitals/labs",
    "- A1c: 8.1",
    "- HR: 92",
])

print("\nSingle-note normalization:")
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    build(d, None)
    db = OntologyDB(d)
    bundle = reconstruct(NOTE, db)
    print("  note:")
    for line in NOTE.split("\n"):
        print(f"    {line}")
    print("  recovered:")
    for e in bundle.get("entry", []):
        for c in e.get("resource", {}).get("code", {}).get("coding", []):
            if c.get("code"):
                print(f"    {c['code']:>10}  {c['system']:<9}  {c['display']}")
