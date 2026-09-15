# fhir-normalize

**Recover canonical SNOMED/LOINC codes from messy free-text clinical data — with
a measured recovery benchmark on synthetic FHIR.**

Real clinical feeds (legacy claims, discharge notes, HL7 dumps) are free-text,
abbreviated, mis-coded, and flattened — `"pt c/o HTN, poorly controlled DM"`
where a clean record would carry `59621000` (Essential hypertension) and
`44054006` (Type 2 diabetes). This toolkit maps that mess back to canonical
codes, and **measures how well it does it** against ground truth.

Everything is synthetic — no PHI — and the default path needs no model, no GPU,
and no heavy dependencies.

---

## The result

```
$ python -m fhir_normalize.benchmark --n 200
  patients scored    : 200
  code recovery rate : 96.5%   (fraction of original SNOMED/LOINC codes recovered from messy text)
  schema uniform     : 100.0%  (bundles that re-validate as canonical FHIR)
```

The pipeline: **generate** synthetic FHIR with real codes → **corrupt** it into
clinician shorthand and flattened CSV → **recover** the codes with a
retrieval-grounded normalizer → **score** recovery against the known ground
truth. The 96.5% is the quantitative claim that "messy clinical dialects can be
mapped back to uniform, canonical codes" — measured, not asserted.

## How it works

| Module | Role |
|--------|------|
| `synth.py` | pure-Python synthetic FHIR R4 generator (a tiny stand-in for Synthea; real SNOMED/LOINC codes so recovery is measurable) |
| `corrupt.py` | the noise generator: strip codes, replace terms with clinician shorthand, flatten the nested Bundle into a legacy CSV row |
| `ontology.py` | a local concept index (in-process cosine + a clinical-alias table; uses Chroma + sentence-transformers if installed, else runs anywhere) |
| `validate.py` | the normalizer: retrieve candidate codes for each free-text line, reconstruct a strict FHIR bundle, never invent a code |
| `benchmark.py` | the end-to-end generate → corrupt → recover → score harness |

The deterministic RAG path is the default. An optional LLM normalizer can be
enabled via an HTTP endpoint (`FHIR_NORMALIZE_LLM_ENDPOINT`) for harder,
open-vocabulary text — see `REPORT.md`.

## Install & run

```bash
pip install fhir-normalize           # core (no model needed)
python -m fhir_normalize.benchmark --n 200
pytest -q                            # end-to-end + unit tests
```

```python
from fhir_normalize import build, OntologyDB, reconstruct
build("./db", None)                                  # seed the concept index
db = OntologyDB("./db")
bundle = reconstruct("pt c/o HTN, poorly controlled DM. A1c elevated.", db)
# → FHIR bundle coded with 59621000 (hypertension), 44054006 (T2DM), 4548-4 (A1c)
```

## Honesty about the benchmark

This is a **controlled** benchmark: the corruption and the seed ontology share a
closed concept set, so 96.5% measures the *mechanism* (retrieval-grounded
recovery of shorthand) cleanly — it is **not** a real-world accuracy claim on
open clinical vocabularies. Recovery on a full SNOMED/LOINC load with real notes
will be lower and is the honest next step. See `REPORT.md` → "What 96.5% does and
doesn't mean". No real or PHI-bearing data is included or required.

MIT.
