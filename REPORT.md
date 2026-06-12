# Clinical Code Recovery: a Measured Benchmark

A short note on turning messy free-text clinical data back into canonical,
uniform FHIR codes — and on measuring it honestly.

Reproducible from this repository: `python -m fhir_normalize.benchmark --n 200`,
`pytest -q`.

---

## 1. The problem

Federated learning, cohort building, registry submission, and computer-assisted
coding all assume *uniform* clinical data: the same condition carries the same
SNOMED code everywhere. Reality is the opposite. Each site speaks a dialect —
`"HTN"`, `"high blood pressure"`, `"pt c/o htn"`, a stray ICD code, or nothing
but a flattened CSV cell. Before any cross-site model can aggregate aligned
signal, the dialects have to be mapped to one canonical vocabulary.

## 2. The pipeline

To measure a normalizer you need ground truth, which real notes (being PHI and
un-coded) don't give you. So the benchmark manufactures it:

1. **Generate** (`synth.py`) — synthetic FHIR R4 bundles with *real* SNOMED/LOINC
   codes drawn from a known concept pool. The codes are the ground truth.
2. **Corrupt** (`corrupt.py`) — three independent degradations that mimic real
   feeds: drop the codes (keep mangled display text), replace standardized terms
   with clinician shorthand (`"Essential hypertension" → "pt c/o HTN"`), and
   flatten the nested Bundle into a wide, poorly-typed CSV row.
3. **Recover** (`ontology.py` + `validate.py`) — for each free-text line, retrieve
   candidate concepts from a local index (embedding similarity over canonical
   displays *plus* a clinical-alias table, so `"T2DM"` and `"poorly controlled
   diabetes"` both land near `44054006`), then reconstruct a strict FHIR bundle
   choosing only from retrieved candidates — never inventing a code.
4. **Score** (`benchmark.py`) — recall of each patient's original codes from the
   corrupted text, plus whether the reconstructed bundle re-validates as
   canonical FHIR.

The retrieval index runs in-process (a tiny cosine store + a deterministic
hashing embedder) so the whole thing executes anywhere with no model, no GPU,
and no service. If `sentence-transformers` and/or `chromadb` are installed they
are used transparently; results are unchanged because both paths share the same
embedder.

## 3. Result

```
patients scored    : 200
code recovery rate : 96.5%
schema uniform     : 100.0%
```

Identical on the in-process and Chroma paths. The mechanism works: shorthand and
flattened feeds are mapped back to the canonical codes, and the output is
schema-valid FHIR every time.

## 4. What 96.5% does and doesn't mean

This is the part most "AI for clinical coding" demos skip.

**It is a controlled benchmark.** The corruption harness and the seed ontology
share a *closed* concept set — the aliases the harness emits are the aliases the
index knows. So 96.5% cleanly isolates the *mechanism* — retrieval-grounded
recovery of dialect to canonical code — without confounding it with vocabulary
coverage. The residual ~3.5% comes from ambiguous shorthand and flattened rows
that drop the term entirely.

**It is not a real-world accuracy number.** On open clinical text against a full
SNOMED/LOINC load, recovery will be lower: unseen synonyms, true ambiguity
(`"discharge"` the event vs. the fluid), negation, and concepts outside the seed
all cost recall. Closing that gap is the honest roadmap:

1. Load the full terminologies (UMLS/SNOMED/LOINC) into the index instead of a
   seed pool.
2. Add a real synonym source (UMLS `MRCONSO`) rather than a hand-seeded alias
   table.
3. Enable the optional LLM normalizer (`FHIR_NORMALIZE_LLM_ENDPOINT`) for the
   long tail of free text the deterministic path can't resolve, still constrained
   to retrieved candidates so it cannot hallucinate codes.
4. Validate against a *held-out* corruption profile the index has never seen —
   the analogue of a held-out test set — to get a number that generalises.

The value of the controlled benchmark is that it is **reproducible, honest about
its scope, and a fixed yardstick** to measure those improvements against. A
demo that quotes a real-world percentage without a held-out, open-vocabulary
evaluation is quoting noise.

## 5. Where this fits

Clinical data normalization is the unglamorous layer under interoperability,
quality reporting, cohort discovery, and any federation of clinical models. The
contribution here is not a state-of-the-art coder; it is a **clean, synthetic,
reproducible harness** for measuring one honestly — and a deterministic baseline
that runs on a laptop with no PHI in sight.

---

### Reproduce

```bash
pip install fhir-normalize
python -m fhir_normalize.benchmark --n 200
pytest -q
```
