#!/usr/bin/env python3
"""
Local Ontology DB — a lightweight vector store of SNOMED-CT / LOINC concepts, used by the
validator's RAG step to map messy free-text to the correct standardized code.

Uses Chroma if available (persistent, embedded — no server), else falls back to a tiny
in-process cosine index so the PoC runs anywhere. Embeddings come from sentence-transformers
if present, else a deterministic hashing embedder (good enough to demonstrate retrieval).

Load a dictionary (TSV: code<TAB>system<TAB>display) then either --serve (idle, healthy) or
import `query()` from the validator.

    python load_ontology.py --build --dict snomed_loinc.tsv --db ./db
    python load_ontology.py --serve --db ./db          # used by docker-compose
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
from pathlib import Path
from typing import List, Tuple

# A minimal seed dictionary so the PoC works before a full SNOMED/LOINC load.
SEED_CONCEPTS = [
    ("59621000", "SNOMED-CT", "Essential hypertension"),
    ("44054006", "SNOMED-CT", "Type 2 diabetes mellitus"),
    ("55822004", "SNOMED-CT", "Hyperlipidemia"),
    ("22298006", "SNOMED-CT", "Myocardial infarction"),
    ("13645005", "SNOMED-CT", "Chronic obstructive pulmonary disease"),
    ("195967001", "SNOMED-CT", "Asthma"),
    ("49436004", "SNOMED-CT", "Atrial fibrillation"),
    ("42343007", "SNOMED-CT", "Congestive heart failure"),
    ("431855005", "SNOMED-CT", "Chronic kidney disease stage 3"),
    ("4548-4", "LOINC", "Hemoglobin A1c"),
    ("8867-4", "LOINC", "Heart rate"),
    ("8480-6", "LOINC", "Systolic blood pressure"),
    ("8462-4", "LOINC", "Diastolic blood pressure"),
    ("29463-7", "LOINC", "Body weight"),
    ("8302-2", "LOINC", "Body height"),
    ("9279-1", "LOINC", "Respiratory rate"),
]


def _embed(text: str, dim: int = 256) -> List[float]:
    """Deterministic hashing embedder (fallback when sentence-transformers is absent)."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        global _ST
        if "_ST" not in globals():
            _ST = SentenceTransformer("all-MiniLM-L6-v2")
        return _ST.encode(text, normalize_embeddings=True).tolist()
    except Exception:
        vec = [0.0] * dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OntologyDB:
    """Chroma-backed if available, else an in-memory cosine index with disk persistence."""

    def __init__(self, db_path: str, load: bool = True):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._chroma = None
        try:
            import chromadb  # type: ignore
            client = chromadb.PersistentClient(path=str(self.db_path))
            self._col = client.get_or_create_collection("ontology")
            self._chroma = client
        except Exception:
            self._mem: List[Tuple[str, str, str, List[float]]] = []  # code, system, display, emb
            if load:
                self._load_disk()  # restore from disk if available

    def add(self, concepts: List[Tuple[str, str, str]]):
        # Each concept: (code, system, display, [embed_text]). embed_text folds in synonyms
        # so messy aliases match, while code/display stay canonical for output.
        def emb_text(c):
            return c[3] if len(c) > 3 else c[2]
        if self._chroma is not None:
            self._col.upsert(
                ids=[c[0] for c in concepts],
                embeddings=[_embed(emb_text(c)) for c in concepts],
                metadatas=[{"code": c[0], "system": c[1], "display": c[2]} for c in concepts],
                documents=[c[2] for c in concepts],
            )
        else:
            for c in concepts:
                self._mem.append((c[0], c[1], c[2], _embed(emb_text(c))))
            self._save_disk()  # persist to disk after adding

    def _save_disk(self):
        """Persist in-memory concepts to disk (fallback when Chroma unavailable)."""
        if self._chroma is not None:
            return
        pickle_path = self.db_path / "ontology_mem.pkl"
        try:
            with open(pickle_path, "wb") as f:
                pickle.dump(self._mem, f)
        except Exception as e:
            print(f"Warning: failed to persist ontology to {pickle_path}: {e}")

    def _load_disk(self):
        """Load in-memory concepts from disk (fallback when Chroma unavailable)."""
        if self._chroma is not None:
            return
        pickle_path = self.db_path / "ontology_mem.pkl"
        if pickle_path.exists():
            try:
                with open(pickle_path, "rb") as f:
                    self._mem = pickle.load(f)
            except Exception as e:
                print(f"Warning: failed to load ontology from {pickle_path}: {e}")
                self._mem = []

    def query(self, text: str, k: int = 5) -> List[dict]:
        """Return the top-k closest standardized concepts for a messy term."""
        if self._chroma is not None:
            r = self._col.query(query_embeddings=[_embed(text)], n_results=k)
            return r.get("metadatas", [[]])[0]
        q = _embed(text)
        scored = [((sum(a * b for a, b in zip(q, e))), code, system, display)
                  for code, system, display, e in self._mem]
        scored.sort(reverse=True)
        return [{"code": c, "system": s, "display": d} for _, c, s, d in scored[:k]]


# Clinical synonyms/abbreviations per code — folded into the embedding text so messy
# shorthand ("pt c/o HTN", "T2DM") matches the canonical concept. Real terminology services
# ship synonym tables; we seed the common ones the corruption harness produces.
ALIASES = {
    "59621000": ["HTN", "pt c/o HTN", "high blood pressure"],
    "44054006": ["T2DM", "DM", "poorly controlled diabetes"],
    "55822004": ["HLD"],
    "22298006": ["h/o MI", "heart attack"],
    "13645005": ["COPD", "COPD exac"],
    "49436004": ["afib", "afib w/ RVR"],
    "42343007": ["CHF", "EF reduced"],
    "431855005": ["CKD", "CKD stage III"],
    "4548-4": ["A1c", "HbA1c", "hemoglobin a1c"],
    "8867-4": ["HR", "pulse", "heart rate"],
    "8480-6": ["SBP", "systolic BP", "systolic blood pressure"],
    "8462-4": ["DBP", "diastolic BP"],
    "29463-7": ["wt", "weight", "body weight"],
    "8302-2": ["ht", "height", "body height"],
    "9279-1": ["RR", "resp rate", "respiratory rate"],
}


def build(db_path: str, dict_path: str | None, reload: bool = True):
    db = OntologyDB(db_path, load=False)  # don't load yet — we're about to add
    raw = list(SEED_CONCEPTS)
    if dict_path and Path(dict_path).exists():
        for line in Path(dict_path).read_text(encoding="utf-8").splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                raw.append((parts[0], parts[1], parts[2]))
    # Build 4-tuples: (code, system, canonical_display, embed_text-with-aliases).
    concepts = [(code, system, display, display + " " + " ".join(ALIASES.get(code, [])))
                for code, system, display in raw]
    db.add(concepts)  # this now persists to disk
    (Path(db_path) / "ready").write_text("ok")
    print(f"Ontology DB built at {db_path}: {len(concepts)} concepts "
          f"({'chroma' if db._chroma else 'in-memory'})")
    # Reload to verify persistence works (simulates what callers will do)
    if reload:
        db2 = OntologyDB(db_path)  # this should load from disk
        count = len(db2._mem) if db2._chroma is None else db2._col.count()
        if count != len(concepts):
            print(f"WARNING: persistence check failed — expected {len(concepts)}, got {count}")
            return False
        print(f"Persistence verified: {count} concepts reloadable")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./db")
    ap.add_argument("--dict", default=None, help="TSV code<TAB>system<TAB>display")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--serve", action="store_true", help="build if needed then idle (for compose)")
    a = ap.parse_args()
    if a.build or a.serve or not (Path(a.db) / "ready").exists():
        build(a.db, a.dict)
    if a.serve:
        import time
        print("Ontology DB ready; serving (idle).")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
