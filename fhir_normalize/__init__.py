"""
fhir-normalize — recover canonical SNOMED/LOINC codes from messy free-text
clinical data, with a measured recovery benchmark on synthetic FHIR.

All-synthetic; no PHI. The deterministic RAG path (ontology retrieval + a
crosswalk over clinical aliases) runs anywhere with no model and no GPU.
"""
from .benchmark import run as run_benchmark
from .ontology import OntologyDB, build
from .validate import reconstruct, validate_bundle

__version__ = "0.1.0"
__all__ = ["run_benchmark", "OntologyDB", "build", "reconstruct", "validate_bundle"]
