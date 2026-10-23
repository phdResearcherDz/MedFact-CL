"""Builds the fact-checking components from command-line options."""
import argparse
import os

from ..utils import CLEAN_SPLITS_DIR, list_split_files, load_json, normalize_whitespace
from .index import DEFAULT_EMBEDDER

CACHE_DIRNAME = "FactCheckCache"


def add_component_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("fact-checking components")
    g.add_argument("--llm", default="gemma3:12b", help="Ollama model used for claim extraction")
    g.add_argument("--ollama_host", default=None)
    g.add_argument("--spacy_model", default="en_core_sci_sm")
    g.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    g.add_argument("--device", default=None)


def cache_dir(ds_dir: str) -> str:
    return os.path.join(ds_dir, CACHE_DIRNAME)


def index_dir(ds_dir: str) -> str:
    return os.path.join(cache_dir(ds_dir), "evidence_index")


def make_claim_store(ds_dir: str, args, need_extractor: bool = True):
    from .pipeline import ClaimStore
    extractor = normalizer = None
    if need_extractor:
        from ..llm import OllamaJSON
        from .extraction import ClaimExtractor
        from .normalization import UMLSNormalizer
        extractor = ClaimExtractor(OllamaJSON(model=args.llm, host=args.ollama_host))
        normalizer = UMLSNormalizer(args.spacy_model)
    return ClaimStore(os.path.join(cache_dir(ds_dir), "claims.jsonl"), extractor, normalizer)


def reference_documents(ds_dir: str, reference_file: str = None):
    """Trusted reference documents: by default the clean contexts of the dataset's training
    splits (PubMed snippets for BioASQ, PubMed abstracts for PubMedQA); otherwise a JSON list
    of {"id", "text"} records."""
    docs = {}
    if reference_file:
        for i, r in enumerate(load_json(reference_file)):
            text = normalize_whitespace(r.get("text") or r.get("context") or "")
            if text:
                docs.setdefault(text, str(r.get("id", i)))
    else:
        split_dir = os.path.join(ds_dir, CLEAN_SPLITS_DIR)
        for f in list_split_files(split_dir):
            for r in load_json(os.path.join(split_dir, f)):
                text = normalize_whitespace(r.get("context"))
                if text:
                    docs.setdefault(text, str(r.get("id")))
    return [(doc_id, text) for text, doc_id in docs.items()]

