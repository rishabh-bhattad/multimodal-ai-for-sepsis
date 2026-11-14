"""Extract latest-only window findings and link them locally to clinical PrimeKG nodes."""

import argparse
import hashlib
import json
import operator
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.utils import setup_logger

logger = setup_logger("Step2-Extraction")


def extract_study_id(cxr_path):
    """Recover the study identity for label lookup without exposing its image path."""
    if pd.isna(cxr_path) or not cxr_path:
        return None
    match = re.search(r"/s(\d+)/", str(cxr_path))
    return int(match.group(1)) if match else None


def load_radiographic_findings(study_ids):
    """Retain positive-label provenance and distinguish missing studies from covered studies."""
    studies = {int(sid) for sid in study_ids}
    result = {sid: {"findings": {}, "coverage": {}} for sid in studies}
    for source in ("chexpert", "negbio"):
        path = config.MIMIC_CXR_DIR / f"mimic-cxr-2.0.0-{source}.csv"
        if not path.is_file():
            path = path.with_suffix(".csv.gz")
        if not path.is_file():
            raise RuntimeError(f"Required {source} label file is missing")
        for chunk in pd.read_csv(path, chunksize=config.STEP2_CSV_CHUNK_SIZE,
                                 usecols=["study_id", *config.TARGET_PATHOLOGIES]):
            for row in chunk[chunk.study_id.isin(studies)].to_dict("records"):
                record = result[int(row["study_id"])]
                positive = [name for name in config.TARGET_PATHOLOGIES if row[name] == 1.0]
                previous = record["coverage"].get(source)
                record["coverage"][source] = (
                    "positive" if positive or previous == "positive" else "covered_no_positive"
                )
                for name in positive:
                    record["findings"].setdefault(name, set()).add(source)
        for record in result.values():
            record["coverage"].setdefault(source, "missing_study")
    return result


def extract_lab_vital_findings(window):
    """Translate latest measurements into observations, choosing one severity per measurement."""
    comparisons = {"<": operator.lt, ">": operator.gt, ">=": operator.ge}
    findings = []
    for measurement, rules in config.LAB_SEVERITY_THRESHOLDS.items():
        field = f"{measurement}_latest"
        value = window[field]
        if pd.isna(value):
            continue
        for threshold, comparison, name in rules:
            if comparisons[comparison](value, threshold):
                findings.append({"name": name, "predicate": "exhibits",
                                 "source": field, "value": float(value)})
                break
    return findings


def select_windows(windows, sample_size):
    """Validate four-window inputs and select reproducible image/no-image smoke subjects."""
    required = {"subject_id", "stay_id", "window_idx", "window_start", "window_end", "split", "cxr_path"}
    required.update(f"{name}_latest" for name in config.LAB_SEVERITY_THRESHOLDS)
    if not required.issubset(windows.columns) or windows.empty:
        raise RuntimeError("Window input is empty or lacks required latest-observation fields")
    if windows.duplicated(["stay_id", "window_idx"]).any():
        raise RuntimeError("Duplicate window keys")
    if not windows.window_idx.isin(range(config.NUM_WINDOWS)).all():
        raise RuntimeError("Unexpected window indices")
    if not windows.groupby("stay_id").size().eq(config.NUM_WINDOWS).all():
        raise RuntimeError("Each stay must retain four windows")
    if not windows.groupby("subject_id").split.nunique().eq(1).all():
        raise RuntimeError("Patient split isolation failed")
    starts = windows.groupby("stay_id").window_start.transform("min")
    if not windows.window_start.eq(starts + pd.to_timedelta(windows.window_idx * config.WINDOW_SIZE_HOURS, unit="h")).all():
        raise RuntimeError("Windows are not contiguous")
    if not (windows.window_end - windows.window_start).eq(pd.Timedelta(hours=config.WINDOW_SIZE_HOURS)).all():
        raise RuntimeError("Unexpected window duration")
    if sample_size:
        stays = windows.assign(has_image=windows.cxr_path.notna()).groupby(
            ["subject_id", "stay_id"], as_index=False
        ).has_image.any().sort_values(["has_image", "stay_id"], ascending=[False, True])
        patients = stays.drop_duplicates("subject_id").sort_values("subject_id").sample(
            frac=1, random_state=config.RANDOM_SEED
        )
        if len(patients) < sample_size:
            raise RuntimeError("Fewer eligible subjects than requested")
        coverage_cases = patients.groupby("has_image", sort=True).head(1).head(sample_size)
        selected = pd.concat([coverage_cases, patients.drop(coverage_cases.index)]).head(sample_size)
        windows = windows[windows.stay_id.isin(selected.stay_id)]
    return windows.sort_values(["stay_id", "window_idx"]).copy()


def load_clinical_nodes():
    """Use native node identities from the same clinical subgraph configured for retrieval."""
    with duckdb.connect(config={"threads": config.DUCKDB_THREADS}) as con:
        nodes = con.execute("""
            WITH clinical AS (
                SELECT * FROM read_csv(?, header=true, all_varchar=true, delim=',', quote='"', escape='"')
                WHERE x_type IN (SELECT unnest(?)) AND y_type IN (SELECT unnest(?))
                  AND relation IN (SELECT unnest(?))
            )
            SELECT x_type AS node_type, x_id AS node_id, x_name AS node_name FROM clinical
            UNION
            SELECT y_type, y_id, y_name FROM clinical
        """, [str(config.PRIMEKG_CSV), config.PRIMEKG_ALLOWED_NODE_TYPES,
              config.PRIMEKG_ALLOWED_NODE_TYPES, config.PRIMEKG_RELATION_WHITELIST]).df()
    nodes = nodes.dropna().sort_values(["node_type", "node_id", "node_name"])
    nodes = nodes.drop_duplicates(["node_type", "node_id"])
    if nodes.empty:
        raise RuntimeError("No eligible clinical PrimeKG nodes")
    return nodes.to_dict("records")


def load_sapbert_component(loader):
    """Prefer cached pinned public weights; download on a first online run."""
    kwargs = {"revision": config.SAPBERT_MODEL_REVISION}
    try:
        return loader.from_pretrained(config.SAPBERT_MODEL_NAME, local_files_only=True, **kwargs)
    except OSError as exc:
        if config.SAPBERT_OFFLINE:
            logger.error("SapBERT cache is incomplete in offline mode. Run once with "
                         "SAPBERT_OFFLINE=0 and Hugging Face offline flags disabled to download "
                         "the pinned public model, then retry offline.")
            raise RuntimeError("SapBERT is not cached for offline execution") from exc
        logger.info("Downloading missing pinned public SapBERT files to the Hugging Face cache")
        return loader.from_pretrained(config.SAPBERT_MODEL_NAME, local_files_only=False, **kwargs)


def sapbert_device():
    """Select CUDA automatically, retaining an explicit CPU option."""
    requested = config.SAPBERT_DEVICE
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for SapBERT but unavailable; use SAPBERT_DEVICE=cpu")
    return device


def embed_terms(terms, tokenizer, model):
    """Encode public vocabulary with normalized CLS vectors from pinned local SapBERT."""
    device = next(model.parameters()).device
    batches = []
    for start in range(0, len(terms), config.SAPBERT_BATCH_SIZE):
        inputs = tokenizer(terms[start:start + config.SAPBERT_BATCH_SIZE], padding=True,
                           truncation=True, max_length=config.SAPBERT_MAX_LENGTH, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            vectors = model(**inputs).last_hidden_state[:, 0, :]
            batches.append(torch.nn.functional.normalize(vectors, p=2, dim=1).cpu())
        if start and start % (config.SAPBERT_BATCH_SIZE * 20) == 0:
            logger.info("Embedded %d/%d public vocabulary terms", start, len(terms))
    return torch.cat(batches)


def rank_matches(terms, vectors, nodes, node_vectors):
    """Keep candidate identities and scores so the provisional cutoff can be audited."""
    scores = vectors @ node_vectors.T
    result = {}
    for term, similarities in zip(terms, scores):
        indices = torch.argsort(similarities, descending=True, stable=True)[:3]
        result[term] = [{"node": nodes[int(i)], "similarity": float(similarities[i])} for i in indices]
    return result


def resolve_mappings(terms, candidates):
    """Match source-qualified queries while preserving original finding names and unresolved terms."""
    mappings = {}
    for term in terms:
        query = config.CXR_MAPPING_QUERIES.get(term, term) if term in config.TARGET_PATHOLOGIES else term
        best = candidates[query][0]
        matched = best["similarity"] >= config.SAPBERT_SIMILARITY_THRESHOLD
        mappings[term] = {"node": best["node"] if matched else None,
                          "similarity": best["similarity"], "matched_query": query,
                          "mapping_status": "matched" if matched else "unresolved"}
    return mappings


def build_mappings():
    """Cache public-vocabulary mappings independently of patient/test outcomes."""
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(config.SAPBERT_THREADS)
    stat = config.PRIMEKG_CSV.stat()
    provenance = {"model": config.SAPBERT_MODEL_NAME, "revision": config.SAPBERT_MODEL_REVISION,
                  "pooling": "normalized_cls", "max_length": config.SAPBERT_MAX_LENGTH,
                  "graph_size": stat.st_size, "graph_mtime_ns": stat.st_mtime_ns,
                  "node_types": config.PRIMEKG_ALLOWED_NODE_TYPES,
                  "relations": config.PRIMEKG_RELATION_WHITELIST}
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    terms = sorted(set(config.TARGET_PATHOLOGIES) | {
        name for rules in config.LAB_SEVERITY_THRESHOLDS.values() for _, _, name in rules
    })
    mapping_signature = hashlib.sha256(json.dumps({
        "vocabulary_signature": signature, "terms": terms, "mapping_method": "direct_sapbert",
        "cxr_mapping_queries": config.CXR_MAPPING_QUERIES,
        "cutoff": config.SAPBERT_SIMILARITY_THRESHOLD, "controls": config.SAPBERT_UNMATCHED_CHECKS,
    }, sort_keys=True).encode()).hexdigest()
    if config.SAPBERT_MAPPINGS_JSON.is_file():
        cached = json.loads(config.SAPBERT_MAPPINGS_JSON.read_text())
        if cached.get("signature") == mapping_signature:
            logger.info("Using cached public finding mappings")
            return cached
    device = sapbert_device()
    logger.info("SapBERT embedding device: %s", device)
    tokenizer = load_sapbert_component(AutoTokenizer)
    model = load_sapbert_component(AutoModel).to(device).eval()
    cached = None
    if config.PRIMEKG_CLINICAL_EMBEDDINGS.is_file():
        cached = torch.load(config.PRIMEKG_CLINICAL_EMBEDDINGS, map_location="cpu", weights_only=True)
    if cached is None or cached.get("signature") != signature:
        nodes = load_clinical_nodes()
        logger.info("Embedding %d clinical PrimeKG nodes locally", len(nodes))
        vectors = embed_terms([node["node_name"] for node in nodes], tokenizer, model)
        cached = {"signature": signature, "provenance": provenance, "nodes": nodes, "vectors": vectors}
        torch.save(cached, config.PRIMEKG_CLINICAL_EMBEDDINGS)
    queries = sorted(set(terms + list(config.CXR_MAPPING_QUERIES.values()) + config.SAPBERT_UNMATCHED_CHECKS))
    candidates = rank_matches(queries, embed_terms(queries, tokenizer, model), cached["nodes"], cached["vectors"])
    mappings = resolve_mappings(terms, candidates)
    result = {"signature": mapping_signature, "provenance": provenance,
              "clinical_nodes": len(cached["nodes"]), "threshold": config.SAPBERT_SIMILARITY_THRESHOLD,
              "threshold_status": "provisional", "correctness_review": "pending",
              "mapping_method": "direct_sapbert",
              "cxr_mapping_queries": config.CXR_MAPPING_QUERIES,
              "mappings": mappings, "candidates": candidates,
              "expected_unmatched_checks": config.SAPBERT_UNMATCHED_CHECKS}
    config.SAPBERT_MAPPINGS_JSON.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def assemble_trajectories(windows, radiographic, mappings):
    """Preserve independent windows, local grouping metadata, and unresolved findings."""
    result = {}
    for row in windows.to_dict("records"):
        stay = result.setdefault(str(row["stay_id"]), {
            "subject_id": int(row["subject_id"]), "split": row["split"], "windows": {},
        })
        findings = extract_lab_vital_findings(row)
        sid = extract_study_id(row["cxr_path"])
        if pd.notna(row["cxr_path"]) and sid is None:
            raise RuntimeError("Aligned image path lacks a study identifier")
        cxr = radiographic[sid] if sid is not None else None
        if cxr is not None:
            for name, sources in sorted(cxr["findings"].items()):
                findings.append({"name": name, "predicate": "shows_radiographic",
                                 "source": "cxr_labels", "value": None, "label_sources": sorted(sources)})
        for finding in findings:
            finding.update(mappings[finding["name"]])
        stay["windows"][f"window_{row['window_idx']}"] = {
            "start": row["window_start"].isoformat(), "end": row["window_end"].isoformat(),
            "cxr_coverage": cxr["coverage"] if cxr is not None else {
                source: "no_image" for source in ("chexpert", "negbio")
            }, "findings": findings,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.PATIENT_WINDOWS_PARQUET)
    parser.add_argument("--output", type=Path, default=config.TEMPORAL_QUADRUPLES_JSON)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample_size", type=int, default=config.STEP2_SAMPLE_SIZE)
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not args.full and args.sample_size < 1:
        parser.error("--sample_size must be positive")
    try:
        windows = select_windows(pd.read_parquet(args.input), None if args.full else args.sample_size)
        study_ids = [extract_study_id(path) for path in windows.cxr_path.dropna()]
        radiographic = load_radiographic_findings([sid for sid in study_ids if sid is not None])
        mapping = build_mappings()
        trajectories = assemble_trajectories(windows, radiographic, mapping["mappings"])
        all_windows = [window for stay in trajectories.values() for window in stay["windows"].values()]
        findings = [finding for window in all_windows for finding in window["findings"]]
        summary = {
            "validation": "passed", "unique_patients": int(windows.subject_id.nunique()),
            "stays": len(trajectories), "windows": len(all_windows),
            "patients_per_split": windows.groupby("split").subject_id.nunique().to_dict(),
            "windows_with_image": int(windows.cxr_path.notna().sum()),
            "empty_finding_windows": sum(not window["findings"] for window in all_windows),
            "findings": len(findings),
            "matched_findings": sum(f["mapping_status"] == "matched" for f in findings),
            "unresolved_findings": sum(f["mapping_status"] == "unresolved" for f in findings),
            "cxr_coverage_by_window": {
                source: pd.Series([window["cxr_coverage"][source] for window in all_windows]).value_counts().to_dict()
                for source in ("chexpert", "negbio")
            }, "clinical_nodes": mapping["clinical_nodes"],
            "threshold": mapping["threshold"], "threshold_status": "provisional",
            "mapping_correctness_review": "pending", "observation_policy": "latest_only",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(trajectories, indent=2, allow_nan=False) + "\n")
        (args.output.parent / "step2_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output.parent / "sapbert_mapping_evidence.json").write_text(json.dumps(mapping, indent=2) + "\n")
        logger.info("Step 2 summary: %s", json.dumps(summary))
    except Exception as exc:
        # Parser and filesystem exceptions can contain patient rows or identifying paths.
        logger.error("Step 2 failed (%s); inspect locally using synthetic inputs", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
