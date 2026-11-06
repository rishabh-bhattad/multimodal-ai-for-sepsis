"""Load faithful clinical PrimeKG context for isolated temporal retrieval."""

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import random
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

logger = logging.getLogger("Step3-GraphRAG")


def load_primekg_graph(path=None):
    """Stream public CSV rows into a native-ID directed multigraph.

    Keys are (node_type, node_id), with IDs retained as strings. Each native
    relation is its own edge key; repeated identical triples are deduplicated.
    Source direction is preserved even when a later retrieval traverses an
    undirected projection. Negative associations are context, never positive
    evidence or connecting-path support.
    """
    path = Path(config.PRIMEKG_CSV if path is None else path)
    initial_stat = path.stat()
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()

    allowed_types = set(config.PRIMEKG_ALLOWED_NODE_TYPES)
    allowed_relations = set(config.PRIMEKG_RELATION_WHITELIST)
    graph = nx.MultiDiGraph()
    source_rows = retained_rows = 0
    required = {"relation", "x_type", "x_id", "x_name", "y_type", "y_id", "y_name"}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("PrimeKG CSV lacks required native-node columns")
        for row in reader:
            source_rows += 1
            if None in row or any(row[field] is None for field in required):
                raise ValueError("Malformed PrimeKG CSV row")
            relation = row["relation"]
            if (row["x_type"] not in allowed_types or row["y_type"] not in allowed_types
                    or relation not in allowed_relations):
                continue
            endpoints = []
            for prefix in ("x", "y"):
                node_type, node_id, name = (
                    row[f"{prefix}_type"], row[f"{prefix}_id"], row[f"{prefix}_name"]
                )
                if not node_id.strip() or not name.strip():
                    raise ValueError("Eligible PrimeKG endpoint lacks identity or name")
                key = (node_type, node_id)
                # Match Step 2's deterministic name choice for a native identity.
                previous_name = graph.nodes[key]["node_name"] if key in graph else name
                graph.add_node(key, node_type=node_type, node_id=node_id,
                               node_name=min(previous_name, name))
                endpoints.append(key)
            negative = relation == "disease_phenotype_negative"
            polarity = "negative" if negative else (
                "positive" if relation == "disease_phenotype_positive" else "unspecified"
            )
            graph.add_edge(*endpoints, key=relation, relation=relation,
                           association_polarity=polarity,
                           evidence_role="negative_association" if negative else "biomedical_context",
                           eligible_for_connecting_paths=not negative)
            retained_rows += 1

    final_stat = path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (
            final_stat.st_size, final_stat.st_mtime_ns):
        raise RuntimeError("PrimeKG source changed during ingestion")
    if not graph:
        raise ValueError("No eligible clinical PrimeKG nodes")
    graph.graph["provenance"] = {
        "source_name": path.name,
        "source_sha256": digest,
        "source_bytes": initial_stat.st_size,
        "source_release": "unverified_local_file",
        "allowed_node_types": sorted(allowed_types),
        "relation_whitelist": sorted(allowed_relations),
        "representation": "networkx.MultiDiGraph",
        "node_identity": ["node_type", "native_node_id_string"],
        "direction_policy": "preserve_native_x_to_y",
        "duplicate_policy": "deduplicate_identical_directed_relation_triples",
        "negative_association_policy": "explicit_context_excluded_from_connecting_paths",
        "source_rows": source_rows,
        "retained_rows": retained_rows,
        "duplicate_rows": retained_rows - graph.number_of_edges(),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
    }
    return graph


def build_path_connectivity(graph):
    """Index eligible associations in either direction without changing source edges.

    The projection is only a connectivity index, not a patient subgraph or an
    assertion that biomedical relationships have reversible meanings. Build it
    once per loaded, unchanged PrimeKG graph and reuse it across windows.
    """
    connectivity = nx.Graph()
    connectivity.add_nodes_from(graph)
    connectivity.add_edges_from(
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data.get("eligible_for_connecting_paths", False)
    )
    return connectivity


def retrieve_connecting_path_edges(graph, active_nodes, *, connectivity=None,
                                   max_path_length=None):
    """Return sorted native edges on every shortest path of at most two hops.

    For each distinct pair of present seeds, a direct connection takes priority.
    Otherwise each common neighbor supplies a shortest two-edge path. This
    bounds search structurally without enumerating longer paths or calling
    ``all_shortest_paths`` with an unsupported cutoff. Missing, isolated, and
    singleton seeds yield no invented edges; trajectory assembly retains those
    seeds separately. Negative associations never enter the path edge set.

    ``connectivity`` may reuse ``build_path_connectivity(graph)`` provided the
    source graph has not changed. Returned (source, target, relation) triples
    preserve every eligible parallel relation and original edge direction.
    Window edge budgeting is applied after path selection.
    """
    limit = config.STEP3_MAX_PATH_LENGTH if max_path_length is None else max_path_length
    if type(limit) is not int or limit not in (1, 2):
        raise ValueError("Connecting paths must be limited to one or two edges")
    if connectivity is None:
        connectivity = build_path_connectivity(graph)
    seeds = sorted(set(active_nodes).intersection(graph))
    selected = set()

    def retain_native_edges(first, second):
        for source, target in ((first, second), (second, first)):
            for relation, data in graph.get_edge_data(source, target, default={}).items():
                if data.get("eligible_for_connecting_paths", False):
                    selected.add((source, target, relation))

    for source, target in combinations(seeds, 2):
        if connectivity.has_edge(source, target):
            retain_native_edges(source, target)
        elif limit == 2:
            # Every shared neighbor is a shortest path; never expand past it.
            common = set(connectivity.neighbors(source)).intersection(connectivity.neighbors(target))
            for middle in sorted(common):
                retain_native_edges(source, middle)
                retain_native_edges(middle, target)
    return tuple(sorted(selected))


def retrieve_window_subgraph(graph, active_nodes, *, connectivity=None,
                             max_path_length=None, max_edges=None):
    """Retrieve one window's independent, bounded clinical context graph.

    Connecting-path edges have priority over one-hop edges incident to active
    seeds. Each directed relation counts once against the configured cap;
    ``max_edges`` can lower that cap, including to zero, but cannot raise it.
    Within each priority group, native (source, target, relation) ordering
    determines pruning. If paths alone overflow, individual path edges may be
    omitted; metadata records this explicitly rather than implying that every
    retrieved path is complete.

    Neighbor expansion includes incoming and outgoing native edges, including
    explicitly negative context, but never recursively expands new neighbors.
    Present seeds survive even without retained edges. Missing seeds remain
    metadata instead of becoming invented PrimeKG nodes. All retained graph,
    node, and edge attributes are copied so later windows cannot mutate them.
    """
    cap = config.STEP3_MAX_EDGES_PER_WINDOW
    budget = cap if max_edges is None else max_edges
    if type(cap) is not int or cap < 0:
        raise ValueError("Configured edge cap must be a nonnegative integer")
    if type(budget) is not int or not 0 <= budget <= cap:
        raise ValueError("Window edge budget must be an integer within the configured cap")
    requested = set(active_nodes)
    seeds = requested.intersection(graph)
    path_edges = retrieve_connecting_path_edges(
        graph, seeds, connectivity=connectivity, max_path_length=max_path_length
    )
    path_edge_set = set(path_edges)
    allowed_types = set(config.PRIMEKG_ALLOWED_NODE_TYPES)
    allowed_relations = set(config.PRIMEKG_RELATION_WHITELIST)
    neighbor_edges = set()
    for seed in sorted(seeds):
        incident = set(graph.in_edges(seed, keys=True)) | set(graph.out_edges(seed, keys=True))
        for source, target, relation in incident:
            if (graph.nodes[source]["node_type"] in allowed_types
                    and graph.nodes[target]["node_type"] in allowed_types
                    and relation in allowed_relations
                    and (source, target, relation) not in path_edge_set):
                neighbor_edges.add((source, target, relation))

    retained_paths = path_edges[:budget]
    retained_neighbors = tuple(sorted(neighbor_edges))[:budget - len(retained_paths)]
    selected = set(retained_paths) | set(retained_neighbors)
    selected_nodes = seeds | {node for source, target, _ in selected for node in (source, target)}
    result = nx.MultiDiGraph()
    if "provenance" in graph.graph:
        result.graph["provenance"] = deepcopy(graph.graph["provenance"])
    for node in sorted(selected_nodes):
        attributes = deepcopy(graph.nodes[node])
        attributes["is_active_seed"] = node in seeds
        result.add_node(node, **attributes)
    for source, target, relation in sorted(selected):
        attributes = deepcopy(graph.edges[source, target, relation])
        attributes["retrieval_role"] = (
            "connecting_path" if (source, target, relation) in path_edge_set else "one_hop_neighbor"
        )
        result.add_edge(source, target, key=relation, **attributes)
    result.graph["retrieval"] = {
        "max_edges": budget,
        "max_path_length": config.STEP3_MAX_PATH_LENGTH if max_path_length is None else max_path_length,
        "neighbor_expansion": "one_hop_from_active_seeds",
        "pruning_order": "path_edges_then_neighbor_edges_native_identity",
        "active_seed_nodes": tuple(sorted(seeds)),
        "missing_seed_nodes": tuple(sorted(requested - seeds)),
        "isolated_seed_nodes": tuple(node for node in sorted(seeds) if result.degree(node) == 0),
        "seeds_without_connecting_paths": tuple(sorted(
            seeds - {node for source, target, _ in path_edges for node in (source, target)}
        )),
        "path_edges_available": len(path_edges),
        "path_edges_retained": len(retained_paths),
        "path_edges_pruned": len(path_edges) - len(retained_paths),
        "neighbor_edges_available": len(neighbor_edges),
        "neighbor_edges_retained": len(retained_neighbors),
        "neighbor_edges_pruned": len(neighbor_edges) - len(retained_neighbors),
        "path_edges_truncated": len(retained_paths) < len(path_edges),
        "truncated": len(selected) < len(path_edges) + len(neighbor_edges),
        "negative_association_edges": sum(
            data["association_polarity"] == "negative" for _, _, data in result.edges(data=True)
        ),
    }
    return result


def _finding_node_key(finding):
    """Validate the Step 2 mapping contract without changing or rematching it."""
    required = {"name", "predicate", "source", "value", "node", "similarity",
                "matched_query", "mapping_status"}
    if not isinstance(finding, dict) or not required.issubset(finding):
        raise ValueError("Finding lacks required Step 2 fields")
    if finding["predicate"] not in ("exhibits", "shows_radiographic"):
        raise ValueError("Unexpected finding predicate")
    status, node = finding["mapping_status"], finding["node"]
    if status == "unresolved":
        if node is not None:
            raise ValueError("Unresolved findings must not supply graph seeds")
        return None
    if status != "matched" or not isinstance(node, dict):
        raise ValueError("Finding mapping must be matched or explicitly unresolved")
    if (node.get("node_type") not in config.PRIMEKG_ALLOWED_NODE_TYPES
            or not isinstance(node.get("node_id"), str) or not node["node_id"].strip()
            or not isinstance(node.get("node_name"), str) or not node["node_name"].strip()):
        raise ValueError("Matched finding lacks a native clinical node identity")
    return node["node_type"], node["node_id"]


def validate_temporal_quadruples(trajectories):
    """Check all stays and subject splits before any retrieval or sampling.

    Validation is read-only. Errors describe contract failures without quoting
    patient identifiers, findings, or timestamps. Existing mapping decisions
    and cohort membership are never recomputed here.
    """
    if not isinstance(trajectories, dict) or not trajectories:
        raise ValueError("Temporal quadruples must contain at least one stay")
    window_keys = [f"window_{index}" for index in range(config.NUM_WINDOWS)]
    coverage_states = {"positive", "covered_no_positive", "missing_study", "no_image"}
    subject_splits = {}
    for stay_id, stay in trajectories.items():
        if not isinstance(stay_id, str) or not stay_id.isdecimal() or not isinstance(stay, dict):
            raise ValueError("Stay records must use native stay-ID string keys")
        subject, split = stay.get("subject_id"), stay.get("split")
        if type(subject) is not int or subject <= 0 or split not in ("train", "val", "test"):
            raise ValueError("Stay lacks valid subject and split metadata")
        if subject in subject_splits and subject_splits[subject] != split:
            raise ValueError("Subject split isolation failed")
        subject_splits[subject] = split
        windows = stay.get("windows")
        if not isinstance(windows, dict) or set(windows) != set(window_keys):
            raise ValueError("Each stay must contain exactly four configured windows")
        previous_end = None
        for key in window_keys:
            window = windows[key]
            if not isinstance(window, dict):
                raise ValueError("Window must contain timestamps, findings, and CXR coverage")
            try:
                start = datetime.fromisoformat(window["start"])
                end = datetime.fromisoformat(window["end"])
                valid_interval = end - start == timedelta(hours=config.WINDOW_SIZE_HOURS)
                valid_continuity = previous_end is None or start == previous_end
            except (KeyError, TypeError, ValueError):
                raise ValueError("Window timestamps must be consistent ISO intervals") from None
            if not valid_interval or not valid_continuity:
                raise ValueError("Windows must be contiguous six-hour intervals")
            previous_end = end
            coverage = window.get("cxr_coverage")
            if (not isinstance(coverage, dict) or set(coverage) != {"chexpert", "negbio"}
                    or any(not isinstance(state, str) or state not in coverage_states
                           for state in coverage.values())):
                raise ValueError("Window must preserve both CXR source coverage states")
            if not isinstance(window.get("findings"), list):
                raise ValueError("Window findings must be an explicit list, including when empty")
            for finding in window["findings"]:
                _finding_node_key(finding)


def assemble_subgraph_trajectories(graph, trajectories, *, max_path_length=None, max_edges=None):
    """Return four independent window graphs per stay with unchanged splits.

    The result retains Step 2's stay/subject grouping. Its ``windows`` mapping
    contains separate MultiDiGraphs whose metadata holds window index,
    timestamps, original findings, and separate CXR coverage. Labels, anchors,
    and other cohort metadata are not graph seeds. Unresolved findings remain
    observations; matched nodes absent from this graph are recorded separately.
    No graph or mutable metadata object is shared across stays or windows.
    """
    validate_temporal_quadruples(trajectories)
    connectivity = build_path_connectivity(graph)
    result = {}
    for stay_id in sorted(trajectories):
        stay = trajectories[stay_id]
        windows = {}
        for index in range(config.NUM_WINDOWS):
            key = f"window_{index}"
            window = stay["windows"][key]
            finding_keys = [_finding_node_key(finding) for finding in window["findings"]]
            active_nodes = [node for node in finding_keys if node is not None]
            subgraph = retrieve_window_subgraph(
                graph, active_nodes, connectivity=connectivity,
                max_path_length=max_path_length, max_edges=max_edges,
            )
            subgraph.graph.update({
                "window_idx": index,
                "start": window["start"], "end": window["end"],
                "cxr_coverage": deepcopy(window["cxr_coverage"]),
                "findings": deepcopy(window["findings"]),
            })
            retrieval = subgraph.graph["retrieval"]
            missing = set(retrieval["missing_seed_nodes"])
            retrieval.update({
                "matched_findings": len(active_nodes),
                "unresolved_findings": finding_keys.count(None),
                "missing_graph_findings": sum(node in missing for node in active_nodes),
            })
            windows[key] = subgraph
        result[stay_id] = {"subject_id": stay["subject_id"], "split": stay["split"], "windows": windows}
    return result


def load_temporal_quadruples(path):
    """Load exact JSON bytes and their checksum, rejecting ambiguous records."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON keys in temporal quadruples")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("Nonfinite JSON constants are not valid Step 2 output")

    payload = Path(path).read_bytes()
    trajectories = json.loads(payload, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    return trajectories, hashlib.sha256(payload).hexdigest()


def select_trajectories(trajectories, sample_size=config.STEP3_SAMPLE_SIZE):
    """Select unique smoke subjects reproducibly, or retain all stays with None.

    Prefer an image-bearing representative stay for each subject and include
    both image/no-image representatives when available and sample size permits.
    Full mode preserves multiple stays for a subject and existing splits.
    """
    validate_temporal_quadruples(trajectories)
    if sample_size is None:
        return {key: trajectories[key] for key in sorted(trajectories)}
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("Sample size must be a positive integer")

    def has_image(stay_id):
        return any(state != "no_image" for window in trajectories[stay_id]["windows"].values()
                   for state in window["cxr_coverage"].values())

    representatives = {}
    for stay_id in sorted(trajectories, key=lambda key: (not has_image(key), key)):
        representatives.setdefault(trajectories[stay_id]["subject_id"], stay_id)
    if len(representatives) < sample_size:
        raise ValueError("Fewer unique subjects than requested; full mode was not selected")
    candidates = sorted(representatives.items())
    random.Random(config.RANDOM_SEED).shuffle(candidates)
    coverage_cases = {}
    for subject, stay_id in candidates:
        coverage_cases.setdefault(has_image(stay_id), (subject, stay_id))
    selected = [coverage_cases[case] for case in (True, False) if case in coverage_cases][:sample_size]
    selected_subjects = {subject for subject, _ in selected}
    selected.extend((subject, stay_id) for subject, stay_id in candidates
                    if subject not in selected_subjects)
    return {key: trajectories[key] for _, key in sorted(selected[:sample_size], key=lambda pair: pair[1])}


def summarize_trajectories(trajectories, *, mode, input_sha256):
    """Return aggregate-only execution diagnostics and file provenance."""
    stays = list(trajectories.values())
    windows = [window for stay in stays for window in stay["windows"].values()]
    subjects = {stay["subject_id"] for stay in stays}
    patients_per_split = {
        split: len({stay["subject_id"] for stay in stays if stay["split"] == split})
        for split in ("train", "val", "test")
    }
    retrievals = [window.graph["retrieval"] for window in windows]
    totals = {field: sum(item[field] for item in retrievals) for field in (
        "matched_findings", "unresolved_findings", "missing_graph_findings",
        "path_edges_available", "path_edges_retained", "path_edges_pruned",
        "neighbor_edges_available", "neighbor_edges_retained", "neighbor_edges_pruned",
        "negative_association_edges",
    )}
    return {
        "validation": "passed", "mode": mode, "input_sha256": input_sha256,
        "unique_patients": len(subjects), "stays": len(stays), "windows": len(windows),
        "patients_per_split": patients_per_split,
        "stays_per_split": dict(sorted(Counter(stay["split"] for stay in stays).items())),
        "findings": sum(len(window.graph["findings"]) for window in windows),
        "empty_finding_windows": sum(not window.graph["findings"] for window in windows),
        "empty_graph_windows": sum(window.number_of_nodes() == 0 for window in windows),
        "edgeless_windows": sum(window.number_of_edges() == 0 for window in windows),
        "windows_with_image": sum(any(state != "no_image" for state in window.graph["cxr_coverage"].values())
                                  for window in windows),
        "cxr_coverage_by_window": {
            source: dict(sorted(Counter(window.graph["cxr_coverage"][source] for window in windows).items()))
            for source in ("chexpert", "negbio")
        },
        "retrieved_edges": sum(window.number_of_edges() for window in windows),
        "max_edges_in_window": max(window.number_of_edges() for window in windows),
        "truncated_windows": sum(item["truncated"] for item in retrievals),
        "path_truncated_windows": sum(item["path_edges_truncated"] for item in retrievals),
        "isolated_seeds": sum(len(item["isolated_seed_nodes"]) for item in retrievals),
        "seeds_without_connecting_paths": sum(len(item["seeds_without_connecting_paths"]) for item in retrievals),
        "limits": {"max_path_length": config.STEP3_MAX_PATH_LENGTH,
                   "max_edges_per_window": config.STEP3_MAX_EDGES_PER_WINDOW},
        "graph_provenance": deepcopy(windows[0].graph.get("provenance", {})),
        "mapping_policy": "preserve_step2_decisions_without_rematching",
        **totals,
    }


def write_trajectory_outputs(trajectories, report, output_path):
    """Stage both artifacts before replacing outputs; publish the report last.

    The pickle is streamed, avoiding a second full in-memory serialized copy.
    Its checksum in the report identifies the matching artifact. Temporary
    files are created beside the outputs and removed on failure.
    """
    output_path = Path(output_path)
    report_path = output_path.parent / "step3_validation.json"
    if output_path.resolve() == report_path.resolve():
        raise ValueError("Pickle and validation report paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=output_path.parent,
                                         prefix=".step3-", suffix=".tmp", delete=False) as staged:
            pickle_path = Path(staged.name)
            temporary_paths.append(pickle_path)
            pickle.dump(trajectories, staged, protocol=pickle.HIGHEST_PROTOCOL)
            staged.flush()
            staged.seek(0)
            digest = hashlib.file_digest(staged, "sha256").hexdigest()
        published_report = {**report, "pickle_sha256": digest, "pickle_protocol": pickle.HIGHEST_PROTOCOL}
        report_bytes = (json.dumps(published_report, indent=2, allow_nan=False) + "\n").encode("utf-8")
        with tempfile.NamedTemporaryFile(mode="wb", dir=output_path.parent,
                                         prefix=".step3-", suffix=".tmp", delete=False) as staged:
            json_path = Path(staged.name)
            temporary_paths.append(json_path)
            staged.write(report_bytes)
        os.replace(pickle_path, output_path)
        os.replace(json_path, report_path)
        return published_report
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrieve four isolated temporal PrimeKG subgraphs per stay.")
    parser.add_argument("--input", type=Path, default=config.TEMPORAL_QUADRUPLES_JSON)
    parser.add_argument("--primekg", type=Path, default=config.PRIMEKG_CSV)
    parser.add_argument("--output", type=Path, default=config.SUBGRAPH_TRAJECTORIES_PKL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample_size", type=int, default=config.STEP3_SAMPLE_SIZE)
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    if not args.full and args.sample_size < 1:
        parser.error("--sample_size must be positive")
    if args.output.suffix.lower() != ".pkl":
        parser.error("--output must use the .pkl extension")
    try:
        protected = {args.input.resolve(), args.primekg.resolve()}
        targets = {args.output.resolve(), (args.output.parent / "step3_validation.json").resolve()}
        if protected.intersection(targets):
            raise ValueError("Output paths must not overwrite input artifacts")
        trajectories, digest = load_temporal_quadruples(args.input)
        selected = select_trajectories(trajectories, None if args.full else args.sample_size)
        graph = load_primekg_graph(args.primekg)
        result = assemble_subgraph_trajectories(graph, selected)
        summary = summarize_trajectories(result, mode="full" if args.full else "sample", input_sha256=digest)
        summary = write_trajectory_outputs(result, summary, args.output)
        logger.info("Step 3 summary: %s", json.dumps(summary, sort_keys=True))
    except Exception as exc:
        # Filesystem/parser exceptions may contain patient paths or record data.
        logger.error("Step 3 failed (%s); inspect locally using synthetic inputs", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    sys.exit(main())
