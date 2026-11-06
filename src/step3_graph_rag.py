"""Load faithful clinical PrimeKG context for isolated temporal retrieval."""

import csv
import hashlib
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


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
