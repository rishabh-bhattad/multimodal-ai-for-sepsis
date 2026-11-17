"""Lossless encoding of the existing displayed KG triples, local to each stay."""
from collections import defaultdict
import json

DICTIONARY = "- Compact KG dictionary v1:"
WINDOW = "- Retrieved KG Prior (compact v1):"
STATIC = "- Static KG Prior (compact v1, shared across all windows):"


def displayed_triples(graph, max_edges):
    """Match legacy name/relation deduplication and edge-cap semantics."""
    if graph is None:
        return []
    triples = set()
    for source, target, key, data in graph.edges(keys=True, data=True):
        def name(node):
            return str(graph.nodes[node].get("node_name")
                       or (node[1] if isinstance(node, tuple) else node)).strip()
        triples.add((name(source), str(data.get("relation", key)).strip(), name(target)))
    return sorted(triples, key=lambda t: (tuple(x.lower() for x in t), t))[:max_edges]


def encode_graphs(groups):
    """Return a dictionary and one directed adjacency list per graph group."""
    entities = sorted({name for triples in groups for s, _, t in triples for name in (s, t)})
    relations = sorted({r for triples in groups for _, r, _ in triples})
    entity_ids = {name: i for i, name in enumerate(entities)}
    relation_ids = {name: i for i, name in enumerate(relations)}
    encoded = []
    for triples in groups:
        adjacency = defaultdict(set)
        for source, relation, target in triples:
            adjacency[entity_ids[source], relation_ids[relation]].add(entity_ids[target])
        encoded.append([[s, r, sorted(targets)] for (s, r), targets in sorted(adjacency.items())])
    return {"entities": entities, "relations": relations}, encoded


def json_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def decode_prompt(prompt):
    """Reconstruct displayed triples for each window; used for fidelity checks."""
    lines = prompt.splitlines()
    if DICTIONARY not in lines:
        return [[] for _ in range(4)]
    dictionary = json.loads(lines[lines.index(DICTIONARY) + 1])
    def decode(payload):
        return sorted((dictionary["entities"][s], dictionary["relations"][r], dictionary["entities"][t])
                      for s, r, targets in payload for t in targets)
    if STATIC in lines:
        triples = decode(json.loads(lines[lines.index(STATIC) + 1]))
        return [list(triples) for _ in range(4)]
    groups = []
    current = -1
    for index, line in enumerate(lines):
        if line.startswith("[TIMELINE:"):
            groups.append([])
            current += 1
        elif line == WINDOW:
            groups[current] = decode(json.loads(lines[index + 1]))
    if len(groups) != 4:
        raise ValueError("Expected four compact graph windows")
    return groups
