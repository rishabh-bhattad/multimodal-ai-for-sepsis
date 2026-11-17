"""Compact encoding preserves legacy displayed evidence without outcome information."""
import networkx as nx
import pytest
from src.compact_kg import DICTIONARY, STATIC, decode_prompt
from src.step4_serialization import assemble_multimodal_prompt, build_stay_example, fit_graph_context, ContextBudgetError
from src.step5_ablations import build_static_rag_prompt, build_no_kg_prompt


def windows():
    return [dict(window_idx=i, map_latest=65+i, cxr_path="image.jpg" if i==1 else None) for i in range(4)]


def graphs():
    graph=nx.MultiDiGraph()
    graph.add_node(("disease","1"), node_name='Disease "A"')
    graph.add_node(("effect/phenotype","2"), node_name="Finding [B]")
    graph.add_edge(("disease","1"),("effect/phenotype","2"), key="negative",
                   relation="disease_phenotype_negative")
    graph.add_edge(("effect/phenotype","2"),("disease","1"), key="positive",
                   relation="disease_phenotype_positive")
    return {"window_0":graph,"window_3":graph.copy()}


def test_directed_negative_relations_and_empty_windows_roundtrip():
    prompt,images=assemble_multimodal_prompt(windows(),graphs())
    expected=sorted([('Disease "A"',"disease_phenotype_negative","Finding [B]"),
                     ("Finding [B]","disease_phenotype_positive",'Disease "A"')])
    assert decode_prompt(prompt)==[expected,[],[],expected]
    assert prompt.count(DICTIONARY)==1
    assert len(images)==prompt.count("<image>")==1
    for i in range(4):
        assert f"MAP {65+i} mmHg" in prompt


def test_static_edges_stored_once_but_apply_to_every_window():
    prompt,_=build_static_rag_prompt(windows(),graphs())
    decoded=decode_prompt(prompt)
    assert all(group==decoded[0] for group in decoded)
    assert len(decoded[0])==2
    assert prompt.count(STATIC)==1


def test_prompt_is_independent_of_outcome_label():
    args=dict(stay_id=1,subject_id=2,split="train",stay_windows=windows(),subgraphs=graphs())
    assert build_stay_example(label=0,**args)["prompt_text"]==build_stay_example(label=1,**args)["prompt_text"]


def test_no_kg_has_no_dictionary():
    prompt,_=build_no_kg_prompt(windows())
    assert DICTIONARY not in prompt


def test_compact_prompt_is_never_silently_pruned():
    prompt,_=assemble_multimodal_prompt(windows(),graphs())
    with pytest.raises(ContextBudgetError,match="no edges were removed"):
        fit_graph_context(prompt,lambda text:{"input_ids":list(range(100))},10)


def test_dictionary_is_deterministic_with_graph_insertion_order():
    original=graphs()
    reversed_graph=nx.MultiDiGraph()
    for node,data in reversed(list(original["window_0"].nodes(data=True))):
        reversed_graph.add_node(node,**data)
    for s,t,k,data in reversed(list(original["window_0"].edges(keys=True,data=True))):
        reversed_graph.add_edge(s,t,key=k,**data)
    changed={"window_0":reversed_graph,"window_3":reversed_graph}
    assert assemble_multimodal_prompt(windows(),original)[0]==assemble_multimodal_prompt(windows(),changed)[0]


def test_cap_preserves_legacy_whole_triple_order_before_case_tiebreak():
    from src.compact_kg import displayed_triples
    g=nx.MultiDiGraph()
    g.add_node(1,node_name="X")
    g.add_node(2,node_name="x")
    g.add_node(3,node_name="Y")
    g.add_edge(1,3,relation="b")
    g.add_edge(2,3,relation="a")
    assert displayed_triples(g,1)==[("x","a","Y")]
