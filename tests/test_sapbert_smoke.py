"""Synthetic mapping checks; actual PrimeKG score evidence comes from the local smoke run."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.step2_extraction import rank_matches, resolve_mappings


def test_cosine_ranking_preserves_node_identity():
    nodes = [{'node_type': 'effect/phenotype', 'node_id': str(i), 'node_name': f'Finding {i}'} for i in range(3)]
    candidates = rank_matches(['Observation'], torch.tensor([[1., 0.]]), nodes,
                              torch.tensor([[0., 1.], [1., 0.], [-1., 0.]]))
    assert candidates['Observation'][0]['node']['node_id'] == '1'
    assert [match['similarity'] for match in candidates['Observation']] == [1., 0., -1.]


@pytest.mark.parametrize('score,status', [(0.699, 'unresolved'), (0.70, 'matched'), (0.71, 'matched')])
def test_provisional_cutoff_is_applied_consistently(monkeypatch, score, status):
    monkeypatch.setattr(config, 'SAPBERT_SIMILARITY_THRESHOLD', 0.70)
    candidates = {'Observation': [{'node': {'node_id': '1'}, 'similarity': score}]}
    match = resolve_mappings(['Observation'], candidates)['Observation']
    assert match['mapping_status'] == status
    assert (match['node'] is not None) == (status == 'matched')


def test_direct_mapping_does_not_substitute_another_query():
    candidates = {
        'Pyrexia': [{'node': {'node_id': '1'}, 'similarity': 0.60}],
        'Fever': [{'node': {'node_id': '2'}, 'similarity': 0.85}],
    }
    result = resolve_mappings(['Pyrexia'], candidates)
    assert result['Pyrexia']['node'] is None
    assert result['Pyrexia']['mapping_status'] == 'unresolved'
    assert result['Pyrexia']['matched_query'] == 'Pyrexia'
    assert result['Pyrexia']['similarity'] == 0.60


def test_cxr_edema_uses_pulmonary_query_but_preserves_original_label():
    candidates = {
        'Edema': [{'node': {'node_id': 'general', 'node_name': 'Edema'}, 'similarity': 1.0}],
        'Pulmonary edema': [{'node': {'node_id': 'pulmonary', 'node_name': 'Pulmonary edema'}, 'similarity': 0.99}],
    }
    mappings = resolve_mappings(['Edema'], candidates)
    assert set(mappings) == {'Edema'}
    assert mappings['Edema']['matched_query'] == 'Pulmonary edema'
    assert mappings['Edema']['node']['node_id'] == 'pulmonary'
