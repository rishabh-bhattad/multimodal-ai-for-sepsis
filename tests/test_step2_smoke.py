"""Smoke test for chest radiograph study identifier parsing and validation."""

import sys
import json
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src import step2_extraction as step2
from src.step2_extraction import extract_study_id


def test_study_id_extraction():
    path = "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384082.jpg"
    assert extract_study_id(path) == 50414267
    assert extract_study_id(None) is None
    assert extract_study_id("") is None
    print("[PASS] test_study_id_extraction")


@pytest.fixture
def windows():
    """Create synthetic four-window trajectories with both image and no-image subjects."""
    rows = []
    for subject in range(1, 13):
        for idx in range(4):
            start = pd.Timestamp('2020-01-01') + pd.Timedelta(hours=idx * 6)
            rows.append({
                'subject_id': subject, 'stay_id': subject, 'window_idx': idx,
                'window_start': start, 'window_end': start + pd.Timedelta(hours=6),
                'split': 'train' if subject < 10 else 'test',
                'cxr_path': 'p00/p1/s100/image.jpg' if subject == 1 and idx == 0 else None,
                **{f'{name}_latest': float('nan') for name in config.LAB_SEVERITY_THRESHOLDS},
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize('measurement,value,expected', [
    ('map', 69.9, 'Systemic Hypotension'), ('map', 70, None),
    ('platelets', 149, 'Thrombocytopenia'), ('platelets', 150, None),
    ('creatinine', 1.2, 'Elevated Creatinine'), ('creatinine', 1.19, None),
    ('lactate', 4.1, 'Elevated Lactate above 4 mmol/L'),
    ('lactate', 4, 'Hyperlactatemia'), ('lactate', 2, None),
    ('wbc', 12, None), ('wbc', 12.1, 'Leukocytosis'), ('wbc', 3.9, 'Leukopenia'),
    ('temp', 38.3, None), ('temp', 38.4, 'Pyrexia'), ('temp', 35.9, 'Hypothermia'),
])
def test_latest_finding_boundaries(windows, measurement, value, expected):
    row = windows.iloc[0].to_dict()
    row[f'{measurement}_latest'] = value
    row[f'{measurement}_earlier_max'] = 999  # Earlier values must not affect extraction.
    findings = step2.extract_lab_vital_findings(row)
    assert [finding['name'] for finding in findings] == ([expected] if expected else [])


def test_sampling_keeps_patient_splits_and_all_four_windows(windows):
    sample = step2.select_windows(windows, 10)
    assert sample.subject_id.nunique() == 10
    assert len(sample) == 40
    assert sample.cxr_path.notna().any() and sample.cxr_path.isna().any()
    pd.testing.assert_frame_equal(sample, step2.select_windows(windows.sample(frac=1, random_state=8), 10))
    assert sample.groupby('subject_id').split.first().to_dict() == {
        subject: windows[windows.subject_id == subject].split.iloc[0] for subject in sample.subject_id.unique()
    }
    assert len(step2.select_windows(windows, None)) == len(windows)


def test_invalid_windows_fail_before_processing(windows):
    with pytest.raises(RuntimeError, match='Duplicate'):
        step2.select_windows(pd.concat([windows, windows.iloc[:1]]), 10)
    with pytest.raises(RuntimeError, match='four windows'):
        step2.select_windows(windows.iloc[1:], 10)
    leaked = windows.copy()
    leaked.loc[0, 'split'] = 'test'
    with pytest.raises(RuntimeError, match='split isolation'):
        step2.select_windows(leaked, 10)


def test_cxr_positive_union_and_source_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'MIMIC_CXR_DIR', tmp_path)
    baseline = {name: 0.0 for name in config.TARGET_PATHOLOGIES}
    pd.DataFrame([
        {'study_id': 100, **baseline, 'Pneumonia': 1, 'Edema': -1},
        {'study_id': 101, **baseline, 'Pneumonia': None},
    ]).to_csv(tmp_path / 'mimic-cxr-2.0.0-chexpert.csv', index=False)
    pd.DataFrame([
        {'study_id': 100, **baseline, 'Pneumonia': 1, 'Edema': 1},
    ]).to_csv(tmp_path / 'mimic-cxr-2.0.0-negbio.csv.gz', index=False)
    findings = step2.load_radiographic_findings([100, 101, 102])
    assert findings[100]['findings'] == {'Pneumonia': {'chexpert', 'negbio'}, 'Edema': {'negbio'}}
    assert findings[101]['coverage'] == {'chexpert': 'covered_no_positive', 'negbio': 'missing_study'}
    assert findings[102]['findings'] == {}
    assert findings[102]['coverage'] == {'chexpert': 'missing_study', 'negbio': 'missing_study'}
    (tmp_path / 'mimic-cxr-2.0.0-negbio.csv.gz').unlink()
    with pytest.raises(RuntimeError, match='negbio label file'):
        step2.load_radiographic_findings([100])


def test_assembly_retains_empty_windows_and_unresolved_findings(windows):
    windows = windows[windows.subject_id == 2].copy()
    windows.loc[windows.window_idx == 1, 'platelets_latest'] = 90
    mapping = {'Thrombocytopenia': {'node': None, 'similarity': 0.5,
               'matched_query': 'Thrombocytopenia', 'mapping_status': 'unresolved'}}
    result = step2.assemble_trajectories(windows, {}, mapping)
    trajectory = result['2']['windows']
    assert len(trajectory) == 4
    assert trajectory['window_0']['findings'] == []
    assert trajectory['window_1']['findings'][0]['mapping_status'] == 'unresolved'
    assert trajectory['window_1']['findings'][0]['value'] == 90
    assert trajectory['window_2']['findings'] == []
    assert trajectory['window_0']['cxr_coverage']['chexpert'] == 'no_image'
    json.dumps(result, allow_nan=False)


def test_cxr_edema_output_retains_source_label_and_mapping_query(windows):
    windows = windows[windows.subject_id == 1].copy()
    mapping = {'Edema': {'node': {'node_type': 'effect/phenotype', 'node_id': '100598',
               'node_name': 'Pulmonary edema'}, 'similarity': 0.99,
               'matched_query': 'Pulmonary edema', 'mapping_status': 'matched'}}
    labels = {100: {'findings': {'Edema': {'chexpert'}},
                   'coverage': {'chexpert': 'positive', 'negbio': 'covered_no_positive'}}}
    result = step2.assemble_trajectories(windows, labels, mapping)
    finding = result['1']['windows']['window_0']['findings'][0]
    assert finding['name'] == 'Edema'
    assert finding['source'] == 'cxr_labels'
    assert finding['label_sources'] == ['chexpert']
    assert finding['matched_query'] == 'Pulmonary edema'
    assert finding['node']['node_name'] == 'Pulmonary edema'


def test_clinical_vocabulary_retains_native_ids(tmp_path, monkeypatch):
    path = tmp_path / 'kg.csv'
    pd.DataFrame([
        {'relation': 'disease_phenotype_positive', 'x_type': 'disease', 'x_id': '001',
         'x_name': 'Synthetic disease', 'y_type': 'effect/phenotype', 'y_id': 'HP:001', 'y_name': 'Synthetic finding'},
        {'relation': 'disease_protein', 'x_type': 'disease', 'x_id': '002',
         'x_name': 'Excluded disease', 'y_type': 'gene/protein', 'y_id': '123', 'y_name': 'Synthetic protein'},
    ]).to_csv(path, index=False)
    monkeypatch.setattr(config, 'PRIMEKG_CSV', path)
    nodes = step2.load_clinical_nodes()
    assert {(node['node_type'], node['node_id']) for node in nodes} == {
        ('disease', '001'), ('effect/phenotype', 'HP:001'),
    }


def test_primekg_quoted_names_after_sniff_sample(tmp_path, monkeypatch):
    row = {'relation': 'disease_phenotype_positive', 'x_type': 'disease', 'x_id': '001',
           'x_name': 'Synthetic disease', 'y_type': 'effect/phenotype',
           'y_id': 'HP:001', 'y_name': 'Synthetic finding'}
    path = tmp_path / 'kg.csv'
    pd.DataFrame([row] * 20500 + [{**row, 'x_id': '002', 'x_name': 'Synthetic, quoted disease'}]).to_csv(path, index=False)
    monkeypatch.setattr(config, 'PRIMEKG_CSV', path)
    assert len(step2.load_clinical_nodes()) == 3


if __name__ == "__main__":
    test_study_id_extraction()
    print("[SUCCESS] Study ID extraction smoke test passed.")
