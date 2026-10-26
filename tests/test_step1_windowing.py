"""Synthetic clinical and temporal regressions; no MIMIC records are fixtures."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import step1_windowing as step1

START = datetime(2020, 1, 1)


def add(con, name, **values):
    """Insert a small synthetic event with explicit columns."""
    con.execute(f"INSERT INTO {name} ({', '.join(values)}) VALUES ({', '.join('?' for _ in values)})", list(values.values()))


@pytest.fixture
def con():
    db = duckdb.connect()
    for schema in ("mimiciv_icu", "mimiciv_hosp", "mimiciv_derived"):
        db.execute(f"CREATE SCHEMA {schema}")
    db.execute("CREATE MACRO observation_hours() AS 24")
    for name, columns in step1.RAW_COLUMNS.items():
        schema, table = name.split('/')
        fields = []
        for col in columns.split():
            dtype = 'VARCHAR'
            if col.endswith('_id') or col in {'itemid', 'org_itemid', 'linkorderid', 'anchor_age', 'anchor_year'}:
                dtype = 'BIGINT'
            elif col in step1.TIME_COLUMNS:
                dtype = 'TIMESTAMP'
            elif col == 'chartdate':
                dtype = 'DATE'
            elif col in step1.NUMBER_COLUMNS or (table == 'outputevents' and col == 'value'):
                dtype = 'DOUBLE'
            fields.append(f'{col} {dtype}')
        db.execute(f"CREATE TABLE mimiciv_{schema}.{table} ({', '.join(fields)})")
    db.execute("CREATE TABLE candidate_subjects (subject_id BIGINT)")
    db.execute("CREATE TABLE cohort_cxrs (subject_id BIGINT, hadm_id BIGINT, study_id BIGINT, dicom_id VARCHAR, cxr_view VARCHAR, cxr_time TIMESTAMP)")
    yield db
    db.close()


def chart(con, itemid, value, hour, subject=1, text=None):
    """Add a synthetic bedside observation to a single stay."""
    add(con, 'mimiciv_icu.chartevents', subject_id=subject, hadm_id=subject, stay_id=subject,
        charttime=START + timedelta(hours=hour), storetime=START + timedelta(hours=hour),
        itemid=itemid, valuenum=value, value=text if text is not None else str(value))


def seed_stay(con, subject=1):
    """Create three ICU days with normal HR/MAP and a normal measured GCS."""
    add(con, 'mimiciv_icu.icustays', subject_id=subject, hadm_id=subject, stay_id=subject, intime=START, outtime=START + timedelta(hours=72), los=3)
    add(con, 'mimiciv_hosp.admissions', subject_id=subject, hadm_id=subject, admittime=START, dischtime=START + timedelta(hours=72))
    add(con, 'candidate_subjects', subject_id=subject)
    for hour in range(73):
        chart(con, 220045, 80, hour, subject)
        chart(con, 220052, 82, hour, subject)
    for item, value in ((220739, 4), (223900, 5), (223901, 6)):
        chart(con, item, value, 0, subject)


def lab(con, itemid, value, hour=30, text=None, specimen=1):
    """Keep gas specimen components together, as the reference expects."""
    add(con, 'mimiciv_hosp.labevents', subject_id=1, hadm_id=1, specimen_id=specimen,
        charttime=START + timedelta(hours=hour), storetime=START + timedelta(hours=hour),
        itemid=itemid, valuenum=value, value=text if text is not None else str(value))


def infection(con, culture_hour=30, antibiotic_hour=31, subject=1):
    """Pair a nonpositive culture and an IV prescription for the reference label."""
    add(con, 'mimiciv_hosp.prescriptions', subject_id=subject, hadm_id=subject,
        starttime=START + timedelta(hours=antibiotic_hour), stoptime=START + timedelta(hours=antibiotic_hour + 1),
        drug='Cefepime', drug_type='MAIN', route='IV')
    add(con, 'mimiciv_hosp.microbiologyevents', subject_id=subject, hadm_id=subject, micro_specimen_id=subject,
        chartdate=(START + timedelta(hours=culture_hour)).date(), charttime=START + timedelta(hours=culture_hour),
        spec_type_desc='BLOOD CULTURE', org_itemid=90856, org_name='NEGATIVE')


def test_normal_gcs_and_missing_components(con):
    seed_stay(con)
    step1.derive_reference(con)
    assert con.execute('SELECT gcs FROM mimiciv_derived.gcs').fetchone()[0] == 15
    assert con.execute('SELECT MAX(sofa_24hours) FROM mimiciv_derived.sofa').fetchone()[0] == 0
    assert con.execute('SELECT COUNT(*) FROM mimiciv_derived.sofa WHERE renal IS NOT NULL').fetchone()[0] == 0


def test_all_six_sofa_components(con):
    seed_stay(con)
    for item, value in ((220739, 1), (223900, 1), (223901, 1)):
        chart(con, item, value, 30)
    for item, value in ((51265, 10), (50885, 13), (50912, 6), (50821, 80), (50816, 100)):
        lab(con, item, value)
    lab(con, 52033, None, text='ART.')
    for hour in (29, 31):
        chart(con, 223849, None, hour, text='CMV')
    add(con, 'mimiciv_icu.inputevents', stay_id=1, itemid=221906, linkorderid=1,
        starttime=START + timedelta(hours=29), endtime=START + timedelta(hours=33), rate=.2, rateuom='mcg/kg/min', amount=1, patientweight=70)
    step1.derive_reference(con)
    row = con.execute('SELECT respiration, coagulation, liver, cardiovascular, cns, renal, sofa_24hours FROM mimiciv_derived.sofa WHERE endtime = ?', [START + timedelta(hours=30)]).fetchone()
    assert row == (4, 4, 4, 4, 4, 4, 24)
    assert con.execute('SELECT cardiovascular FROM mimiciv_derived.sofa WHERE endtime = ?', [START + timedelta(hours=32)]).fetchone()[0] == 4
    assert con.execute('SELECT cardiovascular FROM mimiciv_derived.sofa WHERE endtime = ?', [START + timedelta(hours=34)]).fetchone()[0] == 0


def test_gcs_ett_reference_handling(con):
    seed_stay(con)
    chart(con, 223900, 0, 30, text='No Response-ETT')
    step1.derive_reference(con)
    assert con.execute('SELECT gcs FROM mimiciv_derived.gcs WHERE charttime = ?', [START + timedelta(hours=30)]).fetchone()[0] == 15


def test_rolling_score_expires_after_24_hours(con):
    seed_stay(con)
    lab(con, 51265, 80)
    step1.derive_reference(con)
    values = dict(con.execute('SELECT hr, coagulation_24hours FROM mimiciv_derived.sofa WHERE hr IN (30, 53, 54)').fetchall())
    assert values == {30: 2, 53: 2, 54: 0}


def test_urine_output_can_score_renal_without_creatinine(con):
    seed_stay(con)
    for hour in range(1, 31):
        add(con, 'mimiciv_icu.outputevents', stay_id=1, charttime=START + timedelta(hours=hour), itemid=226559, value=5)
    step1.derive_reference(con)
    assert con.execute('SELECT renal FROM mimiciv_derived.sofa WHERE endtime = ?', [START + timedelta(hours=30)]).fetchone()[0] == 4


@pytest.mark.parametrize('culture_hour,antibiotic_hour,expected', [(10, 82, 1), (10, 83, 0), (54, 30, 1), (55, 30, 0)])
def test_infection_pair_intervals(con, culture_hour, antibiotic_hour, expected):
    seed_stay(con)
    infection(con, culture_hour, antibiotic_hour)
    # The reference attaches prescriptions to an ICU stay only if they start there.
    con.execute("UPDATE mimiciv_icu.icustays SET outtime = ?", [START + timedelta(hours=100)])
    step1.derive_reference(con)
    assert con.execute('SELECT MAX(suspected_infection) FROM mimiciv_derived.suspicion_of_infection').fetchone()[0] == expected


def test_future_recovery_does_not_change_earlier_label(con):
    seed_stay(con)
    lab(con, 51265, 80)
    infection(con)
    step1.derive_reference(con)
    before = con.execute('SELECT sofa_time, sepsis3 FROM mimiciv_derived.sepsis3').fetchone()
    lab(con, 51265, 240, hour=60, specimen=2)
    step1.derive_reference(con)
    assert con.execute('SELECT sofa_time, sepsis3 FROM mimiciv_derived.sepsis3').fetchone() == before
    assert before == (START + timedelta(hours=30), True)


def test_control_and_anchor_eligibility(con):
    seed_stay(con)
    infection(con, culture_hour=80, antibiotic_hour=70)
    step1.derive_reference(con)
    cohort = step1.select_cohort(con, None)
    assert cohort.empty is False  # Earlier antibiotic is the valid in-stay anchor.
    assert cohort.t0.iloc[0] == START + timedelta(hours=70)
    con.execute("UPDATE mimiciv_derived.suspicion_of_infection SET suspected_infection_time = ?", [START + timedelta(hours=100)])
    con.execute((config.STEP1_SQL_DIR / 'step1_cohort.sql').read_text())
    assert con.execute('SELECT COUNT(*) FROM candidate_cohort').fetchone()[0] == 0


def test_later_dysfunction_excludes_control_a(con):
    seed_stay(con)
    infection(con, culture_hour=28, antibiotic_hour=29)
    lab(con, 51265, 80, hour=65)
    step1.derive_reference(con)
    con.execute((config.STEP1_SQL_DIR / 'step1_cohort.sql').read_text())
    assert con.execute('SELECT COUNT(*) FROM candidate_cohort').fetchone()[0] == 0


def test_window_boundaries_temperature_slopes_and_images(con):
    seed_stay(con)
    infection(con, culture_hour=48, antibiotic_hour=49)
    chart(con, 223761, 98.6, 25)
    chart(con, 223762, 37, 26)
    lab(con, 50813, 2, hour=29, specimen=2)
    lab(con, 50813, 4, hour=30, specimen=3)
    lab(con, 50813, 99, hour=48, specimen=4)
    for hour, image in ((29, 'first'), (30, 'boundary'), (48, 'excluded')):
        add(con, 'cohort_cxrs', subject_id=1, hadm_id=1, study_id=hour, dicom_id=image, cxr_view='AP', cxr_time=START + timedelta(hours=hour))
    step1.derive_reference(con)
    cohort = step1.select_cohort(con, 1)
    windows = step1.aggregate_windows(con)
    assert len(windows) == 4
    assert windows.temp_mean.iloc[0] == pytest.approx(37)
    assert windows.lactate_latest.iloc[:2].tolist() == [2, 4]
    assert windows.cxr_dicom_id.dropna().tolist() == ['first', 'boundary']
    assert windows.hr_slope.tolist() == [0, 0, 0, 0]
    summary = step1.validate_windows(con, cohort, windows, 1)
    assert summary['validation'] == 'passed'
    assert summary['temperatures_outside_25_to_45_celsius'] == 0
    json.dumps(summary)
    windows.loc[0, ['window_start', 'window_end']] += pd.Timedelta(hours=1)
    with pytest.raises(RuntimeError, match='Window boundaries'):
        step1.validate_windows(con, cohort, windows, 1)


def test_ten_unique_patients_and_split_isolation(con):
    for subject in range(1, 11):
        seed_stay(con, subject)
        infection(con, subject=subject)
    step1.derive_reference(con)
    cohort = step1.select_cohort(con, 10)
    windows = step1.aggregate_windows(con)
    summary = step1.validate_windows(con, cohort, windows, 10)
    assert summary['unique_patients'] == 10
    assert summary['windows'] == 40
    assert summary['patients_per_split'] == {'test': 2, 'train': 7, 'val': 1}
    with pytest.raises(RuntimeError, match='Only 10 eligible subjects'):
        step1.select_cohort(con, 11)


def test_cxr_fractional_time_and_admission_matching(con, tmp_path, monkeypatch):
    seed_stay(con)
    con.execute('DROP TABLE cohort_cxrs')
    metadata = pd.DataFrame([
        {'subject_id': 1, 'study_id': 1, 'dicom_id': 'front', 'ViewPosition': 'AP', 'StudyDate': '20200102', 'StudyTime': '010203.5'},
        {'subject_id': 1, 'study_id': 2, 'dicom_id': 'lateral', 'ViewPosition': 'LATERAL', 'StudyDate': '20200102', 'StudyTime': '010203'},
        {'subject_id': 1, 'study_id': 3, 'dicom_id': 'outside', 'ViewPosition': 'PA', 'StudyDate': '20200110', 'StudyTime': '010203'},
    ])
    metadata.to_csv(tmp_path / 'mimic-cxr-2.0.0-metadata.csv', index=False)
    monkeypatch.setattr(config, 'MIMIC_CXR_DIR', tmp_path)
    step1.load_cxr_metadata(con)
    rows = con.execute('SELECT dicom_id, cxr_time FROM cohort_cxrs').fetchall()
    assert rows == [('front', datetime(2020, 1, 2, 1, 2, 3, 500000))]


def test_cxr_quoted_descriptions_after_csv_sniff_sample(con, tmp_path, monkeypatch):
    seed_stay(con)
    con.execute('DROP TABLE cohort_cxrs')
    ordinary = {'subject_id': 1, 'study_id': 1, 'dicom_id': 'front', 'ViewPosition': 'AP',
                'StudyDate': '20200102', 'StudyTime': '010203', 'PerformedProcedureStepDescription': 'CHEST'}
    metadata = pd.DataFrame([ordinary] * 20500 + [{**ordinary, 'dicom_id': 'quoted',
        'PerformedProcedureStepDescription': 'CHEST, PORTABLE "AP"'}])
    metadata.to_csv(tmp_path / 'mimic-cxr-2.0.0-metadata.csv', index=False)
    monkeypatch.setattr(config, 'MIMIC_CXR_DIR', tmp_path)
    step1.load_cxr_metadata(con)
    assert con.execute('SELECT COUNT(*) FROM cohort_cxrs').fetchone()[0] == 20501


def test_unobserved_windows_and_single_measurement_slope_remain_missing(con):
    seed_stay(con)
    infection(con, culture_hour=48, antibiotic_hour=49)
    step1.derive_reference(con)
    step1.select_cohort(con, 1)
    con.execute('DELETE FROM mimiciv_derived.vitalsign WHERE charttime != ?', [START + timedelta(hours=25)])
    windows = step1.aggregate_windows(con)
    assert windows.hr_mean.iloc[0] == 80
    assert windows.hr_mean.iloc[1:].isna().all()
    assert windows.hr_slope.isna().all()
    assert windows.lactate_latest.isna().all()


def test_control_b_requires_completed_nonpositive_cultures(con):
    seed_stay(con)
    lab(con, 51265, 80)
    infection(con)
    con.execute('DELETE FROM mimiciv_hosp.prescriptions')
    step1.derive_reference(con)
    cohort = step1.select_cohort(con, 1)
    assert cohort.cohort_type.tolist() == ['control_non_infectious_dysfunction']
    con.execute("UPDATE mimiciv_hosp.microbiologyevents SET org_itemid=90760, org_name='CANCELLED'")
    con.execute((config.STEP1_SQL_DIR / 'step1_cohort.sql').read_text())
    assert con.execute('SELECT COUNT(*) FROM candidate_cohort').fetchone()[0] == 0
    con.execute('DELETE FROM mimiciv_hosp.microbiologyevents')
    con.execute((config.STEP1_SQL_DIR / 'step1_cohort.sql').read_text())
    assert con.execute('SELECT COUNT(*) FROM candidate_cohort').fetchone()[0] == 0
