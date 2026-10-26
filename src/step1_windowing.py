"""Build a local MIT-LCP Sepsis-3 cohort and four six-hour observation windows."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.utils import setup_logger

logger = setup_logger("Step1-Windowing")

# Only fields needed for scoring/windowing are loaded, never clinical notes.
RAW_COLUMNS = {
    "icu/icustays": "subject_id hadm_id stay_id intime outtime los",
    "hosp/patients": "subject_id anchor_age anchor_year",
    "hosp/admissions": "subject_id hadm_id admittime dischtime",
    "icu/chartevents": "subject_id hadm_id stay_id charttime storetime itemid value valuenum valueuom",
    "hosp/labevents": "subject_id hadm_id specimen_id charttime storetime itemid value valuenum",
    "icu/inputevents": "stay_id itemid linkorderid starttime endtime rate rateuom amount patientweight",
    "icu/outputevents": "stay_id charttime itemid value",
    "hosp/prescriptions": "subject_id hadm_id starttime stoptime drug drug_type route",
    "hosp/microbiologyevents": "subject_id hadm_id micro_specimen_id chartdate charttime spec_type_desc org_itemid org_name",
}
TIME_COLUMNS = {"intime", "outtime", "admittime", "dischtime", "charttime", "storetime", "starttime", "endtime", "stoptime"}
NUMBER_COLUMNS = {"los", "valuenum", "rate", "amount", "patientweight"}


def load_raw_table(con, name, where="TRUE"):
    """Load a typed, selected projection so compressed inputs are scanned once."""
    schema, table = name.split("/")
    path = config.MIMIC_IV_DIR / schema / f"{table}.csv.gz"
    if not path.is_file():
        raise RuntimeError(f"Required local table is missing: {name}")
    fields = []
    for col in RAW_COLUMNS[name].split():
        dtype = None
        if col.endswith("_id") or col in {"itemid", "org_itemid", "linkorderid", "anchor_age", "anchor_year"}:
            dtype = "BIGINT"
        elif col in TIME_COLUMNS:
            dtype = "TIMESTAMP"
        elif col == "chartdate":
            dtype = "DATE"
        elif col in NUMBER_COLUMNS or (table == "outputevents" and col == "value"):
            dtype = "DOUBLE"
        fields.append(f"TRY_CAST({col} AS {dtype}) AS {col}" if dtype else col)
    con.execute(f"""
        CREATE OR REPLACE TABLE mimiciv_{schema}.{table} AS
        SELECT {', '.join(fields)}
        FROM read_csv(?, header=true, all_varchar=true, delim=',', quote='"', escape='"') raw
        WHERE {where}
    """, [str(path)])
    count = con.execute(f"SELECT COUNT(*) FROM mimiciv_{schema}.{table}").fetchone()[0]
    logger.info("Loaded %s: %d selected rows", name, count)


def load_candidates(con, sample_size, candidate_size=None):
    """Bound smoke runs by subject while retaining the history needed for labels."""
    for schema in ("mimiciv_icu", "mimiciv_hosp", "mimiciv_derived"):
        con.execute(f"CREATE SCHEMA {schema}")
    con.execute(f"CREATE MACRO observation_hours() AS {int(config.OBSERVATION_HOURS)}")
    load_raw_table(con, "icu/icustays")
    load_raw_table(con, "hosp/patients")
    load_raw_table(con, "hosp/admissions")
    limit = candidate_size or (sample_size * config.STEP1_CANDIDATE_MULTIPLIER if sample_size else None)
    limit_clause = "LIMIT ?" if limit else ""
    con.execute(f"""
        CREATE TEMP TABLE candidate_subjects AS
        SELECT DISTINCT s.subject_id, md5(s.subject_id::VARCHAR || ?) AS selection_key
        FROM mimiciv_icu.icustays s JOIN mimiciv_hosp.patients p USING (subject_id)
        WHERE s.outtime >= s.intime + observation_hours() * INTERVAL '1 hour'
          AND p.anchor_age + year(s.intime) - p.anchor_year >= 18
        ORDER BY selection_key {limit_clause}
    """, [str(config.RANDOM_SEED), *([limit] if limit else [])])
    con.execute("DELETE FROM mimiciv_icu.icustays WHERE subject_id NOT IN (SELECT subject_id FROM candidate_subjects)")
    con.execute("DELETE FROM mimiciv_hosp.admissions WHERE hadm_id NOT IN (SELECT hadm_id FROM mimiciv_icu.icustays)")
    con.execute("DROP TABLE mimiciv_hosp.patients")
    count = con.execute("SELECT COUNT(*) FROM candidate_subjects").fetchone()[0]
    logger.info("Candidate pool: %d subjects; output target: %s", count, sample_size or "full cohort")

    chart_ids = ','.join(map(str, config.STEP1_CHART_ITEM_IDS))
    lab_ids = ','.join(map(str, config.STEP1_LAB_ITEM_IDS))
    vaso_ids = ','.join(map(str, config.STEP1_VASO_ITEM_IDS))
    load_raw_table(con, "icu/chartevents", f"TRY_CAST(stay_id AS BIGINT) IN (SELECT stay_id FROM mimiciv_icu.icustays) AND TRY_CAST(itemid AS BIGINT) IN ({chart_ids})")
    load_raw_table(con, "hosp/labevents", f"TRY_CAST(subject_id AS BIGINT) IN (SELECT subject_id FROM candidate_subjects) AND TRY_CAST(itemid AS BIGINT) IN ({lab_ids})")
    load_raw_table(con, "icu/inputevents", f"TRY_CAST(stay_id AS BIGINT) IN (SELECT stay_id FROM mimiciv_icu.icustays) AND TRY_CAST(itemid AS BIGINT) IN ({vaso_ids})")
    load_raw_table(con, "icu/outputevents", "TRY_CAST(stay_id AS BIGINT) IN (SELECT stay_id FROM mimiciv_icu.icustays)")
    for name in ("hosp/prescriptions", "hosp/microbiologyevents"):
        load_raw_table(con, name, "TRY_CAST(subject_id AS BIGINT) IN (SELECT subject_id FROM candidate_subjects)")

    # Reference doses assume mcg/kg/min except the documented norepinephrine correction.
    unsupported = con.execute("""
        SELECT COUNT(*) FROM mimiciv_icu.inputevents WHERE rate > 0 AND (
            rateuom IS NULL OR NOT (
                rateuom = 'mcg/kg/min' OR (itemid = 221906 AND rateuom = 'mg/kg/min')
            )
        )
    """).fetchone()[0]
    if unsupported:
        raise RuntimeError(f"Unsupported vasopressor rate units in {unsupported} selected rows")


def derive_reference(con):
    """Execute the pinned MIT-LCP DuckDB dependencies required for Sepsis-3."""
    for name in config.MIT_LCP_CONCEPTS:
        logger.info("Deriving %s", name.removesuffix(".sql"))
        con.execute((config.MIT_LCP_SQL_DIR / name).read_text())


def load_cxr_metadata(con):
    """Assign frontal studies to admission intervals without logging identifiers."""
    path = config.MIMIC_CXR_DIR / "mimic-cxr-2.0.0-metadata.csv"
    if not path.is_file():
        path = path.with_suffix(".csv.gz")
    if not path.is_file():
        raise RuntimeError("CXR metadata is missing; multimodal alignment cannot be verified")
    con.execute("""
        CREATE TEMP TABLE cohort_cxrs AS
        WITH parsed AS (
            SELECT TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(study_id AS BIGINT) AS study_id, dicom_id,
                ViewPosition AS cxr_view,
                try_strptime(StudyDate, '%Y%m%d') AS study_date,
                TRY_CAST(StudyTime AS DOUBLE) AS study_time
            FROM read_csv(?, header=true, all_varchar=true, delim=',', quote='"', escape='"')
            WHERE ViewPosition IN ('AP', 'PA')
              AND TRY_CAST(subject_id AS BIGINT) IN (SELECT subject_id FROM candidate_subjects)
        ), timed AS (
            SELECT *, study_date + (
                floor(study_time / 10000) * 3600
                + floor(study_time / 100) % 100 * 60
                + study_time % 100
            ) * INTERVAL '1 second' AS cxr_time
            FROM parsed
            WHERE study_time >= 0 AND floor(study_time / 10000) < 24
              AND floor(study_time / 100) % 100 < 60 AND study_time % 100 <= 60
        )
        SELECT t.subject_id, a.hadm_id, t.study_id, t.dicom_id, t.cxr_view, t.cxr_time
        FROM timed t JOIN mimiciv_hosp.admissions a
          ON t.subject_id = a.subject_id
         AND t.cxr_time >= a.admittime AND t.cxr_time < a.dischtime
        WHERE t.cxr_time IS NOT NULL AND t.study_id IS NOT NULL AND t.dicom_id IS NOT NULL
    """, [str(path)])


def select_cohort(con, sample_size):
    """Apply the study controls and select exactly N unique subjects for a smoke run."""
    con.execute((config.STEP1_SQL_DIR / "step1_cohort.sql").read_text())
    df = con.execute("""
        SELECT c.*, EXISTS (
            SELECT 1 FROM cohort_cxrs x WHERE x.hadm_id = c.hadm_id
              AND x.cxr_time >= c.t0 - observation_hours() * INTERVAL '1 hour'
              AND x.cxr_time < c.t0
        ) AS has_cxr
        FROM candidate_cohort c
        ORDER BY md5(c.subject_id::VARCHAR || ?), c.t0, c.stay_id
    """, [str(config.RANDOM_SEED)]).df()
    logger.info("Eligible candidates: %d stays, %d subjects", len(df), df.subject_id.nunique())
    if df.empty:
        raise RuntimeError("No eligible cohort; inspect aggregate exclusions or increase the candidate pool")
    if sample_size:
        # These coverage-oriented smoke cases are not a research performance sample.
        patients = df.sort_values(["label", "has_cxr"], ascending=[False, False], kind="stable").drop_duplicates("subject_id")
        if len(patients) < sample_size:
            raise RuntimeError(f"Only {len(patients)} eligible subjects; requested {sample_size}. Increase --candidate_size")
        selected = pd.concat([patients[patients.label == label].head(sample_size // 2) for label in (1, 0)])
        remaining = patients[~patients.subject_id.isin(selected.subject_id)]
        df = pd.concat([selected, remaining.head(sample_size - len(selected))]).copy()
    subjects = pd.Series(df.subject_id.unique()).sample(frac=1, random_state=config.RANDOM_SEED).tolist()
    n_train = int(len(subjects) * config.TRAIN_FRACTION)
    n_val = int(len(subjects) * config.VALIDATION_FRACTION)
    splits = {s: "train" if i < n_train else "val" if i < n_train + n_val else "test" for i, s in enumerate(subjects)}
    df["split"] = df.subject_id.map(splits)
    df["reference_revision"] = config.MIT_LCP_REVISION
    con.register("selected_cohort", df)
    return df


def aggregate_windows(con):
    """Aggregate the same four six-hour windows with explicit missing values."""
    con.execute("""
        CREATE TEMP TABLE windows AS
        SELECT c.*, idx AS window_idx,
            t0 - (observation_hours() - idx * ?) * INTERVAL '1 hour' AS window_start,
            t0 - (observation_hours() - (idx + 1) * ?) * INTERVAL '1 hour' AS window_end
        FROM selected_cohort c CROSS JOIN range(?) r(idx)
    """, [config.WINDOW_SIZE_HOURS, config.WINDOW_SIZE_HOURS, config.NUM_WINDOWS])
    expressions = []
    for col, prefix in (("heart_rate", "hr"), ("mbp", "map")):
        expressions.extend(f"{fn}({col}) AS {prefix}_{fn}" for fn in ("mean", "min", "max"))
        expressions.append(f"""(arg_max({col}, charttime) - arg_min({col}, charttime)) /
            NULLIF(epoch(MAX(charttime) FILTER (WHERE {col} IS NOT NULL)
                       - MIN(charttime) FILTER (WHERE {col} IS NOT NULL)) / 3600.0, 0)
            AS {prefix}_slope""")
    labs = [f"arg_max(valuenum, charttime) FILTER (WHERE itemid IN ({','.join(map(str,ids))}) AND valuenum > 0) AS {name}_latest" for name, ids in config.LAB_ITEM_IDS.items()]
    return con.execute(f"""
        WITH vitals AS (
            SELECT w.stay_id, w.window_idx, {', '.join(expressions)},
                AVG(spo2) AS spo2_mean, AVG(temperature) AS temp_mean, AVG(resp_rate) AS resp_rate_mean
            FROM windows w LEFT JOIN mimiciv_derived.vitalsign v
              ON w.stay_id = v.stay_id AND v.charttime >= w.window_start AND v.charttime < w.window_end
            GROUP BY w.stay_id, w.window_idx
        ), labs AS (
            SELECT w.stay_id, w.window_idx, {', '.join(labs)}
            FROM windows w LEFT JOIN mimiciv_hosp.labevents l
              ON w.subject_id = l.subject_id AND w.hadm_id = l.hadm_id
             AND l.charttime >= w.window_start AND l.charttime < w.window_end
            GROUP BY w.stay_id, w.window_idx
        ), images AS (
            SELECT w.stay_id, w.window_idx, x.study_id AS cxr_study_id, x.dicom_id AS cxr_dicom_id,
                x.cxr_view, x.cxr_time,
                CASE WHEN x.dicom_id IS NOT NULL THEN
                    'p' || left(w.subject_id::VARCHAR, 2) || '/p' || w.subject_id::VARCHAR
                    || '/s' || x.study_id::VARCHAR || '/' || x.dicom_id || '.jpg'
                END AS cxr_path
            FROM windows w LEFT JOIN cohort_cxrs x
              ON w.subject_id = x.subject_id AND w.hadm_id = x.hadm_id
             AND x.cxr_time >= w.window_start AND x.cxr_time < w.window_end
            QUALIFY ROW_NUMBER() OVER (PARTITION BY w.stay_id, w.window_idx ORDER BY x.cxr_time DESC, x.dicom_id) = 1
        )
        SELECT w.stay_id, w.subject_id, w.hadm_id, w.t0, w.cohort_type, w.label,
            w.window_idx, w.window_start, w.window_end, w.split,
            v.* EXCLUDE (stay_id, window_idx), l.* EXCLUDE (stay_id, window_idx),
            x.* EXCLUDE (stay_id, window_idx)
        FROM windows w JOIN vitals v USING (stay_id, window_idx)
        JOIN labs l USING (stay_id, window_idx) JOIN images x USING (stay_id, window_idx)
        ORDER BY w.stay_id, w.window_idx
    """).df()


def validate_windows(con, cohort, windows, sample_size):
    """Enforce artifact contracts and return aggregate diagnostics without patient rows."""
    if len(windows) != len(cohort) * config.NUM_WINDOWS or windows.duplicated(["stay_id", "window_idx"]).any():
        raise RuntimeError("Window count/key validation failed")
    if (windows.groupby("subject_id").split.nunique() > 1).any():
        raise RuntimeError("Subject split isolation failed")
    if sample_size and windows.subject_id.nunique() != sample_size:
        raise RuntimeError("Smoke output does not contain the requested unique subjects")
    if not ((windows.window_end - windows.window_start) == pd.Timedelta(hours=config.WINDOW_SIZE_HOURS)).all():
        raise RuntimeError("Window duration validation failed")
    expected_start = windows.t0 - pd.to_timedelta(
        config.OBSERVATION_HOURS - windows.window_idx * config.WINDOW_SIZE_HOURS, unit="h"
    )
    if not windows.window_idx.isin(range(config.NUM_WINDOWS)).all() or not windows.window_start.eq(expected_start).all():
        raise RuntimeError("Window boundaries do not cover the agreed observation period")
    if not (cohort.t0 >= cohort.intime + pd.Timedelta(hours=config.OBSERVATION_HOURS)).all() or not (cohort.t0 <= cohort.outtime).all():
        raise RuntimeError("Cohort observation eligibility failed")
    result = {
        "reference_revision": config.MIT_LCP_REVISION,
        "reference_sql_sha256": {p: hashlib.sha256((config.MIT_LCP_SQL_DIR / p).read_bytes()).hexdigest() for p in config.MIT_LCP_CONCEPTS},
        "observation_hours": config.OBSERVATION_HOURS,
        "unique_patients": int(windows.subject_id.nunique()), "stays": len(cohort), "windows": len(windows),
        "stays_per_label": cohort.label.value_counts().to_dict(),
        "stays_per_cohort_type": cohort.cohort_type.value_counts().to_dict(),
        "patients_per_split": cohort.groupby("split").subject_id.nunique().to_dict(),
        "windows_with_cxr": int(windows.cxr_path.notna().sum()),
        "observed_temperatures": int(windows.temp_mean.notna().sum()),
        "temperatures_outside_25_to_45_celsius": int(((windows.temp_mean < 25) | (windows.temp_mean > 45)).sum()),
        "missing_fraction": windows[[c for c in windows if c.endswith(("_mean", "_latest", "_slope"))]].isna().mean().round(4).to_dict(),
        "candidate_subjects": con.execute("SELECT COUNT(*) FROM candidate_subjects").fetchone()[0],
        "eligible_candidate_stays": con.execute("SELECT COUNT(*) FROM candidate_cohort").fetchone()[0],
        "validation": "passed",
        "scope": "24-hour observations for sepsis probability; no future prediction interval",
    }
    result["sofa_observed_hours_by_component"] = {
        name: con.execute(f"SELECT COUNT(*) FROM mimiciv_derived.sofa s JOIN selected_cohort c USING (stay_id) WHERE s.{name} IS NOT NULL").fetchone()[0]
        for name in ("respiration", "coagulation", "liver", "cardiovascular", "cns", "renal")
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample_size", type=int, default=config.STEP1_SAMPLE_SIZE, help="Eligible unique subjects; defaults to ten")
    mode.add_argument("--full", action="store_true", help="Explicit full run; only after ten-patient validation")
    parser.add_argument("--candidate_size", type=int, help="Bounded candidate subjects for smoke eligibility")
    parser.add_argument("--output", type=Path, default=config.PATIENT_WINDOWS_PARQUET)
    args = parser.parse_args()
    sample_size = None if args.full else args.sample_size
    if sample_size is not None and sample_size < 1:
        parser.error("--sample_size must be positive")
    if args.candidate_size is not None and (args.full or args.candidate_size < (sample_size or 1)):
        parser.error("--candidate_size must be at least --sample_size and only used for smoke runs")
    if config.NUM_WINDOWS * config.WINDOW_SIZE_HOURS != config.OBSERVATION_HOURS:
        parser.error("Window settings must partition the observation period exactly")
    settings = {"threads": config.DUCKDB_THREADS, "memory_limit": config.DUCKDB_MEMORY_LIMIT, "temp_directory": ""}
    with duckdb.connect(config=settings) as con:
        try:
            load_candidates(con, sample_size, args.candidate_size)
            derive_reference(con)
            load_cxr_metadata(con)
            cohort = select_cohort(con, sample_size)
            windows = aggregate_windows(con)
            summary = validate_windows(con, cohort, windows, sample_size)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            windows.to_parquet(args.output, index=False)
            cohort.drop(columns="has_cxr").to_parquet(config.COHORT_PARQUET, index=False)
            config.STEP1_VALIDATION_JSON.write_text(json.dumps(summary, indent=2) + "\n")
            logger.info("Validation passed: %d patients, %d stays, %d windows, %d image windows", summary["unique_patients"], summary["stays"], summary["windows"], summary["windows_with_cxr"])
            logger.info("Cohort counts: %s; split counts: %s", summary["stays_per_label"], summary["patients_per_split"])
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        except Exception as exc:
            # CSV exceptions can contain source rows; never emit their details.
            logger.error("Step 1 failed (%s); inspect schemas or reproduce with synthetic inputs", type(exc).__name__)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
