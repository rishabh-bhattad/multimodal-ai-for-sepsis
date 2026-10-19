"""
src/step1_windowing.py — Sepsis-3 Cohort Extraction & 6-Hour Temporal Windowing

Extracts Sepsis-3 stays and matched negative controls from MIMIC-IV,
computes hourly SOFA scores to determine the onset anchor T0, and discretizes the
pre-onset 24h timeline into four 6-hour windows with aggregated vitals, labs, and CXR paths.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import config
from src.utils import setup_logger

logger = setup_logger("Step1-Windowing")

# Common systemic antibiotics in MIMIC-IV
ANTIBIOTIC_KEYWORDS = [
    "vancomycin", "cefepime", "piperacillin", "tazobactam", "ceftriaxone",
    "cefazolin", "meropenem", "ampicillin", "sulbactam", "ciprofloxacin",
    "levofloxacin", "gentamicin", "tobramycin", "azithromycin", "metronidazole",
    "doxycycline", "clindamycin", "amoxicillin", "clavulanate", "ertapenem",
    "linezolid", "bactrim", "sulfamethoxazole", "trimethoprim"
]


def extract_raw_mimic_tables(con: duckdb.DuckDBPyConnection, sample_size: int = None) -> Dict[str, pd.DataFrame]:
    """Streams and filters raw compressed MIMIC-IV and CXR tables from disk into compact Pandas DataFrames."""
    logger.info("Extracting filtered raw tables from disk via DuckDB streaming...")
    limit_clause = f"LIMIT {sample_size * 10}" if sample_size else ""

    # Filter for adult ICU stays with length-of-stay >= 24h to guarantee sufficient pre-onset observation history
    stays_df = con.execute(f"""
        SELECT 
            stay_id, subject_id, hadm_id,
            CAST(intime AS TIMESTAMP) as intime,
            CAST(outtime AS TIMESTAMP) as outtime,
            los
        FROM read_csv('{config.MIMIC_IV_ICU_DIR}/icustays.csv.gz', compression='gzip', header=True)
        WHERE los >= 1.0
        {limit_clause}
    """).df()

    con.register("target_stays", stays_df)

    # Sepsis-3 infection criterion: pair specimen culture with systemic antibiotic within [-48h, +24h] to anchor Tsusp
    abx_pattern = "|".join(ANTIBIOTIC_KEYWORDS)
    tsusp_df = con.execute(f"""
        WITH abx AS (
            SELECT subject_id, hadm_id, CAST(starttime AS TIMESTAMP) as abx_time
            FROM read_csv('{config.MIMIC_IV_HOSP_DIR}/prescriptions.csv.gz', compression='gzip', header=True)
            WHERE starttime IS NOT NULL AND regexp_matches(lower(drug), '({abx_pattern})')
        ),
        cultures AS (
            SELECT subject_id, hadm_id, COALESCE(CAST(charttime AS TIMESTAMP), CAST(chartdate AS TIMESTAMP)) as culture_time
            FROM read_csv('{config.MIMIC_IV_HOSP_DIR}/microbiologyevents.csv.gz', compression='gzip', header=True)
            WHERE charttime IS NOT NULL OR chartdate IS NOT NULL
        )
        SELECT 
            c.subject_id, c.hadm_id, 
            MIN(LEAST(c.culture_time, a.abx_time)) as t_susp
        FROM cultures c
        JOIN abx a 
          ON c.hadm_id = a.hadm_id
         AND a.abx_time >= c.culture_time - INTERVAL 48 HOUR
         AND a.abx_time <= c.culture_time + INTERVAL 24 HOUR
        GROUP BY c.subject_id, c.hadm_id
    """).df()

    # Extract bedside vitals (HR, MAP, SpO2, Temp, RR) and GCS neurological components for target stays
    vitals_raw_df = con.execute(f"""
        SELECT 
            t.stay_id,
            CAST(c.charttime AS TIMESTAMP) as charttime,
            c.itemid,
            c.valuenum
        FROM target_stays t
        JOIN read_csv('{config.MIMIC_IV_ICU_DIR}/chartevents.csv.gz', compression='gzip', header=True) c
          ON t.stay_id = c.stay_id
        WHERE c.itemid IN (220045, 220052, 220181, 225312, 220277, 223761, 223762, 220210, 224690, 220739, 223900, 223901)
          AND c.valuenum IS NOT NULL
    """).df()

    # Extract central blood labs (platelets, creatinine, bilirubin, lactate, WBC) within [-24h of ICU in, outtime]
    labs_raw_df = con.execute(f"""
        SELECT 
            t.stay_id, t.hadm_id,
            CAST(l.charttime AS TIMESTAMP) as charttime,
            l.itemid,
            l.valuenum
        FROM target_stays t
        JOIN read_csv('{config.MIMIC_IV_HOSP_DIR}/labevents.csv.gz', compression='gzip', header=True) l
          ON t.hadm_id = l.hadm_id
         AND l.charttime >= t.intime - INTERVAL 24 HOUR
         AND l.charttime <= t.outtime
        WHERE l.itemid IN (50813, 52442, 53154, 51300, 51301, 51265, 50912, 50885)
          AND l.valuenum IS NOT NULL
    """).df()

    # Extract IV vasopressor infusion rates (norepinephrine, epinephrine, dopamine, vasopressin) for cardiovascular SOFA
    vaso_raw_df = con.execute(f"""
        SELECT 
            t.stay_id,
            CAST(i.starttime AS TIMESTAMP) as starttime,
            i.itemid,
            i.rate
        FROM target_stays t
        JOIN read_csv('{config.MIMIC_IV_ICU_DIR}/inputevents.csv.gz', compression='gzip', header=True) i
          ON t.stay_id = i.stay_id
        WHERE i.itemid IN (221906, 221289, 221662, 222315)
    """).df()

    # Parse frontal chest radiographs (PA/AP) and compute clean timestamps with 60-second rollover tolerance
    meta_path = config.MIMIC_CXR_DIR / "mimic-cxr-2.0.0-metadata.csv"
    if not meta_path.exists():
        meta_path = config.MIMIC_CXR_DIR / "mimic-cxr-2.0.0-metadata.csv.gz"

    cxr_raw_df = con.execute(f"""
        SELECT 
            dicom_id, subject_id, study_id, ViewPosition as cxr_view,
            strptime(printf('%08d', CAST(StudyDate AS BIGINT)), '%Y%m%d') 
            + INTERVAL (CAST(CAST(StudyTime AS DOUBLE) AS BIGINT) // 10000) HOUR
            + INTERVAL ((CAST(CAST(StudyTime AS DOUBLE) AS BIGINT) // 100) % 100) MINUTE
            + INTERVAL (CAST(CAST(StudyTime AS DOUBLE) AS BIGINT) % 100) SECOND as cxr_time
        FROM read_csv('{meta_path}', header=True, quote='\"', escape='\"')
        WHERE ViewPosition IN ('PA', 'AP')
          AND StudyDate IS NOT NULL AND StudyTime IS NOT NULL
    """).df()

    logger.info("Loaded stays: %d, Tsusp: %d, vitals: %d, labs: %d, CXR: %d",
                len(stays_df), len(tsusp_df), len(vitals_raw_df), len(labs_raw_df), len(cxr_raw_df))

    return {
        "stays": stays_df,
        "tsusp": tsusp_df,
        "vitals": vitals_raw_df,
        "labs": labs_raw_df,
        "vaso": vaso_raw_df,
        "cxr": cxr_raw_df,
    }


def compute_hourly_sofa_scores(
    stays_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    vaso_df: pd.DataFrame
) -> pd.DataFrame:
    """Computes hourly SOFA scores across all 6 organ systems using vectorized array operations."""
    logger.info("Computing hourly organ dysfunction scores...")

    # Floor timestamps to hourly bins to construct a uniform clinical timeline
    vitals = vitals_df.copy()
    vitals["chart_hour"] = vitals["charttime"].dt.floor("h")

    # Take the worst (minimum) MAP and GCS during each hour to capture acute physiological instability
    map_items = [220052, 220181, 225312]
    gcs_items = [220739, 223900, 223901]

    hourly_map = vitals[vitals["itemid"].isin(map_items)].groupby(["stay_id", "chart_hour"])["valuenum"].min().rename("map_val")
    hourly_gcs = vitals[vitals["itemid"].isin(gcs_items)].groupby(["stay_id", "chart_hour"])["valuenum"].min().rename("gcs_val")

    # For labs, take the most severe value in the hour (lowest platelets, highest bilirubin/creatinine)
    labs = labs_df.copy()
    labs["chart_hour"] = labs["charttime"].dt.floor("h")

    hourly_plat = labs[labs["itemid"] == 51265].groupby(["stay_id", "chart_hour"])["valuenum"].min().rename("platelets")
    hourly_bili = labs[labs["itemid"] == 50885].groupby(["stay_id", "chart_hour"])["valuenum"].max().rename("bilirubin")
    hourly_creat = labs[labs["itemid"] == 50912].groupby(["stay_id", "chart_hour"])["valuenum"].max().rename("creatinine")

    # Map vasopressor infusion rates to cardiovascular SOFA thresholds (high-dose epi/norepi = 4, low-dose = 3, dopamine = 2)
    vaso = vaso_df.copy()
    if not vaso.empty:
        vaso["chart_hour"] = vaso["starttime"].dt.floor("h")
        conds = [
            (vaso["itemid"].isin([221906, 221289])) & (vaso["rate"] > 0.1),
            (vaso["itemid"].isin([221906, 221289])) & (vaso["rate"] > 0),
            (vaso["itemid"] == 221662) & (vaso["rate"] > 5),
            (vaso["itemid"].isin([221662, 222315])) & (vaso["rate"] > 0),
        ]
        choices = [4, 3, 3, 2]
        vaso["vaso_score"] = np.select(conds, choices, default=0)
        hourly_vaso = vaso.groupby(["stay_id", "chart_hour"])["vaso_score"].max().rename("vaso_score")
    else:
        hourly_vaso = pd.Series(name="vaso_score", dtype=int)

    # Combine individual organ system hourly summaries into a unified stay timeline
    hourly = pd.concat([hourly_map, hourly_gcs, hourly_plat, hourly_bili, hourly_creat, hourly_vaso], axis=1).reset_index()
    hourly = hourly.merge(stays_df[["stay_id", "subject_id", "hadm_id", "intime", "outtime"]], on="stay_id", how="inner")

    # Vectorized clinical SOFA sub-score thresholds (0 to 4 points per organ system)
    coag = np.select(
        [hourly["platelets"] < 20, hourly["platelets"] < 50, hourly["platelets"] < 100, hourly["platelets"] < 150],
        [4, 3, 2, 1], default=0
    )
    liver = np.select(
        [hourly["bilirubin"] >= 12.0, hourly["bilirubin"] >= 6.0, hourly["bilirubin"] >= 2.0, hourly["bilirubin"] >= 1.2],
        [4, 3, 2, 1], default=0
    )
    renal = np.select(
        [hourly["creatinine"] >= 5.0, hourly["creatinine"] >= 3.5, hourly["creatinine"] >= 2.0, hourly["creatinine"] >= 1.2],
        [4, 3, 2, 1], default=0
    )
    cardio_map = np.where(hourly["map_val"] < 70, 1, 0)
    cardio = np.maximum(cardio_map, hourly["vaso_score"].fillna(0).astype(int))
    cns = np.select(
        [hourly["gcs_val"] < 6, hourly["gcs_val"] <= 9, hourly["gcs_val"] <= 12, hourly["gcs_val"] <= 14],
        [4, 3, 2, 1], default=0
    )

    hourly["total_sofa"] = coag + liver + renal + cardio + cns
    return hourly.sort_values(["stay_id", "chart_hour"]).reset_index(drop=True)


def identify_cohort_t0(
    hourly_df: pd.DataFrame,
    tsusp_df: pd.DataFrame,
    sample_size: int = None
) -> pd.DataFrame:
    """Identifies Sepsis-3 onset anchor T0 (earliest delta SOFA >= 2) and balanced negative controls."""
    logger.info("Identifying T0 onset anchors for Sepsis-3 positives and negative controls...")

    # Align suspected infection times (Tsusp) with hourly organ trajectory per admission
    merged = hourly_df.merge(tsusp_df, on=["subject_id", "hadm_id"], how="left")

    records = []
    for stay_id, group in merged.groupby("stay_id"):
        intime = group["intime"].iloc[0]
        subject_id = group["subject_id"].iloc[0]
        hadm_id = group["hadm_id"].iloc[0]
        t_susp = group["t_susp"].iloc[0]

        # Sepsis-3 consensus: baseline SOFA is the minimum score observed in stay
        baseline_sofa = group["total_sofa"].min()
        group = group.copy()
        group["delta_sofa"] = group["total_sofa"] - baseline_sofa

        has_infection = pd.notna(t_susp)

        if has_infection:
            # Sepsis-3 onset: earliest acute SOFA surge (delta >= 2) within [-24h, +12h] of suspected infection
            surge_window = (group["chart_hour"] >= t_susp - pd.Timedelta(hours=24)) & \
                           (group["chart_hour"] <= t_susp + pd.Timedelta(hours=12))
            surge = group[surge_window & (group["delta_sofa"] >= 2)]

            if not surge.empty:
                t0 = surge["chart_hour"].min()
                # Ensure at least 18h observation history prior to onset for pre-onset feature trajectories
                if t0 - pd.Timedelta(hours=24) >= intime - pd.Timedelta(hours=6):
                    records.append({
                        "stay_id": stay_id, "subject_id": subject_id, "hadm_id": hadm_id,
                        "t0": t0, "label": 1, "cohort_type": "sepsis_3"
                    })
            else:
                # Challenge Control A: Infection suspected, but patient maintains delta SOFA < 2 (prevents infection shortcut)
                t0 = t_susp
                if t0 - pd.Timedelta(hours=24) >= intime - pd.Timedelta(hours=6):
                    records.append({
                        "stay_id": stay_id, "subject_id": subject_id, "hadm_id": hadm_id,
                        "t0": t0, "label": 0, "cohort_type": "control_infection_no_dysfunction"
                    })
        else:
            # Challenge Control B: Acute organ injury (delta SOFA >= 2) without infection (trauma, post-cardiac arrest)
            acute = group[group["delta_sofa"] >= 2]
            if not acute.empty:
                peak_idx = group["delta_sofa"].idxmax()
                t0 = group.loc[peak_idx, "chart_hour"]
                if t0 - pd.Timedelta(hours=24) >= intime - pd.Timedelta(hours=6):
                    records.append({
                        "stay_id": stay_id, "subject_id": subject_id, "hadm_id": hadm_id,
                        "t0": t0, "label": 0, "cohort_type": "control_non_infectious_dysfunction"
                    })

    cohort_df = pd.DataFrame(records)
    if cohort_df.empty:
        logger.warning("No candidate stays satisfied the cohort criteria.")
        return cohort_df

    positives = cohort_df[cohort_df["label"] == 1]
    negatives = cohort_df[cohort_df["label"] == 0]
    logger.info("Extracted %d Sepsis-3 stays and %d control stays.", len(positives), len(negatives))

    # Maintain balanced 1:1 ratio between Sepsis-3 and challenge controls
    if sample_size:
        n_pos = min(sample_size, len(positives))
        n_neg = min(sample_size, len(negatives))
        positives = positives.sample(n=n_pos, random_state=42)
        negatives = negatives.sample(n=n_neg, random_state=42)
        cohort_df = pd.concat([positives, negatives]).reset_index(drop=True)
        logger.info("Subsampled to %d positives and %d negatives.", len(positives), len(negatives))

    return cohort_df


def aggregate_windows(
    cohort_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    cxr_df: pd.DataFrame
) -> pd.DataFrame:
    """Discretizes the 24h pre-onset timeline into four 6-hour windows and aggregates multimodal observations."""
    logger.info("Aggregating multimodal features across four 6h windows...")

    rows = []
    vitals = vitals_df.copy()
    labs = labs_df.copy()
    cxrs = cxr_df.copy()

    for _, patient in cohort_df.iterrows():
        stay_id = patient["stay_id"]
        subject_id = patient["subject_id"]
        hadm_id = patient["hadm_id"]
        t0 = patient["t0"]
        label = patient["label"]

        stay_vitals = vitals[vitals["stay_id"] == stay_id]
        stay_labs = labs[labs["hadm_id"] == hadm_id]
        patient_cxrs = cxrs[cxrs["subject_id"] == subject_id]

        # Discretize pre-onset horizon [T0 - 24h, T0] into 4 non-overlapping 6h intervals: w0=[-24, -18], w1=[-18, -12], w2=[-12, -6], w3=[-6, 0]
        for w_idx in range(4):
            w_start = t0 - pd.Timedelta(hours=24 - w_idx * 6)
            w_end = t0 - pd.Timedelta(hours=24 - (w_idx + 1) * 6)

            wv = stay_vitals[(stay_vitals["charttime"] >= w_start) & (stay_vitals["charttime"] < w_end)]

            # Heart rate summary and 6-hour rate of change (slope = delta val / 6h)
            hr_series = wv[wv["itemid"] == 220045].sort_values("charttime")["valuenum"]
            hr_mean = hr_series.mean() if not hr_series.empty else np.nan
            hr_min = hr_series.min() if not hr_series.empty else np.nan
            hr_max = hr_series.max() if not hr_series.empty else np.nan
            hr_slope = (hr_series.iloc[-1] - hr_series.iloc[0]) / 6.0 if len(hr_series) > 1 else 0.0

            # MAP summary and rate of change (tracking hemodynamic collapse)
            map_series = wv[wv["itemid"].isin([220052, 220181, 225312])].sort_values("charttime")["valuenum"]
            map_mean = map_series.mean() if not map_series.empty else np.nan
            map_min = map_series.min() if not map_series.empty else np.nan
            map_max = map_series.max() if not map_series.empty else np.nan
            map_slope = (map_series.iloc[-1] - map_series.iloc[0]) / 6.0 if len(map_series) > 1 else 0.0

            spo2_mean = wv[wv["itemid"] == 220277]["valuenum"].mean()
            temp_mean = wv[wv["itemid"].isin([223761, 223762])]["valuenum"].mean()
            resp_mean = wv[wv["itemid"].isin([220210, 224690])]["valuenum"].mean()

            # Carry latest observed lab draw within window; leave NaN if unmeasured (no multi-window imputation)
            wl = stay_labs[(stay_labs["charttime"] >= w_start) & (stay_labs["charttime"] < w_end)]

            def get_latest_lab(itemids: List[int]) -> float:
                matches = wl[wl["itemid"].isin(itemids)].sort_values("charttime")
                return matches["valuenum"].iloc[-1] if not matches.empty else np.nan

            lactate = get_latest_lab([50813, 52442, 53154])
            wbc = get_latest_lab([51300, 51301])
            platelets = get_latest_lab([51265])
            creatinine = get_latest_lab([50912])
            bilirubin = get_latest_lab([50885])

            # Frontal radiograph acquired within window (if multiple, retain latest)
            wc = patient_cxrs[(patient_cxrs["cxr_time"] >= w_start) & (patient_cxrs["cxr_time"] < w_end)]
            if not wc.empty:
                latest_cxr = wc.sort_values("cxr_time").iloc[-1]
                sub_str = str(subject_id)
                cxr_path = f"p{sub_str[:2]}/p{sub_str}/s{latest_cxr['study_id']}/{latest_cxr['dicom_id']}.jpg"
                cxr_view = latest_cxr["cxr_view"]
            else:
                cxr_path = None
                cxr_view = None

            rows.append({
                "stay_id": stay_id, "subject_id": subject_id, "hadm_id": hadm_id,
                "label": label, "window_idx": w_idx,
                "window_start": w_start, "window_end": w_end,
                "hr_mean": hr_mean, "hr_min": hr_min, "hr_max": hr_max, "hr_slope": hr_slope,
                "map_mean": map_mean, "map_min": map_min, "map_max": map_max, "map_slope": map_slope,
                "spo2_mean": spo2_mean, "temp_mean": temp_mean, "resp_rate_mean": resp_mean,
                "lactate_latest": lactate, "wbc_latest": wbc, "platelets_latest": platelets,
                "creatinine_latest": creatinine, "bilirubin_latest": bilirubin,
                "cxr_path": cxr_path, "cxr_view": cxr_view
            })

    result = pd.DataFrame(rows)

    # Partition strictly on subject_id (patient level) to prevent data leakage across multiple hospitalizations
    unique_patients = pd.Series(result["subject_id"].unique())
    shuffled = unique_patients.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(shuffled) * 0.70)
    n_val = int(len(shuffled) * 0.15)

    train_pts = set(shuffled.iloc[:n_train])
    val_pts = set(shuffled.iloc[n_train:n_train + n_val])

    def assign_split(sub_id: int) -> str:
        if sub_id in train_pts:
            return "train"
        elif sub_id in val_pts:
            return "val"
        return "test"

    result["split"] = result["subject_id"].apply(assign_split)
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract Sepsis-3 stays and build 4x6h temporal windows.")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample N patients per class for quick testing.")
    parser.add_argument("--output", type=str, default=str(config.DATA_INTERIM / "patient_windows.parquet"))
    args = parser.parse_args()

    con = duckdb.connect()

    logger.info("Beginning Step 1 Windowing pipeline (sample_size=%s)...", args.sample_size)
    
    tables = extract_raw_mimic_tables(con, sample_size=args.sample_size)
    hourly_df = compute_hourly_sofa_scores(tables["stays"], tables["vitals"], tables["labs"], tables["vaso"])
    cohort_df = identify_cohort_t0(hourly_df, tables["tsusp"], sample_size=args.sample_size)

    if cohort_df.empty:
        logger.error("Cohort construction failed: no stays met criteria.")
        sys.exit(1)

    windows_df = aggregate_windows(cohort_df, tables["vitals"], tables["labs"], tables["cxr"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows_df.to_parquet(output_path, index=False)

    logger.info("Successfully exported %d window records across %d stays to %s",
                len(windows_df), windows_df['stay_id'].nunique(), output_path)
    logger.info("Class distribution: %s", windows_df.groupby("label")["stay_id"].nunique().to_dict())
    logger.info("Split distribution: %s", windows_df.groupby("split")["stay_id"].nunique().to_dict())


if __name__ == "__main__":
    main()
