-- Local capstone eligibility and controls; clinical scoring stays in MIT-LCP SQL.
CREATE OR REPLACE TEMP TABLE candidate_cohort AS
WITH severity AS (
    SELECT stay_id, MAX(sofa_24hours) AS peak_sofa
    FROM mimiciv_derived.sofa GROUP BY stay_id
), infection AS (
    SELECT stay_id, MIN(suspected_infection_time) AS t_susp
    FROM mimiciv_derived.suspicion_of_infection
    WHERE suspected_infection = 1 GROUP BY stay_id
), cultures AS (
    SELECT hadm_id,
        COUNT(DISTINCT micro_specimen_id) FILTER (
            WHERE org_itemid IS DISTINCT FROM 90760
              AND COALESCE(org_name, '') != 'CANCELLED'
        ) AS completed_cultures,
        MAX(CASE WHEN org_name IS NOT NULL AND org_name != ''
                      AND org_itemid NOT IN (90856, 90760)
                      AND org_name != 'CANCELLED' THEN 1 ELSE 0 END) AS positive_culture
    FROM mimiciv_hosp.microbiologyevents GROUP BY hadm_id
), eligible_peak AS (
    SELECT s.stay_id, s.endtime AS t0, s.sofa_24hours AS sofa_score
    FROM mimiciv_derived.sofa s JOIN mimiciv_icu.icustays i USING (stay_id)
    WHERE s.endtime >= i.intime + observation_hours() * INTERVAL '1 hour'
      AND s.endtime <= i.outtime
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY s.stay_id ORDER BY s.sofa_24hours DESC, s.endtime
    ) = 1
), anchored AS (
    SELECT i.*, p.sofa_time AS t0, p.suspected_infection_time AS t_susp,
        p.sofa_score, 1 AS label, 'sepsis_3' AS cohort_type
    FROM mimiciv_icu.icustays i JOIN mimiciv_derived.sepsis3 p USING (stay_id)
    WHERE p.sepsis3

    UNION ALL

    SELECT i.*, inf.t_susp AS t0, inf.t_susp, s.peak_sofa AS sofa_score,
        0 AS label, 'control_infection_no_dysfunction' AS cohort_type
    FROM mimiciv_icu.icustays i JOIN infection inf USING (stay_id)
    JOIN severity s USING (stay_id)
    WHERE s.peak_sofa < 2
      AND NOT EXISTS (SELECT 1 FROM mimiciv_derived.sepsis3 p WHERE p.stay_id = i.stay_id AND p.sepsis3)

    UNION ALL

    SELECT i.*, pk.t0, NULL::TIMESTAMP AS t_susp, pk.sofa_score,
        0 AS label, 'control_non_infectious_dysfunction' AS cohort_type
    FROM mimiciv_icu.icustays i JOIN eligible_peak pk USING (stay_id)
    JOIN cultures c USING (hadm_id)
    WHERE pk.sofa_score >= 2 AND c.completed_cultures > 0 AND c.positive_culture = 0
      AND NOT EXISTS (
          SELECT 1 FROM mimiciv_derived.suspicion_of_infection inf
          WHERE inf.hadm_id = i.hadm_id AND inf.suspected_infection = 1
      )
      AND NOT EXISTS (SELECT 1 FROM mimiciv_derived.sepsis3 p WHERE p.stay_id = i.stay_id AND p.sepsis3)
)
SELECT a.* FROM anchored a JOIN mimiciv_hosp.admissions adm USING (hadm_id)
WHERE a.t0 >= a.intime + observation_hours() * INTERVAL '1 hour'
  AND a.t0 <= a.outtime
  AND a.t0 - observation_hours() * INTERVAL '1 hour' >= adm.admittime
  AND a.t0 <= adm.dischtime;
