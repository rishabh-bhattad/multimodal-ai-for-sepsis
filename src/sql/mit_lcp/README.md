# MIT-LCP DuckDB concepts used by Step 1

These 22 SQL files are copied from MIT-LCP/mimic-code, revision
`303d26c623dcc9c49cc0f204468d4acc2f063797`, under
[`mimic-iv/concepts_duckdb`](https://github.com/MIT-LCP/mimic-code/tree/303d26c623dcc9c49cc0f204468d4acc2f063797/mimic-iv/concepts_duckdb).
Their MIT license is included in LICENSE. They are the upstream DuckDB exports;
there is no runtime transpiler, network download, or cloud database dependency.

The filenames are flattened here. Execution order lives in `config.MIT_LCP_CONCEPTS`.
The Python runner supplies typed, cohort-filtered raw tables in `mimiciv_hosp`
and `mimiciv_icu`, then executes these queries in `mimiciv_derived`.
The reference's rolling score and internal timing intervals label the cohort;
the model still receives four six-hour observation windows, with no future
forecast interval.

Upstream source paths:

- `demographics/icustay_times.sql`
- `demographics/icustay_hourly.sql`
- `measurement/vitalsign.sql`
- `measurement/gcs.sql`
- `measurement/bg.sql`
- `measurement/enzyme.sql`
- `measurement/chemistry.sql`
- `measurement/complete_blood_count.sql`
- `measurement/ventilator_setting.sql`
- `measurement/oxygen_delivery.sql`
- `treatment/ventilation.sql`
- `demographics/weight_durations.sql`
- `measurement/urine_output.sql`
- `measurement/urine_output_rate.sql`
- `medication/dobutamine.sql`
- `medication/dopamine.sql`
- `medication/epinephrine.sql`
- `medication/norepinephrine.sql`
- `medication/antibiotic.sql`
- `sepsis/suspicion_of_infection.sql`
- `score/sofa.sql`
- `sepsis/sepsis3.sql`

Local cohort eligibility, negative-control selection, CXR alignment, and window
aggregation live outside the reference SQL. They are capstone study choices,
not claims that MIT-LCP defines this exact cohort. In particular, requiring
24 hours before the anchor excludes early-onset ICU cases. Reference SOFA
assumes a zero baseline and imputes missing components to zero after rolling
aggregation; model-input labs and vitals remain missing when unobserved.
