# Compact KG prompt check

KG serialization defaults to `compact` in `config.STEP4_KG_FORMAT`.
This changes the representation, not Step 3 retrieval. The 200-edge/window cap,
two-hop path limit, source graphs, cohort, and patient splits remain unchanged.

Temporal prompts use a dictionary local to each stay for entity display names
and exact relation names, with separate directed adjacency lists per window.
Each adjacency entry is `[source_index, relation_index, [target_indices]]`.
Negative relation names remain explicit. All four clinical windows, image
markers, and final task instructions are retained.

Static prompts store their bounded merged graph once and state that it is
shared across all four windows. This preserves the exact evidence previously
repeated four times. It does not expand the old static cap to the complete
uncapped union. The no-KG prompt is unchanged.

Dictionaries depend only on retrieved public KG triples for the current stay;
they do not use outcome labels or another patient's observations. At the single
assessment time all four observation windows are available. This encoding does
not resolve retrospective-anchor or input-availability limitations.

## Full-cohort measurement

Qwen3-VL-8B-Instruct processor; 262144-pixel image ceiling; complete prompts,
chat template, image tokens, and the longer binary completion allowance.
All 9119 stays were checked in each variant.

| Variant | Verbose prompts over 8192 | Compact prompts over 8192 | Maximum compact length |
| --- | ---: | ---: | ---: |
| Temporal RAG | 2540 | 200 | 11017 |
| Static RAG | 7104 | 0 | 5079 |
| No KG | 0 | 0 | 1454 |

Temporal total tokens decreased from 56,482,108 to 26,141,087.
Static total tokens decreased from 119,891,780 to 18,540,309.
No-KG total tokens remained 3,995,912.

Every compact prompt was decoded and checked against its saved verbose
counterpart. All clinical window text, image paths, cohort metadata, labels,
and targets matched. Reconstructed displayed triples matched in every window,
including direction and negative relations. This verified 2,093,516 temporal
and 4,639,808 static window-edge occurrences; static occurrences count the
shared graph's logical application to four windows, not repeated storage.
The longest examples and longest image-bearing examples in every split were
also checked with actual processor outputs, matching the calculated lengths.

Fidelity here means equality with the prior bounded, name-based displayed
triples. It does not mean reconstruction of every native graph attribute or
native-node identity that the old text serializer did not expose.
Ordering follows the prior case-insensitive whole-triple sort, with a
deterministic exact-text tie-break.

Local artifacts:

- `data/interim/compact_kg_preflight/context_report.json`
- `data/interim/compact_kg_preflight/datasets/{trag,static_rag,no_kg}/`
- `data/interim/compact_kg_fidelity_final.log`

The token counts are specific to this model and image ceiling. The default
processor resolution is not automatically changed by the compact format.

## Using the format

Fresh Step 4 outputs and experiment-suite datasets use the compact format.
Existing datasets and adapters are not rewritten. Rebuild Step 4, run a fresh
suite, or explicitly select the matching prepared dataset for a single run:

```bash
./run_training.sh --dataset data/interim/compact_kg_preflight/datasets/trag \
  --smoke --max-seq-length 8192 --max-image-pixels 262144 \
  --output checkpoints/compact_8k_smoke --wandb-mode online
```

This example is not a worst-case full-cohort test. Training was not launched
during the compaction check. The new representation still needs a real
training smoke test before a full run.

W&B run settings record the actual observed format as `compact_v1`,
`verbose`, or `no_edges`; suite manifests record the selected format.
Compact prompts that exceed a token budget raise an error rather than entering
the legacy edge-pruning fitter, even if legacy KG reduction is requested.
Thus an 8K temporal full-cohort run would still stop on 200 examples.

A 12288-token budget covers these measured prompts including completion
allowance, but GPU memory fit has not been tested at that length. Changing to
12K requires a separate memory test; alternatively further lossless encoding
could be measured. Do not drop patients or edges silently.
