# Experiment Runner

## Tau3 BS/ECR/SUR Extension

The execution-annotated metric extension and its controlled version-shift
boundary are specified in
[`TAU3_METRIC_EXTENSION_PROTOCOL.md`](TAU3_METRIC_EXTENSION_PROTOCOL.md).
Build it with `scripts/build_tau3_metric_extension.py`; the builder writes a
new dataset and refuses to overwrite existing data. Never relabel controlled
SUR as natural production drift.

## Real Longitudinal Track

The strict executable path for natural chronology, version transitions,
independent communication checks, VSR, BS, ECR, and SUR is documented in
[`real_tracks/README.md`](real_tracks/README.md). Start with
`scripts/audit_real_track_dataset.py`; the analyzer emits no partial metrics
when provenance, reset, completeness, or checker-independence gates fail.

Run the complete locally supported evaluation from the repository root:

```sh
python3 experiments/run_experiments.py
```

The runner uses only the Python standard library plus the repository code. It
trains the unary outcome model on observed RapidAPI endpoint tests, selects the
classification threshold and beam width on validation data, and evaluates once
on the held-out split. It also runs exact-EMP reconstruction proxies, matched
ablations, retrieval-budget and efficiency grids, a bounded exhaustive search
reference, and a controlled retention simulation. Five fixed seeds change only
the proxy distractor banks because the implementation itself is deterministic.

Outputs are written to `experiments/results/`: publication tables in CSV,
Markdown, and LaTeX; plots in TeX, PostScript, PDF, and 300-dpi PNG; raw
task-level measurements; model weights; checksums; and a reproducibility stamp.
Ghostscript and `pdftoppm` are used for figure conversion when available.

Do not report the EMP proxy as Verified Success Rate. The private dataset has no
verified multi-operation executions, trustworthy timestamps, or versioned drift
traces. See `results/artifacts/skipped_experiments.md` for the paper-protocol
experiments that remain blocked by those data and environment requirements.

The optional neural backend has a separate gradient smoke run:

```sh
conda activate corem-www2027
OMP_NUM_THREADS=1 python experiments/run_hgt_smoke.py --epochs 20
```

This controlled fixture verifies that HGT, cross-attention, and relation/set
heads train end to end. It is a software check, not a paper effectiveness result.

To run the neural model on a deterministic sample of observed private endpoint
outcomes and their operation/service graph:

```sh
OMP_NUM_THREADS=1 python experiments/run_hgt_endpoint.py
OMP_NUM_THREADS=1 python experiments/run_hgt_endpoint_multiseed.py \
  --output experiments/results_hgt_final_v2 \
  --train-size 7845 --train-negative-fraction 0.2 \
  --class-balance-power 1
python experiments/analyze_hgt_final.py \
  --results experiments/results_hgt_final_v2
```

The optimized neural run uses all observed training failures plus four sampled
successes per failure with inverse-frequency weighting, keeps
threshold selection and affine probability calibration on separate validation
halves, evaluates all held-out endpoints over ten fixed seeds, and stores paired
test predictions. See `results_hgt_final_v2/README.md` for validation-only
ablations, the measured comparison, and statistical analysis. It is still a
singleton endpoint prediction result and cannot be reported as multi-operation
workflow success.

## Public Multi-operation Benchmark

The public tau3-derived dataset adds execution-verified final-state labels for
strict multi-operation tasks. Build it in the isolated environment (the builder
refuses to overwrite an existing output), validate it, tune on validation only,
then run the frozen ten-seed experiment:

```sh
conda run -n corem-tau3 python scripts/build_tau3_corem_dataset.py \
  --source third_party/tau2-bench --output data/tau3_corem_new
conda run -n corem-www2027 python -m corem.dataset data/tau3_corem
OMP_NUM_THREADS=1 conda run -n corem-www2027 \
  python experiments/tune_tau3_multistep.py
OMP_NUM_THREADS=1 conda run -n corem-www2027 \
  python experiments/run_tau3_multistep.py
conda run -n corem-www2027 python experiments/analyze_tau3_multistep.py
```

For CUDA-enabled PyTorch, add `--device cuda` to both tuning and final-run
commands. On this server, the isolated `.venv-corem-gpu` environment exposes
the H800 and the final run uses `--device cuda --workers 5`. The selected
device, GPU, PyTorch/CUDA versions, and worker count are recorded in the result
artifact.

Labels cover each task's deterministic final-state criteria: agent/user database
hashes, official environment assertions, or both. Communication and
natural-language criteria are not evaluated. Outcome-derived irreducible edges
are stored for auditing but are excluded from the default graph and all reported
experiments.

The harder controlled derivative adds fresh-replayed, complete-operation-group
failures so the redundancy ablation is identifiable. It never overwrites the
base dataset:

```sh
python scripts/augment_tau3_counterexamples.py \
  --input data/tau3_corem --output data/tau3_corem_counterexamples
OMP_NUM_THREADS=1 .venv-corem-gpu/bin/python \
  experiments/tune_tau3_multistep.py \
  --dataset data/tau3_corem_counterexamples \
  --output experiments/results_tau3_counterexample_tuning \
  --device cuda --include-duplicates
OMP_NUM_THREADS=1 .venv-corem-gpu/bin/python \
  experiments/run_tau3_multistep.py \
  --dataset data/tau3_corem_counterexamples \
  --output experiments/results_tau3_counterexamples_10seed \
  --seeds 17,29,41,53,67,79,97,109,127,149 \
  --device cuda --include-duplicates --epochs 20 --workers 5
```

`analyze_tau3_paper_suite.py` generates predictor, logged-candidate policy,
retrieval-budget, controlled retention, controlled drift, and depth-stratified
tables. Its report explicitly lists experiments that cannot be supported by the
available data or pinned implementations; controlled proxies must not be copied
into natural-drift or live-system result cells.

## Frozen Complement-Head Score Audit

The candidate-matched 2-by-2 graph-representation by complement-utility
experiment and its evidence boundaries are documented in
`FACTORIAL_ABLATION_PROTOCOL.md`. It has five-seed old-validation summaries
under `results_public_multiservice_v5_factorial_summary_w{09,10}/` and post-hoc
v6 executed-test summaries under
`results_public_multiservice_v6_factorial_summary_w{09,10}_v2/`.
Neither is independent new-task/provider confirmation.

The frozen v6 complement-head score diagnostic is reproducible with:

```sh
python3 experiments/audit_complement_influence.py \
  --dataset data/public_multiservice_confirmation_v6 \
  --full experiments/results_public_multiservice_confirmation_v6_hgt_ensemble/candidate_scores.jsonl \
  --head-off experiments/results_public_multiservice_confirmation_v6_hgt_head_off/candidate_scores.jsonl \
  --output experiments/results_public_multiservice_confirmation_v6_complement_influence_audit_new
```

The audit refuses to overwrite output and checks that both scoring passes use
identical model states and residual weights. Its score and ranking effects are
exploratory on an inspected test set, not new execution or held-out-provider
evidence.

## Longitudinal v5 Extensions

The E3/E4 online stage uses a separate fixed logistic predictor. Fit its
256-feature artifact on independently checked development executions, with a
disjoint test dataset supplied to the trainer for a split check:

The GPT-4.1 adapter requires the optional SDK in `requirements-composer.txt`
and an `OPENAI_API_KEY` supplied by the execution environment.

```sh
python3 experiments/train_online_predictor.py \
  --development-dataset data/<development-dataset> \
  --development-executions <verified-development-executions.jsonl> \
  --test-dataset data/public_multiservice_challenge_v5 \
  --output <frozen-online-model.json>

python3 experiments/run_online_retention.py \
  --dataset data/public_multiservice_challenge_v5 \
  --online-model <frozen-online-model.json> \
  --composer-module experiments.gpt41_composer \
  --service-catalog <public-service-catalog.json> \
  --storage-budget 640 \
  --retrieval-budget 160 \
  --workload-window 16 \
  --update-time-limit-ms 3000 \
  --seed 2027 \
  --output experiments/results_online_retention_paper
```

At event `t`, retrieval uses the bank frozen before the event; the runner and
independent checker finish before request `t` enters the trailing workload and
the next bank is produced. The output includes execution trajectories,
per-event bank history, churn, update latency, and checksums. The 640-unit
budget and online controls follow Table 8. Reportable runs require the frozen
model and its provenance sidecar, including the test dataset manifest hash, and
a fixed composer adapter implementing `create_runner()` with `reset`, `execute`,
and `snapshot`, plus complete public operation catalogs. `experiments.gpt41_composer` supplies a pinned prompt and a
structured three-step plan for GPT-4.1, then uses the same resettable services
and independent checker. The repository's default controlled plan runner is
available for development prefixes. The paper's original prompt and decoding
settings were not published, so this adapter does not establish numerical
identity with the author's implementation.
The E3 maintenance cap defaults to 3000 ms; add `--uncapped-maintenance` for
the uncapped E4 setting.
The five HGT/BM25 scores belong to Eq. (7) candidate freezing; `--model-dirs`
remains available for development comparisons only. The paper does not include
the exact online feature template or fitted weights, so this implementation
does not establish numerical reproduction of its E3/E4 tables by itself.
At a release transition, only version-invalid records are removed before
retrieval. `--source-executions` may reuse a previously checked trajectory only
for the exact same event and selected-memory tuple; missing tuples are executed
against the pinned service. See `NATURAL_TRAFFIC_AUDIT.md` for the missing
source evidence required by a natural production-traffic drift claim.

`prepare_rolling_hgt_folds.py` creates the two additional past-only folds;
`train_public_multiservice_hgt.py --exclude-provider P --test-provider P`
implements outcome-label leave-provider-out training. Exact historical
`(event_id, selected_memory_ids)` executions can be reused with
`materialize_cached_executions.py`; the script fails if any selected bundle was
not physically executed. Finally, `analyze_longitudinal_extensions.py` emits
non-empty CSV tables plus PDF and 300-dpi PNG figures for online retention,
three rolling windows, four provider holdouts, and four documented release
transitions. These are public-release-drift measurements over controlled
queries, not natural production-traffic drift.
The reportable aligned runs are
`results_online_retention_hgt_aligned_b640_v1/`,
`results_online_retention_hgt_aligned_controls_b640_v1/`, and
`results_online_retention_hgt_aligned_nomemory_b640_v1/`; their consolidated
statistics are in `results_online_retention_hgt_aligned_analysis_v1/`.
Deterministic feature caches change latency only: compare selected IDs, bank
states, and scored outcomes before citing an efficiency rerun. Exact prior
trajectories can be supplied with `--source-executions`, but the source path
and checksum are recorded in every result manifest.

Rebuild the consolidated statistics with:

```sh
.venv-corem-gpu/bin/python experiments/analyze_aligned_online_retention.py \
  --dataset data/public_multiservice_challenge_v5 \
  --runs \
    experiments/results_online_retention_hgt_aligned_b640_v1 \
    experiments/results_online_retention_hgt_aligned_controls_b640_v1 \
    experiments/results_online_retention_hgt_aligned_nomemory_b640_v1 \
  --output experiments/results_online_retention_hgt_aligned_analysis_v1
```

The analyzer verifies source checksums and common model metadata before
computing family-cluster intervals and paired event contrasts. The E4 analyzer
`analyze_aligned_e4.py` applies the same gates to the nine `bq*/bm*` budget
runs and additionally rejects retrieval, storage, checker, and bank-containment
violations.

The earlier directories without
`_strict` are retained for trajectory provenance and carry `REJECTED.md`
markers because their online policy updated before the first incident of a
new release.

## External Baseline Adapters

The external comparison is an explicitly adapted, same-bank retrieval track:

```sh
OMP_NUM_THREADS=1 .venv-corem-gpu/bin/python \
  experiments/run_tau3_external_baseline_adapters.py
TAU2_DATA_DIR=third_party/tau2-bench/data conda run -n corem-tau3 \
  python experiments/replay_tau3_retrieval_ablations.py \
  --selections experiments/results_tau3_external_baselines/external_baseline_selections.json \
  --dataset data/tau3_corem_counterexamples \
  --output experiments/results_tau3_external_baselines
.venv-corem-gpu/bin/python experiments/analyze_tau3_external_baselines.py
```

This produces five retrieval controls and method-inspired MAGMA, ACGM,
HyperMem, HGMem, and AgeMem adapters over 43 held-out tasks and five seeds.
It does not execute the native systems. Their pinned source revisions and the
adaptation boundary are recorded under `experiments/external_baselines/`.

## Matched Retention Objective and Maintenance-Budget Audits

The retention audit changes only the bank-value objective. Item value sums
singleton utilities from Eq. (16), Bundle value scores fixed complete bundles
without invoking the deployed retriever, and CoReM scores the set returned by
that retriever. A new paper-aligned run uses one frozen online logistic scorer,
the same chronological stream, budgets, proposal ordering, and group-swap
limits for all three methods. Older five-checkpoint comparisons are exploratory.
Each cold-process run is stored under
`results_retention_objective_{item,bundle,corem}_cold_v1/`; consolidate it with:

```sh
.venv-corem-gpu/bin/python experiments/analyze_retention_objective_ablation.py \
  --retention-runs \
    experiments/results_retention_objective_item_cold_v1 \
    experiments/results_retention_objective_bundle_cold_v1 \
    experiments/results_retention_objective_corem_cold_v1 \
  --output experiments/results_retention_objective_tables_new
```

The analyzer leaves the independent and E2 contrast rows pending unless their
source files are explicitly supplied with `--independent-contrast` and the
paired `--e2-full`/`--e2-additive` options. It verifies checksums and matching
online model, composer, checker, dataset, and budget contracts for the
retention runs.

To render the online-retention paper table from a complete run, use
`python experiments/generate_online_retention_paper_table.py --run RUN_DIR --output TABLE.tex`.
The renderer checks the run's checksums and requires the frozen online
logistic model and GPT-4.1 composer provenance. Existing exploratory HGT
outputs cannot satisfy that contract.

The 25/100/500-ms anytime protocol is fixed in
`RETENTION_COMPUTE_BUDGET_PROTOCOL.md`. Analyze the six separately started
`results_retention_budget_*_v2/` runs with
`analyze_retention_compute_budgets.py`; the first eight events are calibration
and warm-up, and table metrics use events 9--320. Actual latency may exceed a
nominal limit because an in-flight request score is atomic. The dependency
table is reproduced by `analyze_dependency_complexity.py` from the frozen HSC
factorial logs. Do not relabel its optimum-recovery metric as execution VSR.

The post-hoc coherence repair is enabled with `--require-hyperedge-seed`. It
forbids retrieval expansions that are not rooted in a provenance-attested
dependency hyperedge. Diagnose the original failure with
`analyze_retention_failure_modes.py` and compare the repaired trajectory with
`analyze_corem_coherence_repair.py`. Because this repair was motivated after
the 320-event outcomes were inspected, its result is exploratory until it is
run on a newly frozen chronological stream.

`forecast_corem` is an exploratory past-only extension. After event $t$ has
been executed and checked, it updates a first-order transition table over
latent workflow signatures and may prefetch the highest-confidence next
bundles for $t+1$. Low-support, low-confidence, or low-margin cases fall back
to matched Bundle-value retention. The selected gates are margin 0.015,
transition support 2, and confidence 0.75. The consolidated chronological and
two seeded shuffled-order results are in
`results_retention_adaptive_forecast_analysis_v2/`. The second shuffle was
generated only after these gates were frozen, but it reorders known v5 events
and is not independent natural traffic.

