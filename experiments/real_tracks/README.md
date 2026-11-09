# Real-Track Evaluation

This directory defines the promotion gate for claims based on real chronology,
service-version drift, executable workflows, and communication quality. The
existing tau3 and private datasets remain useful controlled benchmarks, but
they must not be relabelled as this track.

## Dataset Contract

A candidate directory must contain `manifest.json`, `events.jsonl`,
`memories.jsonl`, and `version_transitions.jsonl`. Run:

```bash
python scripts/audit_real_track_dataset.py /path/to/candidate \
  --output experiments/results_real_track/audit.json
```

Every event needs a provenance-backed UTC `event_time`, `incident_id`, provider,
service and version identifiers, a checker contract, and non-empty lists of
required binding atoms, evidence atoms, and communication criteria. Every
memory needs verified provenance, a validity interval, version scope, and
evidence atoms. Each evaluated service must have two observed versions and an
independently evidenced transition. The manifest records source/licence,
immutable source ID, file checksums, a resettable execution contract, and an
independent checker digest.

The audit intentionally rejects response-payload timestamps as collection
timestamps. A field inside an API response is not evidence of when the
benchmark execution occurred.

## Single-Request Public Release-Reconstruction Pilot

The repository includes a T2 pilot over official `json-server` 0.17.4 and
1.0.0-beta.3 npm releases plus `go-httpbin` v2.14.1 and v2.15.0 OCI images.
Build and reproduce it with:

```bash
python scripts/build_public_versioned_service_dataset.py data/public_versioned_services
python scripts/audit_real_track_dataset.py data/public_versioned_services \
  --output experiments/results_public_versioned_audit
python scripts/split_real_track_dataset.py data/public_versioned_services \
  experiments/public_versioned_splits --stratify-by service_id
python experiments/generate_public_service_selections.py \
  --dataset data/public_versioned_services \
  --output experiments/public_versioned_selections
python experiments/run_real_track.py \
  --dataset data/public_versioned_services \
  --splits experiments/public_versioned_splits/chronological_splits.jsonl \
  --selections experiments/public_versioned_selections/selections.jsonl \
  --runner-plugin experiments.public_services_runner:create_runner \
  --checker-plugin experiments.public_services_checker:create_checker \
  --split all --output experiments/results_public_versioned_all
python experiments/analyze_real_track.py \
  --dataset data/public_versioned_services \
  --splits experiments/public_versioned_splits/chronological_splits.jsonl \
  --executions experiments/results_public_versioned_all/execution_results.jsonl \
  --split test --allow-execution-superset \
  --output experiments/results_public_versioned_test_analysis
```

Before the first run, install the two exact npm releases from their committed
lock files with `npm ci --prefix experiments/real_tracks/runtime/json-server-v017`
and the analogous `json-server-v1` command. Docker pulls the immutable
`go-httpbin` image digests recorded in the dataset manifest.

The logical event clock is anchored to official release timestamps; it is not
claimed to be production telemetry. Runtime wall-clock observations are kept
in the execution export. The pilot uses real released software, but its
single-request contracts and deterministic communication checks do not turn it
into a natural-drift or free-form composition result.

## Two-Step Public Workflow Extension

The exploratory extension runs two HTTP operations per incident and binds a
scalar observed in step one into step two. It uses five deterministic
tie-break seeds, a 144-token retrieval budget, and a checker implemented
separately from the runner. Rebuild and run all 2,000 method/seed/event cases:

```bash
python scripts/build_public_versioned_workflow_dataset.py \
  data/public_versioned_workflows_rebuild
python scripts/audit_real_track_dataset.py \
  data/public_versioned_workflows_rebuild \
  --output experiments/results_public_workflows_rebuild_audit
python scripts/split_real_track_dataset.py \
  data/public_versioned_workflows_rebuild \
  experiments/public_workflows_rebuild_splits --stratify-by service_id
python experiments/generate_public_workflow_selections.py \
  --dataset data/public_versioned_workflows_rebuild \
  --output experiments/public_workflows_rebuild_selections \
  --token-budget 144 --seeds 2027 2028 2029 2030 2031
python experiments/run_real_track.py \
  --dataset data/public_versioned_workflows_rebuild \
  --splits experiments/public_workflows_rebuild_splits/chronological_splits.jsonl \
  --selections experiments/public_workflows_rebuild_selections/selections.jsonl \
  --runner-plugin experiments.public_workflows_runner:create_runner \
  --checker-plugin experiments.public_workflows_checker:create_checker \
  --split all --output experiments/results_public_workflows_rebuild_all
python experiments/analyze_real_track.py \
  --dataset data/public_versioned_workflows_rebuild \
  --splits experiments/public_workflows_rebuild_splits/chronological_splits.jsonl \
  --executions experiments/results_public_workflows_rebuild_all/execution_results.jsonl \
  --split test --allow-execution-superset \
  --output experiments/results_public_workflows_rebuild_test_analysis
```

Install the same pinned npm runtimes described above; the runner pulls the
recorded immutable OCI images as needed. Existing audited results use the `_v2`
paths and are summarized in
[`../results_public_workflows_v2_summary.md`](../results_public_workflows_v2_summary.md).
The task revision was informed by a rejected compatible-stale candidate, so
these results are exploratory rather than confirmatory.

## Query-Blind Three-Step Challenge

Challenge v4 removes task keys and exact evidence IDs from retrieval, adds
same-version hard negatives and held-out query paraphrases, and executes three
HTTP operations with two runtime handoffs. Its oracle feasibility gate passes
all 80 incidents. The frozen test contains 16 incidents, five methods, and five
deterministic tie-break seeds (400 executions) at a 160-token budget:

```bash
python scripts/build_public_challenge_dataset_v2.py \
  data/public_versioned_challenge_v4
python scripts/audit_real_track_dataset.py \
  data/public_versioned_challenge_v4 \
  --output experiments/results_public_challenge_v4_audit
python scripts/split_real_track_dataset.py \
  data/public_versioned_challenge_v4 \
  experiments/public_challenge_v4_splits --stratify-by service_id
python experiments/generate_public_challenge_oracle.py \
  --dataset data/public_versioned_challenge_v4 \
  --output experiments/public_challenge_v4_oracle
python experiments/generate_public_challenge_selections.py \
  --dataset data/public_versioned_challenge_v4 \
  --output experiments/public_challenge_v4_selections \
  --token-budget 160 --seeds 2027 2028 2029 2030 2031
python experiments/run_real_track.py \
  --dataset data/public_versioned_challenge_v4 \
  --splits experiments/public_challenge_v4_splits/chronological_splits.jsonl \
  --selections experiments/public_challenge_v4_selections/selections.jsonl \
  --runner-plugin experiments.public_challenge_runner:create_runner \
  --checker-plugin experiments.public_challenge_checker_v2:create_checker \
  --split test --output experiments/results_public_challenge_v4_test
python experiments/analyze_real_track.py \
  --dataset data/public_versioned_challenge_v4 \
  --splits experiments/public_challenge_v4_splits/chronological_splits.jsonl \
  --executions experiments/results_public_challenge_v4_test/execution_results.jsonl \
  --split test --output experiments/results_public_challenge_v4_test_analysis
```

The measured table, paired statistics, and plot inputs are in
[`../results_public_challenge_v4_summary/SUMMARY.md`](../results_public_challenge_v4_summary/SUMMARY.md).
CoReM query-only reaches 56.25% test VSR versus 25.00% for version and recency
BM25, but the incident-paired exact sign test is not significant ($p=0.125$).
This post-hoc controlled challenge tests dependency-bundle retrieval; it is not
a trained-HGT, natural-production-drift, or universal-superiority result. The
rejected infeasible v3 contract is recorded in
[`multistep_analysis_log.md`](multistep_analysis_log.md).

### Four-Provider Frozen V5 Extension

V5 keeps the three-operation/two-handoff contract and adds pinned WireMock and
MockServer releases. It contains 320 controlled incidents, 384 memories, four
providers, eight releases, and 32 task/version families. Install the new Java
runtimes, build and audit the dataset, then run the oracle gate:

```bash
python scripts/install_public_multiservice_runtimes.py
python scripts/build_public_multiservice_challenge_dataset.py \
  data/public_multiservice_challenge_v5
python scripts/audit_real_track_dataset.py \
  data/public_multiservice_challenge_v5 \
  --output experiments/results_public_multiservice_challenge_v5_audit.json
python scripts/split_real_track_dataset.py \
  data/public_multiservice_challenge_v5 \
  experiments/public_multiservice_challenge_v5_splits \
  --stratify-by service_id
python experiments/generate_public_challenge_oracle.py \
  --dataset data/public_multiservice_challenge_v5 \
  --output experiments/public_multiservice_challenge_v5_oracle
python experiments/run_real_track.py \
  --dataset data/public_multiservice_challenge_v5 \
  --splits experiments/public_multiservice_challenge_v5_splits/chronological_splits.jsonl \
  --selections experiments/public_multiservice_challenge_v5_oracle/selections.jsonl \
  --runner-plugin experiments.public_multiservice_challenge_runner:create_runner \
  --checker-plugin experiments.public_multiservice_challenge_checker:create_checker \
  --split all --output /path/to/oracle_output
```

The accepted oracle rerun passes all 320 events with no execution, checker,
reset, binding, evidence, or communication failure. Generate the five-method,
five-seed frozen selections and execute the 64-incident test as follows:

```bash
python experiments/generate_public_challenge_selections.py \
  --dataset data/public_multiservice_challenge_v5 \
  --output experiments/public_multiservice_challenge_v5_selections \
  --token-budget 160 --seeds 2027 2028 2029 2030 2031
python experiments/run_real_track.py \
  --dataset data/public_multiservice_challenge_v5 \
  --splits experiments/public_multiservice_challenge_v5_splits/chronological_splits.jsonl \
  --selections experiments/public_multiservice_challenge_v5_selections/selections.jsonl \
  --runner-plugin experiments.public_multiservice_challenge_runner:create_runner \
  --checker-plugin experiments.public_multiservice_challenge_checker:create_checker \
  --split test --output experiments/results_public_multiservice_challenge_v5_test
```

The primary run contains 1,600 executions with no runner, checker, reset, or
budget error. CoReM query-only reaches 50.00% VSR versus 14.06% for version
BM25. A cold run reproduces all selections, status traces, error fields, and
checker decisions. The family-clustered statistics and per-service breakdown
are in the [v5 summary](../results_public_multiservice_challenge_v5_summary/SUMMARY.md).
This is controlled public-release reconstruction, not production traffic or a
trained-HGT evaluation. The first rejected oracle run and bounded runner repair
are retained in the v5 route card and analysis log.

### Retrieval-budget robustness and efficiency

The frozen post-hoc grid uses 80, 120, 160, 200, and 240 tokens. Generate each
selection artifact with `generate_public_challenge_selections.py`, then execute
all five points with the same normalization wrapper:

```bash
python experiments/run_real_track.py \
  --dataset data/public_versioned_challenge_v4 \
  --splits experiments/public_challenge_v4_splits/chronological_splits.jsonl \
  --selections /path/to/budget_selections.jsonl \
  --runner-plugin experiments.public_challenge_budget_runner:create_runner \
  --checker-plugin experiments.public_challenge_checker_v2:create_checker \
  --split test --output /path/to/budget_executions
python experiments/profile_public_challenge_retrieval.py \
  --dataset data/public_versioned_challenge_v4 \
  --splits experiments/public_challenge_v4_splits/chronological_splits.jsonl \
  --reference-selections experiments/public_challenge_v4_selections/selections.jsonl \
  --output experiments/results_public_challenge_v4_retrieval_profile
```

The accepted five-point run contains 2,000 checked executions with no budget,
runner, or checker violations. CoReM query-only VSR is
0/0/56.25/56.25/50.00%; complete retrieval p95 at 160 tokens is 144.62 ms.
See the [budget summary](../results_public_challenge_v4_budget_summary/SUMMARY.md)
and rejection/repair trail in `multistep_analysis_log.md`. This varies only the
retrieval cap.

### Controlled active-bank storage sweep

The separate past-only sweep fixes retrieval at 160 tokens and evaluates
storage budgets 640, 1280, 1920, 2560, 3840, and 7680. Generate one bank and
selection artifact per point, run it with the standard challenge runner and
checker, then validate the complete sweep:

```bash
python experiments/generate_public_challenge_retention_selections.py \
  --dataset data/public_versioned_challenge_v4 \
  --splits experiments/public_challenge_v4_splits/chronological_splits.jsonl \
  --storage-budget 1920 \
  --output experiments/public_challenge_v4_retention_bm_1920_selections
python experiments/summarize_public_challenge_retention.py \
  --dataset data/public_versioned_challenge_v4 \
  --splits experiments/public_challenge_v4_splits/chronological_splits.jsonl \
  --sweep-root experiments/results_public_challenge_v4_retention \
  --primary-execution experiments/results_public_challenge_v4_test/execution_results.jsonl \
  --output experiments/results_public_challenge_v4_retention_summary \
  --figures figures
```

Across 1,920 checked executions, group-aware VSR is
12.5/43.75/56.25/56.25/56.25/56.25%; independent relevance is
12.5/12.5/12.5/31.25/43.75/56.25%; item recency is
12.5/12.5/25/31.25/50/56.25%. All gates pass, including highest-budget parity
with all 80 frozen CoReM selections and checker objects. See the
[storage summary](../results_public_challenge_v4_retention_summary/SUMMARY.md).
The forecast uses only non-test incidents, but this remains a post-hoc
controlled mechanism comparison rather than online or production retention.

## Split and Execute

Create transitive incident/duplicate-safe chronological 60/10/10/20 splits and
a deterministic provider holdout:

```bash
python scripts/split_real_track_dataset.py /path/to/candidate \
  experiments/real_splits
```

Export retrieval output as JSONL with `event_id`, `method`, integer `seed`,
`selected_memory_ids`, integer `token_cost`, and `latency_ms`. Rows must be
generated prequentially; the executor sorts each method/seed stream by event
time before making service calls. Then run it:

```bash
python experiments/run_real_track.py \
  --dataset /path/to/candidate \
  --splits experiments/real_splits/chronological_splits.jsonl \
  --selections selections.jsonl --split test \
  --runner-plugin my_runner:create_runner \
  --checker-plugin my_checker:create_checker \
  --output experiments/results_real_execution
```

The runner object implements `reset(event)`, `execute(event, memories)`, and
`snapshot(event)`. Reset returns `{"reset_verified": true, "snapshot_id":
"..."}`. The separately implemented checker exposes `check(event, execution)`
and returns goal evidence, exact binding/evidence/communication checks, and
memory-reuse validity checks. Plugins are loaded only from explicit CLI flags;
commands embedded in data are never executed.

## Analyze and Import Native Baselines

```bash
python experiments/analyze_real_track.py \
  --dataset /path/to/candidate \
  --splits experiments/real_splits/chronological_splits.jsonl \
  --executions experiments/results_real_execution/execution_results.jsonl \
  --split test --output experiments/results_real_track
```

This emits VSR, binding success (BS), evidence coverage (ECR), stale-use rate
(SUR), clustered intervals, and per-transition recovery. If any task or checker
label is missing, `metrics` is `null` for the whole run.

For an external system's native output, first run
`experiments/validate_native_baseline_export.py`. Its manifest must pin the
repository and commit, declare seeds and token budget, and attest that the
system's core is native. It must also declare storage, LLM-call, and offline
training-example budgets. Each selection reports method, retrieval latency,
active-bank cost, and LLM calls. The validator rejects unknown memories,
outcome/gold-label fields, missing tasks, duplicates, and budget violations
before service execution.

