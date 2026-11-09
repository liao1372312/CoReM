# Paper-to-Code Alignment

This matrix tracks the method in `7965_CoReM_Dependency_Aware_Jo.pdf` against the executable reference implementation.

| Paper component | Implementation | Status / boundary |
|---|---|---|
| Memory tuple, service/version/time scope, provenance | `corem/types.py` (`MemoryRecord`, `Request`) | Implemented; scope is hard-filtered before retrieval. |
| Heterogeneous evidence graph and grounded relations | `EvidenceNode`, `EvidenceEdge`, `DependencyHyperedge`, `EvidenceGraph`, `PrivateCoremDataset.build_graph` | Implemented verified graph admission, operation/service/schema entities, typed provenance links, and bounded hypergraph paths. |
| HGT and request cross-attention, Eqs. (2–3) | `corem/torch_encoding.py` | Optional trainable PyG `HGTConv` and multi-head request cross-attention implemented; `StableHashEncoder` remains the zero-dependency default. |
| Non-additive set utility, Eq. (5) | `corem/torch_encoding.py`, `corem/model.py` | Implements unary, softplus complement, and softplus duplicate penalty terms. The frozen HGT training path enables the redundancy head; its penalty is active when the graph contains annotated duplicate pairs. |
| Frozen HGT loss, Eq. (6) | `experiments/train_public_multiservice_hgt.py`, `TorchHGTEncoder.fit_supervised` | Five seeds 17/29/41/53/67, 16 epochs, learning rate .01, ranking/calibration/relation/interaction weights 2/.1/0/.001, class-balance power .5, dimensions 32/64, two graph layers, one CPU thread. Zero relation weight leaves graph message passing enabled. The transparent reference model remains a separate diagnostic. |
| Compatibility and evidence coverage, Eq. (4) | `EvidenceGraph.is_admissible`, `is_evidence_complete` | Implemented for record, relation, version, temporal, budget, and evidence constraints. |
| Frozen candidate score, Eq. (7) | `experiments/evaluate_public_multiservice_hgt_ensemble.py` | Five sigmoid probabilities are averaged; a request-local normalized BM25-max residual is combined at fixed .9/.1 weights. This is a ranking score, not a calibrated success probability. |
| Online retrieval, Eqs. (8–10) | `corem/retrieval.py`, `corem/online_predictor.py` | Separate 256-feature logistic success predictor, token cost, complete witness alternatives, bounded beam, hard feasibility, and explicit fallback. The exact feature template and fitted weights are not published in the PDF. |
| Nested bank objective, Eqs. (11–13) | `corem/retention.py` | Uses the same online retriever on past-only workload samples and supports activation/removal costs. Matched controls expose item-value and fixed-complete-bundle objectives. |
| Optional continuous proposal guidance | `_relaxed_request_value`, `_relaxed_sensitivities` | Uses finite differences and alternating projection to order proposals. Appendix B does not prescribe an exact gradient or projector; discrete acceptance remains retriever-consistent. |
| Dependency-group swap | `GroupAwareRetention.optimize` | Implements bounded group proposals and evictions, followed by strict exact discrete acceptance checks. |
| Online Algorithm 1 | `corem/engine.py` | Implemented with executed/verified/eligible feedback and service-version admission gates. External composition and verification remain caller responsibilities. |
| E3/E4 strict prequential evaluation | `experiments/run_online_retention.py`, `experiments/analyze_aligned_online_retention.py`, `experiments/analyze_aligned_e4.py` | Reportable runs require the fixed online logistic model, matching provenance, and a composer adapter. E3 uses a 3000 ms maintenance deadline; `--uncapped-maintenance` selects E4. Frozen HGT/BM25 is available only for development comparison. Manifests pin model and execution-cache provenance. |
| Shared GPT-4.1 composer | `experiments/gpt41_composer.py`, `docs/composer_contract.md` | Pinned model, prompt version, temperature, strict schema, selected-memory plan grounding, one execution, and independent checking. Public operation catalogs may be supplied with events; current v5 events expose only service/version and skeleton. The original private prompt and full catalog are unavailable. |
| Matched retention-objective and compute-budget audits | `experiments/run_online_retention.py`, `experiments/analyze_retention_objective_ablation.py`, `experiments/analyze_retention_compute_budgets.py` | Holds the retriever, predictor, sequence, candidates, bank initialization, budgets, group operations, and proposal cap fixed. Online uncertainty uses 16 non-overlapping 20-event blocks. Anytime deadlines return the best completed feasible bank and report deadline exhaustion explicitly. |
| Exploratory forecast-aware retention | `forecast_corem` in `experiments/run_online_retention.py`, `experiments/analyze_forecast_retention.py` | Learns first-order latent-workflow transitions strictly after execution and checking, activates forecasts only above fixed support/confidence gates, and otherwise falls back to Bundle value. The current gain is confined to the predictable controlled chronology; a second unseen shuffled order is flat against Bundle, so this is not yet a confirmed general CoReM contribution. |
| Dependency-complexity stratification | `experiments/analyze_dependency_complexity.py` | Groups the frozen HSC E2 outcomes by annotated service-chain depth, averages five seeds within family, and never groups by method-selected memory count. This is offline optimum recovery, not live execution VSR. |
| Public multi-operation evaluation | `scripts/augment_tau3_counterexamples.py`, `experiments/run_tau3_multistep.py`, `experiments/analyze_tau3_multistep.py` | Reproducible validation-selected, ten-seed prediction on fresh execution-verified tau3 final states. A1--A3 are matched model ablations; the augmented release includes complete-but-failing duplicate counterexamples. |
| Retrieval, retention, and controlled drift protocol | `experiments/run_tau3_retrieval_ablations.py`, `experiments/replay_tau3_retrieval_ablations.py`, `experiments/analyze_tau3_paper_suite.py` | A4--A5 use fresh deterministic replay. Budget, A6--A8 retention, and drift tables are controlled logged/derived analyses and are labelled accordingly. |
| Real chronology, native exports, execution, and metrics | `corem/real_eval.py`, `scripts/audit_real_track_dataset.py`, `scripts/split_real_track_dataset.py`, `experiments/run_real_track.py`, `experiments/analyze_real_track.py`, `experiments/validate_native_baseline_export.py` | Code path implemented with provenance, reset, completeness, and independent-checker gates. The current local dataset fails the timestamp/version contract, so natural-drift, communication-quality, and native-system measurements remain empirically blocked rather than fabricated. |

Run `python3 -m unittest discover -s tests -v` for behavioral checks,
`python3 experiments/run_experiments.py` for the dependency-free measurements,
and `python3 experiments/run_hgt_endpoint_multiseed.py --output
experiments/results_hgt_final` for the validation-selected neural endpoint
experiment. Run `python3 experiments/analyze_hgt_final.py` for paired tests and
task-level bootstrap intervals. For the public multi-operation suite, follow the
commands in `experiments/README.md`; the frozen machine-readable outputs are
under `experiments/results_tau3_counterexamples_10seed/` and
`experiments/results_tau3_counterexample_suite/`.
For the real longitudinal track and its strict promotion gate, follow
`experiments/real_tracks/README.md`; the current local-source audit is stored in
`experiments/results_real_track_candidate_audit/audit.json`.
