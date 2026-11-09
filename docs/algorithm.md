# CoReM Algorithm and Implementation Notes

## Purpose and Guarantees

CoReM predicts the observed execution success of a retrieved memory **set**, selects an admissible evidence-complete set under a token budget, and retains records under a separate storage budget. Within the online stage, retrieval and retention use the same fixed set-level probability model and cost weights. The output is predictive—not a causal attribution—and an evidence-complete set is not a runtime-success guarantee.

## Inputs and Outputs

`MemoryRecord` stores text, referenced operations, evidence atoms, an optional execution outcome, service/version and temporal scope, provenance, and retrieval/storage costs. `EvidenceGraph` contains memory and heterogeneous evidence nodes, verified typed edges, dependency hyperedges, soft duplicate pairs, hard pairwise conflicts, and forbidden subsets. `Request` supplies required and pre-verified atoms, active operation/version scopes, time, and event-local conflicts.

`SetUtilityModel.fit(examples, graph, config)` accepts actually executed, independently verified, admissible, evidence-complete set outcomes. It returns `FitResult` and updates base, unary, nonnegative complement, nonnegative redundancy, and typed-relation parameters. Examples explicitly marked ineligible cannot supervise interaction terms.

`BundleBeamRetriever.retrieve(request, bank, graph)` returns the best found feasible `RetrievalResult`, or an explicit `fallback`. Witness paths are enumerated separately; alternative support edges are not unioned into a mandatory closure. When `require_hyperedge_seed` is enabled, retrieval considers only structurally attested dependency bundles rather than constructing candidates from ungrouped singleton records. `partition_attested_hyperedges` validates structured runtime handoffs against an independently supplied verification contract before an edge enters that trusted bundle pool. `GroupAwareRetention.optimize(bank, new_ids, past_workload, graph, workload_weights=...)` returns a storage-feasible `RetentionResult` using normalized past-only weights. New evidence enters the online engine only after execution, independent checking, and service/version validation. Fallback skips workflow execution.

The candidate-freezing HGT/BM25 mixture in Eq. (7) is separate from E3/E4 online valuation. `OnlineLogisticPredictor` implements a fixed, 256-dimensional hashed logistic scorer with request/evidence, version, dependency, and cross-record features. Its complete mapping is specified in [online_features.md](online_features.md). `experiments/train_online_predictor.py` fits it only on independently checked development executions and writes a model plus provenance file. Reportable `run_online_retention.py` runs require this frozen artifact and a matching test manifest. The PDF does not publish the exact feature template or fitted coefficients, so this new feature template is an explicit implementation choice; matching the paper's reported numerical results requires the original frozen model and execution inputs.

Paper online evaluation also uses a fixed GPT-4.1 workflow composer. `experiments.gpt41_composer` contains this adapter's exact `SYSTEM_PROMPT` and pins a model, prompt version, temperature, and strict output schema; it sends only the request, exposed service/version catalog, operation skeleton, and selected memories. Each returned plan must match its cited memory, and incomplete or invalid output never reaches the executor. The adapter records configuration, catalog, and input hashes for audit. The included controlled plan runner is a development fixture. A paper run requires `--composer-module experiments.gpt41_composer` and `--service-catalog` covering every event scope. Cached trajectories must match the composer, catalog, visible input, and checker contracts. The PDF does not print the historical experiment's full prompt, and the repository has no preserved provenance linking that historical prompt to the current adapter; behavior-level identity with the composer that produced the paper results therefore remains unverified. The independent checker remains separate from the composer.

## Algorithms

```text
TRAIN(D, G):
  keep only executed, independently checked, eligible feasible examples
  repeat to epoch cap or convergence patience:
    compute BCE + Brier + typed-relation loss
    add sparsity penalty for provenance-backed complement coefficients
    update unary, complement, redundancy, calibration, and relation parameters

RETRIEVE(q, M):
  C <- rank admissible records in M by request-conditioned similarity
  B <- singletons plus bounded, separately enumerated grounded witness paths from C
  W <- {empty}; best <- none
  for depth = 1..L_b:
    for S in W and bundle B_i:
      S' <- S union B_i; charge shared records once
      discard S' if token budget, scope, version, time, or conflict fails
      prioritize expansions by marginal Phi per newly consumed token
      score S' by set utility minus context and uncovered-evidence penalties
      update best only if S' is evidence-complete
    W <- top K_b distinct states
  return best or fallback

RELAXED_SENSITIVITY(q, y):
  initialize fractional inclusion z
  repeat K times:
    x <- y * z
    ascend multilinear set utility plus binary entropy
    alternately project onto box, token-budget, and evidence-coverage constraints
  estimate dV/dy by finite differences through the complete K-step computation

RETAIN(M, N, past Q):
  form singleton and bounded path-connected insertion groups from N
  rank groups and eviction candidates using weighted relaxed sensitivities
  for at most K_s proposals and K_e evictions:
    M' <- (M union inserted_group) minus evicted_group
    value M' by running the same discrete RETRIEVE on every q in Q
    subtract storage-weighted activation and removal costs relative to pre-update M
    accept only a strict improvement
  return M
```

Intermediate retrieval states may be incomplete; returned states may not. Verified request metadata stops closure expansion only for dependency atoms it already establishes. Retention sensitivities propose moves only—every accepted bank is validated by the discrete retriever.

## Complexity and Termination

For embedding width `d`, graph layers `L_g`, nodes `V`, and typed edges `E`, deterministic graph encoding costs `O(L_g(V+E)d)` time and `O(Vd)` space. A set evaluation is `O(n + hr + p)`, for selected records `n`, eligible hyperedges `h` of maximum arity `r`, and duplicate pairs `p`.

With `b` exposed bundles, beam width `K_b`, and depth `L_b`, retrieval performs at most `O(L_b K_b b)` moves. Witness enumeration is separately bounded by the dependency-hop, path-count, and closure-size caps.

For universe size `u`, workload size `K_q`, and `K` relaxation steps, finite-difference proposal ranking uses `O(K_q(u+1)K)` relaxed evaluations. Discrete retention then uses at most `O(K_s K_e K_q)` bounded retrieval calls. The frozen HGT configuration trains for 16 epochs; the dependency-free reference model has its own patience rule. The learned objective and relaxation are non-convex; beam search, projected ascent, and group swaps have no global-optimality guarantee.

## Encoder Backends and Boundaries

`StableHashEncoder` remains the dependency-free default. The optional
`TorchHGTEncoder` in `corem/torch_encoding.py` uses PyG `HGTConv`, trainable
request cross-attention, optional relation classification, and an auxiliary
non-additive set-outcome objective. The paper's frozen training configuration
sets relation supervision to zero while preserving graph message passing.
`TorchSetUtilityModel` exposes these neural
heads directly to bundle retrieval and differentiates their multilinear
extension for retention. The transparent `SetUtilityModel` also detects this
encoder and supports staged fitting when interpretability is preferred.

The optional continuous proposal ordering uses deterministic finite differences and alternating projection. Appendix B leaves this guidance optional; it does not affect the exact discrete acceptance rule in Algorithm 2. Private endpoint data supports only singleton outcome supervision; the multi-operation paths require independently checked workflow executions.
