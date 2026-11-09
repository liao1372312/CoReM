# CoReM Code and Data Guide

This repository contains a Python research implementation of CoReM, controlled Web service workflow datasets, and experiment scripts. It accompanies the paper *CoReM: Dependency-Aware Joint Retrieval and Retention for Web Service Agents*.

## Project Structure and Main Code

| Location | Purpose |
| --- | --- |
| [corem/types.py](corem/types.py) | Memory records, requests, evidence graphs, dependency hyperedges, and service, version, time, and hard conflict constraints. |
| [corem/dataset.py](corem/dataset.py) | Loading and validation of the normalized private CoReM dataset. |
| [corem/encoding.py](corem/encoding.py), [corem/model.py](corem/model.py) | Dependency-free text and graph encoding and the non-additive set utility model. |
| [corem/torch_encoding.py](corem/torch_encoding.py), [corem/torch_model.py](corem/torch_model.py) | Optional PyTorch/PyG heterogeneous graph encoding, attention, and trainable set predictor. |
| [corem/online_predictor.py](corem/online_predictor.py) | Fixed online logistic set predictor, separate from the candidate-stage model. |
| [experiments/models/online_logistic_weights.json](experiments/models/online_logistic_weights.json) | A 256-dimensional online weight file for loading and interface checks; [create_online_model_weights.py](experiments/create_online_model_weights.py) can recreate it. |
| [experiments/gpt41_composer.py](experiments/gpt41_composer.py) | GPT-4.1 workflow composer adapter with a fixed prompt, structured plans, and an independent checking boundary. |
| [corem/retrieval.py](corem/retrieval.py) | Alternative witness paths, budget and conflict constraints, beam search, and explicit fallback. |
| [corem/retention.py](corem/retention.py) | Retriever-based valuation of candidate memory banks and group updates under a storage budget. |
| [corem/engine.py](corem/engine.py) | Coordination of retrieval, external execution feedback, verified evidence admission, training, and retention. The caller supplies workflow execution and independent checking. |
| [corem/real_eval.py](corem/real_eval.py), [corem/attestation.py](corem/attestation.py) | Experiment dataset audits, execution scoring, and dependency evidence verification. |
| [corem/__main__.py](corem/__main__.py) | A small end-to-end demo that needs no external service. |
| [scripts/](scripts/) | Dataset building, splitting, and audit tools. They normally write to new directories; generated data should not be edited by hand. |
| [experiments/](experiments/) | Model training, candidate selection, real execution, statistical summaries, and historical exploration scripts. |
| [tests/](tests/) | Unit tests for the core algorithms, data contracts, and experiment scripts. |

See [docs/algorithm.md](docs/algorithm.md) for algorithm interfaces and complexity, [docs/online_features.md](docs/online_features.md) for the online feature definitions, and [docs/composer_contract.md](docs/composer_contract.md) for the GPT-4.1 composer's fixed input and output contract. Core retrieval and retention use bounded heuristic search. Evidence completeness alone does not guarantee workflow execution success.

## Main Datasets

The paths below describe local research datasets. Dataset contents are not part of this code upload. Check their provenance and redistribution terms before sharing them separately.

| Directory | Contents and purpose | Scope |
| --- | --- | --- |
| `data/public_multiservice_challenge_v5/` | Four service providers, two releases, 320 controlled three-step events, and 384 memories for dependency retrieval and sequential retention. | Queries and hard negatives are constructed. Event time represents a logical replay based on release information, not production traffic. |
| `data/public_multiservice_confirmation_v6/` | New query wording over the v5 services, task families, and memories; its manifest records 640 events. | Tests new queries within existing families, not generalization to new service providers. |
| `data/public_independent_v7_candidate_v4/` | Sixteen Gitea/Meilisearch task families, 128 events, and candidate memories. | The manifest identifies a controlled candidate stage. Verify frozen selections and execution provenance before citing independent-test paper values. |
| `data/tau3_corem_counterexamples/` | Multi-operation examples derived from tau3, including execution outcomes and complete but failing counterexamples. | Use its README and manifest to distinguish replayable state checks from agent success in a natural environment. |
| `data/private_corem/` | Normalized private endpoint and memory data; see the `data/private_corem/CODEBOOK.md`. | Observed single-endpoint labels are not multi-operation workflow successes. Do not publish private data. |
| `data/public_versioned_workflows_v2/`, `data/public_versioned_challenge_v4/` | Earlier two-step workflows and a three-step query-blind challenge. | Separate controlled experiments; do not pool their results with v5/v6 or paper tables. |
| `data/public_independent_v10_e2/` | A later 128-event Typesense/Solr candidate experiment. | These events come from previously designed task families and do not replace the paper's Gitea/Meilisearch test source. |

Read each dataset's `manifest.json` and, when present, its README and `checksums.sha256` to confirm counts, provenance, and integrity. Pilot, staging, development, and other versions under `data/` should not be combined into one test set.

## Entry Points

Run the dependency-free demo from the repository root:

```sh
python -m corem
```

Run the core algorithm tests:

```sh
python -m unittest tests.test_corem tests.test_online_predictor tests.test_gpt41_composer tests.test_online_retention -v
```

Run the full test suite:

```sh
python -m unittest discover -s tests -v
```

The full suite also depends on optional PyTorch, scikit-learn, external runtimes, and historical paper artifacts; see [environment.yml](environment.yml) and [requirements-ml.txt](requirements-ml.txt). The GPT-4.1 composer additionally needs [requirements-composer.txt](requirements-composer.txt) and an API key supplied by the caller. Some scripts also depend on Unix commands when run on Windows. Passing the demo or unit tests does not establish reproduction of the paper's experiments.

Choose an experiment entry point for the task: [experiments/run_experiments.py](experiments/run_experiments.py) runs dependency-free reference measurements; [experiments/run_real_track.py](experiments/run_real_track.py) executes prepared selections and sends outcomes to an independent checker; [experiments/train_online_predictor.py](experiments/train_online_predictor.py) trains and freezes the online logistic model from verified development executions; and [experiments/run_online_retention.py](experiments/run_online_retention.py) runs controlled sequential retention with that model. The paper's HGT/BM25 candidate-stage scores are separate from the online model. See [experiments/README.md](experiments/README.md) and [experiments/real_tracks/README.md](experiments/real_tracks/README.md) for parameters and data contracts. Check inputs, models, and output directories before running.





