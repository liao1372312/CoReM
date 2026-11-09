# GPT-4.1 Composer Contract

`experiments/gpt41_composer.py` is the fixed composer adapter for the online
workflow path. It uses `gpt-4.1-2025-04-14`, temperature `0`, no response
storage, and a strict JSON schema. `PROMPT_VERSION` and
`composer_config_sha256` identify the exact prompt and generation contract;
`composer_input_sha256` identifies the visible input for one call.

The model sees the request, the current service and version, an ordered
operation skeleton, a public operation catalog, and only the
retrieved memories. Event fields containing checker requirements, reference
plans, task keys, or outcomes are excluded. A supplied `--service-catalog`
JSON file contributes only public operation ID, method, path, and description
fields. It must use schema `corem-service-catalog-v1`, with `entries` covering
every event service/version and a nonempty `operations` list for each entry.
The catalog SHA-256 is stored with every execution. A reportable run refuses
to start if the catalog or any event scope is missing.

The catalog format is:

```json
{
  "schema_version": "corem-service-catalog-v1",
  "entries": [{
    "service_id": "example-service",
    "version_id": "v1",
    "operations": [{
      "operation_id": "read-item",
      "method": "GET",
      "path": "/items/{id}",
      "description": "Read one item"
    }]
  }]
}
```

The model returns one cited memory and a plan for every skeleton step. Every
memory must be selected, have the matching step role and index, and supply the
same method, path, body, and capture rule as the returned plan. The existing
runner resolves captured bindings and executes the workflow once. An
incomplete model response, missing step, or ungrounded plan returns a failed
composition without a workflow execution. An independent checker determines
success after execution.

Reportable online runs require a frozen 256-feature online logistic model,
this composer contract, and the independent checker contract. Exact cached
executions are accepted only when their dataset manifest, composer model,
prompt, configuration, catalog, visible-input hash, and checker contract all
match the current run.

The repository contains the exact prompt used by this adapter as
`SYSTEM_PROMPT` in `experiments/gpt41_composer.py`. The paper identifies a
shared GPT-4.1 composer and its input/output boundaries, but does not print
the historical experiment's full prompt, catalog presentation, or decoding
configuration. No preserved run record links that historical prompt to this
adapter, so byte-for-byte identity with the composer that produced the paper
results is unverified.
