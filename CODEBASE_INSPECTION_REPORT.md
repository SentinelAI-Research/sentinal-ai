# SentinelAI Codebase Inspection Report

Inspection date: 2026-10-27  
Repository: `SentinelAI-Research/sentinal-ai`  
Inspection scope: read-only source/config/artifact review, plus the existing pytest suite. No packages were installed and no LLM integration was attempted.

## 1. Executive Summary

SentinelAI has a working in-process monitor pipeline: a `ToolCallProposal` is evaluated against YAML policy and a heuristic score, a `Decision` is returned, and the executor invokes a registered tool only when the verdict is exactly `ALLOW`. DataObjects and a NetworkX provenance DAG exist, and a trajectory store records proposed steps in memory. A benchmark-specific runner and a local research dashboard also exist.

This is not yet a complete runtime service or LLM agent. There is no LLM adapter, no API for submitting proposals to the runtime, no durable decision/trajectory/provenance store, and no audit event stream. ML training and evaluation are offline experiments; trained models do not participate in runtime decisions. The dashboard reads saved evaluation JSON, not live runtime state, and most page content is hardcoded.

The V3 model comparison artifact reports 91.67% accuracy / 82.22% macro F1 for the 7-feature baseline and 100% / 100% for the 13-feature provenance model, on 720 synthetic test rows. The pair-aware split reports zero pair and trajectory overlap. These scores are not evidence of real-world generalization: the generator uses fixed action templates, synthetic security labels, and only two source-family values.

Observed test result: `python -B -m pytest -q -p no:cacheprovider` -> **74 passed in 237.03s**. The test suite appended two lines to tracked `data/sample/memory.txt`; those test-generated lines were removed, returning the worktree to its initially clean state before this report was created.

## 2. Repository and Git State

| Item | Observed state |
|---|---|
| Current branch | `feature/ml-risk-model` |
| HEAD | `9b6ff8a` (`SentinelAI Update before LLM Integration`) |
| Initial tracked/untracked status | Clean; no project untracked files |
| Final status before report creation | Clean after cleanup of the test side effect |
| Important source files untracked | None observed |
| Python | Python 3.11.9 from the existing system interpreter |

Recent five commits, newest first:

1. `9b6ff8a` SentinelAI Update before LLM Integration
2. `8cf0dfd` Frontend update
3. `cbdcc3a` Implement provenance-aware gradient boosting risk models
4. `7db5477` Implement provenance-aware gradient boosting risk models
5. `19e11f2` Build AMI pipeline and security trajectory labeling

The checked-in V3 comparison artifact is `models/hard_negative/v3/model_comparison_v3.json`. Generated V3 JSONL and split files live under `data/processed/`, which `.gitignore` excludes; the V3 split validation observed locally is therefore not guaranteed to be present in a clean clone. Model binaries are present in the repository, but are intentionally not embedded or described by binary contents here. No remote URL or secret value is included.

## 3. Project Structure

Concise source-oriented tree (caches, environments, dependencies, datasets, and binary/generated artifacts omitted):

```text
.
├── README.md
├── requirements.txt
├── pytest.ini
├── backend/
│   ├── agent/          # ProposalBuilder only; no planner/LLM adapter
│   ├── dataobjects/   # DataObject models, source loading, registry, memory lineage
│   ├── monitor/       # Runtime assembly, monitor, service, tool executor
│   ├── policy/        # YAML policy evaluation
│   ├── provenance/    # NetworkX DAG
│   ├── risk/          # Heuristic scorer and result model
│   ├── schemas/       # ToolCallProposal and Decision contracts
│   ├── tools/         # File, memory, communication, and external API functions
│   ├── trajectory/    # In-memory ordered steps
│   └── dashboard_server.py
├── benchmark/
│   ├── runners/       # Scenario execution, memory workflow, security gate
│   └── scenarios/     # Four initial scenario definitions and registry
├── configs/datasets.yaml
├── data/              # manifests, sample input; raw/processed data excluded
├── frontend/          # index.html, app.js, styles.css; no package manifest
├── ml/
│   ├── data/          # AMI/Enron ingestion, normalization, labeling, conversion
│   ├── features/      # dataset and provenance feature builders
│   ├── scenarios/     # scenario definitions/execution and label building
│   ├── evaluation/
│   └── training/
├── models/            # JSON metadata/results and saved model artifacts
├── policies/          # default.yaml and strict.yaml
├── scripts/           # dataset, split, audit, train, evaluate, and ablation CLIs
└── tests/
    ├── integration/   # runtime, policies, lineage, AMI/Enron, memory workflows
    └── unit/           # policy, risk, runtime, provenance features, dataset helpers
```

The README describes several future directories and components (for example `backend/storage`, a FastAPI API, React/TypeScript frontend, and live monitor views) that do not appear in the inspected tree. Treat those passages as intended architecture, not current implementation.

## 4. Architecture and Data Flow

Implemented core flow:

1. `backend.agent.proposal_builder.ProposalBuilder.build` constructs the Pydantic `ToolCallProposal` (`step_id`, `agent_id`, `tool_name`, `arguments`, `input_data_objects`, timestamp).
2. `MonitorService.process` calls `SentinelMonitor.evaluate` before attempting execution.
3. The monitor inserts a `TrajectoryStep`, resolves the last listed input DataObject, evaluates policy using its sensitivity and caller-supplied destination, looks up provenance depth and prior history, then obtains a heuristic risk score and constructs a `Decision`.
4. `RuntimeExecutor.apply_decision` updates the trajectory decision and calls `ToolExecutor.execute` only for `ALLOW`; both `WARN` and `BLOCK` prevent invocation.
5. On successful execution, `OutputDataObjectHandler` hashes the stringified result, registers an output DataObject, inherits the selected source sensitivity, and links every proposal parent to the output node.

The monitor/service boundary is useful for provider neutrality, but important end-to-end gaps remain:

- `ToolCallProposal` has no session ID. `SentinelMonitor` derives `session-{agent_id}`, so separate runs from one agent share trajectory history.
- The monitor and benchmark security gate evaluate only `input_data_objects[-1]`, even though the graph supports multiple parents.
- The output object records all parents, but its sensitivity comes from one source (the last input), not the maximum sensitivity of all ancestors/parents.
- The trajectory step receives verdict and risk score but does not receive tool result, error, or generated output-object ID in `output_data_objects` through `MonitorService`.
- `TrajectoryStore` and `ProvenanceGraph` are in-memory only. There is no durable audit log, replay layer, or cross-process state.
- The runtime factory is programmatic (`create_sentinel_runtime`); there is no production backend API or standalone agent command.

## 5. Backend Component Inventory

| Component / files | Main classes/functions and callers | Current behavior | Dependencies and integration risks |
|---|---|---|---|
| Proposal contract: `backend/schemas/contracts.py`, `backend/agent/proposal_builder.py` | `ToolCallProposal`, `Decision`, `ProposalBuilder.build`; proposal builder used by `benchmark/runners/memory_pipeline.py` | Validates basic Pydantic shape and constructs proposals. | No session ID, typed tool argument schemas, capability scope, destination, idempotency key, or structured agent observation/result contract. `verdict` is a free string rather than an enum. |
| Runtime assembly: `backend/monitor/factory.py` | `create_sentinel_runtime`, `SentinelRuntime`; used by memory pipeline and tests | Wires in-memory trajectory/provenance, YAML policy, heuristic scorer, tool executor, monitor service. | Relative default `policies/default.yaml` depends on current working directory. No API/server lifecycle, persistence, or model loader. |
| Orchestration: `backend/monitor/service.py`, `monitor.py`, `runtime.py` | `MonitorService.process`, `SentinelMonitor.evaluate`, `RuntimeExecutor.apply_decision` | Enforces evaluate-before-execute, updates the step, then creates provenance after successful execution. Runtime blocks every verdict except exact `ALLOW`. | Caller determines destination. Only final input parent is inspected. Risk does not change policy verdict. Tool exceptions are converted to `ExecutionResult.error`; monitor/lookup failures propagate. `WARN` is treated as a hard stop, not an approval workflow. |
| Tool boundary: `backend/monitor/executor.py`, `tool_registry.py` | `ToolExecutor.register_tool/execute`, `create_tool_executor`; runtime factory calls registry | Executes named functions in an allowlisted in-process dictionary and rejects unknown tool names. Registered: `read_file`, `summarize`, `paraphrase`, `calculate`, `write_memory`, `read_memory`, `send_message` (seven tools). | `call_external_api` exists in `backend/tools/external_tools.py` but is not registered. No per-tool authorization schema, argument validation, filesystem confinement, external host allowlist, or separate process sandbox. |
| Tools: `backend/tools/file_tools.py` | `read_file`, `summarize`, `paraphrase`, `calculate`; registry invokes | Reads UTF-8 paths, performs deterministic extractive transformations, and evaluates arithmetic after character filtering. | `read_file` accepts a caller-selected local path. Tools return ordinary Python values, not DataObjects; the monitor wraps output as `str(output)`. |
| Memory and communication: `backend/tools/memory_tools.py`, `communication_tools.py` | `write_memory`, `read_memory`, `send_message`; registry invokes | Memory appends/reads local text; `send_message` returns a timestamped simulated result and sends nothing. | Default memory path is a shared `data/sample/memory.txt`; test suite mutates it. Per-session isolation is optional and not automatically enforced. |
| Data object lifecycle: `backend/dataobjects/models.py`, `factory.py`, `registry.py`, `source_loader.py`, `memory_service.py` | `DataObject`, `create_data_object`, `DataObjectRegistry`, `FileSourceLoader`, `MemoryDataObjectService` | Stores IDs, source, sensitivity, parent IDs, transformation, SHA-256 content hash, timestamps and step. Source text is hashed, not held in the graph model. | Sensitivity is an unconstrained string. Hashing does not prove content integrity across storage; there is no content store, embedding store, classification engine, or durable registry. Memory lineage helpers exist but are not part of the central tool executor’s automatic output handling. |
| Dashboard: `backend/dashboard_server.py` | `DashboardHandler`, `build_payload`, `main` | Serves static assets and read-only `GET /api/evaluation`, `GET /api/health`; defaults to `127.0.0.1:5000` or `PORT`. | This is an `http.server` dashboard, not a runtime API. Payload is assembled from experiment JSON artifacts; no live monitor, proposal, trajectory, or graph endpoints. |

## 6. Policy Engine and Provenance Implementation

**Policy.** `backend/policy/engine.py` loads YAML via `yaml.safe_load`, creates `PolicyRule` models, matches exact `(source_sensitivity, destination)` pairs, then gives `BLOCK` precedence over `WARN` and `ALLOW`. With no matching rule it returns `ALLOW`. `policies/default.yaml` includes confidential-to-external `BLOCK`, internal-to-external `WARN`, public-to-external `ALLOW`, and confidential-to-memory `WARN`. `policies/strict.yaml` contains stricter rules. Its YAML `priority` fields are not declared in `PolicyRule`; verdict resolution uses the fixed verdict ordering, not priority. Invalid verdict strings are not constrained by an enum.

**Provenance.** `backend/provenance/graph.py` uses `networkx.DiGraph`; edges point parent-to-child. It implements object registration, direct parents/children, all ancestors, shortest paths, graph depth, transformation depth, and object lookup. Despite the README calling it a DAG, `add_relationship` does not independently reject cycles. The central output handler does create multiple parent edges, but monitoring/risk currently evaluates only one input object and the last input’s sensitivity.

**Decision and risk.** `HeuristicRiskScorer.assess` assigns score 1.0 for policy block, 0.6 for warning, 0.5 for provenance depth >=3 or external destination after prior history, and 0.4 for history length >=4. `SentinelMonitor` puts this score and reason in `Decision`, but sets `verdict=policy_result.verdict`; therefore risk never promotes an `ALLOW` to `WARN`/`BLOCK`. `risk_probability` is always `None`. Runtime enforcement is deterministic policy-only.

**Trajectory/audit.** `TrajectoryStore` retains ordered `TrajectoryStep` objects in process memory, can query by agent/session and previous steps, and stores verdict/risk. It does not serialize to disk. Execution errors/results and decisions are not written to a separate append-only audit record. Dashboard does not query this store.

## 7. ML Pipeline and Benchmark Results

**Data sources and labels.** `configs/datasets.yaml` describes AMI, Enron, and a PII dataset. AMI scripts normalize the corpus and build transformed records; Enron scripts parse emails and construct provenance-linked six-step trajectories. The experiment builders assign step labels using fixed scenario mappings (for Enron, step 1 `SAFE`, steps 2–5 `EARLY_RISK`, step 6 `VIOLATION`). In the hard-negative and V3 datasets, sensitivity and risk labels are explicitly synthetic experimental metadata, not claims about actual AMI/Enron confidentiality. No raw source files appeared in the inspected `data/raw/` inventory; download/source availability is a clean-clone prerequisite.

**Feature schema.** The current hard-negative feature views have seven baseline features: `has_input_object`, `is_external_action`, `is_memory_operation`, `is_transformation`, `sensitivity_encoded`, `tool_encoded`, `transformation_encoded`. Six added provenance features are `provenance_depth`, `transformation_depth`, `has_provenance`, `has_sensitive_ancestor`, `sensitive_ancestor_count`, `max_ancestor_sensitivity_encoded`. Dataset rows also carry identifiers, scenario/variant, labels and related metadata; training scripts explicitly exclude several IDs/target fields. AMI transformation features include lengths/compression ratio, provenance depth, parent presence, sensitivity/transformation names, and content hashes. Enron features include action/tool, step/progress, sensitivity, memory/external/transformation flags, lineage depths and labels.

**Generation, split, training.** Scripts under `scripts/` perform AMI/Enron ingestion/building, hard-negative construction, feature-view generation, split validation, model training, evaluation, and ablation. V2 split logic groups paired benign/risky variants by original source group (80/20, seed 42); V3 assigns each `pair_id` wholly to one split, stratified/shuffled within the two source families (80/20, seed 42), then validates zero pair/trajectory overlap. V3 explicitly does not hold out scenario templates and uses a fixed action sequence. Models are `sklearn.ensemble.HistGradientBoostingClassifier`, learning rate 0.05, 200 iterations, max leaf nodes 15, L2 1.0, random state 42. Training scripts save joblib artifacts and JSON reports. No online model inference is implemented.

**Observed result artifacts (not just README claims).**

| Experiment | Artifact | Baseline | Provenance-aware | Notes |
|---|---|---:|---:|---|
| V2 split | `models/hard_negative/split_v2/model_comparison_v2.json` | Accuracy 91.67%; macro F1 61.90% | Accuracy 100%; macro F1 100% | 1,920 train / 480 test rows in ablation metadata; trajectory overlap 0. |
| V2 ablation | `models/hard_negative/ablation_v1/ablation_summary_v1.json` | 7-feature accuracy 91.67%; macro F1 61.90% | Full 13-feature accuracy / macro F1 100% / 100% | Removing both sensitive-ancestor indicator features alone keeps the perfect score; removing the ancestor-sensitivity group returns to baseline-level performance. |
| V2 shortcut audit | `models/hard_negative/ablation_v1/ablation_audit_v1.json` | — | — | Test audit flags `is_external_action`, `provenance_depth`, `sensitive_ancestor_count`, `tool_encoded`, `transformation_depth`, and `transformation_encoded` as possible label shortcuts. Audit warns these are synthetic correlations, not causal proof. |
| V3 matched pairs | `models/hard_negative/v3/model_comparison_v3.json` | Accuracy 91.67%; macro F1 82.22% | Accuracy 100%; macro F1 100% | 2,880 train / 720 test rows; 240 train / 60 test pairs; overlap counts 0. Baseline and provenance model both have 100% VIOLATION recall; baseline errors include false-positive VIOLATION predictions on SAFE rows. |

V3 is clearly labeled `synthetic_only` in its result artifact. The paired generator makes baseline/action features identical within each benign/risky pair and changes the intended provenance ancestry at the final decision point; however, fixed action sequence, fixed label construction, two source families, and controlled ancestry make this a mechanism test, not a generalization result. V2 audit also identifies feature-label shortcuts. Do not present 100% as a real-world performance guarantee. Permutation importance is included as diagnostic evidence, with the expected correlated-feature caveat. Early-warning lead time, calibration, runtime overhead, external-task success, and confidence intervals are not reported in these artifacts.

**Reproduction.** V3 commands (from repository root; will write ignored/generated data and models, so not run during this inspection):

```powershell
python scripts/build_shortcut_resistant_benchmark_v3.py --pairs 300 --seed 42
python scripts/split_shortcut_resistant_benchmark_v3.py --test-fraction 0.20 --seed 42
python scripts/train_compare_provenance_models_v3.py --seed 42
```

The defaults use `data/processed/hard_negative/v3/` and `models/hard_negative/v3/`. For V2, ensure the expected AMI/Enron processed experiment inputs exist, then run `build_provenance_hard_negative_dataset.py`, `prepare_provenance_hard_negative_features.py`, `split_provenance_hard_negative_dataset_v2.py`, and `train_provenance_hard_negative_models_v2.py`; run `run_provenance_feature_ablation_v1.py` for V2 ablation. The V2 scripts use repository-relative paths and fixed seed/configuration. Check existing outputs first: some scripts deliberately refuse to overwrite prior results. Raw dataset download and the precise order/availability of each upstream dataset build are prerequisites not satisfied by this inspection.

For the original/V1 pipeline, the script defaults indicate this order (the AMI input and Enron archive must first be available under `data/raw/`):

```powershell
python scripts/build_ami_security_experiment_dataset.py
python scripts/build_enron_experiment_dataset.py
python scripts/build_provenance_hard_negative_dataset.py
python scripts/prepare_provenance_hard_negative_features.py
python scripts/split_provenance_hard_negative_dataset.py
python scripts/train_provenance_hard_negative_models.py
python scripts/evaluate_provenance_hard_negative_models.py
```

The V1 splitter groups complete trajectories and keeps both feature views on the same partition (seed 42, 20% test). These commands write under ignored `data/processed/` and `results/`, and train saved models; inspect existing outputs before running. The README does not provide a single canonical end-to-end command, and this sequence was derived from the scripts' paths and entry points rather than executed.

## 8. Frontend and Dashboard

`frontend/index.html`, `app.js`, and `styles.css` implement a static, multi-page research dashboard with Overview, Initial Benchmark, V2, Ablation/Audit, V3, and Roadmap views. `app.js` renders cards, metrics, tables, and text; charts are mostly CSS-built bars/tables rather than a chart library. Metrics and experiment narratives in the main pages are hardcoded. The only frontend fetch is `GET /api/evaluation` from the Initial Benchmark page, used to display a generic artifact-availability message; it does not bind returned model metrics into the experiment pages.

The dashboard server dynamically reads saved JSON/JSONL summaries for `/api/evaluation` and returns `{ok:true}` from `/api/health`. Its V3 snapshot hardcodes fallback counts (3,600 rows, 600 trajectories, 300 pairs) when local generated artifacts are absent. This can make a dashboard appear populated without those dataset files. No live trajectory/provenance/decision API, mutation endpoint, WebSocket, or SSE is implemented. Despite README plans, there is no `package.json`, React, TypeScript, chart package, or frontend build step.

Start the current dashboard from repository root with `python -m backend.dashboard_server`, then open `http://127.0.0.1:5000` (or the `PORT` selected in the environment). It serves both frontend assets and its read-only API. This starts only the dashboard, not an agent/runtime API.

## 9. Tests and Current Health

The repository has **62 test files** under `tests/unit/` and `tests/integration/`. Unit coverage includes runtime, policy, heuristic risk, tool registry/executor, trajectory storage/conversion, provenance feature extraction, dataset manifest, hashing, and scenario/security gate helpers. Integration coverage includes source loading, AMI/Enron normalization/conversion/features, policy matrix, runtime allow/block behavior, multi-parent provenance, transformations, and memory flows. Notable missing coverage: real LLM adapter (none exists), external API authorization/sandbox, HTTP/API proposal endpoint (none exists), durable audit/restart behavior, concurrent session isolation, tool argument schemas, and end-to-end dashboard/runtime event flow.

| Check | Observed |
|---|---|
| Exact command | `python -B -m pytest -q -p no:cacheprovider` |
| Python interpreter | Python 3.11.9, existing system interpreter |
| Outcome | **74 passed in 237.03s**; no failures/errors reported |
| Configuration | `pytest.ini`: `pythonpath = .`, `testpaths = tests` |
| Environment changes | None; no packages installed |

The cache-suppressed command still caused integration tests to append two lines to `data/sample/memory.txt` via the default memory tool. Because that file was clean before testing, the two newly appended lines were removed and `git status --short --untracked-files=all` returned clean before report creation. Future test runs should isolate or reset the sample memory file, preferably using a temporary path; do not assume the suite is side-effect-free.

## 10. Dependencies and Run Instructions

`requirements.txt` lists unpinned `pydantic`, `PyYAML`, `networkx`, `requests`, `pytest`, `datasets`, `scikit-learn`, and `joblib`. There is no `pyproject.toml`, `package.json`, or lockfile in the inspected root. Setup instructions are not a complete reproducible environment specification; the existing environment already had dependencies sufficient for the observed test run. A conventional setup for a fresh environment would be `python -m venv .venv`, activate it, then `python -m pip install -r requirements.txt`; not performed here.

Current commands from the repository root:

```powershell
# Tests (observed above; test suite may append to data/sample/memory.txt)
python -B -m pytest -q -p no:cacheprovider

# Dashboard and static frontend
python -m backend.dashboard_server

# V3 benchmark, after reviewing existing outputs
python scripts/build_shortcut_resistant_benchmark_v3.py --pairs 300 --seed 42
python scripts/split_shortcut_resistant_benchmark_v3.py --test-fraction 0.20 --seed 42
python scripts/train_compare_provenance_models_v3.py --seed 42
```

README does not document these operational commands; they are derived from Python module/script entry points and defaults. There is no separate frontend start command. The source dashboard is not an API backend for proposal submission. AMI/Enron data downloads/builds have separate scripts and are prerequisites for their experiment pipelines; generated `data/processed/` is ignored. Avoid rerunning scripts over existing model outputs without checking their overwrite guards and destination paths.

## 11. Real LLM Integration Readiness

| Requested integration point | Current state |
|---|---|
| Proposal / decision structures | Present: `ToolCallProposal`, `Decision`, `ExecutionResult`; missing session, structured tool schemas, explicit approval state, normalized execution observation, and stable request/run identity. |
| Agent and tool interfaces | `ProposalBuilder` and `ToolExecutor` exist. Tools are Python callables and are invoked through the monitor service in the memory pipeline. No general planner/agent loop exists. |
| Scripted/simulated agent | No autonomous scripted planner found. `benchmark/scenarios` describes scenarios; `benchmark/runners` scripts fixed workflows. `MemoryPipelineRunner` is the closest monitored scripted workflow, but only some steps go through the monitor. `send_message` is simulated. |
| Provider-neutral adapter location | Add an adapter/planner alongside `backend/agent/` that converts provider function-call output into validated `ToolCallProposal` objects. It should not receive `ToolExecutor` or direct tool callables. Keep provider-specific SDK types outside `backend/monitor/`. |
| Prevent pre-authorization execution | Route every proposal through one `MonitorService.process` boundary; do not expose executor/tools to the LLM adapter. Preserve executor’s exact-ALLOW check. Make destination/tool risk context derived by trusted server-side registry rather than model-supplied. Route WARN to an explicit pause/approval state; do not execute while pending. |
| Record object lineage | Resolve input IDs to registered DataObjects; validate every parent; execute only after authorization; create output object from actual result and link all parent IDs. Fix multi-parent sensitivity aggregation before relying on lineage labels. Ensure memory reads/writes use the same path instead of manual service calls. |
| Log decision and results | Add session/run identity and durable append-only records for proposal, policy/risk decision, execution outcome/error, output object IDs, parent IDs, and timestamps. Current trajectory is transient and omits outputs/errors. |
| Deterministic trajectory testing | Use a fake planner/provider returning fixed proposal sequences and mocked tools; assert each decision precedes invocation, `WARN/BLOCK` never invoke a tool, no unauthorized external call happens, lineage and result IDs are correct, and full safe/unsafe multi-step trajectories are replayable. Use temporary memory and filesystem roots. |
| Implementation order | 1) Harden/complete contracts and monitor boundary, including session and multi-input sensitivity/decision records. 2) Implement deterministic scripted planner and route all eight tools through monitor with simulated external effects and trajectory tests. 3) Add provider-neutral function-calling adapter plus live dashboard/event visibility after the contract is stable. |
| Hosted or local model | Either can be added without changing core monitor if both only produce the same provider-neutral proposal schema and receive results/decisions as explicit messages. The repository has no provider dependencies or adapter today; no provider should be selected as part of this inspection. |

Specific missing requested tools: `call_external_api` is implemented as a raw `requests.get` function but not registered and has no allowlist/sandbox; keep it disabled in the initial demo. `send_message` is registered but returns a simulation object and performs no delivery. Memory/file tools operate on local filesystem state, which needs scoped paths and isolated storage for a real agent.

## 12. Bugs, Risks, and Missing Functionality

1. **Risk score does not govern execution.** A proposal can have a high heuristic risk and still be `ALLOW` if no matching policy rule blocks/warns it. No ML model participates.
2. **Multi-parent lineage is not fully policy-safe.** Only the last input object is checked; output sensitivity is copied from that one object. A lower-sensitivity last parent can mask a more sensitive earlier parent.
3. **Caller-controlled destination.** Policy outcome depends on `destination` supplied by the call site; there is no authoritative mapping from tool and arguments to destination/trust class.
4. **No production authorization API.** The dashboard server is static/evaluation-only; no service accepts an external proposal or proves it passed the monitor.
5. **No durable audit/history.** Restart loses graph and trajectory; execution output/error and output object IDs are not fully written back to the trajectory.
6. **WARN lacks approval semantics.** Runtime treats WARN as non-execution, but no user approval/resume flow exists.
7. **Tool exposure is broader than demo needs.** `read_file` accepts any local path; memory defaults to a shared tracked sample file; external API function can make real network requests if someone registers/calls it.
8. **Policy configuration validation.** Verdict/sensitivity/destination strings are unconstrained; unknown policy verdict handling and ignored `priority` values can surprise maintainers; no-match defaults to ALLOW.
9. **Shared session identity and in-memory state.** Sessions are derived from agent ID and may merge across task runs; concurrency and restart semantics are undefined.
10. **Dashboard can overstate dynamic integration.** Hardcoded text and fallback V3 counts can show results even without current local data; results pages don't render values from API response.
11. **Experiment generalization caveats.** V2 shortcut audit and synthetic labels, plus V3 fixed sequences/two source families, mean current results do not demonstrate learned real-world provenance reasoning. Paired split prevents pair leakage only, not template leakage.
12. **Test mutation.** Full pytest currently changes `data/sample/memory.txt`; temporary storage would make CI and repeated local runs safer.
13. **Reproducibility / clean clone.** Unpinned dependencies, absent raw data, ignored processed datasets/split reports, and possibly local-only outputs limit exact reproduction from a clean checkout.

## 13. Recommended Implementation Plan

1. **Stabilize monitor contract:** add session/run identity, constrain verdict/sensitivity/destination/tool names, define an explicit WARN-pending/approval outcome, and specify typed arguments/results. Make destination trusted metadata, not LLM input.
2. **Correct lineage decision inputs:** evaluate all input parents and relevant ancestors; define conservative sensitivity aggregation; validate parent IDs and prevent cycles/duplicate output IDs. Persist generated output IDs and execution outcomes into trajectory events.
3. **Make deterministic monitored workflows complete:** implement a scripted planner that emits proposals only; ensure each intended operation including memory and simulated messaging uses the same service; use temporary file/memory roots. Add tests proving no tool executes before ALLOW and proving full safe/unsafe trajectories.
4. **Add audit persistence and lifecycle:** introduce a small store abstraction (memory for tests, durable implementation for demo), replayable decision/result records, and explicit error semantics before adding a UI stream.
5. **Integrate LLM adapter behind the proposal contract:** adapter only converts function calls to proposals, awaits monitor decision, and returns approved execution results. Keep provider SDK outside the monitor; keep external network/email simulated and disabled.
6. **Connect UI to actual state:** expose read-only runtime sessions, decisions, trajectory and provenance endpoints; bind existing dashboard to API data and remove fabricated fallbacks. Keep experiment results separately sourced from artifact files.
7. **Keep the research pipeline honest/reproducible:** add held-out scenario templates and varied action sequences, verify no group/template leakage, publish split manifests and environment versions, and report early-warning/classwise metrics and confidence intervals. Preserve current artifacts and add new experiment versions instead of overwriting.

## 14. Files and Functions to Inspect First

1. `backend/monitor/service.py` — `MonitorService.process`: central proposal -> decision -> execution -> output lineage boundary.
2. `backend/monitor/monitor.py` — `SentinelMonitor.evaluate`: session, input selection, policy/risk fusion, decision creation.
3. `backend/monitor/runtime.py` and `backend/monitor/executor.py` — enforcement condition and tool invocation.
4. `backend/provenance/graph.py`, `backend/monitor/output_handler.py`, `backend/dataobjects/memory_service.py` — ancestry and sensitivity propagation.
5. `backend/schemas/contracts.py` and `backend/agent/proposal_builder.py` — provider-neutral adapter contracts.
6. `backend/monitor/tool_registry.py`, `backend/tools/` — tool availability and side effects; note external API is not registered.
7. `backend/policy/engine.py`, `policies/default.yaml`, `policies/strict.yaml` — verdict behavior and policy schema.
8. `benchmark/runners/memory_pipeline.py` and `benchmark/runners/execution_service.py` — closest existing scripted flows and monitor bypass paths.
9. `scripts/build_shortcut_resistant_benchmark_v3.py`, `split_shortcut_resistant_benchmark_v3.py`, `train_compare_provenance_models_v3.py` — current synthetic benchmark design and reproduction.
10. `backend/dashboard_server.py`, `frontend/app.js` — static experiment UI versus live runtime state.

## 15. Inspection Limitations

- This was a source/config/artifact inspection and existing test run, not a live security assessment or hosted LLM integration test.
- Raw AMI/Enron corpus files were not present in the inspected raw-data inventory; source download/reproduction of data-building stages was not verified.
- Dataset JSONL and model binaries were not dumped or independently regenerated. Result values were read from compact JSON artifacts and verified against the corresponding V3 split summary and training script.
- The broad filename search timed out once; targeted PowerShell listings and source reads were used to inspect the requested project areas.
- The tests were run once. Their observed memory-file side effect was cleaned; no source or dependency files were edited.
- At report creation time, the only intended new project file is `CODEBASE_INSPECTION_REPORT.md`. The worktree was clean immediately before this creation.
