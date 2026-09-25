# SentinelAI

## Provenance-Aware Runtime Security & Governance for Long-Horizon LLM Agents

SentinelAI is a research-oriented runtime security and governance layer
designed to monitor and control tool-using LLM agents.

The system is positioned between an AI agent and the tools/data that the
agent is allowed to access. Instead of allowing an agent to directly execute
a tool call, SentinelAI evaluates each proposed action using policy,
trajectory history, data sensitivity, provenance, transformation history,
and learned risk signals before deciding whether the action should be
allowed, warned, or blocked.

---

## 1. Project Overview

Modern LLM agents can perform multi-step tasks involving files, memory,
summarization, calculations, external APIs, and communication tools.

A security problem can arise even when every individual action appears
reasonable in isolation.

For example:

    Step 1: Read a confidential document
            ↓
    Step 2: Summarize the document
            ↓
    Step 3: Store the summary in memory
            ↓
    Step 4: Retrieve the memory
            ↓
    Step 5: Paraphrase the retrieved information
            ↓
    Step 6: Send the resulting content externally

The final action may contain information derived from a sensitive source
even though the final object itself may no longer look identical to the
original source.

This gradual movement toward an unsafe outcome is the central safety-drift
problem studied by SentinelAI.

The system therefore does not evaluate only the current action.

It maintains:

- DataObject identities
- Data sensitivity
- Transformation history
- Provenance relationships
- Ordered trajectory history
- Destination trust
- Explicit security policies
- Content/sensitivity signals
- Learned risk predictions

The goal is to determine whether provenance-aware trajectory monitoring can
detect safety drift earlier than monitoring approaches that lack provenance
information.

---

# 2. Research Question

The central research question is:

> Can provenance-aware trajectory monitoring detect safety drift earlier
> than monitors that only observe the current action or trajectory without
> provenance?

The project is therefore designed as both:

1. a working runtime security system, and
2. a reproducible experimental platform.

The implementation should generate evidence for the research question rather
than relying on hardcoded demonstrations or manually selected outcomes.

---

# 3. Core Design Principle

SentinelAI is not intended to be a prompt-injection detector alone.

The core contribution is provenance-aware runtime monitoring of long-horizon
agent behavior.

The implementation follows these principles:

- Real data should be used wherever practical.
- Real transformations should be executed.
- Data lineage should be recorded automatically.
- Tool executions should produce real DataObjects.
- Agent trajectories should be persisted.
- Security decisions should be derived from actual system state.
- ML models should be trained on reproducibly generated data.
- Baselines should be implemented for comparison.
- Experiments should be reproducible.
- The dashboard should display real backend state.
- No security result should be hardcoded merely for a demonstration.

Synthetic or controlled data may be used where real confidential information
cannot ethically or legally be obtained.

---

# 4. High-Level Architecture

The core execution architecture is:

    USER TASK
        |
        v
    LLM AGENT / PLANNER
        |
        | ToolCallProposal
        v
    +---------------------------------------------+
    |              SENTINELAI MONITOR            |
    |                                             |
    |  1. Policy Check                            |
    |  2. Trajectory Update                       |
    |  3. Provenance Lookup / Graph Traversal     |
    |  4. Content / Sensitivity Analysis          |
    |  5. ML Risk / Drift Prediction              |
    |  6. Decision Fusion                         |
    +---------------------------------------------+
        |
        +------------+-------------+
        |            |             |
      ALLOW         WARN          BLOCK
        |            |             |
        v            v             X
    TOOL EXECUTOR  APPROVAL      STOP
        |
        v
    DATA OBJECT
        |
        +--------------------+
        |                    |
        v                    v
    PROVENANCE           TRAJECTORY
      GRAPH                STORE
        |                    |
        +---------+----------+
                  |
                  v
             AUDIT LOG
                  |
                  v
             WEB DASHBOARD

The agent does not directly control the tools.

The agent proposes an action.

SentinelAI evaluates the proposal.

Only after the security decision is made can the tool execute.

This separation is important because it keeps the security layer independent
from the agent and allows different agent planners, baselines, and models to
be evaluated against the same runtime.

---

# 5. Agent Architecture

The project will initially use a deterministic scripted agent.

This is intentional.

The scripted agent allows controlled and reproducible trajectories to be
generated before introducing the variability of a real LLM.

Later, the scripted planner will be replaced or supplemented with a real
function-calling LLM planner.

Both planners will use the same ToolCallProposal interface.

The architecture therefore becomes:

    Scripted Agent
          |
          v
    ToolCallProposal
          |
          v
    SentinelAI
          |
          v
    Tool Executor


and later:

    Real LLM Agent
          |
          v
    ToolCallProposal
          |
          v
    SentinelAI
          |
          v
    Tool Executor

The security layer remains unchanged.

---

# 6. Tool Registry

SentinelAI will define and control its own tool registry.

The initial tool set contains eight representative capabilities:

1. read_file()
2. summarize()
3. paraphrase()
4. calculate()
5. write_memory()
6. read_memory()
7. send_message()
8. call_external_api()

The purpose of the tools is not simply to demonstrate functionality.

Together they allow the project to create realistic transformation and
information-flow patterns such as:

    confidential source
        ↓
    summary
        ↓
    memory
        ↓
    retrieved memory
        ↓
    paraphrase
        ↓
    external communication

All tools must operate through the same ToolCallProposal contract.

---

# 7. Core Data Model

## ToolCallProposal

Every agent action is represented as a tool-call proposal.

Conceptually:

    ToolCallProposal
        - step_id
        - agent_id
        - tool_name
        - arguments
        - input_data_objects
        - timestamp

The proposal is evaluated before execution.

---

## DataObject

Every security-relevant piece of data is represented as a DataObject.

Conceptually:

    DataObject
        - object_id
        - source
        - sensitivity
        - parents
        - transformation
        - content_hash
        - embedding_ref
        - created_at
        - created_step

Examples include:

- original files
- email messages
- summaries
- paraphrases
- calculated values
- memory entries
- API results
- outbound messages

---

## Decision

Every monitored proposal produces a security decision.

Conceptually:

    Decision
        - step_id
        - verdict
        - risk_score
        - risk_probability
        - policy_hits
        - provenance_trace
        - reason

Possible verdicts:

    ALLOW
    WARN
    BLOCK

The exact outcome should be computed from the active policy, provenance,
trajectory and risk state.

---

# 8. Provenance System

Provenance is a central component of SentinelAI.

The system will represent data lineage as a directed acyclic graph (DAG).

For example:

    Source Object
          |
          v
       Summary
          |
          v
      Memory Object
          |
          v
      Paraphrase
          |
          v
    External Message

If an object is derived from multiple sources, all parents must be retained.

Example:

    Source A ----+
                |
                v
             Combined
                ^
                |
    Source B ----+

The provenance system must therefore support:

- parent-child relationships
- multiple parents
- transformation metadata
- root-source tracing
- ancestry traversal
- provenance snapshots
- provenance-based security features

NetworkX will initially be used for graph operations.

---

# 9. Trajectory System

A provenance graph describes data lineage.

A trajectory describes what the agent did and when it did it.

Example:

    Step 1  read_file       ALLOW
    Step 2  summarize       ALLOW
    Step 3  write_memory    WARN
    Step 4  read_memory     WARN
    Step 5  paraphrase      WARN
    Step 6  send_message    BLOCK

The trajectory store will preserve the ordered history of:

- proposals
- tool executions
- observations
- decisions
- generated DataObjects
- provenance relationships
- risk scores
- policy hits

The trajectory is essential for detecting gradual safety drift.

---

# 10. Policy Engine

Policies will be externalized rather than hardcoded inside the monitor.

Example policy concepts:

    CONFIDENTIAL_TO_EXTERNAL
        -> BLOCK

    INTERNAL_TO_EXTERNAL
        -> WARN

    PUBLIC_TO_EXTERNAL
        -> ALLOW

    CONFIDENTIAL_MEMORY_PERSISTENCE
        -> WARN

Policies should be configurable through YAML/JSON configuration.

The policy engine will remain separate from ML.

This allows the system to distinguish:

- deterministic policy enforcement
- heuristic risk
- learned risk
- provenance-based risk

---

# 11. Risk and ML System

The ML system will estimate security risk from features extracted from the
current action and the historical trajectory.

The initial architecture contains two conceptual stages.

## Stage A — Content / Sensitivity Signals

Potential signals include:

- PII/entity detection
- secret/credential patterns
- semantic similarity to sensitive sources
- sensitivity labels
- optional prompt-injection signals

These become content-related features.

## Stage B — Trajectory Risk Model

The trajectory model will combine:

- content features
- tool/action features
- destination features
- provenance graph features
- transformation depth
- trajectory history
- temporal information
- policy signals

The first learned models will be transparent models such as:

- Logistic Regression
- Random Forest
- Gradient Boosting / HistGradientBoosting

A large language model will not initially be fine-tuned for the risk model.

The purpose is to determine whether provenance itself contributes predictive
value.

A more complex sequence model can be considered as a later extension.

---

# 12. Dataset Strategy

The project follows a layered data strategy.

## Layer 0 — Real Public Source Data

Examples include:

- AMI meeting corpus
- Enron email corpus
- licensed PII datasets
- selected public documents

The original dataset version and checksum should be recorded.

---

## Layer 1 — Transformation Corpus

Real source data is transformed using actual system operations.

Examples:

    source document
        ↓
    summary

    source document
        ↓
    paraphrase

    source document
        ↓
    derived calculation

    source document
        ↓
    memory representation

The transformed objects become provenance-linked DataObjects.

---

## Layer 2 — Trajectory Corpus

The system executes complete scenarios.

Each trajectory records:

- ordered tool calls
- observations
- DataObject IDs
- provenance relationships
- decisions
- risk values

This becomes the main supervised learning/evaluation corpus.

---

## Layer 3 — Security Labels

Each scenario can contain labels such as:

- SAFE
- EARLY_RISK
- VIOLATION
- BENIGN_HARD_NEGATIVE

Additional metadata includes:

- violation step
- drift onset
- destination trust level
- policy hit
- final outcome

---

# 13. Scenario Families

The benchmark will contain both benign and unsafe workflows.

Core scenario families include:

- direct leakage
- summary leakage
- paraphrase leakage
- memory-mediated leakage
- long-horizon leakage
- derived-fact leakage
- mixed-source leakage
- benign internal workflows
- benign external workflows
- hard negatives

Prompt injection can later be incorporated as an auxiliary security signal.

It is not the primary research question.

---

# 14. Benchmark and Baselines

The project will compare multiple monitoring approaches.

Conceptually:

    B0 — No monitor

    B1 — Action-only rules

    B2 — Trajectory-only monitoring

    B3 — Risk model without provenance

    B4 — Provenance-aware risk model

    B5 — Full SentinelAI

The key scientific comparison is between approaches that have access to
provenance and approaches that do not.

This allows the project to investigate whether provenance provides measurable
benefit.

---

# 15. Ablation Studies

The system will support ablations such as:

- remove provenance features
- remove transformation depth
- remove memory information
- remove semantic similarity
- remove policy features
- remove trajectory history
- remove content/PII features

The purpose is to identify which components contribute to performance.

---

# 16. Evaluation Metrics

The benchmark will evaluate metrics including:

- Attack Success Rate (ASR)
- Precision
- Recall
- F1
- AUPRC
- False Positive Rate
- Early-warning lead
- Benign task success
- Runtime overhead
- Calibration error

The exact reported values must always come from executed experiments.

No results should be hardcoded into the documentation before the experiments
are completed.

---

# 17. Real LLM Integration

The real LLM will be introduced after the core runtime, provenance system,
trajectory store, policies, benchmark and initial ML system have been
validated.

The LLM will act as the planner.

It will not receive unrestricted authority over tools.

The execution flow will remain:

    LLM
      ↓
    ToolCallProposal
      ↓
    SentinelAI
      ↓
    ALLOW / WARN / BLOCK
      ↓
    Tool Executor

This ensures that the security layer remains independent of the particular
LLM provider.

---

# 18. Web Dashboard

The final system will include a real-time web dashboard.

Planned technology:

- React
- TypeScript
- FastAPI
- WebSocket/SSE
- Recharts/ECharts
- React Flow/Cytoscape

The dashboard will display actual backend state.

Planned views include:

- Overview
- Live Monitor
- Trajectory Explorer
- Provenance Graph
- Object Inspector
- Policy Center
- Experiment Runner
- Results
- Ablation
- Dataset Registry
- Model Registry
- Audit Log

A live execution may be displayed as:

    CURRENT ACTION
        summarize(object_812)

    RISK
        0.78

    POLICY HITS
        confidential source
        memory persistence

    PROVENANCE

        confidential_document
                ↓
             summary
                ↓
             memory
                ↓
           paraphrase
                ↓
         external_message

    TRAJECTORY

        1. read_file       ALLOW
        2. summarize       ALLOW
        3. write_memory    WARN
        4. read_memory     WARN
        5. paraphrase      WARN
        6. send_message    BLOCK

All values displayed by the final dashboard should originate from the
runtime and experiment systems.

---

# 19. Backend API

The final backend is expected to expose interfaces conceptually similar to:

    POST /api/sessions
    GET  /api/sessions/{id}

    POST /api/tool-proposals

    GET /api/decisions/{session_id}

    GET /api/provenance/{object_id}

    GET /api/trajectory/{session_id}

    POST /api/experiments/run
    GET  /api/experiments/{id}

    GET /api/models

    GET /api/policies
    PUT /api/policies

    GET /api/datasets

    WS /api/stream/{session_id}

These interfaces will be implemented after the core runtime contracts have
been established.

---

# 20. Reproducibility

Reproducibility is a first-class requirement.

Every research experiment should eventually record:

    experiment_id
    git_sha
    dataset_manifest_sha
    model_version
    config_hash
    random_seed
    scenario_suite
    timestamp
    metrics

The goal is that a researcher can determine exactly which code, dataset,
configuration and model produced a reported result.

A clean clone of the repository should eventually be capable of recreating
at least one published experiment.

---

# 21. Repository Structure

    sentinelai/
    │
    ├── README.md
    ├── .gitignore
    ├── .env.example
    │
    ├── docs/
    │
    ├── backend/
    │   ├── agent/
    │   ├── tools/
    │   ├── dataobjects/
    │   ├── provenance/
    │   ├── trajectory/
    │   ├── policy/
    │   ├── risk/
    │   ├── monitor/
    │   └── storage/
    │
    ├── ml/
    │   ├── data/
    │   ├── preprocessing/
    │   ├── features/
    │   ├── training/
    │   ├── evaluation/
    │   └── models/
    │
    ├── benchmark/
    │   ├── scenarios/
    │   ├── generators/
    │   ├── runners/
    │   ├── baselines/
    │   ├── metrics/
    │   └── reports/
    │
    ├── policies/
    ├── configs/
    ├── scripts/
    │
    ├── tests/
    │   ├── unit/
    │   ├── integration/
    │   ├── security/
    │   └── regression/
    │
    ├── data/
    │   ├── manifests/
    │   └── sample/
    │
    └── frontend/
        └── src/
            ├── pages/
            ├── components/
            ├── features/
            ├── hooks/
            ├── api/
            └── types/

The repository is intentionally organized around research and system
boundaries rather than individual features.

---

# 22. Development Roadmap

The implementation will proceed incrementally.

## Phase 0 — Repository and Environment

- repository structure
- development environment
- dependency management
- documentation skeleton
- testing setup

---

## Phase 1 — Core Contracts

Implement:

- ToolCallProposal
- DataObject
- Decision

Add unit tests.

---

## Phase 2 — Tools

Implement the controlled tool registry and the eight tools.

Every tool must consume and/or produce DataObjects through the common
interface.

---

## Phase 3 — Provenance

Implement:

- provenance DAG
- parent-child relationships
- multiple-parent lineage
- root tracing
- provenance traversal

---

## Phase 4 — Trajectory

Implement:

- ordered trajectory events
- proposal records
- execution records
- decision records
- persistence
- replay

---

## Phase 5 — Policy Engine

Implement:

- YAML policy loading
- policy matching
- ALLOW/WARN/BLOCK
- policy explanations
- policy tests

---

## Phase 6 — Heuristic Risk

Create an initial non-ML risk system.

This establishes a working runtime before learned models are introduced.

---

## Phase 7 — Real Data Pipeline

Integrate appropriate public datasets.

Initial candidates include:

- AMI
- Enron
- licensed PII datasets

Every dataset should have a manifest containing its source, version and
checksum.

---

## Phase 8 — Scenario Generation

Generate:

- benign workflows
- attack workflows
- different transformation depths
- long-horizon trajectories
- mixed-source trajectories
- hard negatives

---

## Phase 9 — Feature Engineering

Extract:

- action features
- content features
- destination features
- trajectory features
- provenance features
- transformation-depth features
- policy features

---

## Phase 10 — ML Training

Train the initial risk/drift models.

Create reproducible:

- training splits
- validation splits
- test splits
- model versions
- experiment metadata

Splits should avoid leakage between related source/scenario groups.

---

## Phase 11 — Baselines and Ablations

Run:

- no-monitor baseline
- action-only baseline
- trajectory-only baseline
- risk without provenance
- provenance-aware model
- full system

Then execute ablation studies.

---

## Phase 12 — Real LLM Planner

Connect a real tool-calling LLM.

The LLM will generate ToolCallProposals.

SentinelAI remains responsible for security decisions.

---

## Phase 13 — Web Dashboard

Implement:

- FastAPI backend
- React frontend
- live runtime events
- trajectory visualization
- provenance graph
- risk visualization
- audit log
- experiment results

---

## Phase 14 — Scale and Reproducibility

Increase:

- dataset size
- trajectory count
- scenario diversity
- transformation depth
- experiment automation

Evaluate whether the architecture continues to operate efficiently.