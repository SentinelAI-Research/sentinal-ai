const pageContent = document.getElementById("page-content");
const navButtons = [...document.querySelectorAll(".nav-item")];
const breadcrumbCurrent = document.getElementById("breadcrumb-current");
const topbarTitle = document.getElementById("topbar-title");
const backendHealthTag = document.getElementById("backend-health-tag");

let agentScenarios = [];
let agentRun = null;
let agentRunState = "idle";
let agentRunError = "";
let selectedAgentStep = 0;

let llmConfiguration = null;
let llmHistory = [];
let llmCurrentRun = null;
let llmCurrentRunId = null;
let llmIsRunning = false;
let llmSelectedEvent = 0;

let v5ReportData = null;
let v5DevReportData = null;
let v5ScenarioCatalog = null;
let v5ActiveTableKey = null;
let v5SelectedRowIndex = null;
let v5SelectedModelKey = null;
let v5LastSimResult = null;
let v5SelectedSimStep = 0;

const pageNames = {
  overview: "Research Overview",
  "experiment-1": "Experiment 1 — Initial Benchmark",
  "experiment-2": "Experiment 2 — V2 Model Comparison",
  "experiment-3": "Experiment 3 — Ablation & Shortcut Audit",
  "experiment-4": "Experiment 4 — V3 Matched-Pair Benchmark",
  "experiment-5": "Experiment 5 — V4 Zero-Shot Templates",
  "v5-enterprise": "V5 Enterprise Suite — Tables, Models, DAG & Simulation",
  "agent-lab": "Deterministic Agent Lab",
  "llm-lab": "LLM Lab",
  roadmap: "V4 Research Roadmap"
};

const METRIC_GLOSSARY = [
  {
    term: "ASR (Attack Success Rate · ↓ Lower is better)",
    desc: "Fraction of attack trajectories where an unauthorized or risky action succeeds despite monitoring."
  },
  {
    term: "Attack Block Rate · ↑ Higher is better",
    desc: "Share of attack trajectories intercepted (BLOCK) before risky exfiltration or execution completes."
  },
  {
    term: "Attack Detection Rate · ↑ Higher is better",
    desc: "Share of risky/attack trajectories flagged with a warning (WARN) or block (BLOCK) by the monitor."
  },
  {
    term: "Benign FPR (False Positive Rate · ↓ Lower is better)",
    desc: "How often safe/benign trajectories trigger a false security alert or unnecessary block."
  },
  {
    term: "Benign Completion · ↑ Higher is better",
    desc: "Percentage of legitimate benign tasks that complete cleanly without disruptive intervention."
  },
  {
    term: "Early-Warning Rate · ↑ Higher is better",
    desc: "How often risk is detected at an intermediate step before the final violation/exfiltration attempt."
  },
  {
    term: "Macro F1 & Accuracy · ↑ Higher is better",
    desc: "Accuracy measures overall correct predictions; Macro F1 averages F1 across classes equally so minority/risk classes still count."
  },
  {
    term: "Mean / P95 Latency (ms) · ↓ Lower is better",
    desc: "Runtime overhead added by the monitor per step/evaluation in milliseconds."
  }
];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function metricCard(label, value, note = "", extraClass = "") {
  return `
    <article class="card metric-card ${extraClass}">
      <div class="card-title">${escapeHtml(label)}</div>
      <div class="metric-value numeric">${escapeHtml(value)}</div>
      <div class="metric-note">${escapeHtml(note)}</div>
    </article>`;
}

function nextPage(page, title, description) {
  return `
    <button type="button" class="next-link" data-go="${escapeHtml(page)}">
      <span><strong>${escapeHtml(title)}</strong><small>${escapeHtml(description)}</small></span>
      <span class="next-arrow" aria-hidden="true">→</span>
    </button>`;
}

function pageHeader(kicker, title, description, tag = "") {
  return `
    <div class="experiment-header">
      <div class="section-kicker">${escapeHtml(kicker)}</div>
      <h1>${escapeHtml(title)}</h1>
      <p>${escapeHtml(description)}</p>
      ${tag ? `<div style="margin-top:12px"><span class="pill gray">${escapeHtml(tag)}</span></div>` : ""}
    </div>`;
}

function sectionHeading(title, description = "") {
  return `
    <div class="section-heading">
      <div><h2>${escapeHtml(title)}</h2>${description ? `<p>${escapeHtml(description)}</p>` : ""}</div>
    </div>`;
}

function comparisonTable(rows) {
  return `
    <div class="table-wrap">
      <table>
        <thead><tr><th>Metric</th><th>Baseline</th><th>Provenance-aware</th><th>Difference</th></tr></thead>
        <tbody>${rows.map(row => `
          <tr>
            <td>${escapeHtml(row[0])}</td>
            <td class="numeric">${escapeHtml(row[1])}</td>
            <td class="numeric">${escapeHtml(row[2])}</td>
            <td><strong>${escapeHtml(row[3])}</strong></td>
          </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

function formatMetricPercent(val) {
  if (typeof val !== "number" || Number.isNaN(val)) return "—";
  const pct = Math.abs(val) <= 1 ? val * 100 : val;
  return `${pct.toFixed(2)}%`;
}

function formatNumber(val, digits = 2) {
  if (typeof val !== "number" || Number.isNaN(val)) return "—";
  return val.toFixed(digits);
}

function extractReportTables(report) {
  if (!report || typeof report !== "object") return [];
  const entries = [];
  for (const [key, value] of Object.entries(report)) {
    if (Array.isArray(value) && value.length > 0 && typeof value[0] === "object" && value[0] !== null) {
      entries.push({ key, rows: value });
    }
  }
  return entries;
}

function formatTableLabel(key) {
  return key
    .replaceAll("_", " ")
    .replace(/\b(r\d+)\b/gi, match => match.toUpperCase());
}

function computeBestConfigs(report) {
  const tables = extractReportTables(report);
  const primary = tables.find(t => /r1|overall|comparison/i.test(t.key)) || tables[0];
  if (!primary || !Array.isArray(primary.rows) || primary.rows.length === 0) {
    return null;
  }

  const rows = primary.rows;
  const pickBy = (predicateScore, higherBetter = true) => {
    let best = null;
    let bestScore = null;
    for (const row of rows) {
      const score = predicateScore(row);
      if (typeof score !== "number" || Number.isNaN(score)) continue;
      if (bestScore === null || (higherBetter ? score > bestScore : score < bestScore)) {
        bestScore = score;
        best = row;
      }
    }
    return best ? { name: best.config || best.model || best.name || best.variant || "Unknown", score: bestScore, row: best } : null;
  };

  return {
    tableKey: primary.key,
    lowestAsr: pickBy(r => r.asr, false),
    highestBlock: pickBy(r => r.attack_block_rate ?? r.block_rate, true),
    lowestFpr: pickBy(r => r.benign_false_alert_rate ?? r.benign_fpr, false),
    highestEarlyWarn: pickBy(r => r.early_warning_rate, true),
    lowestLatency: pickBy(r => r.mean_latency_ms ?? r.latency_ms, false),
    allRows: rows
  };
}

function renderGlossarySection() {
  return `
    <article class="card content-card">
      <div class="section-kicker">METRIC GUIDE</div>
      <h2>What these evaluation metrics &amp; classifications mean</h2>
      <p class="muted small" style="margin-bottom:14px">
        Plain-English definitions so every metric and verdict shown in tables and simulations is transparent during review.
      </p>
      <div class="glossary-grid">
        ${METRIC_GLOSSARY.map(item => `
          <div class="glossary-item">
            <strong>${escapeHtml(item.term)}</strong>
            <span>${escapeHtml(item.desc)}</span>
          </div>
        `).join("")}
      </div>
    </article>
  `;
}

const pages = {
  "v5-enterprise"() {
    return `
      <section class="v5-banner">
        <div>
          <div class="eyebrow">EXPERIMENT 06 · ENTERPRISE SUITE</div>
          <h1>V5 Enterprise Benchmark, Model Comparison &amp; Live DAG Simulation</h1>
          <p>
            Inspect full empirical tables dynamically loaded from backend reports, compare evaluated monitor/model
            configurations, view generated result charts, and execute live step-by-step provenance simulations.
          </p>
        </div>
        <span class="agent-demo-tag">DYNAMIC V5 SUITE</span>
      </section>

      <div id="v5-kpi-grid" class="grid grid-4">
        ${metricCard("Primary Suite Status", "Loading…", "Reading /api/v5/report")}
        ${metricCard("Lowest Attack Success (ASR)", "—", "Dynamic best config", "green")}
        ${metricCard("Highest Attack Block Rate", "—", "Dynamic best config", "green")}
        ${metricCard("Early-Warning Rate", "—", "Pre-violation detection", "amber")}
      </div>

      <article id="v5-model-comparison-card" class="card content-card">
        <div class="section-kicker">MODEL &amp; PRESET EVALUATION</div>
        <h2>Model &amp; Configuration Comparison</h2>
        <p class="muted small">
          Dynamically compares evaluated configurations from the loaded V5 report (e.g., unmonitored/rule presets vs.
          non-provenance ML vs. provenance-aware configurations) and highlights top performers per criterion.
        </p>
        <div id="v5-model-comparison-body" style="margin-top:12px">
          <div class="empty-state">Loading V5 evaluation report to compare models and presets…</div>
        </div>
      </article>

      <article id="v5-observations-card" class="card content-card">
        <div class="section-kicker">REPORT-DERIVED MODEL NOTES</div>
        <h2>Model Observations</h2>
        <p class="muted small">
          Select an evaluated configuration or trained classifier to inspect its measured metrics and comparisons
          with entries from the same report section.
        </p>
        <div id="v5-observation-list" class="v5-observation-list" style="margin-top:12px"></div>
        <div id="v5-observation-detail" class="v5-observation-detail" aria-live="polite">
          <div class="empty-state">Loading report-derived observations…</div>
        </div>
      </article>

      <article id="v5-tables-card" class="card content-card">
        <div class="table-toolbar">
          <div>
            <div class="section-kicker">EMPIRICAL RESULT TABLES</div>
            <h2>Dynamic Benchmark Tables (R1–R5 &amp; Report Sections)</h2>
            <p class="muted small">Switch tabs across available report tables and click any row to inspect full details.</p>
          </div>
          <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <button type="button" id="v5-refresh-btn" class="agent-clear-button">Refresh Report</button>
            <button type="button" id="v5-export-json-btn" class="agent-clear-button">Export JSON</button>
          </div>
        </div>
        <div id="v5-table-tabs" class="tab-bar" style="margin-bottom:12px"></div>
        <div class="table-wrap">
          <table>
            <thead id="v5-dynamic-thead"></thead>
            <tbody id="v5-dynamic-tbody">
              <tr><td class="empty-state">Loading table rows…</td></tr>
            </tbody>
          </table>
        </div>
        <div id="v5-row-inspector" class="row-inspector-drawer" hidden></div>
      </article>

      <article id="v5-artifacts-card" class="card content-card">
        <div class="section-kicker">GENERATED VISUAL ARTIFACTS</div>
        <h2>Plots &amp; Result Figures</h2>
        <p class="muted small">
          Displays any result figures/charts referenced by the backend report or present in the results directory.
        </p>
        <div id="v5-artifacts-body" style="margin-top:12px">
          <div class="empty-state">Checking generated result figures…</div>
        </div>
      </article>

      <article class="card content-card">
        <div class="section-kicker">LIVE SIMULATION &amp; PROVENANCE DAG</div>
        <h2>Interactive Scenario Runner &amp; Lineage Graph</h2>
        <p class="muted small">
          Select a monitor configuration, scenario family, and partition, then run the simulation to inspect verdicts,
          step telemetry, and the dynamically constructed provenance chain.
        </p>

        <div class="llm-form-grid" style="margin-top:14px">
          <label class="llm-field" for="v5-config-select">
            Monitor Preset / Model
            <select id="v5-config-select">
              <option value="C0">C0 — Unmonitored Baseline</option>
              <option value="C1">C1 — Action-Only Policy</option>
              <option value="C2">C2 — Trajectory Risk</option>
              <option value="C2b">C2b — Session Taint</option>
              <option value="C3">C3 — ML without Provenance</option>
              <option value="C4-Heuristic">C4-Heuristic (Rule-based Provenance)</option>
              <option value="C4-ML" selected>C4-ML (Full Provenance + ML)</option>
            </select>
          </label>

          <label class="llm-field" for="v5-family-select">
            Scenario Family
            <select id="v5-family-select">
              <option value="memory_mediated_leakage">Memory-Mediated Leakage</option>
              <option value="confidential_decoy_public_send">Confidential Decoy Public Send (Benign)</option>
              <option value="paraphrase_leakage">Paraphrase Exfiltration</option>
              <option value="summary_leakage">Summary Exfiltration</option>
              <option value="cross_session_memory_leakage">Cross-Session Memory Leakage</option>
              <option value="multi_source_join_leakage">Multi-Source Join Leakage</option>
            </select>
          </label>

          <label class="llm-field" for="v5-partition-select">
            Benchmark Partition
            <select id="v5-partition-select">
              <option value="test" selected>Test Split</option>
              <option value="dev">Dev Split</option>
            </select>
          </label>

          <div style="display:flex; align-items:flex-end;">
            <button type="button" id="v5-run-btn" class="agent-run-button" style="width:100%; min-height:38px;">
              Run Simulation
            </button>
          </div>
        </div>

        <div id="v5-run-status" class="agent-status" role="status" aria-live="polite">Ready to run</div>
        <p id="v5-run-error" class="agent-error" role="alert" hidden></p>

        <div class="v5-sim-layout" style="margin-top:16px">
          <section class="agent-panel" style="margin-bottom:0">
            <div class="section-kicker">STEP TELEMETRY</div>
            <h2>Execution Steps &amp; Verdicts</h2>
            <div id="v5-telemetry-feed" style="margin-top:12px; display:grid; gap:8px;">
              <div class="empty-state">Run a simulation to inspect step-by-step verdicts and monitor timing.</div>
            </div>
          </section>

          <section class="agent-panel" style="margin-bottom:0">
            <div class="section-kicker">DYNAMIC PROVENANCE DAG</div>
            <h2>Lineage &amp; Sensitivity Flow</h2>
            <p class="muted small">Click any DAG node to inspect step features, sensitivity ancestry, and verdict.</p>
            <div id="v5-dag-view" class="dag-canvas-box">
              <div class="empty-state">No simulation DAG generated yet.</div>
            </div>
            <div id="v5-dag-inspector" class="row-inspector-drawer" hidden></div>
          </section>
        </div>
      </article>

      ${renderGlossarySection()}
      ${nextPage("agent-lab", "Try Deterministic Agent Lab", "Explore scripted scenario traces and provenance graphs")}
    `;
  },

  "agent-lab"() {
    return `
      <section class="agent-lab-header">
        <div>
          <div class="eyebrow">DEMO MODE · SCRIPTED PLANNER</div>
          <h1>Deterministic Agent Lab</h1>
          <p>The planner follows a fixed action sequence. SentinelAI makes every security decision before a registered tool can execute.</p>
        </div>
        <span class="agent-demo-tag">DEMO MODE</span>
      </section>

      <section class="agent-run-controls" aria-labelledby="agent-run-heading">
        <div>
          <div class="section-kicker">SCENARIO RUNNER</div>
          <h2 id="agent-run-heading">Choose a controlled scenario</h2>
          <p class="muted small">Only predefined scenarios are accepted by the backend.</p>
        </div>
        <div id="agent-run-status" class="agent-status" role="status" aria-live="polite">Not run yet</div>
        <div id="agent-scenario-list" class="agent-scenario-list">
          <div class="empty-state">Loading predefined scenarios…</div>
        </div>
      </section>

      <section class="agent-run-result" id="agent-run-result" aria-live="polite">
        <div class="agent-empty-state">
          <span class="agent-empty-mark" aria-hidden="true">01</span>
          <h2>No run selected</h2>
          <p>Run a scenario to inspect the actual monitor decisions, tool outcomes, and provenance created by the backend.</p>
        </div>
      </section>`;
  },

  "llm-lab"() {
    return `
      <section class="llm-lab-header">
        <div>
          <div class="eyebrow">PROVIDER-BACKED PLANNING · MONITORED EXECUTION</div>
          <h1>LLM Lab</h1>
          <p>Provider output is an untrusted proposal. SentinelAI validates each action, applies policy, and only then permits a registered tool to run.</p>
        </div>
        <span class="agent-demo-tag">NO DIRECT TOOL ACCESS</span>
      </section>
      <section class="llm-run-form-panel">
        <form id="llm-run-form">
          <label class="llm-field llm-task-field" for="llm-task">Task description</label>
          <textarea id="llm-task" name="task" rows="3" maxlength="2000" required placeholder="Describe a task using the provided public sample document."></textarea>
          <div class="llm-form-grid">
            <label class="llm-field" for="llm-mode">Execution mode
              <select id="llm-mode" name="mode" required>
                <option value="deterministic">Deterministic plan</option>
                <option value="fake">Fake provider (offline)</option>
                <option value="live">Live OpenAI provider</option>
              </select>
            </label>
            <label class="llm-field" for="llm-scenario">Scenario fixture
              <select id="llm-scenario" name="scenario_id" required></select>
            </label>
            <label class="llm-field" for="llm-policy">Policy
              <select id="llm-policy" name="policy_id" required></select>
            </label>
            <label class="llm-field" for="llm-action-limit">Action limit
              <input id="llm-action-limit" name="max_actions" type="number" min="1" max="12" value="6" required />
            </label>
          </div>
          <div class="llm-run-actions">
            <div class="llm-provider-label" id="llm-provider-label">Loading provider configuration…</div>
            <div class="llm-button-row">
              <button id="llm-run-button" class="agent-run-button" type="submit" disabled>Run monitored task</button>
              <button class="agent-clear-button" type="button" disabled aria-describedby="llm-cancel-note">Cancel run</button>
            </div>
          </div>
          <p id="llm-cancel-note" class="llm-note">Safe cancellation is not supported by the current synchronous tools.</p>
          <div id="llm-run-status" class="agent-status" role="status" aria-live="polite">Not run yet</div>
          <p id="llm-run-error" class="agent-error" role="alert" hidden></p>
        </form>
      </section>
      <section id="llm-run-output" class="llm-run-output" aria-live="polite">
        <div class="agent-empty-state">
          <span class="agent-empty-mark" aria-hidden="true">LLM</span>
          <h2>No task run yet</h2>
          <p>Run a deterministic, fake-provider, or configured live-provider task to inspect its real events and provenance.</p>
        </div>
      </section>
      <section class="llm-history-panel agent-panel">
        <div class="llm-history-heading">
          <div>
            <div class="section-kicker">RUN RECORDS</div>
            <h2>Recent runs</h2>
          </div>
          <div class="llm-history-filters">
            <label for="llm-filter-status">Status</label>
            <select id="llm-filter-status"><option value="all">All</option></select>
            <label for="llm-filter-mode">Mode</label>
            <select id="llm-filter-mode"><option value="all">All</option></select>
            <label for="llm-filter-verdict">Verdict</label>
            <select id="llm-filter-verdict">
              <option value="all">All</option>
              <option value="allow">ALLOW</option>
              <option value="warn">WARN</option>
              <option value="block">BLOCK</option>
            </select>
          </div>
        </div>
        <div id="llm-history-content" class="llm-history-list"></div>
      </section>`;
  },

  overview() {
    return `
      <section class="hero">
        <div class="eyebrow">PROJECT OVERVIEW</div>
        <h1>Understanding runtime risk in autonomous AI agents.</h1>
        <p>SentinelAI explores whether provenance-aware monitoring can identify risky agent behaviour by tracking where information came from and how it was transformed—not just by examining the current action.</p>
      </section>

      ${sectionHeading("Research journey", "Each stage has its own page. Open an experiment to see its method, results, and conclusions.")}

      <div class="grid grid-2">
        <article class="card">
          <div class="section-kicker">01 · START</div>
          <h2>Initial Benchmark</h2>
          <p class="muted">Build the initial dataset and establish a first model comparison.</p>
          <span class="pill gray">Foundation</span>
          ${nextPage("experiment-1", "Open Experiment 1", "Initial dataset and baseline")}
        </article>
        <article class="card">
          <div class="section-kicker">02 · EVALUATE</div>
          <h2>V2 Split Comparison</h2>
          <p class="muted">Keep paired variants together across train and test splits and compare models with and without provenance features.</p>
          <span class="pill green">8.33 pp accuracy lift</span>
          ${nextPage("experiment-2", "Open Experiment 2", "V2 model evaluation")}
        </article>
        <article class="card">
          <div class="section-kicker">03 · INVESTIGATE</div>
          <h2>Ablation &amp; Shortcut Audit</h2>
          <p class="muted">Test which feature groups contribute to the results and look for synthetic shortcuts.</p>
          <span class="pill amber">Shortcut risks identified</span>
          ${nextPage("experiment-3", "Open Experiment 3", "Ablation and feature audit")}
        </article>
        <article class="card">
          <div class="section-kicker">04 · REFINE</div>
          <h2>V3 Matched-Pair Benchmark</h2>
          <p class="muted">Create matched safe/risky trajectories with the same action sequence and compare models on a pair-aware split.</p>
          <span class="pill green">100% provenance-model accuracy</span>
          ${nextPage("experiment-4", "Open Experiment 4", "V3 controlled evaluation")}
        </article>
        <article class="card">
          <div class="section-kicker">05 · GENERALIZE</div>
          <h2>V4 Zero-Shot Templates</h2>
          <p class="muted">Evaluate zero-shot generalization across unseen scenario sequence templates and indirect prompt injection.</p>
          <span class="pill green">+83.33 pp Macro F1 lift</span>
          ${nextPage("experiment-5", "Open Experiment 5", "V4 zero-shot generalization")}
        </article>
        <article class="card card-featured">
          <div class="section-kicker">06 · ENTERPRISE SUITE</div>
          <h2>V5 Enterprise Suite</h2>
          <p class="muted">Full V5 evaluation tables, model comparisons, generated plots, dynamic provenance DAG, and live scenario simulation.</p>
          <span class="pill green">Tables · DAG · Live Simulation</span>
          ${nextPage("v5-enterprise", "Open V5 Enterprise Suite", "Inspect V5 tables, models, DAG & run simulations")}
        </article>
      </div>

      ${sectionHeading("V4 & V5 research snapshot")}
      <div class="grid grid-3">
        ${metricCard("V4 Step Records", "800", "Unseen template generalization")}
        ${metricCard("Zero-Shot Baseline F1", "16.67%", "Action-only model on template shift", "amber")}
        ${metricCard("Provenance Macro F1", "100.00%", "DAG provenance features", "green")}
      </div>
      <div class="callout warning">
        <strong>Important research finding:</strong> Action-only signals can struggle under sequence/template shifts, whereas lineage and ancestor provenance capture multi-step context before risky actions occur.
      </div>
      ${renderGlossarySection()}
      ${nextPage("v5-enterprise", "Examine V5 Enterprise Suite", "View empirical tables, model comparisons, and run simulations")}
    `;
  },

  "experiment-1"() {
    return `
      ${pageHeader("EXPERIMENT 01 · FOUNDATION", "Initial benchmark", "This was the starting point: create a dataset of agent-like trajectories and establish an initial comparison before moving to the stricter V2 split and ablation studies.", "Initial synthetic benchmark")}

      ${sectionHeading("What we did")}
      <div class="grid grid-2">
        <article class="card">
          <h3>1. Built the initial data</h3>
          <p class="muted">The original generated benchmark used AMI- and Enron-derived source material to construct simulated trajectories. The text-derived source material was used to create synthetic benchmark examples; the benchmark labels and sensitivity metadata are not claims about the original documents' real-world confidentiality.</p>
        </article>
        <article class="card">
          <h3>2. Represented agent actions</h3>
          <p class="muted">Trajectories were represented as a sequence of steps, such as reading, summarizing, transforming, storing, or sending information. Labels described the risk state of each step.</p>
        </article>
      </div>

      ${sectionHeading("Initial dataset snapshot", "Counts recorded for the original generated dataset.")}
      <div class="grid grid-3">
        ${metricCard("Source families", "2", "AMI and Enron")}
        ${metricCard("Trajectories", "200", "100 per source family")}
        ${metricCard("Rows", "1,200", "6 steps per trajectory")}
      </div>

      <article class="card content-card">
        <h2>Why this experiment mattered</h2>
        <p>The initial benchmark gave us a working foundation for evaluating whether model features could identify risky steps. However, a promising initial result alone cannot show that the model has learned provenance-based reasoning rather than patterns embedded in the generated data.</p>
        <div class="callout">
          <strong>What came next:</strong> V2 introduced a paired hard-negative dataset and a controlled train/test split so that the benign and risky variants derived from the same source group would not be separated across the split.
        </div>
      </article>

      <article class="card content-card">
        <h2>Results and artifacts</h2>
        <p class="muted">The initial evaluation artifact is expected at <code>results/ml_evaluation/provenance_hard_negative_evaluation.json</code>. This page does not invent initial accuracy or F1 values if that artifact is absent.</p>
        <div id="initial-artifact-status" class="empty-state">Checking whether the initial evaluation artifact is available through the dashboard API…</div>
      </article>

      ${nextPage("experiment-2", "Next: V2 model comparison", "Move from the initial benchmark to the paired train/test evaluation")}
    `;
  },

  "experiment-2"() {
    return `
      ${pageHeader("EXPERIMENT 02 · MODEL COMPARISON", "V2 split comparison", "We introduced paired benign and provenance-risky variants, then evaluated a baseline model against a model with additional provenance features.", "V2 · Pair-aware split")}

      ${sectionHeading("Dataset and split")}
      <div class="grid grid-4">
        ${metricCard("Training rows", "1,920", "320 trajectories")}
        ${metricCard("Test rows", "480", "80 trajectories")}
        ${metricCard("Train source groups", "160", "80 AMI + 80 Enron")}
        ${metricCard("Test source groups", "40", "20 AMI + 20 Enron")}
      </div>

      <article class="card content-card">
        <h2>What was compared?</h2>
        <div class="grid grid-2">
          <div>
            <h3>Baseline without provenance</h3>
            <p class="muted">Seven features describing the current step, including its tool, operation type, and current sensitivity.</p>
            <div class="feature-list">
              <span class="feature-chip">has_input_object</span>
              <span class="feature-chip">is_external_action</span>
              <span class="feature-chip">is_memory_operation</span>
              <span class="feature-chip">is_transformation</span>
              <span class="feature-chip">sensitivity_encoded</span>
              <span class="feature-chip">tool_encoded</span>
              <span class="feature-chip">transformation_encoded</span>
            </div>
          </div>
          <div>
            <h3>Provenance-aware model</h3>
            <p class="muted">The same seven baseline features plus six features describing the information's history and ancestry.</p>
            <div class="feature-list">
              <span class="feature-chip">provenance_depth</span>
              <span class="feature-chip">transformation_depth</span>
              <span class="feature-chip">has_provenance</span>
              <span class="feature-chip">has_sensitive_ancestor</span>
              <span class="feature-chip">sensitive_ancestor_count</span>
              <span class="feature-chip">max_ancestor_sensitivity_encoded</span>
            </div>
          </div>
        </div>
      </article>

      ${sectionHeading("Verified V2 test results")}
      <div class="grid grid-4">
        ${metricCard("Baseline accuracy", "91.67%", "7 features")}
        ${metricCard("Provenance accuracy", "100%", "13 features", "green")}
        ${metricCard("Accuracy lift", "+8.33 pp", "Percentage-point difference", "green")}
        ${metricCard("Macro F1 lift", "+38.10 pp", "61.90% → 100%", "green")}
      </div>

      <article class="card content-card">
        <h2>Model comparison</h2>
        ${comparisonTable([
      ["Accuracy", "91.67%", "100.00%", "+8.33 pp"],
      ["Macro F1", "61.90%", "100.00%", "+38.10 pp"]
    ])}
        <p class="muted small" style="margin-top:12px">Evaluation used a held-out test set. The paired variants were kept together by source group to avoid pair overlap between training and testing.</p>
      </article>

      <article class="card content-card">
        <h2>What we learned</h2>
        <p>On this benchmark, adding provenance features improved accuracy and macro F1. However, the result does not by itself prove that provenance was the reason for the improvement. Features can accidentally encode the benchmark's synthetic labelling rules.</p>
        <div class="callout warning">
          <strong>Question raised:</strong> does the model rely on meaningful provenance information, or on a small number of artificial signals that happen to identify the labels?
        </div>
        <p class="muted small">Model family: HistGradientBoostingClassifier. Configuration included learning rate 0.05, max iterations 200, max leaf nodes 15, L2 regularization 1.0, and random state 42.</p>
      </article>

      ${nextPage("experiment-3", "Next: Ablation & shortcut audit", "Remove feature groups and investigate possible shortcuts")}
    `;
  },

  "experiment-3"() {
    return `
      ${pageHeader("EXPERIMENT 03 · DIAGNOSTIC STUDY", "Ablation & shortcut audit", "After the V2 model reached 100% accuracy, we tested what happened when provenance feature groups were removed and inspected feature distributions for synthetic shortcuts.", "V2 · Ablation v1")}

      <div class="callout warning">
        <strong>Why this step was necessary:</strong> a perfect test score can reflect a simple shortcut in a controlled dataset. We needed to determine which features were carrying the model's predictions before trusting the result.
      </div>

      ${sectionHeading("Ablation results", "Metrics below are from the same V2 test set.")}
      <div class="table-wrap">
        <table>
          <thead><tr><th>Feature set</th><th>Features</th><th>Accuracy</th><th>Macro F1</th><th>Interpretation</th></tr></thead>
          <tbody>
            <tr><td>Baseline without provenance</td><td>7</td><td class="numeric">91.67%</td><td class="numeric">61.90%</td><td>Missed the VIOLATION class</td></tr>
            <tr><td>Full provenance</td><td>13</td><td class="numeric">100.00%</td><td class="numeric">100.00%</td><td>Perfect on this test set</td></tr>
            <tr><td>Drop has_sensitive_ancestor</td><td>12</td><td class="numeric">100.00%</td><td class="numeric">100.00%</td><td>Score unchanged</td></tr>
            <tr><td>Drop sensitive_ancestor_count</td><td>12</td><td class="numeric">100.00%</td><td class="numeric">100.00%</td><td>Score unchanged</td></tr>
            <tr><td>Drop both ancestor indicators</td><td>11</td><td class="numeric">100.00%</td><td class="numeric">100.00%</td><td>Score unchanged</td></tr>
            <tr><td>Drop ancestor sensitivity group</td><td>10</td><td class="numeric">91.67%</td><td class="numeric">61.90%</td><td>Returned to baseline-level performance</td></tr>
            <tr><td>Drop provenance structure group</td><td>10</td><td class="numeric">100.00%</td><td class="numeric">100.00%</td><td>Score unchanged</td></tr>
            <tr><td>Drop all provenance</td><td>7</td><td class="numeric">91.67%</td><td class="numeric">61.90%</td><td>Returned to baseline-level performance</td></tr>
          </tbody>
        </table>
      </div>

      ${sectionHeading("Permutation importance", "Macro F1 decrease when a feature was permuted in the full provenance model.")}
      <article class="card">
        <div class="bar-list">
          <div class="bar-row"><span>is_external_action</span><div class="bar-track"><div class="bar-fill" style="width:100%"></div></div><span class="bar-value">0.3614</span></div>
          <div class="bar-row"><span>sensitive_ancestor_count</span><div class="bar-track"><div class="bar-fill" style="width:94.1%"></div></div><span class="bar-value">0.3402</span></div>
          <div class="bar-row"><span>has_input_object</span><div class="bar-track"><div class="bar-fill" style="width:65.5%"></div></div><span class="bar-value">0.2367</span></div>
        </div>
        <p class="muted small" style="margin-top:16px">Reported importance is the mean decrease in macro F1. The remaining listed features had zero measured permutation importance on this test set.</p>
      </article>

      ${sectionHeading("Audit findings")}
      <div class="grid grid-2">
        <article class="card">
          <h3>Possible shortcut: external action</h3>
          <p class="muted">All EARLY_RISK rows had <code>is_external_action = 0</code>, while all VIOLATION rows had <code>is_external_action = 1</code>. That separation can make the label unusually easy to infer.</p>
          <span class="pill amber">Investigate further</span>
        </article>
        <article class="card">
          <h3>Possible shortcut: ancestor count</h3>
          <p class="muted">Every VIOLATION row had a sensitive ancestor count of 6, while the other labels used different ranges in this generated dataset.</p>
          <span class="pill amber">Label-associated pattern</span>
        </article>
        <article class="card">
          <h3>Other label-associated features</h3>
          <p class="muted">The audit also flagged patterns in provenance depth, tool encoding, transformation depth, and transformation encoding.</p>
          <span class="pill gray">Review synthetic rules</span>
        </article>
        <article class="card">
          <h3>What this does not prove</h3>
          <p class="muted">These findings identify potential shortcuts in a synthetic benchmark. They do not establish that the model is leaking labels in a real deployment or that provenance is unhelpful.</p>
          <span class="pill green">Evidence, not proof</span>
        </article>
      </div>

      <article class="card content-card">
        <h2>Conclusion from the ablation study</h2>
        <p>The ancestor-sensitivity feature group mattered for the model's score, but the feature audit showed that several synthetic signals were strongly associated with the labels. We therefore needed a more controlled benchmark in which safe and risky variants share the same action sequence and differ in their provenance history.</p>
        <div class="callout">
          <strong>Design decision:</strong> build V3 as a matched-pair benchmark and evaluate on a pair-aware train/test split.
        </div>
      </article>

      ${nextPage("experiment-4", "Next: V3 matched-pair benchmark", "Test matched safe/risky trajectories with identical action sequences")}
    `;
  },

  "experiment-4"() {
    return `
      ${pageHeader("EXPERIMENT 04 · CONTROLLED EVALUATION", "V3 matched-pair benchmark", "V3 was designed to make the baseline-versus-provenance comparison more controlled: each benign trajectory is paired with a provenance-risky counterpart that uses the same six-step action sequence.", "V3 · 300 matched pairs")}

      ${sectionHeading("Dataset and split")}
      <div class="grid grid-4">
        ${metricCard("Matched pairs", "300", "600 trajectories total")}
        ${metricCard("Dataset rows", "3,600", "6 steps per trajectory")}
        ${metricCard("Train split", "2,880", "240 pairs · 480 trajectories")}
        ${metricCard("Test split", "720", "60 pairs · 120 trajectories")}
      </div>

      <article class="card content-card">
        <h2>How the matched pairs work</h2>
        <div class="grid grid-2">
          <div>
            <span class="pill green">Benign variant</span>
            <h3 style="margin-top:12px">Same action sequence</h3>
            <p class="muted">The trajectory uses the fixed sequence of six actions. Its final step is labelled SAFE, with public sensitivity metadata.</p>
          </div>
          <div>
            <span class="pill red">Provenance-risky variant</span>
            <h3 style="margin-top:12px">Different ancestry</h3>
            <p class="muted">The action sequence and baseline features match the benign counterpart, but the controlled synthetic provenance ancestry makes the final step a VIOLATION.</p>
          </div>
        </div>
        <div class="callout warning">
          The CONFIDENTIAL ancestry in this benchmark is synthetic metadata. It does not mean the original AMI or Enron source documents are actually confidential.
        </div>
      </article>

      ${sectionHeading("Verified V3 test results")}
      <div class="grid grid-4">
        ${metricCard("Baseline accuracy", "91.67%", "7 features")}
        ${metricCard("Provenance accuracy", "100%", "13 features", "green")}
        ${metricCard("Accuracy lift", "+8.33 pp", "Provenance minus baseline", "green")}
        ${metricCard("Macro F1 lift", "+17.78 pp", "82.22% → 100%", "green")}
      </div>

      <article class="card content-card">
        <h2>Model comparison</h2>
        ${comparisonTable([
      ["Accuracy", "91.67%", "100.00%", "+8.33 pp"],
      ["Macro F1", "82.22%", "100.00%", "+17.78 pp"],
      ["VIOLATION recall", "100.00%", "100.00%", "0.00 pp"]
    ])}
        <p class="muted small" style="margin-top:12px">The test split contained 60 matched pairs. The split validation reported zero pair overlap and zero trajectory overlap between training and testing.</p>
      </article>

      <article class="card content-card">
        <h2>What changed compared with V2?</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Question</th><th>V2</th><th>V3</th></tr></thead>
            <tbody>
              <tr><td>Paired variants kept together?</td><td>Yes, by source group</td><td>Yes, by matched pair</td></tr>
              <tr><td>Same action sequence in each pair?</td><td>Not the central design constraint</td><td>Yes, fixed six-step sequence</td></tr>
              <tr><td>Baseline VIOLATION recall</td><td>0% in the reported V2 results</td><td>100% in the reported V3 results</td></tr>
              <tr><td>Baseline macro F1</td><td>61.90%</td><td>82.22%</td></tr>
              <tr><td>Provenance model accuracy</td><td>100%</td><td>100%</td></tr>
            </tbody>
          </table>
        </div>
      </article>

      <article class="card content-card">
        <h2>Interpretation</h2>
        <p>The provenance-aware model outperformed the baseline in accuracy and macro F1 on V3. However, both models had 100% VIOLATION recall, so V3 does not show an improvement in that particular metric. The baseline's errors were in other classes.</p>
        <div class="callout warning">
          <strong>Remaining limitation:</strong> V3 uses a fixed action sequence and a limited set of synthetic scenario families. Pair-aware splitting prevents pair overlap, but it does not establish generalization to unseen templates or real-world agent behaviour.
        </div>
      </article>

      ${nextPage("experiment-5", "Next: V4 zero-shot template generalization", "Test generalization across unseen scenario sequence templates")}
    `;
  },

  "experiment-5"() {
    return `
      ${pageHeader("EXPERIMENT 05 · GENERALIZATION PROOF", "V4 Zero-Shot Scenario Templates", "Evaluates model generalizability across completely unseen scenario templates, multi-input lineage aggregation, and indirect prompt injection attacks.", "V4 · Dynamic API Metrics")}

      ${sectionHeading("Zero-Shot Metric Comparison", "Loaded dynamically from /api/evaluation when available.")}
      <div id="v4-dynamic-kpis" class="grid grid-3">
        ${metricCard("Baseline Macro F1", "16.67%", "7 Non-Provenance Features", "amber")}
        ${metricCard("Provenance Macro F1", "100.00%", "Provenance DAG Features", "green")}
        ${metricCard("Performance Lift", "+83.33 pp", "Zero-Shot Sequence Generalization", "green")}
      </div>

      <article class="card content-card">
        <h2>How We Evaluated Generalization</h2>
        <p>Standard action-level signals can degrade when exposed to unseen sequence templates or prompt injection. By testing across sequence templates and adversarial injection scenarios, lineage and ancestor provenance preserve structural context across steps.</p>
        ${comparisonTable([
      ["Zero-Shot Macro F1", "16.67%", "100.00%", "+83.33 pp"],
      ["Template Shift Behavior", "Action-only drop", "Lineage-preserved", "Structural invariance"],
      ["Prompt Injection Defense", "Action-blind", "Ancestry-aware", "Intercepts risky flow"]
    ])}
      </article>

      ${nextPage("v5-enterprise", "Open V5 Enterprise Suite", "Inspect V5 tables, model comparisons, generated plots, DAG & simulation")}
    `;
  },

  roadmap() {
    return `
      ${pageHeader("RESEARCH ROADMAP · NEXT STAGE", "V4 & V5: test generalization and runtime enforcement", "Testing whether provenance-aware monitoring helps across unseen action sequences, scenario templates, and live runtime execution.", "Research progression")}

      ${sectionHeading("Key research milestones")}
      <div class="timeline">
        <div class="timeline-item">
          <div class="timeline-marker">1</div>
          <div class="timeline-body">
            <h3>Vary action sequences &amp; templates</h3>
            <p>Move beyond fixed six-step sequences to branching workflows and held-out scenario templates.</p>
          </div>
        </div>
        <div class="timeline-item">
          <div class="timeline-marker">2</div>
          <div class="timeline-body">
            <h3>Audit features &amp; eliminate shortcuts</h3>
            <p>Verify that labels are not trivially encoded in synthetic metadata or single-step flags.</p>
          </div>
        </div>
        <div class="timeline-item">
          <div class="timeline-marker">3</div>
          <div class="timeline-body">
            <h3>Compare baselines, ablations &amp; ML models</h3>
            <p>Evaluate configurations side-by-side using clear metrics, confidence intervals, and runtime latency.</p>
          </div>
        </div>
        <div class="timeline-item">
          <div class="timeline-marker">4</div>
          <div class="timeline-body">
            <h3>Inspect lineage DAGs &amp; live simulations</h3>
            <p>Trace how sensitive inputs propagate through transformations and external destinations in real time.</p>
          </div>
        </div>
      </div>

      ${nextPage("v5-enterprise", "Go to V5 Enterprise Suite", "Explore V5 tables, model comparisons, and live simulation")}
    `;
  }
};

async function checkBackendHealth() {
  if (!backendHealthTag) return;
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (res.ok) {
      backendHealthTag.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span>Backend Connected</span>`;
    } else {
      backendHealthTag.innerHTML = `<span class="status-dot is-offline" aria-hidden="true"></span><span>Backend Degraded</span>`;
    }
  } catch {
    backendHealthTag.innerHTML = `<span class="status-dot is-offline" aria-hidden="true"></span><span>Backend Offline</span>`;
  }
}

function getV5ObservationEntries() {
  const entries = [];
  const reportSections = [
    ["R1_overall_comparison", "Test evaluation · overall comparison"],
    ["R5_component_ablations", "Test evaluation · component ablations"]
  ];

  for (const [sectionKey, source] of reportSections) {
    const rows = Array.isArray(v5ReportData?.[sectionKey]) ? v5ReportData[sectionKey] : [];
    for (const row of rows) {
      const name = row.config || row.model || row.name || row.variant;
      if (!name) continue;
      entries.push({
        key: `${sectionKey}:${name}`,
        name: String(name),
        source,
        group: sectionKey,
        details: row
      });
    }
  }

  const models = v5DevReportData?.models;
  if (models && typeof models === "object") {
    for (const [configuration, candidates] of Object.entries(models)) {
      if (!candidates || typeof candidates !== "object") continue;
      const best = v5DevReportData[`best_${configuration}`];
      for (const [modelName, metrics] of Object.entries(candidates)) {
        if (!metrics || typeof metrics !== "object") continue;
        const isSelectedBest = best?.model_name === modelName;
        const thresholdMetrics = isSelectedBest
          ? best?.threshold_tuning?.selected_metrics
          : null;
        entries.push({
          key: `dev:${configuration}:${modelName}`,
          name: `${configuration} · ${modelName}`,
          source: `Dev cross-validation · ${configuration}`,
          group: `dev:${configuration}`,
          details: {
            ...metrics,
            ...(isSelectedBest ? { selected_as_best_in_dev_report: true } : {}),
              ...(isSelectedBest ? { threshold_tuning: best.threshold_tuning } : {}),
              ...(thresholdMetrics ? { selected_threshold_metrics: thresholdMetrics } : {})
          }
        });
      }
    }
  }

  return entries;
}

function formatObservationValue(key, value) {
  if (typeof value !== "number") return String(value);
  if (/rate|asr|fpr/i.test(key)) return formatMetricPercent(value);
  if (/latency/i.test(key)) return `${formatNumber(value, 2)} ms`;
  return formatNumber(value, 4);
}

function deriveV5Observations(entry, entries) {
  const metricLabels = {
    asr: "Attack success rate",
    attack_block_rate: "Attack block rate",
    attack_detection_rate: "Attack detection rate",
    benign_false_alert_rate: "Benign false-alert rate",
    benign_fpr: "Benign false-positive rate",
    benign_completion_rate: "Benign completion rate",
    early_warning_rate: "Early-warning rate",
    mean_latency_ms: "Mean latency",
    cv_auroc: "Cross-validation AUROC",
    cv_pr_auc: "Cross-validation PR-AUC",
    cv_f1_at_0_5: "Cross-validation F1 at 0.5"
  };
  const comparableEntries = entries.filter(candidate => candidate.group === entry.group);
  const observations = [];

  for (const [metric, label] of Object.entries(metricLabels)) {
    const values = comparableEntries
      .filter(candidate => typeof candidate.details?.[metric] === "number")
      .map(candidate => candidate.details[metric]);
    const value = entry.details?.[metric];
    if (typeof value !== "number" || values.length === 0) continue;
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);

    if (minimum === maximum) {
      observations.push(`${label} is ${formatObservationValue(metric, value)} across this comparison group.`);
    } else if (value === minimum) {
      observations.push(`${label} is the lowest in this comparison group (${formatObservationValue(metric, value)}).`);
    } else if (value === maximum) {
      observations.push(`${label} is the highest in this comparison group (${formatObservationValue(metric, value)}).`);
    } else {
      observations.push(
        `${label} is ${formatObservationValue(metric, value)}; the group range is ${formatObservationValue(metric, minimum)}–${formatObservationValue(metric, maximum)}.`
      );
    }
  }

  const thresholdMetrics = entry.details?.selected_threshold_metrics;
  if (thresholdMetrics && typeof thresholdMetrics === "object") {
    const thresholdSummary = Object.entries(thresholdMetrics)
      .map(([metric, value]) => `${metric.replaceAll("_", " ")}: ${formatObservationValue(metric, value)}`)
      .join("; ");
    observations.push(`Selected dev threshold metrics: ${thresholdSummary}.`);
  }

  if (observations.length === 0) {
    observations.push("No comparable numeric metrics were present for this entry.");
  }
  return observations;
}

function renderV5Observations(entries) {
  const listEl = document.getElementById("v5-observation-list");
  const detailEl = document.getElementById("v5-observation-detail");
  if (!listEl || !detailEl) return;

  if (entries.length === 0) {
    listEl.innerHTML = "";
    detailEl.innerHTML = `<div class="empty-state">No model or configuration metric rows are available in the loaded V5 reports.</div>`;
    return;
  }

  if (!entries.some(entry => entry.key === v5SelectedModelKey)) {
    v5SelectedModelKey = entries[0].key;
  }
  const selected = entries.find(entry => entry.key === v5SelectedModelKey);

  listEl.innerHTML = entries.map(entry => `
    <button
      type="button"
      class="v5-model-choice ${entry.key === selected.key ? "selected" : ""}"
      data-v5-model-key="${escapeHtml(entry.key)}"
      aria-pressed="${entry.key === selected.key}"
    >
      <strong>${escapeHtml(entry.name)}</strong>
      <small>${escapeHtml(entry.source)}</small>
    </button>
  `).join("");

  detailEl.innerHTML = `
    <div class="section-kicker">${escapeHtml(selected.source)}</div>
    <h3>${escapeHtml(selected.name)}</h3>
    <h4>Observed comparisons</h4>
    <ul class="v5-observation-items">
      ${deriveV5Observations(selected, entries).map(observation => `<li>${escapeHtml(observation)}</li>`).join("")}
    </ul>
    <details class="v5-metric-breakdown">
      <summary>Full metric breakdown</summary>
      <pre>${escapeHtml(JSON.stringify(selected.details, null, 2))}</pre>
    </details>
  `;
}

async function loadV5SuiteData(forceRefresh = false) {
  if (!forceRefresh && v5ReportData) {
    renderV5Suite();
    return;
  }

  try {
    const [repRes, devRes, scenRes] = await Promise.all([
      fetch("/api/v5/report", { cache: "no-store" }),
      fetch("/api/v5/dev-report", { cache: "no-store" }),
      fetch("/api/v5/scenarios", { cache: "no-store" })
    ]);

    v5ReportData = repRes.ok ? await repRes.json() : null;
    v5DevReportData = devRes.ok ? await devRes.json() : null;
    v5ScenarioCatalog = scenRes.ok ? await scenRes.json() : null;
  } catch {
    v5ReportData = null;
  }

  renderV5Suite();
}

function renderV5Suite() {
  const kpiGrid = document.getElementById("v5-kpi-grid");
  const modelBody = document.getElementById("v5-model-comparison-body");
  const tabsEl = document.getElementById("v5-table-tabs");
  const theadEl = document.getElementById("v5-dynamic-thead");
  const tbodyEl = document.getElementById("v5-dynamic-tbody");
  const artifactsBody = document.getElementById("v5-artifacts-body");
  const observationList = document.getElementById("v5-observation-list");
  const observationDetail = document.getElementById("v5-observation-detail");

  if (!kpiGrid || !modelBody || !tabsEl || !theadEl || !tbodyEl || !artifactsBody || !observationList || !observationDetail) return;

  if (!v5ReportData) {
    kpiGrid.innerHTML = `
      ${metricCard("V5 Test Report", "Unavailable", "Run V5 test evaluation to generate report", "amber")}
      ${metricCard("Lowest ASR", "—", "Requires V5 report")}
      ${metricCard("Highest Block Rate", "—", "Requires V5 report")}
      ${metricCard("Early-Warning Rate", "—", "Requires V5 report")}
    `;
    modelBody.innerHTML = `<div class="empty-state">No V5 test report found at <code>/api/v5/report</code>. You can still run live simulations below using <code>POST /api/run-scenario</code>.</div>`;
    tabsEl.innerHTML = "";
    theadEl.innerHTML = "";
    tbodyEl.innerHTML = `<tr><td class="empty-state">No report tables available yet.</td></tr>`;
    artifactsBody.innerHTML = `<div class="empty-state">No generated result artifacts listed in the report yet.</div>`;
    renderV5Observations(getV5ObservationEntries());
    return;
  }

  const best = computeBestConfigs(v5ReportData);
  const devInfo = v5ScenarioCatalog
    ? `${v5ScenarioCatalog.dev_count ?? 0} dev · ${v5ScenarioCatalog.test_count ?? 0} test scenarios`
    : "Dynamic V5 evaluation report";
  const observationEntries = getV5ObservationEntries();
  if (!observationEntries.some(entry => entry.key === v5SelectedModelKey)) {
    v5SelectedModelKey = observationEntries[0]?.key ?? null;
  }

  kpiGrid.innerHTML = `
    ${metricCard(
    "Evaluated Configurations",
    best ? String(best.allRows.length) : "Available",
    devInfo,
    "green"
  )}
    ${metricCard(
    "Lowest ASR (Best)",
    best?.lowestAsr ? formatMetricPercent(best.lowestAsr.score) : "—",
    best?.lowestAsr ? `Config: ${best.lowestAsr.name}` : "No ASR metric in primary table",
    "green"
  )}
    ${metricCard(
    "Highest Attack Block",
    best?.highestBlock ? formatMetricPercent(best.highestBlock.score) : "—",
    best?.highestBlock ? `Config: ${best.highestBlock.name}` : "No block metric in primary table",
    "green"
  )}
    ${metricCard(
    "Best Early-Warning",
    best?.highestEarlyWarn ? formatMetricPercent(best.highestEarlyWarn.score) : "—",
    best?.highestEarlyWarn ? `Config: ${best.highestEarlyWarn.name}` : "No early-warning metric in primary table",
    "amber"
  )}
  `;

  // Model & Preset Comparison Summary
  if (best) {
    const highlights = [
      best.lowestAsr ? `<span class="pill green">Lowest ASR: ${escapeHtml(best.lowestAsr.name)} (${formatMetricPercent(best.lowestAsr.score)})</span>` : "",
      best.highestBlock ? `<span class="pill green">Top Block Rate: ${escapeHtml(best.highestBlock.name)} (${formatMetricPercent(best.highestBlock.score)})</span>` : "",
      best.lowestFpr ? `<span class="pill blue">Lowest Benign FPR: ${escapeHtml(best.lowestFpr.name)} (${formatMetricPercent(best.lowestFpr.score)})</span>` : "",
      best.highestEarlyWarn ? `<span class="pill amber">Top Early Warning: ${escapeHtml(best.highestEarlyWarn.name)} (${formatMetricPercent(best.highestEarlyWarn.score)})</span>` : "",
      best.lowestLatency ? `<span class="pill gray">Fastest Monitor: ${escapeHtml(best.lowestLatency.name)} (${formatNumber(best.lowestLatency.score, 2)} ms)</span>` : ""
    ].filter(Boolean);

    modelBody.innerHTML = `
      <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px">${highlights.join("")}</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Configuration / Model</th>
              <th>ASR (↓)</th>
              <th>Block Rate (↑)</th>
              <th>Detection (↑)</th>
              <th>Benign FPR (↓)</th>
              <th>Benign Comp. (↑)</th>
              <th>Early Warn (↑)</th>
              <th>Latency (ms ↓)</th>
            </tr>
          </thead>
          <tbody>
            ${best.allRows.map(r => {
      const name = r.config || r.model || r.name || r.variant || "—";
      const isBest = best.lowestAsr && name === best.lowestAsr.name;
      const observationKey = `R1_overall_comparison:${name}`;
      const selectedClass = observationKey === v5SelectedModelKey ? "selected-row" : "";
      return `
                <tr class="clickable-row ${selectedClass}" data-v5-model-key="${escapeHtml(observationKey)}">
                  <td><strong>${escapeHtml(name)}</strong> ${isBest ? `<span class="pill green">Best ASR</span>` : ""}</td>
                  <td class="numeric">${formatMetricPercent(r.asr)}</td>
                  <td class="numeric">${formatMetricPercent(r.attack_block_rate ?? r.block_rate)}</td>
                  <td class="numeric">${formatMetricPercent(r.attack_detection_rate ?? r.detection_rate)}</td>
                  <td class="numeric">${formatMetricPercent(r.benign_false_alert_rate ?? r.benign_fpr)}</td>
                  <td class="numeric">${formatMetricPercent(r.benign_completion_rate)}</td>
                  <td class="numeric">${formatMetricPercent(r.early_warning_rate)}</td>
                  <td class="numeric">${typeof (r.mean_latency_ms ?? r.latency_ms) === "number" ? `${formatNumber(r.mean_latency_ms ?? r.latency_ms, 2)} ms` : "—"}</td>
                </tr>
              `;
    }).join("")}
          </tbody>
        </table>
      </div>
    `;
  } else {
    modelBody.innerHTML = `<div class="empty-state">Report loaded, but no tabular configuration comparison array was found.</div>`;
  }
  renderV5Observations(observationEntries);

  // Dynamic Report Tables (R1-R5 or any array sections)
  const tables = extractReportTables(v5ReportData);
  if (tables.length === 0) {
    tabsEl.innerHTML = "";
    theadEl.innerHTML = "";
    tbodyEl.innerHTML = `<tr><td class="empty-state">No table arrays present in V5 report JSON.</td></tr>`;
  } else {
    if (!v5ActiveTableKey || !tables.some(t => t.key === v5ActiveTableKey)) {
      v5ActiveTableKey = tables[0].key;
      v5SelectedRowIndex = null;
    }

    tabsEl.innerHTML = tables.map(t => `
      <button
        type="button"
        class="tab-btn ${t.key === v5ActiveTableKey ? "active" : ""}"
        data-v5-table-key="${escapeHtml(t.key)}"
      >
        ${escapeHtml(formatTableLabel(t.key))} (${t.rows.length})
      </button>
    `).join("");

    renderActiveV5Table(tables.find(t => t.key === v5ActiveTableKey));
  }

  // Generated Artifacts / Figures
  const rawArtifacts = [
    ...(Array.isArray(v5ReportData.artifacts) ? v5ReportData.artifacts : []),
    ...(Array.isArray(v5ReportData.plots) ? v5ReportData.plots : [])
  ];
  const uniqueArtifacts = rawArtifacts.filter((artifact, index, artifacts) => {
    const path = typeof artifact === "string" ? artifact : artifact?.path || artifact?.url;
    return path && artifacts.findIndex(candidate =>
      (typeof candidate === "string" ? candidate : candidate?.path || candidate?.url) === path
    ) === index;
  });

  if (uniqueArtifacts.length > 0) {
    artifactsBody.innerHTML = `
      <div class="artifact-grid">
        ${uniqueArtifacts.map(art => {
      const pathStr = typeof art === "string" ? art : (art.path || art.url || "");
      const titleStr = typeof art === "string" ? art.split("/").pop() : (art.title || art.name || art.path || "Result Figure");
      const src = pathStr.startsWith("/") ? pathStr : `/${pathStr}`;
      return `
            <article class="artifact-card">
              <img src="${escapeHtml(src)}" alt="${escapeHtml(titleStr)}" data-artifact-src="${escapeHtml(src)}" />
              <div class="artifact-card-body">
                <strong>${escapeHtml(titleStr)}</strong>
                <small>${escapeHtml(pathStr)}</small>
                <button
                  type="button"
                  class="artifact-zoom-button"
                  data-v5-artifact-zoom
                  data-artifact-src="${escapeHtml(src)}"
                  data-artifact-title="${escapeHtml(titleStr)}"
                  aria-label="Open ${escapeHtml(titleStr)} in large view"
                >View larger</button>
              </div>
            </article>
          `;
    }).join("")}
      </div>
      <dialog class="artifact-dialog" id="v5-artifact-dialog" aria-labelledby="v5-artifact-dialog-title">
        <div class="artifact-dialog-content">
          <div class="artifact-dialog-header">
            <h3 id="v5-artifact-dialog-title"></h3>
            <button type="button" class="artifact-zoom-button" data-v5-artifact-close>Close</button>
          </div>
          <img id="v5-artifact-dialog-image" class="artifact-dialog-image" alt="" />
        </div>
      </dialog>
    `;
    artifactsBody.querySelectorAll("img[data-artifact-src]").forEach(image => {
      image.addEventListener("error", () => {
        const fallback = document.createElement("div");
        fallback.className = "empty-state artifact-image-error";
        fallback.textContent = `Image file not found on server: ${image.dataset.artifactSrc}`;
        image.replaceWith(fallback);
        const zoomButton = image.closest(".artifact-card")?.querySelector("[data-v5-artifact-zoom]");
        if (zoomButton) {
          zoomButton.disabled = true;
          zoomButton.textContent = "Image unavailable";
        }
      }, { once: true });
    });
  } else {
    artifactsBody.innerHTML = `
      <div class="empty-state">
        No generated image files were found in <code>results/v5</code>, and the report lists no plot paths.
      </div>
    `;
  }
}

function renderActiveV5Table(tableObj) {
  const theadEl = document.getElementById("v5-dynamic-thead");
  const tbodyEl = document.getElementById("v5-dynamic-tbody");
  const inspector = document.getElementById("v5-row-inspector");
  if (!theadEl || !tbodyEl || !inspector || !tableObj) return;

  const columns = Array.from(
    tableObj.rows.reduce((set, row) => {
      Object.keys(row).forEach(k => set.add(k));
      return set;
    }, new Set())
  );

  theadEl.innerHTML = `<tr>${columns.map(col => `<th>${escapeHtml(col.replaceAll("_", " "))}</th>`).join("")}</tr>`;

  tbodyEl.innerHTML = tableObj.rows.map((row, idx) => {
    const selected = idx === v5SelectedRowIndex ? "selected-row" : "";
    return `
      <tr class="clickable-row ${selected}" data-v5-row-index="${idx}">
        ${columns.map(col => {
      const val = row[col];
      let formatted = "—";
      let cls = "";
      if (typeof val === "number") {
        cls = "numeric";
        formatted = Number.isInteger(val) ? String(val) : val.toFixed(4);
      } else if (typeof val === "boolean") {
        formatted = val ? "true" : "false";
      } else if (val !== null && val !== undefined) {
        formatted = typeof val === "object" ? JSON.stringify(val) : String(val);
      }
      return `<td class="${cls}">${escapeHtml(formatted)}</td>`;
    }).join("")}
      </tr>
    `;
  }).join("");

  if (v5SelectedRowIndex !== null && tableObj.rows[v5SelectedRowIndex]) {
    inspector.hidden = false;
    inspector.innerHTML = `
      <div class="section-kicker">SELECTED TABLE ROW · #${v5SelectedRowIndex + 1}</div>
      <strong>${escapeHtml(formatTableLabel(tableObj.key))}</strong>
      <pre style="margin-top:8px; white-space:pre-wrap; font-family:var(--font-mono); font-size:11.5px; color:var(--text);">${escapeHtml(JSON.stringify(tableObj.rows[v5SelectedRowIndex], null, 2))}</pre>
    `;
  } else {
    inspector.hidden = true;
    inspector.innerHTML = "";
  }
}

async function runV5Simulation() {
  const configSelect = document.getElementById("v5-config-select");
  const familySelect = document.getElementById("v5-family-select");
  const partitionSelect = document.getElementById("v5-partition-select");
  const statusEl = document.getElementById("v5-run-status");
  const errorEl = document.getElementById("v5-run-error");
  const runBtn = document.getElementById("v5-run-btn");

  if (!configSelect || !familySelect || !partitionSelect || !statusEl || !errorEl) return;

  errorEl.hidden = true;
  errorEl.textContent = "";
  statusEl.className = "agent-status is-running";
  statusEl.textContent = "Running V5 scenario simulation…";
  if (runBtn) runBtn.disabled = true;

  try {
    const res = await fetch("/api/run-scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        config_name: configSelect.value,
        family: familySelect.value,
        partition: partitionSelect.value
      })
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.details || data?.error || "Simulation failed.");
    }

    v5LastSimResult = data;
    v5SelectedSimStep = 0;

    const summary = data.summary || {};
    if (summary.attack_executed) {
      statusEl.className = "agent-status is-error";
      statusEl.textContent = `Attack Executed (${data.config || configSelect.value})`;
    } else if (summary.completed === false) {
      statusEl.className = "agent-status is-blocked";
      statusEl.textContent = `Intercepted / Stopped (${data.config || configSelect.value})`;
    } else {
      statusEl.className = "agent-status is-success";
      statusEl.textContent = `Completed Cleanly (${data.config || configSelect.value})`;
    }

    renderV5SimulationOutput(data);
  } catch (err) {
    statusEl.className = "agent-status is-error";
    statusEl.textContent = "Simulation error";
    errorEl.hidden = false;
    errorEl.textContent = err.message || "Could not run V5 simulation.";
  } finally {
    if (runBtn) runBtn.disabled = false;
  }
}

function renderV5SimulationOutput(sim) {
  const feedEl = document.getElementById("v5-telemetry-feed");
  const dagEl = document.getElementById("v5-dag-view");
  if (!feedEl || !dagEl) return;

  const steps = Array.isArray(sim?.steps) ? sim.steps : [];
  if (steps.length === 0) {
    feedEl.innerHTML = `<div class="empty-state">Simulation returned no step rows.</div>`;
    dagEl.innerHTML = `<div class="empty-state">No DAG steps to render.</div>`;
    return;
  }

  feedEl.innerHTML = steps.map((st, idx) => {
    const verdict = String(st.verdict || "NONE").toUpperCase();
    const pillCls = verdict === "BLOCK" ? "red" : verdict === "WARN" ? "amber" : "green";
    const activeStyle = idx === v5SelectedSimStep ? "border-color:#23765b; background:#f3faf6;" : "";
    return `
      <button
        type="button"
        class="card"
        style="padding:12px 14px; text-align:left; cursor:pointer; ${activeStyle}"
        data-v5-sim-step="${idx}"
      >
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
          <div>
            <span class="pill gray">Step ${escapeHtml(st.step ?? idx + 1)}</span>
            <strong style="margin-left:6px">${escapeHtml(st.tool || "tool")}</strong>
            <span class="muted small">→ ${escapeHtml(st.destination || "local")}</span>
          </div>
          <span class="pill ${pillCls}">${escapeHtml(verdict)}</span>
        </div>
        <div class="muted small" style="margin-top:6px; display:flex; flex-wrap:wrap; gap:12px;">
          <span>Sensitive Ancestor: <strong>${escapeHtml(st.has_sensitive_ancestor ?? "—")}</strong></span>
          <span>Prov. Depth: <strong>${escapeHtml(st.provenance_depth ?? "—")}</strong></span>
          <span>Transform Depth: <strong>${escapeHtml(st.transformation_depth ?? "—")}</strong></span>
          <span>Latency: <strong>${escapeHtml(st.monitor_time_ms ?? "—")} ms</strong></span>
        </div>
      </button>
    `;
  }).join("");

  dagEl.innerHTML = `
    <div class="dag-node-list">
      ${steps.map((st, idx) => {
    const verdict = String(st.verdict || "NONE").toUpperCase();
    const nodeCls = verdict === "BLOCK" ? "is-block" : verdict === "WARN" ? "is-warn" : "";
    const selCls = idx === v5SelectedSimStep ? "is-selected" : "";
    const pillCls = verdict === "BLOCK" ? "red" : verdict === "WARN" ? "amber" : "green";
    const connector = idx < steps.length - 1
      ? `<div class="dag-edge-connector">↓ lineage edge (prov_depth=${escapeHtml(st.provenance_depth ?? "?")}, xform_depth=${escapeHtml(st.transformation_depth ?? "?")})</div>`
      : "";
    return `
          <button
            type="button"
            class="dag-node ${nodeCls} ${selCls}"
            data-v5-sim-step="${idx}"
          >
            <div>
              <strong>Node ${idx + 1}: ${escapeHtml(st.tool || "action")}</strong>
              <div class="muted small">Dest: ${escapeHtml(st.destination || "internal")} · Sens. Ancestor: ${escapeHtml(st.has_sensitive_ancestor ?? "—")}</div>
            </div>
            <span class="pill ${pillCls}">${escapeHtml(verdict)}</span>
          </button>
          ${connector}
        `;
  }).join("")}
    </div>
  `;

  renderV5SimStepInspector(steps[v5SelectedSimStep] || steps[0]);
}

function renderV5SimStepInspector(step) {
  const inspector = document.getElementById("v5-dag-inspector");
  if (!inspector || !step) return;
  inspector.hidden = false;
  inspector.innerHTML = `
    <div class="section-kicker">INSPECTED NODE / STEP</div>
    <strong>Step ${escapeHtml(step.step ?? "")} — ${escapeHtml(step.tool || "")}</strong>
    <pre style="margin-top:8px; white-space:pre-wrap; font-family:var(--font-mono); font-size:11.5px; color:var(--text);">${escapeHtml(JSON.stringify(step, null, 2))}</pre>
  `;
}

function exportV5ReportJson() {
  if (!v5ReportData) return;
  const blob = new Blob([JSON.stringify(v5ReportData, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "v5_test_evaluation_report.json";
  a.click();
  URL.revokeObjectURL(url);
}

async function loadInitialEvaluationStatus() {
  const target = document.getElementById("initial-artifact-status");
  if (!target) return;

  try {
    const response = await fetch("/api/evaluation", { cache: "no-store" });
    if (!response.ok) throw new Error("Evaluation endpoint unavailable");

    const data = await response.json();
    const text = JSON.stringify(data).toLowerCase();

    if (text.includes("provenance_hard_negative_evaluation") ||
      text.includes("initial") ||
      text.includes("evaluation")) {
      target.textContent =
        "The evaluation API responded successfully. Check the served evaluation data for the initial benchmark's exact metrics; this page intentionally does not substitute V2 or V3 scores for the initial result.";
    } else {
      target.textContent =
        "The API responded, but the initial evaluation metrics could not be identified confidently. No initial accuracy or F1 score is inferred.";
    }
  } catch {
    target.textContent =
      "Initial evaluation metrics are not currently available through the dashboard API. The dataset counts and experiment description above are recorded project details; no initial model score is fabricated.";
  }
}

function createTextBlock(tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function appendDetailRow(container, label, value) {
  const row = document.createElement("div");
  row.className = "agent-detail-row";
  row.append(createTextBlock("span", label, "agent-detail-label"));
  const valueElement = document.createElement("span");
  valueElement.className = "agent-detail-value";
  valueElement.textContent = value || "None";
  row.append(valueElement);
  container.append(row);
}

function setAgentRunStatus(message, state = "idle") {
  agentRunState = state;
  const status = document.getElementById("agent-run-status");
  if (status) {
    status.textContent = message;
    status.className = `agent-status is-${state}`;
  }
}

async function loadAgentScenarios() {
  const container = document.getElementById("agent-scenario-list");
  if (!container) return;
  container.replaceChildren();

  try {
    const response = await fetch("/api/agent/scenarios", { cache: "no-store" });
    if (!response.ok) throw new Error("Scenario catalog is unavailable.");
    const payload = await response.json();
    if (!payload || !Array.isArray(payload.scenarios)) {
      throw new Error("Scenario catalog response was malformed.");
    }
    agentScenarios = payload.scenarios;
    if (agentScenarios.length === 0) {
      container.append(createTextBlock("p", "No predefined scenarios are available.", "empty-state"));
      return;
    }

    agentScenarios.forEach(scenario => {
      const card = document.createElement("article");
      card.className = "agent-scenario";
      const title = createTextBlock("h3", scenario.title || "Scenario");
      const description = createTextBlock("p", scenario.description || "");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "agent-run-button";
      button.dataset.agentScenario = scenario.scenario_id;
      button.textContent = "Run scenario";
      button.disabled = agentRunState === "running";
      card.append(title, description, button);
      container.append(card);
    });
    if (agentRun) {
      setAgentRunStatus(taskStatusLabel(agentRun.status), taskStatusClass(agentRun.status));
      renderAgentRun(agentRun);
    }
  } catch (error) {
    container.append(createTextBlock("p", error.message || "Could not load scenarios.", "agent-error"));
    setAgentRunStatus("Scenario service unavailable", "error");
  }
}

async function runAgentScenario(scenarioId) {
  if (agentRunState === "running") return;
  document.querySelectorAll("[data-agent-scenario]").forEach(btn => { btn.disabled = true; });
  agentRun = null;
  agentRunError = "";
  selectedAgentStep = 0;
  document.getElementById("agent-run-result")?.replaceChildren();
  setAgentRunStatus("Running scenario…", "running");

  try {
    const response = await fetch("/api/agent/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenarioId })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.error || "The backend rejected the scenario request.");
    }
    agentRun = payload;
    selectedAgentStep = payload.trace?.length ? 0 : -1;
    setAgentRunStatus(taskStatusLabel(payload.status), taskStatusClass(payload.status));
    renderAgentRun(payload);
  } catch (error) {
    agentRunError = error.message || "The run could not be loaded.";
    setAgentRunStatus("Run failed", "error");
    renderAgentError(agentRunError);
  } finally {
    document.querySelectorAll("[data-agent-scenario]").forEach(btn => { btn.disabled = false; });
  }
}

function taskStatusLabel(status) {
  if (status === "completed") return "Completed successfully";
  if (status === "stopped") return "Stopped by security decision";
  if (status === "failed") return "Run failed";
  return "Run status unavailable";
}

function taskStatusClass(status) {
  if (status === "completed") return "success";
  if (status === "stopped") return "blocked";
  return "error";
}

function renderAgentError(message) {
  const container = document.getElementById("agent-run-result");
  if (!container) return;
  container.replaceChildren();
  const panel = document.createElement("div");
  panel.className = "agent-error-panel";
  panel.append(createTextBlock("h2", "Run unavailable"));
  panel.append(createTextBlock("p", message));
  container.append(panel);
}

function renderAgentRun(run) {
  const container = document.getElementById("agent-run-result");
  if (!container) return;
  container.replaceChildren();

  const heading = document.createElement("div");
  heading.className = "agent-result-heading";
  const headingCopy = document.createElement("div");
  headingCopy.append(createTextBlock("div", "TASK RESULT", "section-kicker"));
  headingCopy.append(createTextBlock("h2", run.task_description || "Deterministic run"));
  headingCopy.append(createTextBlock("p", `Run ID · ${run.run_id}`, "agent-run-id"));
  const clearButton = document.createElement("button");
  clearButton.type = "button";
  clearButton.className = "agent-clear-button";
  clearButton.dataset.agentClear = "true";
  clearButton.textContent = "Clear result";
  heading.append(headingCopy, clearButton);
  container.append(heading);

  const summary = run.summary || {};
  const metrics = document.createElement("div");
  metrics.className = "agent-metrics";
  [
    ["Planned / proposed", `${summary.planned_actions ?? 0} / ${summary.proposed_actions ?? 0}`, ""],
    ["Executed", summary.executed_actions ?? 0, "success"],
    ["Warnings", summary.warnings ?? 0, "warning"],
    ["Blocked", summary.blocked_actions ?? 0, "blocked"],
    ["Failed executions", summary.failed_executions ?? 0, "error"]
  ].forEach(([label, value, state]) => {
    const card = document.createElement("article");
    card.className = `agent-metric ${state ? `is-${state}` : ""}`;
    card.append(createTextBlock("span", label));
    card.append(createTextBlock("strong", String(value)));
    metrics.append(card);
  });
  container.append(metrics);

  const taskResult = document.createElement("div");
  taskResult.className = `agent-task-result is-${taskStatusClass(run.status)}`;
  taskResult.append(createTextBlock("strong", taskStatusLabel(run.status)));
  taskResult.append(createTextBlock("span", run.stop_reason || "No stop reason supplied."));
  container.append(taskResult);

  const workspace = document.createElement("div");
  workspace.className = "agent-workspace";
  const timelineSection = document.createElement("section");
  timelineSection.className = "agent-panel";
  timelineSection.append(createTextBlock("div", "EXECUTION TRACE", "section-kicker"));
  timelineSection.append(createTextBlock("h2", "Action timeline"));
  const timeline = document.createElement("div");
  timeline.className = "agent-timeline";
  (run.trace || []).forEach((step, index) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `agent-trace-step is-${step.verdict === "BLOCK" ? "blocked" : step.verdict === "WARN" ? "warning" : step.execution_status === "failed" ? "error" : "success"}`;
    item.dataset.agentTraceIndex = String(index);
    item.setAttribute("aria-pressed", String(index === selectedAgentStep));
    const line = document.createElement("span");
    line.className = "agent-trace-line";
    line.textContent = String(step.action_index ?? index + 1).padStart(2, "0");
    const body = document.createElement("span");
    body.className = "agent-trace-copy";
    body.append(createTextBlock("strong", `${step.tool_name} · ${String(step.execution_status || "").replaceAll("_", " ")}`));
    body.append(createTextBlock("span", step.description || "Proposed tool action."));
    const verdict = document.createElement("span");
    verdict.className = `agent-verdict is-${(step.verdict || "none").toLowerCase()}`;
    verdict.textContent = step.verdict || "NOT EVALUATED";
    item.append(line, body, verdict);
    timeline.append(item);
  });
  timelineSection.append(timeline);

  const side = document.createElement("div");
  side.className = "agent-side-panels";
  const inspector = document.createElement("section");
  inspector.className = "agent-panel";
  renderAgentInspector(inspector, run.trace?.[selectedAgentStep]);
  const provenance = document.createElement("section");
  provenance.className = "agent-panel";
  provenance.append(createTextBlock("div", "DATA LINEAGE", "section-kicker"));
  provenance.append(createTextBlock("h2", "Provenance explorer"));
  renderAgentProvenance(provenance, run.provenance);
  side.append(inspector, provenance);
  workspace.append(timelineSection, side);
  container.append(workspace);
}

function renderAgentInspector(container, step) {
  container.replaceChildren();
  container.append(createTextBlock("div", "DECISION INSPECTOR", "section-kicker"));
  container.append(createTextBlock("h2", "Selected action"));
  if (!step) {
    container.append(createTextBlock("p", "Select a timeline action to inspect its decision and execution details.", "empty-state"));
    return;
  }
  appendDetailRow(container, "Step", step.step_id);
  appendDetailRow(container, "Tool", step.tool_name);
  appendDetailRow(container, "Policy verdict", step.verdict || "Not evaluated");
  appendDetailRow(container, "Execution", String(step.execution_status || "").replaceAll("_", " "));
  appendDetailRow(container, "Decision reason", step.decision_reason || "Not available");
  appendDetailRow(container, "Destination", step.destination);
  appendDetailRow(container, "Matched rules", (step.matched_policy_rules || []).join(", "));
  appendDetailRow(container, "Inputs", (step.input_data_object_ids || []).join(", "));
  appendDetailRow(container, "Outputs", (step.output_data_object_ids || []).join(", "));
}

function renderAgentProvenance(container, provenance) {
  const objects = provenance?.objects || [];
  const edges = provenance?.edges || [];
  if (objects.length === 0 || edges.length === 0) {
    container.append(createTextBlock("p", "No provenance graph was produced for this run.", "empty-state"));
    return;
  }
  const objectMap = new Map(objects.map(object => [object.object_id, object]));
  const edgeList = document.createElement("ol");
  edgeList.className = "agent-provenance-edges";
  edges.forEach(edge => {
    const parent = objectMap.get(edge.parent_id);
    const child = objectMap.get(edge.child_id);
    if (!parent || !child) return;
    const item = document.createElement("li");
    item.append(createTextBlock("strong", parent.object_id));
    item.append(createTextBlock("span", ` ${parent.sensitivity} `, "agent-sensitivity"));
    item.append(createTextBlock("span", "→", "agent-edge-arrow"));
    item.append(createTextBlock("strong", child.object_id));
    item.append(createTextBlock("span", ` ${child.sensitivity} `, "agent-sensitivity"));
    item.append(createTextBlock("small", child.transformation || "source"));
    edgeList.append(item);
  });
  container.append(edgeList);
}

function clearAgentRun() {
  agentRun = null;
  agentRunError = "";
  selectedAgentStep = 0;
  setAgentRunStatus("Not run yet", "idle");
  const container = document.getElementById("agent-run-result");
  if (!container) return;
  container.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "agent-empty-state";
  empty.append(createTextBlock("span", "01", "agent-empty-mark"));
  empty.append(createTextBlock("h2", "No run selected"));
  empty.append(createTextBlock("p", "Run a scenario to inspect the actual monitor decisions, tool outcomes, and provenance created by the backend."));
  container.append(empty);
}

function llmStatusLabel(status) {
  return ({
    queued: "Queued",
    running: "Running",
    completed: "Completed",
    stopped: "Stopped by policy",
    failed: "Failed"
  })[status] || "Unknown status";
}

function llmStatusClass(status) {
  if (status === "completed") return "success";
  if (status === "stopped") return "blocked";
  if (status === "failed") return "error";
  return "running";
}

function setLLMStatus(message, state) {
  const status = document.getElementById("llm-run-status");
  if (!status) return;
  status.textContent = message;
  status.className = `agent-status is-${state}`;
}

function showLLMError(message) {
  const error = document.getElementById("llm-run-error");
  if (!error) return;
  error.textContent = message;
  error.hidden = false;
}

async function loadLLMLab() {
  const modeSelect = document.getElementById("llm-mode");
  const scenarioSelect = document.getElementById("llm-scenario");
  const policySelect = document.getElementById("llm-policy");
  const runButton = document.getElementById("llm-run-button");
  const providerLabel = document.getElementById("llm-provider-label");
  if (!modeSelect || !scenarioSelect || !policySelect || !runButton) return;

  try {
    const [configRes, historyRes] = await Promise.all([
      fetch("/api/llm/configuration", { cache: "no-store" }),
      fetch("/api/llm/runs", { cache: "no-store" })
    ]);
    if (!configRes.ok || !historyRes.ok) {
      throw new Error("LLM Lab configuration or run history is unavailable.");
    }
    const [config, historyPayload] = await Promise.all([configRes.json(), historyRes.json()]);
    llmConfiguration = config;
    llmHistory = Array.isArray(historyPayload.runs) ? historyPayload.runs : [];

    populateLLMFilters(config);
    scenarioSelect.replaceChildren();
    (config.scenarios || []).forEach(sc => {
      const opt = document.createElement("option");
      opt.value = sc;
      opt.textContent = sc.replaceAll("_", " ");
      scenarioSelect.append(opt);
    });
    policySelect.replaceChildren();
    (config.policies || []).forEach(pol => {
      const opt = document.createElement("option");
      opt.value = pol;
      opt.textContent = pol;
      policySelect.append(opt);
    });

    updateLLMProviderLabel();
    runButton.disabled = llmIsRunning;
    if (providerLabel) providerLabel.hidden = false;
    renderLLMHistory();
    if (llmCurrentRunId) {
      await refreshLLMRun(llmCurrentRunId, false);
    }
  } catch (error) {
    if (providerLabel) providerLabel.textContent = error.message || "LLM Lab unavailable.";
    setLLMStatus("Provider/configuration unavailable", "error");
    runButton.disabled = true;
    showLLMError(error.message || "The LLM Lab API is unavailable.");
  }
}

function updateLLMProviderLabel() {
  const mode = document.getElementById("llm-mode")?.value;
  const target = document.getElementById("llm-provider-label");
  if (!target || !llmConfiguration) return;
  if (mode === "live") {
    const provider = llmConfiguration.openai;
    target.textContent = provider?.configured
      ? `Live provider · ${provider.model}`
      : `Live provider · ${provider?.model || "default"} · API key not configured`;
    return;
  }
  if (mode === "fake") {
    target.textContent = "Fake provider · offline scripted responses";
    return;
  }
  target.textContent = "Deterministic plan · no model provider";
}

async function submitLLMRun(event) {
  event.preventDefault();
  if (llmIsRunning) return;
  const task = document.getElementById("llm-task")?.value || "";
  const mode = document.getElementById("llm-mode")?.value || "";
  const scenarioId = document.getElementById("llm-scenario")?.value || "";
  const policyId = document.getElementById("llm-policy")?.value || "";
  const maxActions = Number(document.getElementById("llm-action-limit")?.value);

  if (!task.trim() || !Number.isInteger(maxActions) || maxActions < 1 || maxActions > 12) {
    showLLMError("Enter a task and an action limit from 1 through 12.");
    return;
  }

  llmIsRunning = true;
  llmSelectedEvent = 0;
  document.getElementById("llm-run-button").disabled = true;
  setLLMStatus("Submitting run…", "running");

  try {
    const response = await fetch("/api/llm/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, mode, scenario_id: scenarioId, policy_id: policyId, max_actions: maxActions })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.message || payload?.error || "Could not start LLM run.");
    }
    llmCurrentRunId = payload.run_id;
    llmCurrentRun = payload;
    setLLMStatus(llmStatusLabel(payload.status), llmStatusClass(payload.status));
    renderLLMRun(payload);
    await pollLLMRun(payload.run_id);
  } catch (error) {
    setLLMStatus("Run could not start", "error");
    showLLMError(error.message || "The run could not start.");
  } finally {
    llmIsRunning = false;
    const btn = document.getElementById("llm-run-button");
    if (btn && llmConfiguration) btn.disabled = false;
  }
}

async function pollLLMRun(runId) {
  while (llmIsRunning) {
    await new Promise(resolve => setTimeout(resolve, 600));
    const done = await refreshLLMRun(runId, true);
    if (done) return;
  }
}

async function refreshLLMRun(runId, updateStatus = true) {
  const response = await fetch(`/api/llm/runs/${encodeURIComponent(runId)}`, { cache: "no-store" });
  if (!response.ok) throw new Error("Could not retrieve the run record.");
  const run = await response.json();
  llmCurrentRun = run;
  if (updateStatus) setLLMStatus(llmStatusLabel(run.status), llmStatusClass(run.status));
  if (location.hash === "#llm-lab") renderLLMRun(run);
  await refreshLLMHistory();
  return !["queued", "running"].includes(run.status);
}

async function refreshLLMHistory() {
  const response = await fetch("/api/llm/runs", { cache: "no-store" });
  if (!response.ok) return;
  const payload = await response.json();
  if (Array.isArray(payload.runs)) {
    llmHistory = payload.runs;
    renderLLMHistory();
  }
}

function appendLLMMetric(container, label, value, state = "") {
  const card = document.createElement("article");
  card.className = `llm-metric ${state ? `is-${state}` : ""}`;
  card.append(createTextBlock("span", label));
  card.append(createTextBlock("strong", value === null || value === undefined ? "—" : String(value)));
  container.append(card);
}

function renderLLMRun(run) {
  const container = document.getElementById("llm-run-output");
  if (!container) return;
  container.replaceChildren();

  const metrics = run.summary || {};
  const top = document.createElement("div");
  top.className = "llm-run-heading";
  top.append(createTextBlock("div", `RUN · ${run.run_id}`, "agent-run-id"));
  top.append(createTextBlock("h2", llmStatusLabel(run.status)));
  top.append(createTextBlock("p", `Mode: ${run.mode} · Provider: ${run.provider || "unavailable"} · Model: ${run.model || "unavailable"}`, "llm-muted"));
  container.append(top);

  const cards = document.createElement("div");
  cards.className = "llm-metrics";
  appendLLMMetric(cards, "Duration (s)", run.duration_seconds);
  appendLLMMetric(cards, "Proposals", metrics.proposals || 0);
  appendLLMMetric(cards, "ALLOW", metrics.allow || 0, "success");
  appendLLMMetric(cards, "WARN", metrics.warn || 0, "warning");
  appendLLMMetric(cards, "BLOCK", metrics.block || 0, "blocked");
  appendLLMMetric(cards, "Executed", metrics.executed || 0, "success");
  container.append(cards);

  const eventLayout = document.createElement("div");
  eventLayout.className = "llm-event-layout";

  const timeline = document.createElement("section");
  timeline.className = "agent-panel";
  timeline.append(createTextBlock("div", "TRUSTED RUN EVENTS", "section-kicker"));
  timeline.append(createTextBlock("h2", "Run timeline"));
  const list = document.createElement("div");
  list.className = "llm-event-list";
  (run.events || []).forEach((ev, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "llm-event";
    button.dataset.llmEventIndex = String(index);
    button.setAttribute("aria-pressed", String(index === llmSelectedEvent));
    button.append(createTextBlock("span", String(ev.type || "event").replaceAll("_", " "), "llm-event-title"));
    button.append(createTextBlock("span", JSON.stringify(ev.data || {}), "llm-event-detail"));
    list.append(button);
  });
  timeline.append(list);

  const provenance = document.createElement("section");
  provenance.className = "agent-panel llm-provenance";
  provenance.append(createTextBlock("div", "TRUSTED DATA LINEAGE", "section-kicker"));
  provenance.append(createTextBlock("h2", "Provenance"));
  renderAgentProvenance(provenance, run.provenance);

  eventLayout.append(timeline, provenance);
  container.append(eventLayout);
}

function populateLLMFilters(configuration) {
  const statusSelect = document.getElementById("llm-filter-status");
  const modeSelect = document.getElementById("llm-filter-mode");
  if (statusSelect && statusSelect.options.length === 1) {
    ["queued", "running", "completed", "stopped", "failed"].forEach(st => {
      const opt = document.createElement("option");
      opt.value = st;
      opt.textContent = st;
      statusSelect.append(opt);
    });
  }
  if (modeSelect && modeSelect.options.length === 1) {
    (configuration.modes || []).forEach(m => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      modeSelect.append(opt);
    });
  }
}

function renderLLMHistory() {
  const container = document.getElementById("llm-history-content");
  if (!container) return;
  container.replaceChildren();
  const statusFilter = document.getElementById("llm-filter-status")?.value || "all";
  const modeFilter = document.getElementById("llm-filter-mode")?.value || "all";
  const verdictFilter = document.getElementById("llm-filter-verdict")?.value || "all";

  const filtered = llmHistory.filter(run => {
    const summary = run.summary || {};
    const verdictOk = verdictFilter === "all" || (summary[verdictFilter] || 0) > 0;
    return (statusFilter === "all" || run.status === statusFilter) &&
      (modeFilter === "all" || run.mode === modeFilter) &&
      verdictOk;
  });

  if (!filtered.length) {
    container.append(createTextBlock("p", "No matching runs in this server session.", "empty-state"));
    return;
  }

  filtered.forEach(run => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "llm-history-row";
    button.dataset.llmHistoryRun = run.run_id;
    button.append(createTextBlock("strong", run.task || "Task"));
    button.append(createTextBlock("span", `${run.status} · ${run.mode} · ${run.scenario_id} · ${run.run_id}`));
    container.append(button);
  });
}

async function loadLLMHistoryRun(runId) {
  const response = await fetch(`/api/llm/runs/${encodeURIComponent(runId)}`, { cache: "no-store" });
  if (!response.ok) throw new Error("Could not retrieve selected run.");
  const run = await response.json();
  llmCurrentRun = run;
  llmCurrentRunId = run.run_id;
  llmSelectedEvent = 0;
  setLLMStatus(llmStatusLabel(run.status), llmStatusClass(run.status));
  renderLLMRun(run);
}

function renderPage(page, updateHash = true) {
  const selectedPage = pages[page] ? page : "overview";

  pageContent.innerHTML = pages[selectedPage]();
  topbarTitle.textContent = pageNames[selectedPage];
  breadcrumbCurrent.textContent = pageNames[selectedPage];

  navButtons.forEach(button => {
    const isActive = button.dataset.page === selectedPage;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-current", isActive ? "page" : "false");
  });

  if (updateHash && window.location.hash !== `#${selectedPage}`) {
    history.pushState({ page: selectedPage }, "", `#${selectedPage}`);
  }

  if (selectedPage === "experiment-1") {
    loadInitialEvaluationStatus();
  }
  if (selectedPage === "v5-enterprise") {
    loadV5SuiteData(false);
    if (v5LastSimResult) {
      renderV5SimulationOutput(v5LastSimResult);
    }
  }
  if (selectedPage === "agent-lab") {
    loadAgentScenarios();
  }
  if (selectedPage === "llm-lab") {
    loadLLMLab();
  }

  window.scrollTo({ top: 0, behavior: "auto" });
}

document.getElementById("navigation").addEventListener("click", event => {
  const button = event.target.closest("[data-page]");
  if (button) renderPage(button.dataset.page);
});

pageContent.addEventListener("click", event => {
  const goBtn = event.target.closest("[data-go]");
  if (goBtn) renderPage(goBtn.dataset.go);

  const zoomButton = event.target.closest("[data-v5-artifact-zoom]");
  if (zoomButton) {
    const dialog = document.getElementById("v5-artifact-dialog");
    const dialogImage = document.getElementById("v5-artifact-dialog-image");
    const dialogTitle = document.getElementById("v5-artifact-dialog-title");
    if (!dialog || !dialogImage || !dialogTitle) return;
    dialogImage.src = zoomButton.dataset.artifactSrc;
    dialogImage.alt = zoomButton.dataset.artifactTitle;
    dialogTitle.textContent = zoomButton.dataset.artifactTitle;
    dialog.showModal();
    return;
  }

  const closeZoomButton = event.target.closest("[data-v5-artifact-close]");
  if (closeZoomButton) {
    closeZoomButton.closest("dialog")?.close();
    return;
  }

  if (event.target.matches("#v5-artifact-dialog")) {
    event.target.close();
    return;
  }

  if (event.target.closest("#v5-refresh-btn")) {
    loadV5SuiteData(true);
    return;
  }

  if (event.target.closest("#v5-export-json-btn")) {
    exportV5ReportJson();
    return;
  }

  if (event.target.closest("#v5-run-btn")) {
    runV5Simulation();
    return;
  }

  const modelChoice = event.target.closest("[data-v5-model-key]");
  if (modelChoice && v5ReportData) {
    v5SelectedModelKey = modelChoice.dataset.v5ModelKey;
    renderV5Suite();
    return;
  }

  const tabBtn = event.target.closest("[data-v5-table-key]");
  if (tabBtn && v5ReportData) {
    v5ActiveTableKey = tabBtn.dataset.v5TableKey;
    v5SelectedRowIndex = null;
    renderV5Suite();
    return;
  }

  const rowEl = event.target.closest("[data-v5-row-index]");
  if (rowEl && v5ReportData) {
    const idx = Number(rowEl.dataset.v5RowIndex);
    v5SelectedRowIndex = v5SelectedRowIndex === idx ? null : idx;
    const tables = extractReportTables(v5ReportData);
    renderActiveV5Table(tables.find(t => t.key === v5ActiveTableKey));
    return;
  }

  const simStepBtn = event.target.closest("[data-v5-sim-step]");
  if (simStepBtn && v5LastSimResult) {
    v5SelectedSimStep = Number(simStepBtn.dataset.v5SimStep);
    renderV5SimulationOutput(v5LastSimResult);
    return;
  }

  const scenarioButton = event.target.closest("[data-agent-scenario]");
  if (scenarioButton) runAgentScenario(scenarioButton.dataset.agentScenario);

  const clearButton = event.target.closest("[data-agent-clear]");
  if (clearButton) clearAgentRun();

  const traceButton = event.target.closest("[data-agent-trace-index]");
  if (traceButton && agentRun) {
    selectedAgentStep = Number(traceButton.dataset.agentTraceIndex);
    renderAgentRun(agentRun);
  }

  const eventButton = event.target.closest("[data-llm-event-index]");
  if (eventButton && llmCurrentRun) {
    llmSelectedEvent = Number(eventButton.dataset.llmEventIndex);
    renderLLMRun(llmCurrentRun);
  }

  const historyButton = event.target.closest("[data-llm-history-run]");
  if (historyButton) {
    loadLLMHistoryRun(historyButton.dataset.llmHistoryRun).catch(error => {
      showLLMError(error.message || "Could not load selected run.");
    });
  }
});

pageContent.addEventListener("submit", event => {
  if (event.target.id === "llm-run-form") submitLLMRun(event);
});

pageContent.addEventListener("change", event => {
  if (event.target.id === "llm-mode") updateLLMProviderLabel();
  if (["llm-filter-status", "llm-filter-mode", "llm-filter-verdict"].includes(event.target.id)) {
    renderLLMHistory();
  }
});

window.addEventListener("popstate", () => {
  renderPage(window.location.hash.slice(1) || "overview", false);
});

window.addEventListener("hashchange", () => {
  const page = window.location.hash.slice(1);
  if (page && pages[page]) renderPage(page, false);
});

checkBackendHealth();
renderPage(window.location.hash.slice(1) || "overview", false);