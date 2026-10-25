const fmt = (value) =>
  value == null ? "—" : `${(Number(value) * 100).toFixed(2)}%`;
const fmtWhole = (value) =>
  value == null ? "—" : Number(value).toLocaleString();
const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const metricDefinitions = [
  ["accuracy", "Accuracy"],
  ["macro_f1", "Macro F1"],
  ["macro_precision", "Macro precision"],
  ["macro_recall", "Macro recall"],
  ["weighted_f1", "Weighted F1"],
  ["final_step_accuracy", "Final-step accuracy"],
  ["final_safe_vs_violation_accuracy", "Final SAFE / VIOLATION accuracy"],
];

function byId(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  const element = byId(id);
  if (element) element.textContent = value;
}

function renderFeatures(payload) {
  const without = payload.models.without_provenance.features || [];
  const withProvenance = payload.models.with_provenance.features || [];
  setText("feature-with", withProvenance.length || "—");
  setText("feature-without", without.length || "—");
  setText("with-count", withProvenance.length || "—");
  setText("without-count", without.length || "—");
  const render = (features, className = "") =>
    features.length
      ? features
          .map(
            (feature) =>
              `<div class="feature-chip ${className}">${escapeHtml(feature)}</div>`,
          )
          .join("")
      : '<div class="feature-chip">Metadata not available</div>';
  byId("without-features").innerHTML = render(without);
  byId("with-features").innerHTML = render(withProvenance, "green-list");
}

function renderOverview(payload) {
  const evaluation = payload.evaluation;
  const without = evaluation?.without_provenance;
  const withProvenance = evaluation?.with_provenance;
  const improvement = evaluation?.improvement;
  setText("hero-accuracy", fmt(withProvenance?.overall?.accuracy));
  setText("base-accuracy", fmt(without?.overall?.accuracy));
  setText("final-improvement", fmt(improvement?.final_step_accuracy));
  const status = byId("source-status");
  const notice = byId("data-notice");
  // The data notice is intentionally commented out in index.html for now.
  if (evaluation && !payload.evaluation_is_fallback) {
    status.className = "status-pill ready";
    status.innerHTML =
      '<span class="status-dot"></span> Evaluation source loaded';
    if (notice) notice.classList.add("hidden");
  } else if (evaluation) {
    status.className = "status-pill snapshot";
    status.innerHTML = '<span class="status-dot"></span> Using prompt snapshot';
    if (notice) notice.classList.remove("hidden");
  } else {
    status.className = "status-pill loading";
    status.innerHTML =
      '<span class="status-dot"></span> Awaiting evaluation JSON';
    if (notice) notice.classList.remove("hidden");
  }
  setText(
    "updated-at",
    payload.evaluation_updated_at
      ? `Evaluation source: ${new Date(payload.evaluation_updated_at).toLocaleString()}`
      : "Source: waiting for evaluation artifact",
  );
}

function renderDataset(payload) {
  const design = payload.experiment_design || {};
  const dataset = payload.dataset || {};
  const total = dataset.available ? dataset.record_count : design.total_records;
  const trajectories = dataset.available
    ? dataset.trajectory_count
    : design.total_trajectories;
  const sourceCounts = dataset.source_datasets || {};
  const amiRecords = sourceCounts.AMI ?? sourceCounts.ami ?? design.ami_records;
  const enronRecords =
    sourceCounts.ENRON ?? sourceCounts.enron ?? design.enron_records;
  setText("dataset-total", fmtWhole(total));
  setText("trajectory-count", fmtWhole(trajectories));
  setText("steps-count", fmtWhole(design.steps_per_trajectory));
  setText("ami-trajectories", fmtWhole(design.ami_trajectories));
  setText("enron-trajectories", fmtWhole(design.enron_trajectories));
  setText("ami-records", fmtWhole(amiRecords));
  setText("enron-records", fmtWhole(enronRecords));
  const max = Math.max(Number(amiRecords || 0), Number(enronRecords || 0), 1);
  byId("ami-bar").style.width = `${(Number(amiRecords || 0) / max) * 100}%`;
  byId("enron-bar").style.width = `${(Number(enronRecords || 0) / max) * 100}%`;
  setText(
    "dataset-note",
    dataset.available ? "Observed JSONL records" : "Documented scope",
  );
}

function renderComparison(payload) {
  const evaluation = payload.evaluation;
  const without = evaluation?.without_provenance;
  const withProvenance = evaluation?.with_provenance;
  const improvement = evaluation?.improvement;
  const rows = metricDefinitions
    .map(([key, label]) => {
      let base;
      let lineage;
      let lift;
      if (key === "final_step_accuracy") {
        base = without?.final_step?.accuracy;
        lineage = withProvenance?.final_step?.accuracy;
        lift = improvement?.final_step_accuracy;
      } else if (key === "final_safe_vs_violation_accuracy") {
        base = without?.final_safe_vs_violation?.accuracy;
        lineage = withProvenance?.final_safe_vs_violation?.accuracy;
        lift = improvement?.final_safe_vs_violation_accuracy;
      } else {
        base = without?.overall?.[key];
        lineage = withProvenance?.overall?.[key];
        lift = improvement?.[key];
      }
      return `<div class="comparison-row">
      <div class="comparison-metric">${label}</div>
      <div class="comparison-value"><span class="comparison-number">${fmt(base)}</span><div class="mini-track"><div class="mini-fill" style="width:${(base ?? 0) * 100}%"></div></div></div>
      <div class="comparison-value"><span class="comparison-number">${fmt(lineage)}</span><div class="mini-track"><div class="mini-fill green" style="width:${(lineage ?? 0) * 100}%"></div></div></div>
      <div class="lift">${fmt(lift)}</div>
    </div>`;
    })
    .join("");
  byId("comparison-rows").innerHTML = rows;
}

function renderMatrix(id, matrix, score) {
  const element = byId(id);
  setText(score, matrix ? fmt(matrix.score) : "—");
  if (!matrix) {
    element.innerHTML =
      '<div class="matrix-y">True</div><div class="matrix-axis">SAFE</div><div class="matrix-axis">EARLY_RISK</div><div class="matrix-axis">VIOLATION</div><div class="matrix-y">Awaiting evaluation</div>';
    return;
  }
  const values = matrix.values || [];
  const max = Math.max(...values.flat(), 1);
  let html = '<div class="matrix-y">True / Pred.</div>';
  ["SAFE", "EARLY", "VIOL."].forEach((label) => {
    html += `<div class="matrix-axis">${label}</div>`;
  });
  values.forEach((row, rowIndex) => {
    html += `<div class="matrix-y">${["SAFE", "EARLY", "VIOL."][rowIndex]}</div>`;
    row.forEach((value, colIndex) => {
      const diag = rowIndex === colIndex;
      const intensity = value / max;
      const className = value === 0 ? "zero" : diag ? "high" : "diag";
      html += `<div class="${className}" style="${diag ? `opacity:${Math.max(0.35, intensity)}` : ""}">${value}</div>`;
    });
  });
  element.innerHTML = html;
}

function matrixFromResult(result) {
  if (!result || !Array.isArray(result.confusion_matrix)) return null;
  return { values: result.confusion_matrix, score: result.accuracy };
}

function renderMatrices(payload) {
  renderMatrix(
    "without-matrix",
    matrixFromResult(payload.evaluation?.without_provenance?.overall),
    "without-matrix-score",
  );
  renderMatrix(
    "with-matrix",
    matrixFromResult(payload.evaluation?.with_provenance?.overall),
    "with-matrix-score",
  );
}

const referenceTrajectory = [
  ["01", "read_file", "source object"],
  ["02", "summarize", "transformation"],
  ["03", "write_memory", "persistent memory"],
  ["04", "read_memory", "retrieval"],
  ["05", "paraphrase", "transformation"],
  ["06", "send_message", "external action"],
];

function renderTrajectory(payload) {
  const trajectory = payload.dataset?.trajectory || [];
  const items = trajectory.length
    ? trajectory.map((item) => [
        String(item.step ?? "").padStart(2, "0"),
        item.tool || "unknown tool",
        item.transformation || item.label || "observed step",
      ])
    : referenceTrajectory;
  byId("trajectory-chain").innerHTML = items
    .map(
      ([step, tool, detail]) => `
    <div class="trajectory-node">
      <div class="node-dot">${escapeHtml(step)}</div>
      <div class="node-step">STEP ${escapeHtml(step)}</div>
      <div class="node-tool">${escapeHtml(tool)}</div>
      <div class="node-detail">${escapeHtml(detail)}</div>
    </div>`,
    )
    .join("");
  // The empty-state copy is temporarily commented out in index.html.
  const trajectoryEmpty = byId("trajectory-empty");
  if (trajectoryEmpty) {
    trajectoryEmpty.classList.toggle("hidden", trajectory.length > 0);
  }
  setText(
    "trajectory-note",
    trajectory.length ? "Observed trajectory JSONL" : "Reference lineage path",
  );
}

async function loadDashboard() {
  const button = byId("refresh-button");
  button.disabled = true;
  try {
    const response = await fetch("/api/evaluation", { cache: "no-store" });
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const payload = await response.json();
    renderOverview(payload);
    renderDataset(payload);
    renderFeatures(payload);
    renderComparison(payload);
    // Confusion-matrix section is temporarily commented out in index.html.
    // renderMatrices(payload);
    renderTrajectory(payload);
  } catch (error) {
    byId("source-status").className = "status-pill loading";
    byId("source-status").innerHTML =
      '<span class="status-dot"></span> Dashboard API unavailable';
    console.error(error);
  } finally {
    button.disabled = false;
  }
}

byId("refresh-button").addEventListener("click", loadDashboard);
loadDashboard();
