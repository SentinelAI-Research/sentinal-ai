
const pageContent = document.getElementById("page-content");
const navButtons = [...document.querySelectorAll(".nav-item")];
const breadcrumbCurrent = document.getElementById("breadcrumb-current");
const topbarTitle = document.getElementById("topbar-title");

const pageNames = {
  overview: "Research Overview",
  "experiment-1": "Experiment 1 — Initial Benchmark",
  "experiment-2": "Experiment 2 — V2 Model Comparison",
  "experiment-3": "Experiment 3 — Ablation & Shortcut Audit",
  "experiment-4": "Experiment 4 — V3 Matched-Pair Benchmark",
  roadmap: "V4 Research Roadmap"
};

function metricCard(label, value, note = "", extraClass = "") {
  return `
    <article class="card metric-card ${extraClass}">
      <div class="card-title">${label}</div>
      <div class="metric-value numeric">${value}</div>
      <div class="metric-note">${note}</div>
    </article>`;
}

function nextPage(page, title, description) {
  return `
    <button class="next-link" data-go="${page}">
      <span><strong>${title}</strong><small>${description}</small></span>
      <span class="next-arrow">→</span>
    </button>`;
}

function pageHeader(kicker, title, description, tag = "") {
  return `
    <div class="experiment-header">
      <div class="section-kicker">${kicker}</div>
      <h1>${title}</h1>
      <p>${description}</p>
      ${tag ? `<div style="margin-top:13px"><span class="pill gray">${tag}</span></div>` : ""}
    </div>`;
}

function sectionHeading(title, description = "") {
  return `
    <div class="section-heading">
      <div><h2>${title}</h2>${description ? `<p>${description}</p>` : ""}</div>
    </div>`;
}

function comparisonTable(rows) {
  return `
    <div class="table-wrap">
      <table>
        <thead><tr><th>Metric</th><th>Baseline</th><th>Provenance-aware</th><th>Difference</th></tr></thead>
        <tbody>${rows.map(row => `
          <tr>
            <td>${row[0]}</td>
            <td class="numeric">${row[1]}</td>
            <td class="numeric">${row[2]}</td>
            <td><strong>${row[3]}</strong></td>
          </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

const pages = {
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
          <h2>Ablation & Shortcut Audit</h2>
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
      </div>

      ${sectionHeading("Current research position")}
      <div class="grid grid-3">
        ${metricCard("V3 matched pairs", "300", "600 trajectories · 3,600 rows")}
        ${metricCard("Baseline macro F1", "82.22%", "V3 test split")}
        ${metricCard("Provenance macro F1", "100%", "V3 test split; synthetic benchmark")}
      </div>
      <div class="callout warning">
        <strong>Important interpretation:</strong> the current 100% result is from a controlled synthetic benchmark. The ablation findings show that synthetic feature shortcuts must be investigated further before drawing real-world conclusions.
      </div>
      ${nextPage("roadmap", "Continue to V4 planning", "Design a stronger test with varied action sequences and held-out scenario templates")}
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
        ${metricCard("Provenance accuracy", "100%", "13 features")}
        ${metricCard("Accuracy lift", "+8.33 pp", "Percentage-point difference")}
        ${metricCard("Macro F1 lift", "+38.10 pp", "61.90% → 100%")}
      </div>

      <article class="card content-card">
        <h2>Model comparison</h2>
        ${comparisonTable([
      ["Accuracy", "91.67%", "100.00%", "+8.33 pp"],
      ["Macro F1", "61.90%", "100.00%", "+38.10 pp"]
    ])}
        <p class="muted small" style="margin-top:12px">Evaluation used a held-out test split. The paired variants were kept together by source group to avoid pair overlap between training and testing.</p>
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
            <tr><td>Baseline without provenance</td><td>7</td><td>91.67%</td><td>61.90%</td><td>Missed the VIOLATION class</td></tr>
            <tr><td>Full provenance</td><td>13</td><td>100.00%</td><td>100.00%</td><td>Perfect on this test set</td></tr>
            <tr><td>Drop has_sensitive_ancestor</td><td>12</td><td>100.00%</td><td>100.00%</td><td>Score unchanged</td></tr>
            <tr><td>Drop sensitive_ancestor_count</td><td>12</td><td>100.00%</td><td>100.00%</td><td>Score unchanged</td></tr>
            <tr><td>Drop both ancestor indicators</td><td>11</td><td>100.00%</td><td>100.00%</td><td>Score unchanged</td></tr>
            <tr><td>Drop ancestor sensitivity group</td><td>10</td><td>91.67%</td><td>61.90%</td><td>Returned to baseline-level performance</td></tr>
            <tr><td>Drop provenance structure group</td><td>10</td><td>100.00%</td><td>100.00%</td><td>Score unchanged</td></tr>
            <tr><td>Drop all provenance</td><td>7</td><td>91.67%</td><td>61.90%</td><td>Returned to baseline-level performance</td></tr>
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
        ${metricCard("Provenance accuracy", "100%", "13 features")}
        ${metricCard("Accuracy lift", "+8.33 pp", "Provenance minus baseline")}
        ${metricCard("Macro F1 lift", "+17.78 pp", "82.22% → 100%")}
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

      ${nextPage("roadmap", "Next: V4 research roadmap", "Vary action sequences and evaluate on held-out scenario templates")}
    `;
  },

  roadmap() {
    return `
      ${pageHeader("RESEARCH ROADMAP · NEXT STAGE", "V4: test generalization, not just a perfect score", "The next experiment should test whether provenance-aware monitoring continues to help when the model encounters different action sequences and scenario templates.", "Proposed next experiment · Not yet evaluated")}

      ${sectionHeading("What V4 should change")}
      <div class="timeline">
        <div class="timeline-item">
          <div class="timeline-marker">1</div>
          <div class="timeline-body">
            <h3>Vary the action sequences</h3>
            <p>Generate different valid sequences and branching patterns instead of using the same six-step order for every trajectory.</p>
          </div>
        </div>
        <div class="timeline-item">
          <div class="timeline-marker">2</div>
          <div class="timeline-body">
            <h3>Create held-out scenario templates</h3>
            <p>Reserve entire scenario templates for testing. Keep all related variants from a template within one split to reduce template leakage.</p>
          </div>
        </div>
        <div class="timeline-item">
          <div class="timeline-marker">3</div>
          <div class="timeline-body">
            <h3>Audit features before training</h3>
            <p>Check whether action types, tool IDs, depths, and sensitivity counts accidentally encode labels. Avoid using label-derived metadata as model input.</p>
          </div>
        </div>
        <div class="timeline-item">
          <div class="timeline-marker">4</div>
          <div class="timeline-body">
            <h3>Run baseline, provenance, and ablations</h3>
            <p>Use the same split and model settings for all variants. Report accuracy, macro F1, per-class precision/recall, and confidence intervals where appropriate.</p>
          </div>
        </div>
        <div class="timeline-item">
          <div class="timeline-marker">5</div>
          <div class="timeline-body">
            <h3>Evaluate realistic trajectories</h3>
            <p>Where feasible, add carefully designed scenarios with multiple transformations, indirect data flows, and different destinations. Keep external actions simulated for the demo.</p>
          </div>
        </div>
      </div>

      ${sectionHeading("Questions V4 should answer")}
      <div class="grid grid-2">
        <article class="card">
          <h3>Does provenance improve held-out generalization?</h3>
          <p class="muted">Compare the baseline and provenance-aware model on scenario templates that neither model saw during training.</p>
        </article>
        <article class="card">
          <h3>Does the model rely on shortcuts?</h3>
          <p class="muted">Inspect feature distributions and repeat ablations to check whether performance depends on one artificial feature.</p>
        </article>
        <article class="card">
          <h3>Does it detect risk earlier?</h3>
          <p class="muted">Measure how early in a trajectory the monitor identifies a risky information flow—not only whether it classifies the final step correctly.</p>
        </article>
        <article class="card">
          <h3>Can we reproduce the findings?</h3>
          <p class="muted">Fix random seeds, preserve dataset versions, save split manifests, and report per-class metrics alongside aggregate scores.</p>
        </article>
      </div>

      <article class="card content-card">
        <h2>Current status</h2>
        <p>V2 comparison, V2 ablation, and V3 matched-pair evaluation have been run according to the results recorded for this project. V4 is the proposed next research stage; no V4 results are being claimed here.</p>
      </article>

      ${nextPage("overview", "Return to research overview", "Navigate back to the experiment index")}
    `;
  }
};

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

  window.scrollTo({ top: 0, behavior: "auto" });
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
  } catch (error) {
    target.textContent =
      "Initial evaluation metrics are not currently available through the dashboard API. The dataset counts and experiment description above are recorded project details; no initial model score is fabricated.";
  }
}

document.getElementById("navigation").addEventListener("click", event => {
  const button = event.target.closest("[data-page]");
  if (button) renderPage(button.dataset.page);
});

pageContent.addEventListener("click", event => {
  const button = event.target.closest("[data-go]");
  if (button) renderPage(button.dataset.go);
});

window.addEventListener("popstate", () => {
  renderPage(window.location.hash.slice(1) || "overview", false);
});

window.addEventListener("hashchange", () => {
  const page = window.location.hash.slice(1);
  if (page && pages[page]) renderPage(page, false);
});

renderPage(window.location.hash.slice(1) || "overview", false);
