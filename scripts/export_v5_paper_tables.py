from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEV_REPORT_PATH = ROOT / "models" / "v5" / "dev_tuning_report.json"
TEST_REPORT_PATH = ROOT / "results" / "v5" / "v5_test_evaluation_report.json"
OUT_DIR = ROOT / "results" / "v5"


def _fmt_pct_ci(val: float, ci: list[float]) -> str:
    return f"{val * 100:.1f}\\% [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"


def _fmt_pct_ci_md(val: float, ci: list[float]) -> str:
    return f"{val * 100:.1f}% [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"


def build_latex_tables(
    dev_report: dict[str, Any],
    test_report: dict[str, Any],
) -> str:
    lines: list[str] = [
        "% Auto-generated SentinelAI V5 Publication Tables (booktabs format)",
        "% Requires: \\usepackage{booktabs}",
        "",
        "% ==========================================================================",
        "% TABLE B6: 30% DEV SET MULTI-MODEL COMPARISON (GROUPKFOLD BY SOURCE DOC)",
        "% ==========================================================================",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{30\\% Development Set Multi-Model Cross-Validation Comparison (GroupKFold by Source Document).}",
        "\\label{tab:v5_dev_models}",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        "Model Family & \\multicolumn{3}{c}{C3 (No Provenance)} & \\multicolumn{3}{c}{C4 (Full Provenance DAG)} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        " & AUROC & PR-AUC & F1@0.5 & AUROC & PR-AUC & F1@0.5 \\\\",
        "\\midrule",
    ]
    for m_name in dev_report["models"]["C4"]:
        c3_m = dev_report["models"]["C3"][m_name]
        c4_m = dev_report["models"]["C4"][m_name]
        lines.append(
            f"{m_name} & {c3_m['cv_auroc']:.4f} & {c3_m['cv_pr_auc']:.4f} & {c3_m['cv_f1_at_0_5']:.4f} "
            f"& \\textbf{{{c4_m['cv_auroc']:.4f}}} & \\textbf{{{c4_m['cv_pr_auc']:.4f}}} & \\textbf{{{c4_m['cv_f1_at_0_5']:.4f}}} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "% ==========================================================================",
            "% TABLE R1: 70% HELD-OUT TEST SET OVERALL COMPARISON (WITH 95% BOOTSTRAP CI)",
            "% ==========================================================================",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Overall Security, Utility, and Early-Warning Comparison on the 70\\% Held-Out Test Split (95\\% Bootstrap CI).}",
            "\\label{tab:v5_r1_overall}",
            "\\begin{tabular}{@{}lcccccc@{}}",
            "\\toprule",
            "Config & ASR ($\\downarrow$) & Block Rate ($\\uparrow$) & Benign Comp. ($\\uparrow$) & Early Warn ($\\uparrow$) & Mean Lead & Latency (ms) \\\\",
            "\\midrule",
        ]
    )
    for r in test_report["R1_overall_comparison"]:
        cfg = r["config"].replace("_", "\\_")
        lines.append(
            f"{cfg} & {_fmt_pct_ci(r['asr'], r['asr_ci95'])} "
            f"& {_fmt_pct_ci(r['attack_block_rate'], r['attack_block_ci95'])} "
            f"& {_fmt_pct_ci(r['benign_completion_rate'], r['benign_completion_ci95'])} "
            f"& {_fmt_pct_ci(r['early_warning_rate'], r['early_warning_ci95'])} "
            f"& {r['mean_lead_steps']:.2f} & {r['mean_latency_ms']:.2f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "% ==========================================================================",
            "% TABLE R2: ATTACK BLOCK RATE BY ATTACK FAMILY (70% HELD-OUT TEST SPLIT)",
            "% ==========================================================================",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Attack Block Rate (\\%) Across Adversarial Attack Families on the 70\\% Test Split.}",
            "\\label{tab:v5_r2_families}",
            "\\begin{tabular}{@{}lccccccc@{}}",
            "\\toprule",
            "Attack Family & C0 & C1 & C2 & C2b & C3 & C4-Heur. & C4-ML \\\\",
            "\\midrule",
        ]
    )
    cfgs_order = ["C0", "C1", "C2", "C2b", "C3", "C4-Heuristic", "C4-ML"]
    for fam, fam_dict in sorted(test_report["R2_by_attack_family"].items()):
        fam_label = fam.replace("_", "\\_")
        vals = " & ".join(
            f"{fam_dict[c]['block_rate'] * 100:.1f}\\%"
            for c in cfgs_order
            if c in fam_dict
        )
        lines.append(f"{fam_label} & {vals} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "% ==========================================================================",
            "% TABLE R5: CONTROLLED COMPONENT ABLATION STUDY (70% HELD-OUT TEST SPLIT)",
            "% ==========================================================================",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Controlled Component Ablation Study on the 70\\% Test Split.}",
            "\\label{tab:v5_r5_ablations}",
            "\\begin{tabular}{@{}lcccc@{}}",
            "\\toprule",
            "Ablation Variant & ASR ($\\downarrow$) & Block ($\\uparrow$) & Early Warn ($\\uparrow$) & Benign Comp. ($\\uparrow$) \\\\",
            "\\midrule",
        ]
    )
    for r in test_report["R5_component_ablations"]:
        name = r["config"].replace("_", "\\_")
        lines.append(
            f"{name} & {r['asr'] * 100:.1f}\\% & {r['attack_block_rate'] * 100:.1f}\\% "
            f"& {r['early_warning_rate'] * 100:.1f}\\% & {r['benign_completion_rate'] * 100:.1f}\\% \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    if "test_auc_metrics" in test_report:
        lines.extend(
            [
                "% ==========================================================================",
                "% TABLE R6: HELD-OUT STEP CLASSIFICATION & TEST RANKING METRICS",
                "% ==========================================================================",
                "\\begin{table*}[t]",
                "\\centering",
                "\\caption{Held-Out V5 Step Classification and Test Ranking Metrics.}",
                "\\label{tab:v5_r6_metrics}",
                "\\begin{tabular}{@{}lcccccc@{}}",
                "\\toprule",
                "Preset & Family & Accuracy & Macro F1 & Weighted F1 & Test ROC-AUC & Test PR-AUC \\\\",
                "\\midrule",
            ]
        )
        auc = test_report["test_auc_metrics"]
        for preset in ("C3", "C4"):
            for model_name, model_metrics in sorted(auc.get(preset, {}).items()):
                runtime_name = "C3" if preset == "C3" else "C4-ML"
                three = model_metrics.get("step_classification_metrics", {})
                lines.append(
                    f"{runtime_name} & {model_name} & "
                    f"{three.get('accuracy', 0):.4f} & "
                    f"{three.get('macro_f1', 0):.4f} & "
                    f"{three.get('weighted_f1', 0):.4f} & "
                    f"{model_metrics.get('test_roc_auc') if model_metrics.get('test_roc_auc') is not None else 'N/A'} & "
                    f"{model_metrics.get('test_pr_auc') if model_metrics.get('test_pr_auc') is not None else 'N/A'} \\\\"
                )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def build_markdown_tables(
    dev_report: dict[str, Any],
    test_report: dict[str, Any],
) -> str:
    md: list[str] = [
        "# SentinelAI V5 Publication Tables & Empirical Summary",
        "",
        "## Table B6: 30% Development Set Multi-Model Comparison (GroupKFold by Source Document)",
        "",
        "| Model Family | C3 AUROC | C3 PR-AUC | C3 F1@0.5 | C4 AUROC | C4 PR-AUC | C4 F1@0.5 | Δ PR-AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name in dev_report["models"]["C4"]:
        c3_m = dev_report["models"]["C3"][m_name]
        c4_m = dev_report["models"]["C4"][m_name]
        delta = c4_m["cv_pr_auc"] - c3_m["cv_pr_auc"]
        md.append(
            f"| {m_name} | {c3_m['cv_auroc']:.4f} | {c3_m['cv_pr_auc']:.4f} | {c3_m['cv_f1_at_0_5']:.4f} "
            f"| **{c4_m['cv_auroc']:.4f}** | **{c4_m['cv_pr_auc']:.4f}** | **{c4_m['cv_f1_at_0_5']:.4f}** | **+{delta:.4f}** |"
        )

    md.extend(
        [
            "",
            "## Table R1: Overall Test Set Comparison (70% Held-Out Split, 95% Bootstrap CI)",
            "",
            "| Config | ASR (↓) | Block Rate (↑) | Detection Rate (↑) | Benign Comp. (↑) | Early Warning (↑) | Mean Lead Steps | Mean Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in test_report["R1_overall_comparison"]:
        md.append(
            f"| **{r['config']}** | {_fmt_pct_ci_md(r['asr'], r['asr_ci95'])} "
            f"| {_fmt_pct_ci_md(r['attack_block_rate'], r['attack_block_ci95'])} "
            f"| {_fmt_pct_ci_md(r['attack_detection_rate'], r['attack_detection_ci95'])} "
            f"| {_fmt_pct_ci_md(r['benign_completion_rate'], r['benign_completion_ci95'])} "
            f"| {_fmt_pct_ci_md(r['early_warning_rate'], r['early_warning_ci95'])} "
            f"| {r['mean_lead_steps']:.2f} | {r['mean_latency_ms']:.2f} |"
        )

    md.extend(
        [
            "",
            "## Table R2: Attack Block Rate (%) by Attack Family (70% Test Split)",
            "",
            "| Attack Family | C0 | C1 | C2 | C2b | C3 | C4-Heuristic | C4-ML |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    cfgs_order = ["C0", "C1", "C2", "C2b", "C3", "C4-Heuristic", "C4-ML"]
    for fam, fam_dict in sorted(test_report["R2_by_attack_family"].items()):
        vals = " | ".join(
            f"{fam_dict[c]['block_rate'] * 100:.1f}%"
            for c in cfgs_order
            if c in fam_dict
        )
        md.append(f"| `{fam}` | {vals} |")

    md.extend(
        [
            "",
            "## Table R3 & R4: Early-Warning Lead Time & Runtime Latency Breakdown",
            "",
            "| Config | Early Warning Rate (↑) | Mean Lead (steps) | Median Lead (steps) | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in test_report["R1_overall_comparison"]:
        md.append(
            f"| **{r['config']}** | {r['early_warning_rate'] * 100:.1f}% | {r['mean_lead_steps']:.2f} "
            f"| {r['median_lead_steps']:.2f} | {r['mean_latency_ms']:.2f} | {r['median_latency_ms']:.2f} | {r['p95_latency_ms']:.2f} |"
        )

    md.extend(
        [
            "",
            "## Table R5: Controlled Component Ablation Study (70% Test Split)",
            "",
            "| Ablation Variant | ASR (↓) | Block Rate (↑) | Early Warning (↑) | Benign Completion (↑) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in test_report["R5_component_ablations"]:
        md.append(
            f"| `{r['config']}` | {r['asr'] * 100:.1f}% | {r['attack_block_rate'] * 100:.1f}% "
            f"| {r['early_warning_rate'] * 100:.1f}% | {r['benign_completion_rate'] * 100:.1f}% |"
        )
    if "test_auc_metrics" in test_report:
        md.extend(
            [
                "",
                "## Table R6: Held-Out Step Classification & Test Ranking Metrics",
                "",
                "| Preset | Classifier | Step Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Test ROC-AUC | Test PR-AUC |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        auc = test_report["test_auc_metrics"]
        for preset in ("C3", "C4"):
            runtime_name = "C3" if preset == "C3" else "C4-ML"
            for model_name, model_metrics in sorted(auc.get(preset, {}).items()):
                three = model_metrics.get("step_classification_metrics", {})
                md.append(
                    f"| {runtime_name} | {model_name} "
                    f"| {three.get('accuracy', 0):.4f} "
                    f"| {three.get('macro_precision', 0):.4f} "
                    f"| {three.get('macro_recall', 0):.4f} "
                    f"| {three.get('macro_f1', 0):.4f} "
                    f"| {three.get('weighted_f1', 0):.4f} "
                    f"| {model_metrics.get('test_roc_auc') if model_metrics.get('test_roc_auc') is not None else 'N/A'} "
                    f"| {model_metrics.get('test_pr_auc') if model_metrics.get('test_pr_auc') is not None else 'N/A'} |"
                )
    md.append("")
    return "\n".join(md)


def generate_figures(
    dev_report: dict[str, Any],
    test_report: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    saved: list[Path] = []
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    # Figure 1: Security vs. Utility & Early-Warning Comparison (R1)
    r1 = test_report["R1_overall_comparison"]
    configs = [r["config"] for r in r1]
    block_rates = [r["attack_block_rate"] * 100 for r in r1]
    benign_comp = [r["benign_completion_rate"] * 100 for r in r1]
    early_warn = [r["early_warning_rate"] * 100 for r in r1]

    x = np.arange(len(configs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=300)
    ax.bar(x - w, block_rates, width=w, label="Attack Block Rate (↑)", color="#1f77b4")
    ax.bar(x, benign_comp, width=w, label="Benign Task Completion (↑)", color="#2ca02c")
    ax.bar(x + w, early_warn, width=w, label="Early-Warning Rate (↑)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 112)
    ax.set_title(
        "SentinelAI V5: Attack Prevention, Benign Utility & Early Warning (70% Test Set)"
    )
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    f1 = out_dir / "fig1_security_utility_tradeoff.png"
    fig.savefig(f1)
    plt.close(fig)
    saved.append(f1)

    # Figure 2: Attack Family Block Rate Heatmap (R2)
    cfgs_order = ["C0", "C1", "C2", "C2b", "C3", "C4-Heuristic", "C4-ML"]
    families = sorted(test_report["R2_by_attack_family"].keys())
    matrix = np.array(
        [
            [
                test_report["R2_by_attack_family"][fam][c]["block_rate"] * 100
                for c in cfgs_order
            ]
            for fam in families
        ]
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=300)
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(cfgs_order)))
    ax.set_xticklabels(cfgs_order, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families)
    for i in range(len(families)):
        for j in range(len(cfgs_order)):
            val = matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.0f}%",
                ha="center",
                va="center",
                color="white" if val > 55 else "black",
                fontsize=8.5,
            )
    ax.set_title("Attack Block Rate (%) Across Multi-Hop & Memory Attack Families (R2)")
    fig.colorbar(im, ax=ax, label="Block Rate (%)")
    fig.tight_layout()
    f2 = out_dir / "fig2_attack_family_heatmap.png"
    fig.savefig(f2)
    plt.close(fig)
    saved.append(f2)

    # Figure 3: Dev Multi-Model PR-AUC Lift (B6) & Component Ablations (R5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=300)
    model_names = list(dev_report["models"]["C4"].keys())
    c3_ap = [dev_report["models"]["C3"][m]["cv_pr_auc"] for m in model_names]
    c4_ap = [dev_report["models"]["C4"][m]["cv_pr_auc"] for m in model_names]
    xm = np.arange(len(model_names))
    ax1.bar(xm - 0.18, c3_ap, width=0.36, label="C3 (No Provenance)", color="#7f7f7f")
    ax1.bar(xm + 0.18, c4_ap, width=0.36, label="C4 (Provenance DAG)", color="#1f77b4")
    ax1.set_xticks(xm)
    ax1.set_xticklabels(model_names, rotation=18, ha="right")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Dev CV PR-AUC")
    ax1.set_title("(a) Provenance Gain Across ML Models (B6)")
    ax1.legend(loc="lower right", frameon=True)

    r5 = test_report["R5_component_ablations"]
    abl_labels = ["Full C4", "No Inherit.", "No Mem. Lineage", "No Risk Fusion"]
    abl_block = [r["attack_block_rate"] * 100 for r in r5]
    abl_early = [r["early_warning_rate"] * 100 for r in r5]
    xa = np.arange(len(abl_labels))
    ax2.bar(xa - 0.18, abl_block, width=0.36, label="Attack Block (%)", color="#2ca02c")
    ax2.bar(
        xa + 0.18, abl_early, width=0.36, label="Early Warning (%)", color="#ff7f0e"
    )
    ax2.set_xticks(xa)
    ax2.set_xticklabels(abl_labels, rotation=18, ha="right")
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("(b) Component Ablation Impact (R5)")
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    f3 = out_dir / "fig3_dev_models_and_ablations.png"
    fig.savefig(f3)
    plt.close(fig)
    saved.append(f3)

    return saved


def export_all(
    *,
    dev_report_path: Path = DEV_REPORT_PATH,
    test_report_path: Path = TEST_REPORT_PATH,
    out_dir: Path = OUT_DIR,
) -> dict[str, str]:
    dev_report = json.loads(dev_report_path.read_text(encoding="utf-8"))
    test_report = json.loads(test_report_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tex_path = out_dir / "paper_tables.tex"
    md_path = out_dir / "paper_tables.md"
    tex_path.write_text(build_latex_tables(dev_report, test_report), encoding="utf-8")
    md_path.write_text(build_markdown_tables(dev_report, test_report), encoding="utf-8")
    figs = generate_figures(dev_report, test_report, out_dir)

    return {
        "latex_tables": str(tex_path),
        "markdown_tables": str(md_path),
        "figures": [str(f) for f in figs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LaTeX/Markdown paper tables and publication figures (Step 5)."
    )
    parser.add_argument("--dev-report", type=Path, default=DEV_REPORT_PATH)
    parser.add_argument("--test-report", type=Path, default=TEST_REPORT_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    outputs = export_all(
        dev_report_path=args.dev_report,
        test_report_path=args.test_report,
        out_dir=args.out_dir,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
