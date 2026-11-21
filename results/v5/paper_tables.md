# SentinelAI V5 Publication Tables & Empirical Summary

## Table B6: 30% Development Set Multi-Model Comparison (GroupKFold by Source Document)

| Model Family | C3 AUROC | C3 PR-AUC | C3 F1@0.5 | C4 AUROC | C4 PR-AUC | C4 F1@0.5 | Δ PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HistGradientBoosting | 0.7542 | 0.6887 | 0.7005 | **0.9874** | **0.9837** | **0.9483** | **+0.2950** |
| RandomForest | 0.7694 | 0.7237 | 0.6857 | **0.9867** | **0.9830** | **0.9476** | **+0.2593** |
| LogisticRegression | 0.6974 | 0.6163 | 0.5918 | **0.9814** | **0.9765** | **0.9362** | **+0.3602** |
| MLPClassifier | 0.7339 | 0.6671 | 0.6761 | **0.9851** | **0.9797** | **0.9392** | **+0.3126** |

## Table R1: Overall Test Set Comparison (70% Held-Out Split, 95% Bootstrap CI)

| Config | ASR (↓) | Block Rate (↑) | Detection Rate (↑) | Benign Comp. (↑) | Early Warning (↑) | Mean Lead Steps | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0** | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 0.00 | 0.40 |
| **C1** | 85.7% [81.4, 89.6] | 14.3% [10.4, 18.6] | 14.3% [10.4, 18.6] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 0.00 | 0.42 |
| **C2** | 85.7% [81.4, 89.6] | 14.3% [10.4, 18.6] | 14.3% [10.4, 18.6] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 0.00 | 0.35 |
| **C2b** | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 80.0% [74.3, 85.2] | 0.0% [0.0, 0.0] | 0.00 | 0.28 |
| **C3** | 85.7% [81.4, 89.6] | 14.3% [10.4, 18.6] | 47.9% [42.1, 53.6] | 100.0% [100.0, 100.0] | 1.8% [0.4, 3.6] | 0.04 | 68.96 |
| **C4-Heuristic** | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 75.4% [70.4, 80.0] | 2.75 | 0.31 |
| **C4-ML** | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 85.7% [81.4, 89.6] | 3.62 | 6.14 |

## Table R2: Attack Block Rate (%) by Attack Family (70% Test Split)

| Attack Family | C0 | C1 | C2 | C2b | C3 | C4-Heuristic | C4-ML |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `derived_fact_leakage` | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| `direct_leakage` | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `long_horizon_leakage` | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| `memory_mediated_leakage` | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| `mixed_source_leakage` | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| `paraphrase_leakage` | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| `summary_leakage` | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 100.0% |

## Table R3 & R4: Early-Warning Lead Time & Runtime Latency Breakdown

| Config | Early Warning Rate (↑) | Mean Lead (steps) | Median Lead (steps) | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0** | 0.0% | 0.00 | 0.00 | 0.40 | 0.24 | 0.76 |
| **C1** | 0.0% | 0.00 | 0.00 | 0.42 | 0.27 | 0.83 |
| **C2** | 0.0% | 0.00 | 0.00 | 0.35 | 0.25 | 0.71 |
| **C2b** | 0.0% | 0.00 | 0.00 | 0.28 | 0.21 | 0.61 |
| **C3** | 1.8% | 0.04 | 0.00 | 68.96 | 65.53 | 84.07 |
| **C4-Heuristic** | 75.4% | 2.75 | 2.50 | 0.31 | 0.22 | 0.65 |
| **C4-ML** | 85.7% | 3.62 | 3.50 | 6.14 | 5.59 | 11.33 |

## Table R5: Controlled Component Ablation Study (70% Test Split)

| Ablation Variant | ASR (↓) | Block Rate (↑) | Early Warning (↑) | Benign Completion (↑) |
| :--- | :---: | :---: | :---: | :---: |
| `R5.0_Full_C4` | 0.0% | 100.0% | 85.7% | 100.0% |
| `R5.1_No_Sensitivity_Inheritance` | 85.7% | 14.3% | 85.7% | 100.0% |
| `R5.2_No_Memory_Lineage` | 28.2% | 71.8% | 85.7% | 100.0% |
| `R5.3_Policy_Only_No_Risk_Fusion` | 0.0% | 100.0% | 0.0% | 100.0% |

## Table R6: Held-Out Step Classification & Test Ranking Metrics

| Preset | Classifier | Step Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C3 | HistGradientBoosting | 0.4815 | 0.7480 | 0.5180 | 0.4578 | 0.3578 | 0.807823 | 0.775078 |
| C3 | LogisticRegression | 0.4347 | 0.7843 | 0.3526 | 0.2374 | 0.2874 | 0.768556 | 0.729585 |
| C3 | MLPClassifier | 0.4608 | 0.8078 | 0.4027 | 0.3265 | 0.3365 | 0.779146 | 0.753041 |
| C3 | RandomForest | 0.4967 | 0.8137 | 0.5158 | 0.4666 | 0.3868 | 0.813811 | 0.78919 |
| C4-ML | HistGradientBoosting | 0.7380 | 0.8710 | 0.8195 | 0.7962 | 0.7208 | 0.975865 | 0.973219 |
| C4-ML | LogisticRegression | 0.7148 | 0.8642 | 0.8035 | 0.7754 | 0.6917 | 0.966662 | 0.968527 |
| C4-ML | MLPClassifier | 0.7547 | 0.8675 | 0.8301 | 0.8119 | 0.7430 | 0.975503 | 0.974304 |
| C4-ML | RandomForest | 0.7380 | 0.8710 | 0.8195 | 0.7962 | 0.7208 | 0.971437 | 0.970281 |
