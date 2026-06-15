# Reported result files

The `results/` directory contains compact, machine-readable summaries used in
the manuscript.

| File | Contents |
|---|---|
| `multidataset_classification_summary.csv` | TFN classification accuracy and macro-F1 |
| `architecture_baseline_summary.csv` | TFN, CNN, and ResNet1D comparison |
| `adaptive_gate_summary.csv` | Proposed score and uncertainty baselines |
| `adaptive_gate_paired_tests.csv` | Paired Wilcoxon tests |
| `adaptive_gate_decision_summary.csv` | Stream-level gate decisions |
| `extended_robustness_summary.csv` | Mixed, colored, impulse, and mechanical corruption |
| `extended_robustness_paired_tests.csv` | Extended-corruption paired tests |
| `extended_sequence_length_ablation.csv` | Sequence-length ablation |
| `ensemble_uncertainty_summary.csv` | Three-member Deep Ensemble comparison |
| `runtime_benchmark.json` | Parameter count and inference latency |

Headline result under `-6 dB` Gaussian noise:

- mean AURC reduction versus maximum softmax confidence: `0.036499`;
- improved pairs: `8/9`;
- one-sided Wilcoxon p-value: `0.019531`.

The method preserves the confidence ranking under clean, `6 dB`, and `0 dB`
Gaussian conditions. Colored noise has a negative mean AURC reduction and is
reported as a limitation.

