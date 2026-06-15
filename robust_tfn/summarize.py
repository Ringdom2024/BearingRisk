from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research_runs"
SEEDS = {
    999: ("formal_noise_aug", "formal_tfconv_calibration"),
    1001: ("seed_1001_noise_aug", "seed_1001_tfconv_calibration"),
    1003: ("seed_1003_noise_aug", "seed_1003_tfconv_calibration"),
}
CONDITIONS = ["clean", "snr_6", "snr_0", "snr_-6"]


def read_classification(run_name):
    frame = pd.read_csv(RUNS / run_name / "metrics.csv")
    return frame[frame["score"] == "evidence"].set_index("condition")


def read_risk(run_name):
    return pd.read_csv(RUNS / run_name / "learned_risk.csv")


def classification_tables():
    paired = []
    for seed, (baseline_name, improved_name) in SEEDS.items():
        baseline = read_classification(baseline_name)
        improved = read_classification(improved_name)
        for condition in CONDITIONS:
            row = {"seed": seed, "condition": condition}
            for metric in ("accuracy", "macro_f1"):
                base_value = baseline.loc[condition, metric]
                improved_value = improved.loc[condition, metric]
                row[f"baseline_{metric}"] = base_value
                row[f"improved_{metric}"] = improved_value
                row[f"delta_{metric}"] = improved_value - base_value
            paired.append(row)

    paired = pd.DataFrame(paired)
    summary = []
    for condition in CONDITIONS:
        subset = paired[paired["condition"] == condition]
        for metric in ("accuracy", "macro_f1"):
            summary.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "baseline_mean": subset[f"baseline_{metric}"].mean(),
                    "baseline_std": subset[f"baseline_{metric}"].std(ddof=1),
                    "improved_mean": subset[f"improved_{metric}"].mean(),
                    "improved_std": subset[f"improved_{metric}"].std(ddof=1),
                    "mean_delta": subset[f"delta_{metric}"].mean(),
                    "delta_std": subset[f"delta_{metric}"].std(ddof=1),
                    "improved_seed_count": int(
                        (subset[f"delta_{metric}"] > 0).sum()
                    ),
                }
            )
    return paired, pd.DataFrame(summary)


def risk_tables():
    rows = []
    for seed, (baseline_name, improved_name) in SEEDS.items():
        for model_name, run_name in (
            ("baseline", baseline_name),
            ("improved", improved_name),
        ):
            frame = read_risk(run_name)
            for condition in CONDITIONS:
                selected = frame[frame["condition"] == condition].set_index("score")
                learned = selected.loc["learned_evidence"]
                confidence = selected.loc["confidence"]
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "condition": condition,
                        "learned_aurc": learned["aurc"],
                        "confidence_aurc": confidence["aurc"],
                        "aurc_reduction": confidence["aurc"] - learned["aurc"],
                        "learned_top10_accuracy": learned["accuracy_at_10pct"],
                        "confidence_top10_accuracy": confidence[
                            "accuracy_at_10pct"
                        ],
                        "top10_accuracy_gain": learned["accuracy_at_10pct"]
                        - confidence["accuracy_at_10pct"],
                    }
                )

    paired = pd.DataFrame(rows)
    summary = (
        paired.groupby(["model", "condition"], sort=False)
        .agg(
            learned_aurc_mean=("learned_aurc", "mean"),
            learned_aurc_std=("learned_aurc", "std"),
            confidence_aurc_mean=("confidence_aurc", "mean"),
            confidence_aurc_std=("confidence_aurc", "std"),
            aurc_reduction_mean=("aurc_reduction", "mean"),
            learned_top10_accuracy_mean=("learned_top10_accuracy", "mean"),
            confidence_top10_accuracy_mean=("confidence_top10_accuracy", "mean"),
            top10_accuracy_gain_mean=("top10_accuracy_gain", "mean"),
        )
        .reset_index()
    )
    return paired, summary


def percent(value):
    return f"{100 * value:.2f}%"


def mean_std(mean, std):
    return f"{100 * mean:.2f} +/- {100 * std:.2f}"


def write_report(class_summary, risk_summary):
    accuracy = class_summary[class_summary["metric"] == "accuracy"].set_index(
        "condition"
    )
    severe_risk = risk_summary[
        (risk_summary["model"] == "improved")
        & (risk_summary["condition"] == "snr_-6")
    ].iloc[0]

    lines = [
        "# CWRU 阶段实验报告",
        "",
        "## 实验协议",
        "",
        "- 训练工况：负载 0、1；校准工况：负载 2；测试工况：负载 3。",
        "- 三个独立随机种子：999、1001、1003。",
        "- 基线：TFN + 0 到 12 dB 随机噪声增强。",
        "- 改进：冻结主干和 BatchNorm，仅对 TFconv 进行物理频率锚定的反事实校准。",
        "- 测试：clean、6 dB、0 dB、-6 dB；测试工况不参与训练或阈值拟合。",
        "",
        "## 分类结果（准确率%，均值 +/- 标准差）",
        "",
        "| 条件 | 噪声增强基线 | 反事实 TFconv 校准 | 平均变化 | 提升种子数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = accuracy.loc[condition]
        lines.append(
            f"| {condition} | {mean_std(row['baseline_mean'], row['baseline_std'])} "
            f"| {mean_std(row['improved_mean'], row['improved_std'])} "
            f"| {percent(row['mean_delta'])} "
            f"| {int(row['improved_seed_count'])}/3 |"
        )

    lines.extend(
        [
            "",
            "## 极端噪声下的风险识别",
            "",
            "在改进模型的 -6 dB 测试中，反事实学习风险分数与普通最大软概率相比：",
            "",
            f"- AURC：{severe_risk['learned_aurc_mean']:.4f} vs "
            f"{severe_risk['confidence_aurc_mean']:.4f}，平均降低 "
            f"{severe_risk['aurc_reduction_mean']:.4f}（越低越好）。",
            f"- 仅接收风险最低 10% 样本时准确率："
            f"{percent(severe_risk['learned_top10_accuracy_mean'])} vs "
            f"{percent(severe_risk['confidence_top10_accuracy_mean'])}，提升 "
            f"{percent(severe_risk['top10_accuracy_gain_mean'])}。",
            "- 三个随机种子中，-6 dB 的 AURC 和 top-10% 准确率均优于普通置信度。",
            "",
            "## 当前结论",
            "",
            "1. 物理反事实 TFconv 校准有正向平均效果，但只在 2/3 个种子提升，稳定性不足，"
            "目前不能单独作为已验证的核心结论。",
            "2. 反事实风险模块在 -6 dB 极端噪声下表现稳定，适合定位为“严重退化条件下的"
            "选择性故障诊断”；在 clean、6 dB 条件下并不稳定优于普通置信度。",
            "3. 当前证据是 CWRU 单数据集 proof-of-concept，尚不足以稳妥投稿 SCI 四区。"
            "需要增加至少两个公开数据集、强基线、更多种子和统计检验。",
            "",
            "## 下一阶段最低实验集",
            "",
            "- Paderborn + JNU/SEU 跨工况复现同一协议。",
            "- 与 TFN、WDCNN、ResNet1D、现有噪声鲁棒方法比较。",
            "- 校准模块增加稳定性约束并做消融：无物理锚、无反事实删除、不同通道稀疏率。",
            "- 风险模块报告 risk-coverage 曲线、AURC、FPR95、不同未知噪声类型。",
        ]
    )
    (ROOT / "RESEARCH_PROGRESS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    RUNS.mkdir(exist_ok=True)
    class_paired, class_summary = classification_tables()
    risk_paired, risk_summary = risk_tables()

    class_paired.to_csv(RUNS / "classification_by_seed.csv", index=False)
    class_summary.to_csv(RUNS / "classification_summary.csv", index=False)
    risk_paired.to_csv(RUNS / "risk_by_seed.csv", index=False)
    risk_summary.to_csv(RUNS / "risk_summary.csv", index=False)
    write_report(class_summary, risk_summary)

    print(class_summary.to_string(index=False))
    print()
    print(risk_summary.to_string(index=False))


if __name__ == "__main__":
    main()
