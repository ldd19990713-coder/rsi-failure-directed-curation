#!/usr/bin/env python3
"""
阶段1最终分析：扩展数据重新训练后的统计检验
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("./results")


def wilson_conf_interval(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson置信区间"""
    p_hat = k / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    return lower, upper


def binomial_test(k: int, n: int, p0: float = 0.1) -> dict:
    """二项检验"""
    p_value = 1 - stats.binom.cdf(k - 1, n, p0)
    return {
        "k": k,
        "n": n,
        "p_hat": k / n,
        "p0": p0,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
    }


def mcnemar_test(correct1: list, correct2: list) -> dict:
    """McNemar检验"""
    a = sum(1 for x, y in zip(correct1, correct2) if x and y)
    b = sum(1 for x, y in zip(correct1, correct2) if x and not y)
    c = sum(1 for x, y in zip(correct1, correct2) if not x and y)
    d = sum(1 for x, y in zip(correct1, correct2) if not x and not y)

    if b + c == 0:
        return {"p_value": 1.0, "significant_005": False, "discordant_pairs": 0}

    statistic = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "contingency_table": {"a": a, "b": b, "c": c, "d": d},
        "mcnemar_statistic": statistic,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "discordant_pairs": b + c,
    }


def main():
    print("=" * 70)
    print("阶段1最终分析：扩展数据重新训练")
    print("=" * 70)

    # 加载训练汇总
    summary_path = RESULTS_DIR / "expanded_training_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\n训练配置:")
    print(f"  训练集: {summary['train_set_size']} 样本")
    print(f"  评估集: {summary['eval_set_size']} 样本")

    rounds = summary["rounds"]
    print(f"\n训练结果:")
    for r in rounds:
        print(f"  轮次{r['round']}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")

    # ============================================================
    # 1. 单轮次分析
    # ============================================================
    print(f"\n{'='*70}")
    print("1. 单轮次准确率分析")
    print(f"{'='*70}")

    for r in rounds:
        k = r["correct"]
        n = r["total"]
        ci = wilson_conf_interval(k, n)
        binom = binomial_test(k, n, p0=0.10)

        print(f"\n轮次{r['round']}:")
        print(f"  准确率: {r['accuracy']:.2%} ({k}/{n})")
        print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  二项检验 (vs 10%): p = {binom['p_value']:.4f}")
        print(f"  显著优于随机? {'✓ Yes' if binom['significant_005'] else '✗ No'}")

    # ============================================================
    # 2. 配对比较
    # ============================================================
    print(f"\n{'='*70}")
    print("2. 配对比较 (McNemar检验)")
    print(f"{'='*70}")

    # 加载各轮次的详细评估结果
    eval_files = [
        RESULTS_DIR / "expanded_eval_round_1.json",
        RESULTS_DIR / "expanded_eval_round_2.json",
        RESULTS_DIR / "expanded_eval_round_3.json",
    ]

    predictions = []
    for eval_file in eval_files:
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
                predictions.append([p["correct"] for p in data["predictions"]])
        else:
            print(f"⚠ 文件不存在: {eval_file}")
            break

    if len(predictions) == 3:
        # 轮次1 vs 轮次2
        test_12 = mcnemar_test(predictions[0], predictions[1])
        acc1 = sum(predictions[0]) / len(predictions[0])
        acc2 = sum(predictions[1]) / len(predictions[1])
        print(f"\n轮次1 vs 轮次2:")
        print(f"  准确率: {acc1:.2%} vs {acc2:.2%} (差异: {acc2-acc1:+.2%})")
        print(f"  McNemar p-value: {test_12['p_value']:.4f}")
        print(f"  显著差异? {'✓ Yes' if test_12['significant_005'] else '✗ No'}")

        # 轮次2 vs 轮次3
        test_23 = mcnemar_test(predictions[1], predictions[2])
        acc3 = sum(predictions[2]) / len(predictions[2])
        print(f"\n轮次2 vs 轮次3:")
        print(f"  准确率: {acc2:.2%} vs {acc3:.2%} (差异: {acc3-acc2:+.2%})")
        print(f"  McNemar p-value: {test_23['p_value']:.4f}")
        print(f"  显著差异? {'✓ Yes' if test_23['significant_005'] else '✗ No'}")

        # 轮次1 vs 轮次3
        test_13 = mcnemar_test(predictions[0], predictions[2])
        print(f"\n轮次1 vs 轮次3:")
        print(f"  准确率: {acc1:.2%} vs {acc3:.2%} (差异: {acc3-acc1:+.2%})")
        print(f"  McNemar p-value: {test_13['p_value']:.4f}")
        print(f"  显著差异? {'✓ Yes' if test_13['significant_005'] else '✗ No'}")

    # ============================================================
    # 3. 按难度分析
    # ============================================================
    print(f"\n{'='*70}")
    print("3. 按难度分层分析")
    print(f"{'='*70}")

    for i, eval_file in enumerate(eval_files):
        if not eval_file.exists():
            break
        with open(eval_file) as f:
            data = json.load(f)

        print(f"\n轮次{i+1}:")
        for difficulty in ["easy", "medium", "hard"]:
            subset = [p for p in data["predictions"] if p["difficulty"] == difficulty]
            if not subset:
                continue
            k = sum(1 for p in subset if p["correct"])
            n = len(subset)
            ci = wilson_conf_interval(k, n)
            print(f"  {difficulty:<8}: {k/n:.2%} ({k}/{n}) CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    # ============================================================
    # 4. 核心结论
    # ============================================================
    print(f"\n{'='*70}")
    print("4. 核心结论")
    print(f"{'='*70}")

    r1_acc = rounds[0]["accuracy"]
    r2_acc = rounds[1]["accuracy"]
    r3_acc = rounds[2]["accuracy"]

    print(f"\n递归微调效果:")
    print(f"  轮次1→2: {r1_acc:.2%} → {r2_acc:.2%} ({r2_acc-r1_acc:+.2%})")
    if test_12['significant_005']:
        print(f"    ✓ 统计显著 (p={test_12['p_value']:.4f})")
    else:
        print(f"    ✗ 不显著 (p={test_12['p_value']:.4f})")

    print(f"\n  轮次2→3: {r2_acc:.2%} → {r3_acc:.2%} ({r3_acc-r2_acc:+.2%})")
    if test_23['significant_005']:
        print(f"    ✓ 统计显著 (p={test_23['p_value']:.4f})")
        if r3_acc < r2_acc:
            print(f"    ⚠ 灾难性遗忘！")
    else:
        print(f"    ✗ 不显著 (p={test_23['p_value']:.4f})")

    print(f"\n  总体改进: {r1_acc:.2%} → {r3_acc:.2%} ({r3_acc-r1_acc:+.2%})")
    if test_13['significant_005']:
        print(f"    ✓ 统计显著 (p={test_13['p_value']:.4f})")
    else:
        print(f"    ✗ 不显著 (p={test_13['p_value']:.4f})")

    # ============================================================
    # 5. 保存分析结果
    # ============================================================
    output = {
        "experiment": "V2_Expanded_Retraining",
        "train_set_size": summary["train_set_size"],
        "eval_set_size": summary["eval_set_size"],
        "rounds": rounds,
        "improvement": summary.get("improvement"),
        "timestamp": summary["timestamp"],
    }

    output_path = RESULTS_DIR / "stage1_final_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*70}")
    print(f"✓ 分析已保存: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
