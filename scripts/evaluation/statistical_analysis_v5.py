#!/usr/bin/env python3
"""
V5实验统计分析 - 多轮训练分析
"""

import json
import math
from pathlib import Path


def wilson_confidence_interval(k, n, z=1.96):
    """计算Wilson置信区间"""
    if n == 0:
        return (0, 0)

    p_hat = k / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return (lower, upper)


def binomial_test(k, n, p0=0.1):
    """二项检验：准确率是否显著优于随机（10%）"""
    if n == 0:
        return 1.0

    # 计算P(X >= k | p = p0)
    p_value = 0.0
    for i in range(k, n + 1):
        # C(n, i) * p0^i * (1-p0)^(n-i)
        log_comb = sum(math.log(n - j) - math.log(j + 1) for j in range(i)) if i > 0 else 0
        try:
            log_prob = log_comb + i * math.log(p0) + (n - i) * math.log(1 - p0)
            p_value += math.exp(log_prob)
        except (ValueError, ZeroDivisionError):
            continue

    return min(1.0, p_value)


def analyze_v5_results():
    """分析V5实验结果"""
    print("=" * 80)
    print("V5实验统计分析 - 多轮训练优化")
    print("=" * 80)

    # 查找结果文件
    results_dir = Path("./results")
    result_files = list(results_dir.glob("v5_optimized_training_*rounds_results.json"))

    if not result_files:
        print("未找到V5训练结果文件")
        return

    results_path = result_files[0]
    print(f"\n分析文件: {results_path.name}")

    with open(results_path) as f:
        results = json.load(f)

    rounds = results["rounds"]
    num_rounds = len(rounds)

    # 1. 各轮次准确率和置信区间
    print("\n【1. 各轮次准确率和置信区间】")
    print("-" * 80)
    print(f"{'轮次':<15} {'准确率':<12} {'95% CI':<20} {'p-value':<12} {'显著性':<10}")
    print("-" * 80)

    for round_data in rounds:
        round_num = round_data["round"]
        k = round_data["correct"]
        n = round_data["total"]
        acc = k / n if n > 0 else 0

        # Wilson置信区间
        ci_lower, ci_upper = wilson_confidence_interval(k, n)

        # 二项检验（vs 10%随机）
        p_value = binomial_test(k, n, p0=0.1)

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(f"Round {round_num:<9} {acc:>8.2%}     [{ci_lower:.2%}, {ci_upper:.2%}]    {p_value:>10.4f}   {sig}")

    print("-" * 80)
    print("注: *** p<0.001, ** p<0.01, * p<0.05, ns: 不显著")

    # 2. 找出最佳轮次
    print("\n【2. 最佳轮次分析】")
    print("-" * 80)

    best_round = max(rounds, key=lambda x: x["accuracy"])
    best_round_num = best_round["round"]
    best_acc = best_round["accuracy"]
    best_k = best_round["correct"]
    best_n = best_round["total"]

    print(f"最佳轮次: Round {best_round_num}")
    print(f"最佳准确率: {best_acc:.2%} ({best_k}/{best_n})")

    ci_lower, ci_upper = wilson_confidence_interval(best_k, best_n)
    print(f"95% 置信区间: [{ci_lower:.2%}, {ci_upper:.2%}]")

    p_value = binomial_test(best_k, best_n, p0=0.1)
    print(f"显著性检验 (vs 10%随机): p = {p_value:.6f}")

    if p_value < 0.05:
        print("✓ 结论: 最佳轮次准确率显著优于随机水平")
    else:
        print("⚠ 结论: 最佳轮次准确率与随机水平无显著差异")

    print("-" * 80)

    # 3. 按难度分析最佳轮次
    print("\n【3. 最佳轮次按难度分析】")
    print("-" * 80)

    if "difficulty_stats" in best_round:
        for diff in ["easy", "medium", "hard"]:
            if diff in best_round["difficulty_stats"]:
                stats = best_round["difficulty_stats"][diff]
                k = stats["correct"]
                n = stats["total"]
                acc = k / n if n > 0 else 0
                ci_lower, ci_upper = wilson_confidence_interval(k, n)
                print(f"{diff:<8}: {acc:>6.2%} ({k}/{n})  95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")

    print("-" * 80)

    # 4. 训练历史分析（过拟合检测）
    print("\n【4. 训练历史与过拟合检测】")
    print("-" * 80)

    if "training_history" in results and results["training_history"]:
        history = results["training_history"]

        print(f"{'轮次':<10} {'Epoch':<10} {'训练Loss':<12} {'验证Acc':<12}")
        print("-" * 80)

        for h in history:
            print(f"Round {h['round']:<4} Epoch {h['epoch']:<5} {h['train_loss']:>10.4f}   {h['eval_accuracy']:>8.2%}")

        # 检测过拟合
        print("\n过拟合检测:")
        round_2_history = [h for h in history if h['round'] == 2]
        if round_2_history:
            first_loss = round_2_history[0]['train_loss']
            final_loss = round_2_history[-1]['train_loss']
            final_eval = round_2_history[-1]['eval_accuracy']

            if final_loss < first_loss * 0.5:
                print(f"  ✓ 训练Loss显著下降: {first_loss:.4f} → {final_loss:.4f}")
            else:
                print(f"  ⚠ 训练Loss下降有限: {first_loss:.4f} → {final_loss:.4f}")

            print(f"  最终验证准确率: {final_eval:.2%}")

    print("-" * 80)

    # 5. 改进趋势分析
    print("\n【5. 改进趋势分析】")
    print("-" * 80)

    # 计算相邻轮次的改进
    improvements = []
    for i in range(1, len(rounds)):
        prev_acc = rounds[i-1]["accuracy"]
        curr_acc = rounds[i]["accuracy"]
        improvement = curr_acc - prev_acc
        improvements.append({
            "from_round": rounds[i-1]["round"],
            "to_round": rounds[i]["round"],
            "improvement": improvement
        })

    print(f"{'比较':<20} {'改进':<12} {'趋势':<10}")
    print("-" * 80)

    for imp in improvements:
        trend = "↑" if imp["improvement"] > 0 else "↓" if imp["improvement"] < 0 else "→"
        print(f"Round {imp['from_round']} → {imp['to_round']:<8} {imp['improvement']:>+8.2%}      {trend}")

    # 总体趋势
    total_improvement = rounds[-1]["accuracy"] - rounds[0]["accuracy"]
    print(f"\n总体改进 (Round 1 → {rounds[-1]['round']}): {total_improvement:+.2%}")

    print("-" * 80)

    # 6. 与V4对比
    print("\n【6. 与V4实验对比】")
    print("-" * 80)

    v4_best_acc = 0.20  # V4最佳准确率20%
    print(f"V4最佳准确率: {v4_best_acc:.2%}")
    print(f"V5最佳准确率: {best_acc:.2%}")
    print(f"改进: {best_acc - v4_best_acc:+.2%}")

    if best_acc > v4_best_acc:
        print("✓ V5优于V4")
    else:
        print("⚠ V5未超越V4")

    print("-" * 80)

    # 7. 结论与建议
    print("\n【7. 结论与建议】")
    print("-" * 80)

    # 结论
    print("主要结论:")
    if best_acc > 0.3:
        print(f"  ✓ 递归微调成功，最佳准确率达到{best_acc:.2%}")
    elif best_acc > 0.2:
        print(f"  ⚠ 递归微调部分成功，最佳准确率为{best_acc:.2%}")
    else:
        print(f"  ✗ 递归微调效果有限，最佳准确率仅为{best_acc:.2%}")

    # Medium样本表现
    if "difficulty_stats" in best_round and "medium" in best_round["difficulty_stats"]:
        medium_stats = best_round["difficulty_stats"]["medium"]
        medium_acc = medium_stats["correct"] / medium_stats["total"] if medium_stats["total"] > 0 else 0
        if medium_acc > 0.1:
            print(f"  ✓ Medium样本优化有效，准确率为{medium_acc:.2%}")
        else:
            print(f"  ⚠ Medium样本仍需改进，准确率为{medium_acc:.2%}")

    # 建议
    print("\n下一步建议:")
    if best_acc < 0.3:
        print("  1. 考虑使用更大的基础模型（如Qwen2.5-7B）")
        print("  2. 扩大训练数据规模（当前53个样本可能不足）")
        print("  3. 尝试不同的数据增强策略")
    else:
        print("  1. 在独立测试集上验证泛化能力")
        print("  2. 尝试更复杂的化学反应预测任务")
        print("  3. 探索其他RSI框架变体")

    print("-" * 80)

    # 保存分析结果
    analysis = {
        "best_round": best_round_num,
        "best_accuracy": best_acc,
        "best_correct": best_k,
        "best_total": best_n,
        "confidence_interval_95": wilson_confidence_interval(best_k, best_n),
        "p_value_vs_random": p_value,
        "total_improvement": total_improvement,
        "comparison_with_v4": {
            "v4_best": v4_best_acc,
            "v5_best": best_acc,
            "improvement": best_acc - v4_best_acc
        },
        "significant": p_value < 0.05
    }

    analysis_path = results_dir / "v5_statistical_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✓ 分析结果已保存: {analysis_path}")


if __name__ == "__main__":
    analyze_v5_results()
