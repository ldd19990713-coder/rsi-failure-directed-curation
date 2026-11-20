#!/usr/bin/env python3
"""
阶段1: 在扩大的评估集上评估所有现有模型
目标：获得每个模型的准确率、置信区间、配对比较结果
"""

import gc
import json
import re
import torch
from pathlib import Path
from datetime import datetime
import sys

EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"


# ============================================================
# SMILES评估函数
# ============================================================

def extract_smiles_from_text(text: str) -> str:
    text = text.strip()
    smiles_pattern = r'[A-Za-z0-9@+\-\[\]\(\)=#$/\\.]+'
    candidates = re.findall(smiles_pattern, text)
    if candidates:
        best = max(candidates, key=len)
        return best
    return text.strip()


def normalize_smiles(smiles: str) -> str:
    s = smiles.strip().replace(" ", "")
    s = s.strip("Product:").strip("product:").strip()
    return s


def check_smiles_match(prediction: str, target: str) -> bool:
    pred_smiles = extract_smiles_from_text(prediction)
    pred_norm = normalize_smiles(pred_smiles)
    target_norm = normalize_smiles(target)
    if pred_norm == target_norm:
        return True
    if target_norm in pred_norm or pred_norm in target_norm:
        return True
    return False


# ============================================================
# 模型加载和评估
# ============================================================

def load_model(model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_model(model, tokenizer, eval_data, device, model_name):
    """评估单个模型，返回详细结果"""
    correct = 0
    failed = 0
    predictions = []

    with torch.no_grad():
        for example in eval_data:
            prompt = f"### Reaction:\n{example['input_text']}\n### Product SMILES:\n"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            target = example["target_text"]
            is_correct = check_smiles_match(generated_text, target)

            pred_smiles = extract_smiles_from_text(generated_text)
            predictions.append({
                "id": example["id"],
                "reaction_type": example["reaction_type"],
                "difficulty": example["difficulty"],
                "target": target,
                "predicted": pred_smiles,
                "correct": is_correct,
            })

            if is_correct:
                correct += 1
            else:
                failed += 1

    return {
        "model_name": model_name,
        "accuracy": correct / len(eval_data),
        "correct": correct,
        "total": len(eval_data),
        "failed": failed,
        "predictions": predictions,
    }


def evaluate_single_model(model_path, eval_data, device, model_name):
    """加载并评估单个模型"""
    print(f"\n[评估] {model_name}")
    print(f"  模型路径: {model_path}")

    model, tokenizer = load_model(model_path, device)
    result = evaluate_model(model, tokenizer, eval_data, device, model_name)

    # 清理显存
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  准确率: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 加载扩大的评估集
    eval_path = DATA_DIR / "eval_reactions_expanded.json"
    with open(eval_path) as f:
        eval_data = json.load(f)
    print(f"\n评估集: {len(eval_data)} 样本")

    # 按难度统计
    diff_counts = {}
    for e in eval_data:
        d = e["difficulty"]
        diff_counts[d] = diff_counts.get(d, 0) + 1
    print(f"难度分布: {diff_counts}")

    # 定义要评估的模型
    models_to_evaluate = [
        {
            "name": "V2_Round1",
            "path": MODEL_DIR / "model_round_1",
            "description": "V2轮次1 (初始微调)",
        },
        {
            "name": "V2_Round2",
            "path": MODEL_DIR / "model_round_2",
            "description": "V2轮次2 (最优模型)",
        },
        {
            "name": "V2_Round3",
            "path": MODEL_DIR / "model_round_3",
            "description": "V2轮次3 (无EWC)",
        },
        {
            "name": "V3_Round3_EWC",
            "path": MODEL_DIR / "model_round_3_ewc",
            "description": "V3轮次3 (带EWC)",
        },
    ]

    # 评估所有模型
    all_results = []
    for model_info in models_to_evaluate:
        if not model_info["path"].exists():
            print(f"\n⚠ 模型不存在: {model_info['path']}")
            continue

        result = evaluate_single_model(
            str(model_info["path"]), eval_data, device, model_info["name"]
        )
        result["description"] = model_info["description"]
        all_results.append(result)

    # 保存结果
    output_path = RESULTS_DIR / "stage1_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "eval_set_size": len(eval_data),
            "difficulty_distribution": diff_counts,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print(f"\n{'='*70}")
    print("阶段1: 扩大评估集评估结果汇总")
    print(f"{'='*70}")
    print(f"评估集: {len(eval_data)} 样本 ({diff_counts})")
    print(f"\n{'模型':<20} {'准确率':>8} {'正确/总数':>12} {'描述':<30}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model_name']:<20} {r['accuracy']:>7.1%} "
              f"{r['correct']:>4}/{r['total']:<6} {r['description']:<30}")

    # 按难度统计每个模型
    print(f"\n{'='*70}")
    print("按难度分析")
    print(f"{'='*70}")
    for r in all_results:
        diff_stats = {}
        for p in r["predictions"]:
            d = p["difficulty"]
            if d not in diff_stats:
                diff_stats[d] = {"correct": 0, "total": 0}
            diff_stats[d]["total"] += 1
            if p["correct"]:
                diff_stats[d]["correct"] += 1

        print(f"\n{r['model_name']}:")
        for d in ["easy", "medium", "hard"]:
            if d in diff_stats:
                s = diff_stats[d]
                rate = s["correct"] / s["total"] if s["total"] > 0 else 0
                print(f"  {d:<8}: {rate:>6.1%} ({s['correct']}/{s['total']})")

    print(f"\n✓ 结果已保存: {output_path}")
    return all_results


if __name__ == "__main__":
    main()
