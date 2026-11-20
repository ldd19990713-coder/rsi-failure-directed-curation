#!/usr/bin/env python3
"""
辅线 Step 3: LoRA adapter 在干净外集评测
=======================================

目的: 在干净外集 (80 样本) 上测试已有 LoRA adapter
优先级: seed43 (已有) → 其他 seed (需补训)
输出: artifacts/canonical/P2p_clean_external/lora_{seed}/
"""

import gc
import json
import re
import sys
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data" / "v9_g1prime_p2_external_clean"
MODEL_PATH = "."
P2B_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "P2b_7b"


def get_7b_snapshot_path():
    base = Path(MODEL_PATH)
    if not base.exists():
        return None
    snapshot_dir = base / "snapshots"
    if not snapshot_dir.exists():
        return None
    snapshots = list(snapshot_dir.iterdir())
    if snapshots:
        return snapshots[0]
    return None


def extract_smiles_fixed(text: str) -> str:
    text = text.strip()
    smiles_chars = r'[A-Za-z0-9@+\-\[\]\(\)=#\$\/\\\.]'
    patterns_priority = [
        r'(?:product\s+)?SMILES\s*is\s*:\s*(' + smiles_chars + r'+)',
        r'(?:product|major\s+product)\s+SMILES\s*:\s*(' + smiles_chars + r'+)',
        r'SMILES\s*:\s*(' + smiles_chars + r'+)',
    ]
    for pattern in patterns_priority:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            smiles = match.group(1).rstrip('.')
            if len(smiles) >= 3:
                return smiles
    smiles_pattern = r'[A-Z][A-Za-z0-9@+\-\[\]\(\)=#\$\/\\.]{2,}'
    matches = re.findall(smiles_pattern, text)
    valid_smiles = [m.rstrip('.') for m in matches if re.search(r'[=()\[\]#0-9]', m.rstrip('.'))]
    if valid_smiles:
        return max(valid_smiles, key=len)
    if len(text) < 100 and re.match(r'^[A-Za-z0-9@+\-\[\]\(\)=#\$\/\\.]+$', text):
        return text.rstrip('.')
    return ""


def evaluate_with_adapter(seed, external_data, tokenizer, base_model, device):
    """使用指定 seed 的 LoRA adapter 评测"""

    # 加载 LoRA adapter
    lora_path = P2B_DIR / f"P2b_A_fail_seed{seed}" / "lora_adapter"
    if not lora_path.exists():
        print(f"  ❌ LoRA adapter not found: {lora_path}")
        return None

    print(f"  加载 LoRA adapter: seed={seed}")
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    # 评测
    correct = 0
    results = []
    type_stats = {}

    for idx, item in enumerate(external_data):
        prompt = f"输入: {item['input_text']}\n输出:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model.generate(**inputs, max_new_tokens=100, num_beams=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred = extract_smiles_fixed(generated[len(prompt):])
            except Exception as e:
                pred = ""

        gt = item.get('product_smiles', item.get('target_text', ''))
        is_correct = (pred == gt) if pred else False

        if is_correct:
            correct += 1

        rtype = item.get('reaction_type', 'unknown')
        if rtype not in type_stats:
            type_stats[rtype] = {'c': 0, 't': 0}
        type_stats[rtype]['t'] += 1
        if is_correct:
            type_stats[rtype]['c'] += 1

        results.append({
            'id': item['id'],
            'reaction_type': rtype,
            'ground_truth': gt,
            'prediction': pred,
            'correct': is_correct
        })

    acc = correct / len(external_data)

    # 保存结果
    output_dir = EXPERIMENT_DIR / "artifacts" / "canonical" / "P2p_clean_external" / f"lora_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': f'lora_seed{seed}_clean_external',
        'model': f'Qwen2.5-7B + LoRA seed{seed}',
        'seed': seed,
        'data': 'v9_g1prime_p2_external_clean',
        'total_samples': len(external_data),
        'accuracy': acc,
        'correct': correct,
        'reaction_type_breakdown': {
            t: {'correct': s['c'], 'total': s['t'], 'accuracy': s['c']/s['t'] if s['t'] > 0 else 0}
            for t, s in type_stats.items()
        },
        'predictions': results
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  准确率: {acc*100:.2f}% ({correct}/{len(external_data)})")

    # 清理 LoRA 权重
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 重新加载基座
    print(f"  重新加载基座模型...")
    snapshot_path = get_7b_snapshot_path()
    base_model = AutoModelForCausalLM.from_pretrained(
        str(snapshot_path), torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True, attn_implementation="eager"
    )

    return acc, output


def main():
    parser = argparse.ArgumentParser(description="LoRA adapter 在干净外集评测")
    parser.add_argument('--seeds', nargs='+', type=int, default=[43],
                        help="要测试的 seed 列表 (默认: 43)")
    args = parser.parse_args()

    seeds = args.seeds
    print("=" * 70)
    print("辅线 Step 3: LoRA adapter 在干净外集评测")
    print(f"种子: {seeds}")
    print("=" * 70)

    # 加载干净外集
    with open(DATA_DIR / "external_eval_clean.json") as f:
        external_data = json.load(f)
    print(f"干净外集: {len(external_data)} 样本")

    # 加载基座模型
    snapshot_path = get_7b_snapshot_path()
    if snapshot_path is None:
        print("ERROR: 7B model not found")
        sys.exit(1)

    print(f"\n加载 Qwen2.5-7B 基座模型 (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(snapshot_path), torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True, attn_implementation="eager"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    print(f"模型加载完成, device={device}")

    # 逐个测试 seed
    results = {}
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"测试 seed={seed}")
        print(f"{'='*50}")

        result = evaluate_with_adapter(seed, external_data, tokenizer, base_model, device)

        if result is not None:
            acc, output = result
            results[seed] = acc
            base_model = result[2] if len(result) > 2 else base_model
        else:
            print(f"  ⚠️ seed{seed} adapter 不存在，跳过")

    # 汇总
    print(f"\n{'='*70}")
    print(f"多种子 LoRA 干净外集汇总")
    print(f"{'='*70}")

    for seed, acc in results.items():
        print(f"  seed{seed}: {acc*100:.2f}%")

    if results:
        import statistics
        accs = list(results.values())
        if len(accs) > 1:
            print(f"\n  Mean: {statistics.mean(accs)*100:.2f}% ± {statistics.stdev(accs)*100:.2f}%")
        else:
            print(f"\n  (仅 1 个 seed，无法计算 SD)")

        # 与 in-split 对比
        in_split_mean = 0.7773
        gap = in_split_mean - statistics.mean(accs)
        print(f"  vs in-split 均值 (77.73%): −{gap*100:.2f}pp")

    # 保存汇总
    summary = {
        'experiment': 'lora_multiseed_clean_external',
        'seeds_tested': list(results.keys()),
        'results': {str(k): v for k, v in results.items()},
        'mean': statistics.mean(results.values()) if results else 0,
        'std': statistics.stdev(results.values()) if len(results) > 1 else 0,
        'in_split_mean': 0.7773,
        'gap_vs_in_split': in_split_mean - statistics.mean(results.values()) if results else 0
    }

    output_dir = EXPERIMENT_DIR / "artifacts" / "canonical" / "P2p_clean_external"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ 汇总已保存: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
