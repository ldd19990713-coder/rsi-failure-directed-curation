#!/usr/bin/env python3
"""
P2′: 加载 LoRA adapter 在外集评测
=================================

目的: 加载基座 + LoRA adapter，在 v9_g1prime_p2_external 上评测
输入: 已训 LoRA adapter (seed43)
输出: 总准确率 + E/M/H + 按反应类型 + 与 in-split 落差
"""

import gc
import json
import re
import sys
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data" / "v9_g1prime_p2_external"
MODEL_PATH = "."
LORA_PATH = EXPERIMENT_DIR / "artifacts" / "canonical" / "P2b_7b" / "P2b_A_fail_seed43" / "lora_adapter"
OUTPUT_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "P2p_external"


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


def main():
    print("=" * 70)
    print("P2′: 加载 LoRA adapter 在外集评测")
    print("模型: Qwen2.5-7B + LoRA (seed43)")
    print("数据: v9_g1prime_p2_external (100 samples)")
    print("=" * 70)

    # 检查 LoRA adapter 是否存在
    if not LORA_PATH.exists():
        print(f"ERROR: LoRA adapter not found at {LORA_PATH}")
        print("请先运行 p2p_save_lora.py 训练并保存 adapter")
        sys.exit(1)

    # 加载 LoRA adapter 配置
    with open(LORA_PATH / "adapter_config.json") as f:
        adapter_config = json.load(f)
    seed = adapter_config.get("_custom_metadata", {}).get("seed", 43)
    r = adapter_config.get("r", 16)
    alpha = adapter_config.get("lora_alpha", 32)
    print(f"\nLoRA adapter: seed={seed}, r={r}, α={alpha}")

    # 加载外部数据
    with open(DATA_DIR / "external_eval.json") as f:
        external_data = json.load(f)
    print(f"外部集: {len(external_data)} samples")

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

    # 加载 LoRA adapter
    print(f"加载 LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    model = model.merge_and_unload()  # 合并 LoRA 权重到基座，提高推理速度

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"模型加载完成, device={device}")

    # 评测
    correct = 0
    results = []
    diff_stats = {'easy': {'c': 0, 't': 0}, 'medium': {'c': 0, 't': 0}, 'hard': {'c': 0, 't': 0}}
    type_stats = {}

    print(f"\n开始评测 {len(external_data)} 个样本...")

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

        diff = item.get('difficulty', 'medium')
        diff_stats[diff]['t'] += 1
        if is_correct:
            diff_stats[diff]['c'] += 1

        rtype = item.get('reaction_type', 'unknown')
        if rtype not in type_stats:
            type_stats[rtype] = {'c': 0, 't': 0}
        type_stats[rtype]['t'] += 1
        if is_correct:
            type_stats[rtype]['c'] += 1

        results.append({
            'id': item['id'],
            'reaction_type': rtype,
            'difficulty': diff,
            'ground_truth': gt,
            'prediction': pred,
            'correct': is_correct
        })

        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx + 1}/{len(external_data)}, 当前准确率: {correct/(idx+1)*100:.1f}%")

    # 汇总
    acc = correct / len(external_data)
    print(f"\n{'='*70}")
    print(f"P2′ L3+LoRA 外集评测结果")
    print(f"{'='*70}")
    print(f"总准确率: {acc*100:.2f}% ({correct}/{len(external_data)})")

    print(f"\n按难度:")
    for d in ['easy', 'medium', 'hard']:
        s = diff_stats[d]
        if s['t'] > 0:
            print(f"  {d}: {s['c']}/{s['t']} = {s['c']/s['t']*100:.1f}%")

    print(f"\n按反应类型 (top 10):")
    sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['t'], reverse=True)[:10]
    for rtype, s in sorted_types:
        if s['t'] > 0:
            print(f"  {rtype}: {s['c']}/{s['t']} = {s['c']/s['t']*100:.1f}%")

    # 与 in-split 对比
    in_split_peak = 0.8144  # seed43 peak
    in_split_final = 0.8041  # seed43 final
    gap_peak = in_split_peak - acc
    gap_final = in_split_final - acc

    print(f"\n与 in-split 对比 (seed43):")
    print(f"  In-split Peak: {in_split_peak*100:.2f}%")
    print(f"  In-split Final: {in_split_final*100:.2f}%")
    print(f"  External (L3+LoRA): {acc*100:.2f}%")
    print(f"  落差 vs Peak: {gap_peak*100:.2f}pp")
    print(f"  落差 vs Final: {gap_final*100:.2f}pp")

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'P2p_external_lora',
        'model': 'Qwen2.5-7B + LoRA (seed43)',
        'lora_path': str(LORA_PATH),
        'recipe_hash': adapter_config.get('recipe_hash', '9e3788dd7a2eb89d66a1b403d5c5b7c425d583adf7c799c35b761b57fbf7f809'),
        'data': 'v9_g1prime_p2_external',
        'total_samples': len(external_data),
        'accuracy': acc,
        'correct': correct,
        'difficulty_breakdown': {
            d: {'correct': s['c'], 'total': s['t'], 'accuracy': s['c']/s['t'] if s['t'] > 0 else 0}
            for d, s in diff_stats.items()
        },
        'reaction_type_breakdown': {
            t: {'correct': s['c'], 'total': s['t'], 'accuracy': s['c']/s['t'] if s['t'] > 0 else 0}
            for t, s in type_stats.items()
        },
        'in_split_comparison': {
            'seed': 43,
            'in_split_peak': in_split_peak,
            'in_split_final': in_split_final,
            'external_acc': acc,
            'gap_vs_peak': gap_peak,
            'gap_vs_final': gap_final
        },
        'predictions': results
    }

    with open(OUTPUT_DIR / "metrics.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ 结果已保存: {OUTPUT_DIR / 'metrics.json'}")

    # 清理
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
