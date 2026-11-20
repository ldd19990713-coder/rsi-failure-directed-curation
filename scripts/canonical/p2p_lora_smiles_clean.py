#!/usr/bin/env python3
"""
辅线 Step 3 (修正版): LoRA adapter 在干净 SMILES 外集评测
"""

import gc
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data" / "v9_g1prime_p2_external_smiles_clean"
MODEL_PATH = "."
P2B_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "P2b_7b"
OUTPUT_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "P2p_clean_external_smiles" / "lora_seed43"


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
    print("辅线 Step 3 (修正): LoRA seed43 在干净 SMILES 外集评测")
    print("=" * 70)

    # 加载干净外集
    with open(DATA_DIR / "external_eval_smiles_clean.json") as f:
        external_data = json.load(f)
    print(f"干净 SMILES 外集: {len(external_data)} 样本")

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
    lora_path = P2B_DIR / "P2b_A_fail_seed43" / "lora_adapter"
    print(f"加载 LoRA adapter: seed=43")
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"模型加载完成, device={device}")

    # 评测
    correct = 0
    results = []
    type_stats = {}

    print(f"\n开始评测 {len(external_data)} 个样本...")

    for idx, item in enumerate(external_data):
        reactants = item.get('reactants_smiles', item.get('input_text', ''))
        if isinstance(reactants, list):
            reactants = '|'.join(reactants)

        rtype = item.get('reaction_type', 'unknown')
        prompt = f"输入: {reactants}\n输出:"

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

        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx + 1}/{len(external_data)}, 当前准确率: {correct/(idx+1)*100:.1f}%")

    # 汇总
    acc = correct / len(external_data)
    print(f"\n{'='*70}")
    print(f"LoRA seed43 干净 SMILES 外集评测结果")
    print(f"{'='*70}")
    print(f"总准确率: {acc*100:.2f}% ({correct}/{len(external_data)})")

    print(f"\n按反应类型:")
    sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['t'], reverse=True)
    for rtype, s in sorted_types:
        if s['t'] > 0:
            print(f"  {rtype}: {s['c']}/{s['t']} = {s['c']/s['t']*100:.1f}%")

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'lora_seed43_smiles_clean',
        'model': 'Qwen2.5-7B + LoRA seed43',
        'seed': 43,
        'data': 'v9_g1prime_p2_external_smiles_clean',
        'total_samples': len(external_data),
        'accuracy': acc,
        'correct': correct,
        'reaction_type_breakdown': {
            t: {'correct': s['c'], 'total': s['t'], 'accuracy': s['c']/s['t'] if s['t'] > 0 else 0}
            for t, s in type_stats.items()
        },
        'predictions': results
    }

    with open(OUTPUT_DIR / "metrics.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ 结果已保存: {OUTPUT_DIR / 'metrics.json'}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
