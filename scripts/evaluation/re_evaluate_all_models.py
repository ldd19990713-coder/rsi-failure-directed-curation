#!/usr/bin/env python3
"""
使用修复版SMILES提取器重新评估所有模型
"""

import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_smiles_fixed(text: str) -> str:
    """修复版SMILES提取 - 只匹配明确的SMILES模式"""
    text = text.strip()
    smiles_chars = r'[A-Za-z0-9@+\-\[\]\(\)=#\$\/\\\.]'

    # 策略1（最优先）：查找明确的"SMILES is:"模式
    patterns_priority = [
        r'(?:product\s+)?SMILES\s+is\s*:\s*(' + smiles_chars + r'+)',
        r'(?:product|major\s+product)\s+SMILES\s*:\s*(' + smiles_chars + r'+)',
        r'SMILES\s*:\s*(' + smiles_chars + r'+)',
    ]

    for pattern in patterns_priority:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            smiles = match.group(1).rstrip('.')
            if len(smiles) >= 3:
                return smiles

    # 策略2：查找"is:"后跟SMILES样式的文本（必须包含化学键符号）
    match = re.search(r'\bis\s*:\s*(' + smiles_chars + r'+)', text, re.IGNORECASE)
    if match:
        smiles = match.group(1).rstrip('.')
        if re.search(r'[=()\[\]#0-9]', smiles) and len(smiles) >= 3:
            return smiles

    # 策略3：查找所有包含化学键符号的SMILES样式的匹配
    smiles_pattern = r'[A-Z][A-Za-z0-9@+\-\[\]\(\)=#\$\/\\.]{2,}'
    matches = re.findall(smiles_pattern, text)

    valid_smiles = []
    for m in matches:
        m = m.rstrip('.')
        if re.search(r'[=()\[\]#0-9]', m):
            valid_smiles.append(m)

    if valid_smiles:
        return max(valid_smiles, key=len)

    # 策略4：如果文本本身就是纯SMILES
    if len(text) < 100 and re.match(r'^[A-Za-z0-9@+\-\[\]\(\)=#\$\/\\.]+$', text):
        return text.rstrip('.')

    return ""


def evaluate_model(model_path: str, eval_data: list, model_name: str) -> dict:
    """评估单个模型"""
    print(f'\n{"="*80}')
    print(f'评估模型: {model_name}')
    print(f'{"="*80}')

    # 加载模型
    print(f'加载模型: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    print('✓ 模型加载完成\n')

    # 评估
    correct = 0
    total = 0
    predictions = []

    for example in eval_data:
        input_text = example['input_text']
        target = example['target_text']

        inputs = tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # 增加以支持解释文本
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        pred_smiles = extract_smiles_fixed(pred_text)
        is_correct = (pred_smiles == target)

        if is_correct:
            correct += 1

        predictions.append({
            'id': example['id'],
            'reaction_type': example['reaction_type'],
            'difficulty': example['difficulty'],
            'target': target,
            'predicted': pred_smiles,
            'correct': is_correct,
            'raw_output': pred_text[-200:]  # 保存最后200字符用于调试
        })

        total += 1

        if total % 10 == 0:
            print(f'  已评估: {total}/{len(eval_data)}')

    accuracy = correct / total if total > 0 else 0.0
    print(f'\n✓ 评估完成: {correct}/{total} = {accuracy:.2%}')

    # 按难度统计
    difficulty_stats = {}
    for pred in predictions:
        diff = pred['difficulty']
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {'correct': 0, 'total': 0}
        difficulty_stats[diff]['total'] += 1
        if pred['correct']:
            difficulty_stats[diff]['correct'] += 1

    print('\n按难度统计:')
    for diff in ['easy', 'medium', 'hard']:
        if diff in difficulty_stats:
            stats = difficulty_stats[diff]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f'  {diff}: {stats["correct"]}/{stats["total"]} = {acc:.2%}')

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'difficulty_stats': difficulty_stats,
        'predictions': predictions
    }


def main():
    print('='*80)
    print('重新评估所有模型（修复版SMILES提取器）')
    print('='*80)

    # 加载评估数据
    eval_path = Path('./data/eval_reactions_expanded.json')
    with open(eval_path) as f:
        eval_data = json.load(f)
    print(f'\n评估集: {len(eval_data)} 样本')

    # 评估所有V2模型
    models = [
        ('./models/expanded_round_1', 'V2_Round1'),
        ('./models/expanded_round_2', 'V2_Round2'),
        ('./models/expanded_round_3', 'V2_Round3'),
    ]

    results = {}
    for model_path, model_name in models:
        if not Path(model_path).exists():
            print(f'\n⚠ 模型不存在: {model_path}')
            continue

        result = evaluate_model(model_path, eval_data, model_name)
        results[model_name] = result

        # 释放显存
        del result
        torch.cuda.empty_cache()

    # 保存结果
    output_path = Path('./results/re_evaluation_fixed.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n\n{"="*80}')
    print('汇总结果')
    print(f'{"="*80}')
    for model_name, result in results.items():
        print(f'{model_name}: {result["accuracy"]:.2%} ({result["correct"]}/{result["total"]})')

    print(f'\n✓ 结果已保存: {output_path}')


if __name__ == '__main__':
    main()
