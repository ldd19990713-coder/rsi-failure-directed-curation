#!/usr/bin/env python3
"""
调试：改进SMILES提取逻辑并重新评估
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def extract_smiles_improved(text: str) -> str:
    """改进版SMILES提取"""
    text = text.strip()

    # 策略1：查找"product SMILES is:"或类似模式
    patterns = [
        r'(?:product|major product|final product)\s+(?:SMILES|smiles|is)[:\s]+([A-Za-z0-9@+\-\[\]\(\)=#\$\/\\\.]+)',
        r'(?:SMILES|smiles)[:\s]+([A-Za-z0-9@+\-\[\]\(\)=#\$\/\\\.]+)',
        r'(?:is|:)\s*([A-Za-z0-9@+\-\[\]\(\)=#\$\/\\\.]+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # 返回最长的匹配
            longest = max(matches, key=len)
            if len(longest) > 3:  # 至少3个字符
                return longest

    # 策略2：查找所有SMILES样式的匹配
    smiles_pattern = r'[A-Za-z0-9@+\-\[\]\(\)=#\$\/\\\.]{3,}'
    matches = re.findall(smiles_pattern, text)
    if matches:
        # 过滤掉常见的英文单词
        filtered = [m for m in matches if not m.isalpha() or len(m) < 8]
        if filtered:
            return max(filtered, key=len)

    # 策略3：返回原文本（如果很短）
    if len(text) < 50:
        return text

    return ""


# 加载数据
with open('../data/eval_reactions_expanded.json') as f:
    eval_data = json.load(f)

# 加载模型
model_path = '../models/expanded_round_2'
print(f'加载模型: {model_path}')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)
print('✓ 模型加载完成\n')

# 测试前10个样本
print('测试前10个样本（改进版SMILES提取）:')
correct = 0
total = 0

for i, example in enumerate(eval_data[:10]):
    input_text = example['input_text']
    target = example['target_text']

    inputs = tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 使用改进版提取
    pred_smiles = extract_smiles_improved(pred_text)

    is_correct = (pred_smiles == target)
    if is_correct:
        correct += 1
    total += 1

    print(f'[{i}] {example["id"]} ({example["difficulty"]})')
    print(f'  目标: {target}')
    print(f'  预测: {pred_smiles}')
    print(f'  正确: {"✓" if is_correct else "✗"}')
    if not is_correct and i < 3:
        print(f'  原始输出: {pred_text[-300:]}')
    print()

print(f'\n前10个样本准确率: {correct}/{total} = {correct/total:.2%}')

# 如果准确率提高，测试完整评估集
if correct / total > 0.2:
    print('\n✓ 准确率改善，测试完整评估集...')
    correct_all = 0
    total_all = 0

    for example in eval_data:
        input_text = example['input_text']
        target = example['target_text']

        inputs = tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_smiles = extract_smiles_improved(pred_text)

        is_correct = (pred_smiles == target)
        if is_correct:
            correct_all += 1
        total_all += 1

    print(f'\n完整评估集准确率: {correct_all}/{total_all} = {correct_all/total_all:.2%}')
