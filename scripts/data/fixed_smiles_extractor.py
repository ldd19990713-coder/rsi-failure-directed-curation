#!/usr/bin/env python3
"""
修复版SMILES提取器 - 解决错误匹配英文单词的问题
"""

import re

def extract_smiles_fixed(text: str) -> str:
    """修复版SMILES提取 - 只匹配明确的SMILES模式"""
    text = text.strip()

    # SMILES字符集（包含元素符号和化学键符号）
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
    # 这样可以排除"is substitution"这样的英文短语
    match = re.search(r'\bis\s*:\s*(' + smiles_chars + r'+)', text, re.IGNORECASE)
    if match:
        smiles = match.group(1).rstrip('.')
        # 验证：必须包含至少一个非字母字符（=, [, ], (, ), #, 数字等）
        if re.search(r'[=()\[\]#0-9]', smiles) and len(smiles) >= 3:
            return smiles

    # 策略3：查找所有包含化学键符号的SMILES样式的匹配
    # 这可以捕获独立的SMILES字符串
    smiles_pattern = r'[A-Z][A-Za-z0-9@+\-\[\]\(\)=#\$\/\\.]{2,}'
    matches = re.findall(smiles_pattern, text)

    # 过滤出真正的SMILES（必须包含特殊字符）
    valid_smiles = []
    for m in matches:
        m = m.rstrip('.')
        # 必须包含至少一个化学键符号或数字
        if re.search(r'[=()\[\]#0-9]', m):
            valid_smiles.append(m)

    if valid_smiles:
        return max(valid_smiles, key=len)

    # 策略4：如果文本本身就是纯SMILES
    if len(text) < 100 and re.match(r'^[A-Za-z0-9@+\-\[\]\(\)=#\$\/\\.]+$', text):
        return text.rstrip('.')

    return ""


# 测试
if __name__ == "__main__":
    test_cases = [
        # 之前提取错误的样本
        ("The reaction type is substitution_SN2, the reactants are CCBr + CCS[Na]. The product SMILES is: CCSCC",
         "CCSCC"),
        ("Reaction type: cyclization. Cyclization of the given molecule OCCCC(=O)O. The product SMILES is: O=C1OCCC1.",
         "O=C1OCCC1"),
        ("we need to follow the steps of the hydroboration reaction. The product SMILES is: CC(C)CO",
         "CC(C)CO"),
        # 纯SMILES
        ("O=C1OCCC1", "O=C1OCCC1"),
        # 其他情况
        ("The product SMILES is: CC(C)O.", "CC(C)O"),
    ]

    print("测试修复版SMILES提取器:")
    correct = 0
    for i, (text, expected) in enumerate(test_cases):
        result = extract_smiles_fixed(text)
        is_correct = result == expected
        if is_correct:
            correct += 1
        print(f"{i+1}. {'✓' if is_correct else '✗'} 期望: {expected}, 得到: {result}")

    print(f"\n准确率: {correct}/{len(test_cases)} = {correct/len(test_cases):.2%}")
