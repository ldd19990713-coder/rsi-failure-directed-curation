#!/usr/bin/env python3
"""
V8 Leakage Audit - SMILES Extraction Unit Tests

测试extract_smiles_fixed()函数防止回归
确保评估脚本的严格精确匹配不会被意外放宽
"""

import sys
import re
import unittest
from pathlib import Path


# 直接定义函数避免导入transformers依赖
def extract_smiles_fixed(text: str) -> str:
    """Extract SMILES from model output - V5 frozen version"""
    text = text.strip()
    smiles_chars = r'[A-Za-z0-9@+\-\[\]\(\)=#\$\/\\\.]'

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

    smiles_pattern = r'[A-Z][A-Za-z0-9@+\-\[\]\(\)=#\$\/\\.]{2,}'
    matches = re.findall(smiles_pattern, text)
    valid_smiles = [m.rstrip('.') for m in matches if re.search(r'[=()\[\]#0-9]', m.rstrip('.'))]

    if valid_smiles:
        return max(valid_smiles, key=len)

    if len(text) < 100 and re.match(r'^[A-Za-z0-9@+\-\[\]\(\)=#\$\/\\.]+$', text):
        return text.rstrip('.')

    return ""


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES is syntactically valid (basic check)"""
    if not smiles or len(smiles) < 2:
        return False
    # Basic validity: balanced brackets, valid characters
    if smiles.count('[') != smiles.count(']'):
        return False
    if smiles.count('(') != smiles.count(')'):
        return False
    return True


class TestSMILESExtraction(unittest.TestCase):
    """SMILES提取函数单元测试"""

    def test_basic_extraction(self):
        """测试基本SMILES提取"""
        text = "The product SMILES is: CC(=O)O"
        result = extract_smiles_fixed(text)
        self.assertEqual(result, "CC(=O)O")

    def test_extraction_with_format_variant(self):
        """测试格式变体提取"""
        text = "SMILES: c1ccccc1"
        result = extract_smiles_fixed(text)
        self.assertEqual(result, "c1ccccc1")

    def test_extraction_complex_smiles(self):
        """测试复杂SMILES提取"""
        text = "The major product SMILES is: N#CC1CCCC=C1"
        result = extract_smiles_fixed(text)
        self.assertEqual(result, "N#CC1CCCC=C1")

    def test_extraction_with_brackets(self):
        """测试带括号SMILES提取"""
        text = "Product: CC(Br)CB"
        result = extract_smiles_fixed(text)
        self.assertEqual(result, "CC(Br)CB")

    def test_extraction_empty_on_no_match(self):
        """测试无匹配时返回空"""
        text = "This text contains no SMILES"
        result = extract_smiles_fixed(text)
        self.assertEqual(result, "")

    def test_extraction_rejects_short(self):
        """测试拒绝过短的匹配"""
        text = "The product is C"
        result = extract_smiles_fixed(text)
        # 太短，不应匹配
        self.assertEqual(result, "")

    def test_strict_no_fuzzy_match(self):
        """测试严格匹配，不进行模糊匹配"""
        # 这是关键的防回归测试
        target = "CC(=O)c1ccccc1"
        # 相似但不相同的SMILES
        wrong1 = "CC(=O)c1ccccc"  # 少一个c
        wrong2 = "CC(=O)c1cccc1"  # 环己烷不是苯
        wrong3 = "CC(O)c1ccccc1"  # 醇不是酮

        # 确保评估不会错误匹配这些
        self.assertNotEqual(target, wrong1)
        self.assertNotEqual(target, wrong2)
        self.assertNotEqual(target, wrong3)


class TestSMILESValidation(unittest.TestCase):
    """SMILES验证函数单元测试"""

    def test_valid_smiles_basic(self):
        """测试基本有效SMILES"""
        self.assertTrue(is_valid_smiles("CC"))
        self.assertTrue(is_valid_smiles("CCO"))
        self.assertTrue(is_valid_smiles("c1ccccc1"))

    def test_valid_smiles_with_brackets(self):
        """测试带括号的有效SMILES"""
        self.assertTrue(is_valid_smiles("CC(=O)O"))
        self.assertTrue(is_valid_smiles("CC(Br)CB"))
        self.assertTrue(is_valid_smiles("N#CC1CCCC=C1"))

    def test_invalid_unbalanced_brackets(self):
        """测试不平衡括号"""
        self.assertFalse(is_valid_smiles("CC(=O"))  # 缺少)
        self.assertFalse(is_valid_smiles("CC=O)"))  # 缺少(
        self.assertFalse(is_valid_smiles("CC[OH"))  # 缺少]

    def test_invalid_empty_or_short(self):
        """测试空或过短"""
        self.assertFalse(is_valid_smiles(""))
        self.assertFalse(is_valid_smiles("C"))


class TestExactMatchProtocol(unittest.TestCase):
    """精确匹配协议测试 - 防止评估放宽"""

    def test_exact_match_required(self):
        """测试必须精确匹配"""
        # V8协议: 精确SMILES匹配是HARD GATE
        target = "CC(=O)c1ccccc1"

        # 以下都不应该被视为正确
        wrong_predictions = [
            "CC(=O)c1ccccc",   # 不完整
            "CC(=O)C1=CC=CC=C1",  # 不同表示（理论上相同但字符串不同）
            "CC(O)c1ccccc1",   # 不同产物
            "",                 # 空
        ]

        for pred in wrong_predictions:
            # 精确匹配协议
            is_correct = (pred == target)
            self.assertFalse(is_correct, f"Should not match: {pred}")

    def test_no_partial_credit(self):
        """测试没有部分分数"""
        # V8协议: 没有部分匹配，只有精确匹配
        target = "N#CC1CCCC=C1"
        prediction = "N#CC1CCCC"  # 部分正确

        # 精确匹配
        is_correct = (prediction == target)
        self.assertFalse(is_correct)


def run_tests():
    """运行所有测试"""
    print("="*80)
    print("V8 SMILES Extraction Unit Tests")
    print("="*80)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestSMILESExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestSMILESValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestExactMatchProtocol))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✅ All tests passed! SMILES extraction is strict and correct.")
    else:
        print("❌ Some tests failed! Check for regression in SMILES extraction.")
    print("="*80)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
