#!/usr/bin/env python3
"""
V8 Phase 2 - Data Deduplication

根据Phase 1近重复扫描结果，移除train中与test有重复产物SMILES的样本
策略：从train中移除重复样本（保持test完整性）
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts" / "v8"


def load_data():
    """加载原始数据"""
    train_path = DATA_DIR / "train_reactions_expanded.json"
    eval_path = DATA_DIR / "eval_reactions_expanded.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    return train_data, eval_data


def extract_product_smiles(item):
    """提取产物SMILES"""
    product = item.get("product_smiles", "")
    target = item.get("target_text", "")
    return (product or target).strip()


def find_duplicates(train_data, eval_data):
    """找出train中与test有重复产物的样本"""
    # 建立test产物索引
    test_products = {}
    for item in eval_data:
        smiles = extract_product_smiles(item)
        if smiles:
            test_products[smiles] = item.get("id")

    # 找出train中的重复
    duplicates = []
    for item in train_data:
        smiles = extract_product_smiles(item)
        if smiles in test_products:
            duplicates.append({
                "train_id": item.get("id"),
                "test_id": test_products[smiles],
                "product_smiles": smiles,
                "reaction_type": item.get("reaction_type", "unknown")
            })

    return duplicates


def deduplicate(train_data, duplicates):
    """从train中移除重复样本"""
    duplicate_ids = {d["train_id"] for d in duplicates}
    deduped = [item for item in train_data if item.get("id") not in duplicate_ids]
    return deduped, duplicate_ids


def save_deduped_data(train_data, eval_data, duplicates, removed_ids):
    """保存去重后的数据"""
    # 创建版本化目录
    version = "v8_deduped"
    versioned_dir = DATA_DIR / version
    versioned_dir.mkdir(parents=True, exist_ok=True)

    # 保存去重后的train
    train_path = versioned_dir / "train_reactions_deduped.json"
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    # 复制eval（保持不变）
    eval_path = versioned_dir / "eval_reactions_deduped.json"
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    # 保存去重清单
    manifest = {
        "version": version,
        "created": datetime.now().isoformat(),
        "original_train_size": len(train_data) + len(removed_ids),
        "deduped_train_size": len(train_data),
        "eval_size": len(eval_data),
        "removed_samples": list(removed_ids),
        "removed_count": len(removed_ids),
        "duplicates": duplicates,
        "strategy": "remove_from_train",
        "reason": "Ensure train ∩ test = ∅ by product SMILES"
    }

    manifest_path = versioned_dir / "dedup_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return train_path, eval_path, manifest_path


def main():
    print("="*80)
    print("V8 Phase 2 - Data Deduplication")
    print("="*80)
    print(f"\n开始时间: {datetime.now().isoformat()}")

    # 加载数据
    print("\n[1/4] 加载原始数据...")
    train_data, eval_data = load_data()
    print(f"  Train: {len(train_data)} 样本")
    print(f"  Test:  {len(eval_data)} 样本")

    # 找出重复
    print("\n[2/4] 检测重复产物...")
    duplicates = find_duplicates(train_data, eval_data)
    print(f"  发现 {len(duplicates)} 个重复:")
    for dup in duplicates:
        print(f"    - {dup['train_id']} ↔ {dup['test_id']}: {dup['product_smiles'][:40]}...")

    # 去重
    print("\n[3/4] 执行去重...")
    deduped_train, removed_ids = deduplicate(train_data, duplicates)
    print(f"  移除样本: {sorted(removed_ids)}")
    print(f"  去重后Train: {len(deduped_train)} 样本 (-{len(removed_ids)})")

    # 保存
    print("\n[4/4] 保存去重数据...")
    train_path, eval_path, manifest_path = save_deduped_data(deduped_train, eval_data, duplicates, removed_ids)
    print(f"  ✓ 去重Train: {train_path}")
    print(f"  ✓ 去重Eval:  {eval_path}")
    print(f"  ✓ 清单:      {manifest_path}")

    # 统计
    print(f"\n{'='*80}")
    print("去重统计")
    print(f"{'='*80}")
    print(f"  原始Train: {len(train_data) + len(removed_ids)}")
    print(f"  移除数量:  {len(removed_ids)}")
    print(f"  去重Train: {len(deduped_train)}")
    print(f"  Eval:      {len(eval_data)} (未变)")

    # 难度分布
    print(f"\n难度分布 (去重后Train):")
    difficulty_counts = defaultdict(int)
    for item in deduped_train:
        diff = item.get("difficulty", "unknown")
        difficulty_counts[diff] += 1
    for diff in ["easy", "medium", "hard"]:
        print(f"  {diff}: {difficulty_counts[diff]}")

    print(f"\n✓ 去重完成!")
    print(f"  下一步: 使用新数据路径重新运行G1")

    return 0


if __name__ == "__main__":
    sys.exit(main())
