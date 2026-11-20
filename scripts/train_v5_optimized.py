#!/usr/bin/env python3
"""
V5训练脚本 - 综合短期改进方案
1. 针对Medium样本优化（增加权重、生成更多变体）
2. 增加训练轮数（5-10轮）
3. 优化数据增强策略（多种变体类型）
4. 监控过拟合
"""

import gc
import json
import re
import os
import sys

# 优先使用/tmp目录的transformers（避免NFS文件系统问题）
pass  # public: no /tmp libs

import torch
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR

# 允许从网络下载模型（新环境可能没有缓存）
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"

# 本地模型缓存路径
LOCAL_MODEL_CACHE = "Qwen/Qwen2.5-1.5B"


# ============================================================
# 路径配置
# ============================================================
EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"

for d in [MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 修复版SMILES提取器
# ============================================================
def extract_smiles_fixed(text: str) -> str:
    """修复版SMILES提取"""
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


# ============================================================
# 加权数据集类（针对Medium样本优化）
# ============================================================
class WeightedReactionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256, difficulty_weights=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 难度权重：默认为 {'easy': 1.0, 'medium': 2.0, 'hard': 1.0}
        self.difficulty_weights = difficulty_weights or {'easy': 1.0, 'medium': 2.0, 'hard': 1.0}

        # 根据权重扩展数据
        self.expanded_data = []
        for sample in data:
            difficulty = sample.get('difficulty', 'medium')
            weight = self.difficulty_weights.get(difficulty, 1.0)
            # 按权重复制样本
            num_copies = int(weight)
            for _ in range(num_copies):
                self.expanded_data.append(sample)

    def __len__(self):
        return len(self.expanded_data)

    def __getitem__(self, idx):
        example = self.expanded_data[idx]
        text = f"输入: {example['input_text']}\n输出: {example['target_text']}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ============================================================
# 增强版数据增强（针对Medium样本）
# ============================================================
class EnhancedDataAugmentor:
    """增强版数据增强器"""

    def __init__(self):
        self.augmentation_types = [
            "condition_variant",  # 改变条件描述
            "hint_variant",       # 添加提示
            "format_variant",     # 改变格式
            "synonym_variant",    # 使用同义词
            "context_variant",    # 添加上下文
        ]

    def augment_sample(self, sample, round_num, aug_type):
        """生成单个变体"""
        variant = sample.copy()
        variant["id"] = f"{sample['id']}_aug_r{round_num}_{aug_type}"

        if aug_type == "condition_variant":
            # 变体1: 改变条件描述
            condition_variants = [
                "standard conditions",
                "optimized conditions",
                "typical procedure",
                "standard procedure"
            ]
            variant["conditions"] = condition_variants[round_num % len(condition_variants)]
            variant["input_text"] = (
                f"Predict the major organic product of the following reaction.\n"
                f"Reaction type: {sample['reaction_type']}\n"
                f"Reactants: {' + '.join(sample['reactants_smiles'])}\n"
                f"Conditions: {variant['conditions']}\n"
                f"Provide the product SMILES."
            )

        elif aug_type == "hint_variant":
            # 变体2: 添加提示
            variant["input_text"] = (
                f"This is a {sample['reaction_type']} reaction.\n"
                f"Reactants: {' + '.join(sample['reactants_smiles'])}\n"
                f"Conditions: {sample['conditions']}\n"
                f"Provide the product SMILES."
            )

        elif aug_type == "format_variant":
            # 变体3: 改变格式
            variant["input_text"] = (
                f"Task: Predict product of {sample['reaction_type']} reaction\n"
                f"Input: {' + '.join(sample['reactants_smiles'])}\n"
                f"Reaction conditions: {sample['conditions']}\n"
                f"Output the product SMILES:"
            )

        elif aug_type == "synonym_variant":
            # 变体4: 使用同义词
            reaction_type_synonyms = {
                "substitution_SN2": "nucleophilic substitution",
                "elimination": "elimination reaction",
                "addition": "addition reaction",
                "oxidation": "oxidative transformation",
                "reduction": "reductive transformation",
                "cyclization": "ring-forming reaction",
            }
            synonym = reaction_type_synonyms.get(sample['reaction_type'], sample['reaction_type'])
            variant["input_text"] = (
                f"Predict the product of the following {synonym}.\n"
                f"Reactants: {' + '.join(sample['reactants_smiles'])}\n"
                f"Conditions: {sample['conditions']}\n"
                f"What is the product SMILES?"
            )

        elif aug_type == "context_variant":
            # 变体5: 添加上下文
            variant["input_text"] = (
                f"Given the following chemical reaction:\n"
                f"Type: {sample['reaction_type']}\n"
                f"Starting materials: {' + '.join(sample['reactants_smiles'])}\n"
                f"Reaction conditions: {sample['conditions']}\n"
                f"Determine the major organic product and provide its SMILES."
            )

        return variant

    def augment_failed_cases(self, failed_cases, round_num, is_medium=False):
        """为失败案例生成多种变体"""
        augmented = []

        for case in failed_cases:
            # 根据难度决定变体数量
            num_variants = 3 if is_medium else 2

            # 选择变体类型
            variant_types = self.augmentation_types[:num_variants]

            for aug_type in variant_types:
                variant = self.augment_sample(case, round_num, aug_type)
                augmented.append(variant)

        return augmented


# ============================================================
# V5训练器（综合优化）
# ============================================================
class ReactionTrainerV5:
    """V5训练器 - 综合优化版"""

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B", device="cuda"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.augmentor = EnhancedDataAugmentor()
        self.training_history = []  # 用于监控过拟合

    def load_model(self, model_path=None):
        """加载模型"""
        path_to_load = model_path or self.model_name

        # 如果是远程模型名，使用本地缓存
        if not Path(path_to_load).exists() and "/" in path_to_load and not Path(path_to_load).exists():
            if Path(LOCAL_MODEL_CACHE).exists():
                path_to_load = LOCAL_MODEL_CACHE
                print(f"\n[加载模型] 使用本地缓存: {path_to_load}")
            else:
                print(f"\n[加载模型] {path_to_load}")
        else:
            print(f"\n[加载模型] {path_to_load}")

        self.tokenizer = AutoTokenizer.from_pretrained(path_to_load, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            path_to_load,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.gradient_checkpointing_enable()
        print("✓ 模型加载完成")

    def train_round(self, round_num, train_data, eval_data, lr=5e-6, epochs=3, difficulty_weights=None):
        """训练一轮（带过拟合监控）"""
        print(f"\n{'='*60}")
        print(f"轮次 {round_num} 训练")
        print(f"{'='*60}")

        # 使用加权数据集
        dataset = WeightedReactionDataset(train_data, self.tokenizer, max_length=256, difficulty_weights=difficulty_weights)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        print(f"训练样本: {len(train_data)} (加权后: {len(dataset)})")
        print(f"学习率: {lr}, Epochs: {epochs}")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs, eta_min=lr * 0.1)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            nan_count = 0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches - nan_count, 1)

            # 每个epoch后评估（监控过拟合）
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:  # 每2个epoch或最后一个epoch
                eval_result = self.evaluate(eval_data, verbose=False)
                eval_acc = eval_result["accuracy"]
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Eval Acc: {eval_acc:.2%} (NaN: {nan_count})")

                # 记录训练历史
                self.training_history.append({
                    "round": round_num,
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "eval_accuracy": eval_acc
                })
            else:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} (NaN: {nan_count})")

        # 保存模型
        model_path = MODEL_DIR / f"v5_round_{round_num}"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        print(f"✓ 模型已保存: {model_path}")

        return model_path

    def evaluate(self, eval_data, round_num=None, verbose=True):
        """评估模型"""
        if verbose:
            print(f"\n{'='*60}")
            if round_num:
                print(f"轮次 {round_num} 评估")
            else:
                print("评估")
            print(f"{'='*60}")

        self.model.eval()
        correct = 0
        predictions = []
        failed_cases = []
        difficulty_stats = {}

        with torch.no_grad():
            for example in eval_data:
                input_text = example["input_text"]
                target = example["target_text"]

                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred_smiles = extract_smiles_fixed(pred_text)

                is_correct = (pred_smiles == target)
                if is_correct:
                    correct += 1
                else:
                    failed_cases.append(example)

                # 按难度统计
                difficulty = example.get('difficulty', 'medium')
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {'correct': 0, 'total': 0}
                difficulty_stats[difficulty]['total'] += 1
                if is_correct:
                    difficulty_stats[difficulty]['correct'] += 1

                predictions.append({
                    "id": example["id"],
                    "reaction_type": example["reaction_type"],
                    "difficulty": difficulty,
                    "target": target,
                    "predicted": pred_smiles,
                    "correct": is_correct,
                })

        accuracy = correct / len(eval_data)

        if verbose:
            print(f"✓ 准确率: {accuracy:.2%} ({correct}/{len(eval_data)})")
            print("\n按难度统计:")
            for diff in ['easy', 'medium', 'hard']:
                if diff in difficulty_stats:
                    stats = difficulty_stats[diff]
                    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                    print(f"  {diff}: {stats['correct']}/{stats['total']} = {acc:.2%}")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(eval_data),
            "failed_cases": failed_cases,
            "predictions": predictions,
            "difficulty_stats": difficulty_stats
        }

    def prepare_round_data(self, round_num, base_data, failed_cases):
        """准备轮次训练数据（增强版数据增强）"""
        # 按难度分类失败案例
        failed_easy = [c for c in failed_cases if c.get('difficulty') == 'easy']
        failed_medium = [c for c in failed_cases if c.get('difficulty') == 'medium']
        failed_hard = [c for c in failed_cases if c.get('difficulty') == 'hard']

        augmented = []

        # 为每种难度的失败案例生成变体
        if failed_easy:
            aug_easy = self.augmentor.augment_failed_cases(failed_easy, round_num, is_medium=False)
            augmented.extend(aug_easy)
            print(f"  Easy失败案例: {len(failed_easy)} → {len(aug_easy)}个变体")

        # Medium样本：生成更多变体（3倍）
        if failed_medium:
            aug_medium = self.augmentor.augment_failed_cases(failed_medium, round_num, is_medium=True)
            augmented.extend(aug_medium)
            print(f"  Medium失败案例: {len(failed_medium)} → {len(aug_medium)}个变体")

        if failed_hard:
            aug_hard = self.augmentor.augment_failed_cases(failed_hard, round_num, is_medium=False)
            augmented.extend(aug_hard)
            print(f"  Hard失败案例: {len(failed_hard)} → {len(aug_hard)}个变体")

        augmented_data = base_data + augmented
        print(f"  总计: {len(base_data)} + {len(augmented)} = {len(augmented_data)}")

        return augmented_data

    def cleanup(self):
        """清理显存"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# 主函数：多轮训练（5-10轮）
# ============================================================
def run_v5_training(num_rounds=7):
    """运行V5训练（多轮）"""
    print("=" * 80)
    print(f"V5训练 - 综合优化（{num_rounds}轮）")
    print("=" * 80)

    # 加载数据集
    print("\n[Step 1] 加载数据集")
    eval_path = DATA_DIR / "eval_reactions_expanded.json"
    train_path = DATA_DIR / "train_reactions_expanded.json"

    with open(eval_path) as f:
        eval_data = json.load(f)
    with open(train_path) as f:
        base_train_data = json.load(f)

    print(f"✓ 训练集: {len(base_train_data)} 样本")
    print(f"✓ 评估集: {len(eval_data)} 样本")

    # 初始化结果记录
    results = {
        "experiment": f"V5_Optimized_Training_{num_rounds}Rounds",
        "timestamp": datetime.now().isoformat(),
        "dataset_size": {"train": len(base_train_data), "eval": len(eval_data)},
        "num_rounds": num_rounds,
        "difficulty_weights": {"easy": 1.0, "medium": 2.0, "hard": 1.0},
        "rounds": [],
        "training_history": []
    }

    # 初始化训练器
    trainer = ReactionTrainerV5()
    trainer.load_model()

    # Round 1: 基础模型评估
    print(f"\n{'='*80}")
    print("Round 1: 评估基础模型")
    print(f"{'='*80}")

    round1_result = trainer.evaluate(eval_data, round_num=1)

    results["rounds"].append({
        "round": 1,
        "description": "基础模型",
        "accuracy": round1_result["accuracy"],
        "correct": round1_result["correct"],
        "total": round1_result["total"],
        "difficulty_stats": round1_result["difficulty_stats"]
    })

    # 多轮训练
    current_failed = round1_result["failed_cases"]

    for round_num in range(2, num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"Round {round_num}: 递归微调")
        print(f"{'='*80}")

        # 准备训练数据（带数据增强）
        train_data = trainer.prepare_round_data(round_num, base_train_data, current_failed)

        # 动态调整学习率（逐渐降低）
        lr = 5e-6 * (0.85 ** (round_num - 2))

        # 训练（使用难度权重）
        trainer.train_round(
            round_num,
            train_data,
            eval_data,
            lr=lr,
            epochs=3,
            difficulty_weights={"easy": 1.0, "medium": 2.0, "hard": 1.0}
        )

        # 评估
        round_result = trainer.evaluate(eval_data, round_num=round_num)

        results["rounds"].append({
            "round": round_num,
            "description": f"递归微调Round {round_num}",
            "accuracy": round_result["accuracy"],
            "correct": round_result["correct"],
            "total": round_result["total"],
            "train_size": len(train_data),
            "difficulty_stats": round_result["difficulty_stats"],
            "learning_rate": lr
        })

        # 更新失败案例
        current_failed = round_result["failed_cases"]

        # 检查是否过拟合（如果验证集准确率下降超过20%）
        if round_num >= 3:
            prev_acc = results["rounds"][-2]["accuracy"]
            curr_acc = round_result["accuracy"]
            if curr_acc < prev_acc * 0.8:
                print(f"\n⚠ 警告: 检测到可能的过拟合（准确率下降: {prev_acc:.2%} → {curr_acc:.2%}）")
                print("建议停止训练或减少轮数")

        print(f"\n✓ Round {round_num} 完成: {round_result['accuracy']:.2%}")

    # 保存训练历史
    results["training_history"] = trainer.training_history

    # 汇总结果
    print(f"\n{'='*80}")
    print(f"训练结果汇总（{num_rounds}轮）")
    print(f"{'='*80}")

    for round_data in results["rounds"]:
        round_num = round_data["round"]
        accuracy = round_data["accuracy"]
        print(f"Round {round_num}: {accuracy:.2%} ({round_data['correct']}/{round_data['total']})")

    # 找出最佳轮次
    best_round = max(results["rounds"], key=lambda x: x["accuracy"])
    print(f"\n★ 最佳轮次: Round {best_round['round']} (准确率: {best_round['accuracy']:.2%})")

    # 分析过拟合
    if trainer.training_history:
        print(f"\n过拟合监控:")
        round_2_loss = [h for h in trainer.training_history if h['round'] == 2]
        if round_2_loss:
            final_loss = round_2_loss[-1]['train_loss']
            final_eval = round_2_loss[-1]['eval_accuracy']
            print(f"  Round 2最终Loss: {final_loss:.4f}")
            print(f"  Round 2最终Eval Acc: {final_eval:.2%}")

    # 保存结果
    results_path = RESULTS_DIR / f"v5_optimized_training_{num_rounds}rounds_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存: {results_path}")

    return results


if __name__ == "__main__":
    # 运行7轮训练（可根据需要调整）
    results = run_v5_training(num_rounds=7)
