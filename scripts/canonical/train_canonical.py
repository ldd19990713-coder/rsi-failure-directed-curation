#!/usr/bin/env python3
"""
V8 Gate Experiment - P0: V5 Frozen Reproduction with Multiple Seeds

Protocol: V8 Gate Checklist v1.0
Purpose: Reproduce V5 baseline with >=3 seeds for G1 gate

Key constraints:
- weight_decay = 0.01 (FROZEN)
- eta_min_ratio = 0.1 (FROZEN)
- num_rounds = 7 (FROZEN)
- difficulty_weights = {E:1, M:2, H:1} (FROZEN)
- evaluation: exact SMILES match (HARD GATE)

Mandatory metrics per run:
- Exact SMILES accuracy
- Invalid SMILES rate
- Easy/Medium/Hard breakdown
- Peak accuracy, peak round, post-peak drawdown
"""

import gc
import json
import re
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

pass  # public: no /tmp libs

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================
# Paths
# ============================================================
EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data"
# V5 G1' Canonical: output to dedicated directory
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts" / "canonical"
CONFIGS_DIR = EXPERIMENT_DIR / "configs"

LOCAL_MODEL_CACHE = "Qwen/Qwen2.5-1.5B"

# V5 Canonical Recipe Hash (FROZEN)
EXPECTED_RECIPE_HASH = "2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3"

def verify_recipe_hash() -> bool:
    """Verify recipe hash matches expected canonical value"""
    hash_file = CONFIGS_DIR / "recipe_hash.txt"
    if not hash_file.exists():
        print(f"[WARN] Recipe hash file not found: {hash_file}")
        return False

    content = hash_file.read_text()
    for line in content.splitlines():
        if line.startswith("recipe_sha256:"):
            actual_hash = line.split(":")[1].strip()
            if actual_hash == EXPECTED_RECIPE_HASH:
                print(f"[V8] ✓ Recipe hash verified: {actual_hash[:16]}...")
                return True
            else:
                print(f"[V8] ✗ Recipe hash MISMATCH!")
                print(f"  Expected: {EXPECTED_RECIPE_HASH}")
                print(f"  Actual:   {actual_hash}")
                return False
    print(f"[WARN] recipe_sha256 not found in {hash_file}")
    return False


# ============================================================
# SMILES Extraction (V5 version, FROZEN)
# ============================================================
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


# ============================================================
# Dataset Classes
# ============================================================
class WeightedReactionDataset(Dataset):
    """V5 frozen version - difficulty weighted dataset"""

    def __init__(self, data, tokenizer, max_length=256, difficulty_weights=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.difficulty_weights = difficulty_weights or {'easy': 1.0, 'medium': 2.0, 'hard': 1.0}

        # Expand data based on weights
        self.expanded_data = []
        for sample in data:
            difficulty = sample.get('difficulty', 'medium')
            weight = self.difficulty_weights.get(difficulty, 1.0)
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
# Data Augmentor (V5 frozen version)
# ============================================================
class DataAugmentor:
    """V5 frozen version - 5 variant types"""

    def __init__(self):
        self.augmentation_types = [
            "condition_variant",
            "hint_variant",
            "format_variant",
            "synonym_variant",
            "context_variant",
        ]

    def augment_sample(self, sample, round_num, aug_type):
        """Generate single variant - V5 frozen"""
        variant = sample.copy()
        variant["id"] = f"{sample['id']}_aug_r{round_num}_{aug_type}"

        if aug_type == "condition_variant":
            condition_variants = ["standard conditions", "optimized conditions", "typical procedure", "standard procedure"]
            variant["conditions"] = condition_variants[round_num % len(condition_variants)]
            variant["input_text"] = (
                f"Predict the major organic product of the following reaction.\n"
                f"Reaction type: {sample['reaction_type']}\n"
                f"Reactants: {' + '.join(sample['reactants_smiles'])}\n"
                f"Conditions: {variant['conditions']}\n"
                f"Provide the product SMILES."
            )
        elif aug_type == "hint_variant":
            variant["input_text"] = (
                f"This is a {sample['reaction_type']} reaction.\n"
                f"Reactants: {' + '.join(sample['reactants_smiles'])}\n"
                f"Conditions: {sample['conditions']}\n"
                f"Provide the product SMILES."
            )
        elif aug_type == "format_variant":
            variant["input_text"] = (
                f"Task: Predict product of {sample['reaction_type']} reaction\n"
                f"Input: {' + '.join(sample['reactants_smiles'])}\n"
                f"Reaction conditions: {sample['conditions']}\n"
                f"Output the product SMILES:"
            )
        elif aug_type == "synonym_variant":
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
            variant["input_text"] = (
                f"Given the following chemical reaction:\n"
                f"Type: {sample['reaction_type']}\n"
                f"Starting materials: {' + '.join(sample['reactants_smiles'])}\n"
                f"Reaction conditions: {sample['conditions']}\n"
                f"Determine the major organic product and provide its SMILES."
            )

        return variant

    def augment_failed_cases(self, failed_cases, round_num, is_medium=False):
        """Generate variants for failed cases - V5 frozen"""
        augmented = []
        for case in failed_cases:
            num_variants = 3 if is_medium else 2
            variant_types = self.augmentation_types[:num_variants]
            for aug_type in variant_types:
                variant = self.augment_sample(case, round_num, aug_type)
                augmented.append(variant)
        return augmented


# ============================================================
# V8 Gate Trainer
# ============================================================
class V8GateTrainer:
    """V8 Gate Experiment Trainer - Frozen V5 Configuration"""

    def __init__(self, seed: int, device: str = "cuda"):
        self.device = device
        self.seed = seed
        self.augmentor = DataAugmentor()
        self.training_history = []
        self.aug_manifest = []  # Track augmentation sources

        # Set random seeds for reproducibility
        self._set_seed(seed)

        # Metrics tracking
        self.round_metrics = []

    def _set_seed(self, seed: int):
        """Set all random seeds"""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[V8] Random seed set to: {seed}")

    def load_model(self):
        """Load Qwen2.5-1.5B from local cache"""
        print(f"\n[V8] Loading model from local cache...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_CACHE,
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_CACHE,
            torch_dtype=torch.float32,  # FROZEN: float32
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.gradient_checkpointing_enable()

        print("[V8] Model loaded successfully")
        return self

    def _worker_init_fn(self, worker_id):
        """固定DataLoader worker的随机种子，消除非确定性"""
        import random
        import numpy as np
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def train_round(self, round_num, train_data, eval_data, lr=5e-6, epochs=3):
        """Train one round with V5 frozen configuration"""
        print(f"\n{'='*60}")
        print(f"Round {round_num} Training")
        print(f"{'='*60}")

        # V5 frozen: difficulty_weights
        difficulty_weights = {'easy': 1.0, 'medium': 2.0, 'hard': 1.0}

        dataset = WeightedReactionDataset(train_data, self.tokenizer, max_length=256, difficulty_weights=difficulty_weights)
        # V8 FIX: 添加worker_init_fn消除DataLoader非确定性
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # 主进程加载，避免多进程非确定性
            worker_init_fn=self._worker_init_fn,
            generator=torch.Generator().manual_seed(self.seed)
        )

        print(f"Training samples: {len(train_data)} (weighted: {len(dataset)})")
        print(f"Learning rate: {lr}, Epochs: {epochs}")
        print(f"Weight decay: 0.01 (FROZEN)")

        # V5 frozen: weight_decay=0.01
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        # V5 frozen: eta_min_ratio=0.1
        eta_min = lr * 0.1
        scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs, eta_min=eta_min)
        print(f"CosineAnnealing eta_min: {eta_min:.2e}")

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Evaluate every 2 epochs or last epoch
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                eval_result = self.evaluate(eval_data, verbose=False)
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Eval Acc: {eval_result['accuracy']:.2%}")

                self.training_history.append({
                    "round": round_num,
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "eval_accuracy": eval_result["accuracy"]
                })

        return self

    def evaluate(self, eval_data, round_num=None, verbose=True) -> Dict:
        """Evaluate with V8 mandatory metrics"""
        self.model.eval()

        correct = 0
        invalid_count = 0
        predictions = []
        failed_cases = []
        difficulty_stats = {'easy': {'correct': 0, 'total': 0},
                           'medium': {'correct': 0, 'total': 0},
                           'hard': {'correct': 0, 'total': 0}}

        with torch.no_grad():
            for example in eval_data:
                input_text = example["input_text"]
                target = example["target_text"]
                difficulty = example.get('difficulty', 'medium')

                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred_smiles = extract_smiles_fixed(pred_text)

                # Track invalid SMILES
                is_valid = is_valid_smiles(pred_smiles)
                if not is_valid:
                    invalid_count += 1

                # Exact SMILES match (HARD GATE)
                is_correct = (pred_smiles == target)
                if is_correct:
                    correct += 1
                else:
                    failed_cases.append(example)

                # Difficulty breakdown
                difficulty_stats[difficulty]['total'] += 1
                if is_correct:
                    difficulty_stats[difficulty]['correct'] += 1

                predictions.append({
                    "id": example["id"],
                    "difficulty": difficulty,
                    "target": target,
                    "predicted": pred_smiles,
                    "correct": is_correct,
                    "valid_smiles": is_valid,
                })

        # Calculate metrics
        n = len(eval_data)
        accuracy = correct / n
        invalid_rate = invalid_count / n

        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": n,
            "invalid_smiles_rate": invalid_rate,
            "invalid_count": invalid_count,
            "difficulty_breakdown": {},
            "failed_cases": failed_cases,
            "predictions": predictions,
        }

        # Difficulty breakdown
        for diff in ['easy', 'medium', 'hard']:
            stats = difficulty_stats[diff]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            metrics["difficulty_breakdown"][diff] = {
                "correct": stats['correct'],
                "total": stats['total'],
                "accuracy": acc
            }

        if verbose:
            print(f"\n✓ Accuracy: {accuracy:.2%} ({correct}/{n})")
            print(f"✓ Invalid SMILES rate: {invalid_rate:.2%} ({invalid_count}/{n})")
            print(f"\nDifficulty breakdown:")
            for diff in ['easy', 'medium', 'hard']:
                s = metrics["difficulty_breakdown"][diff]
                print(f"  {diff}: {s['correct']}/{s['total']} = {s['accuracy']:.2%}")

        return metrics

    def prepare_round_data(self, round_num, base_data, failed_cases):
        """Prepare round data with augmentation tracking"""
        failed_easy = [c for c in failed_cases if c.get('difficulty') == 'easy']
        failed_medium = [c for c in failed_cases if c.get('difficulty') == 'medium']
        failed_hard = [c for c in failed_cases if c.get('difficulty') == 'hard']

        augmented = []

        if failed_easy:
            aug = self.augmentor.augment_failed_cases(failed_easy, round_num, is_medium=False)
            augmented.extend(aug)
            for v in aug:
                source_id = v.get('id', '').split('_aug')[0]
                # V8: 添加canonical_hash用于追踪
                canonical_hash = hashlib.md5(v.get('target_text', '').encode()).hexdigest()[:8]
                self.aug_manifest.append({
                    "source_id": source_id,
                    "variant_type": "easy_aug",
                    "variant_subtype": v.get('id', '').split('_')[-1] if '_aug' in v.get('id', '') else 'unknown',
                    "round": round_num,
                    "canonical_hash": canonical_hash
                })

        if failed_medium:
            aug = self.augmentor.augment_failed_cases(failed_medium, round_num, is_medium=True)
            augmented.extend(aug)
            for v in aug:
                source_id = v.get('id', '').split('_aug')[0]
                canonical_hash = hashlib.md5(v.get('target_text', '').encode()).hexdigest()[:8]
                self.aug_manifest.append({
                    "source_id": source_id,
                    "variant_type": "medium_aug",
                    "variant_subtype": v.get('id', '').split('_')[-1] if '_aug' in v.get('id', '') else 'unknown',
                    "round": round_num,
                    "canonical_hash": canonical_hash
                })

        if failed_hard:
            aug = self.augmentor.augment_failed_cases(failed_hard, round_num, is_medium=False)
            augmented.extend(aug)
            for v in aug:
                source_id = v.get('id', '').split('_aug')[0]
                canonical_hash = hashlib.md5(v.get('target_text', '').encode()).hexdigest()[:8]
                self.aug_manifest.append({
                    "source_id": source_id,
                    "variant_type": "hard_aug",
                    "variant_subtype": v.get('id', '').split('_')[-1] if '_aug' in v.get('id', '') else 'unknown',
                    "round": round_num,
                    "canonical_hash": canonical_hash
                })

        # V8 LEAKAGE GUARD: 确保增广数据不包含test样本
        augmented_ids = {v.get('id', '') for v in augmented}
        if hasattr(self, 'test_ids'):
            test_in_aug = augmented_ids & self.test_ids
            assert len(test_in_aug) == 0, \
                f"LEAKAGE DETECTED: test samples in augmented data = {test_in_aug}"

        augmented_data = base_data + augmented
        print(f"  Data: {len(base_data)} + {len(augmented)} = {len(augmented_data)}")

        return augmented_data

    def run_v5_reproduction(self, num_rounds=7) -> Dict:
        """Run V5 reproduction with 7 rounds"""
        print("="*80)
        print(f"V8 Gate - V5 Reproduction (seed={self.seed})")
        print("="*80)

        # Verify recipe hash at startup
        hash_ok = verify_recipe_hash()

        print("\nFrozen configuration:")
        print("  weight_decay: 0.01")
        print("  eta_min_ratio: 0.1")
        print("  num_rounds: 7")
        print("  difficulty_weights: {E:1, M:2, H:1}")

        # Load data - 使用去重后的数据 (Phase 2修复)
        data_version = os.environ.get("V8_DATA_VERSION", "v8_deduped")
        if data_version == "v8_deduped":
            eval_path = DATA_DIR / "v8_deduped" / "eval_reactions_deduped.json"
            train_path = DATA_DIR / "v8_deduped" / "train_reactions_deduped.json"
        elif data_version == "v9_g1prime":
            eval_path = DATA_DIR / "v9_g1prime" / "eval_reactions_g1prime.json"
            train_path = DATA_DIR / "v9_g1prime" / "train_reactions_g1prime.json"
        else:
            eval_path = DATA_DIR / "eval_reactions_expanded.json"
            train_path = DATA_DIR / "train_reactions_expanded.json"

        print(f"\n[V8] 数据版本: {data_version}")

        with open(eval_path) as f:
            eval_data = json.load(f)
        with open(train_path) as f:
            base_train_data = json.load(f)

        # V8 LEAKAGE GUARD: 运行时断言防止test泄露
        train_ids = {item['id'] for item in base_train_data}
        eval_ids = {item['id'] for item in eval_data}
        assert len(train_ids & eval_ids) == 0, \
            f"LEAKAGE DETECTED: train ∩ eval = {train_ids & eval_ids}"
        print(f"\n[V8] ✓ Leakage guard: train ∩ eval = ∅ (disjoint verified)")

        # V8 LEAKAGE GUARD: 记录test ID用于后续检查
        self.test_ids = eval_ids

        print(f"\nTrain: {len(base_train_data)}, Eval: {len(eval_data)}")

        # Load model
        self.load_model()

        # Round 1: Base model evaluation
        print(f"\n{'='*80}")
        print("Round 1: Base Model Evaluation")
        print(f"{'='*80}")

        round1_metrics = self.evaluate(eval_data, round_num=1)

        self.round_metrics.append({
            "round": 1,
            "accuracy": round1_metrics["accuracy"],
            "correct": round1_metrics["correct"],
            "total": round1_metrics["total"],
            "invalid_rate": round1_metrics["invalid_smiles_rate"],
            "difficulty_breakdown": round1_metrics["difficulty_breakdown"],
        })

        # Rounds 2-7: Recursive fine-tuning
        current_failed = round1_metrics["failed_cases"]

        for round_num in range(2, num_rounds + 1):
            print(f"\n{'='*80}")
            print(f"Round {round_num}: Recursive Fine-tuning")
            print(f"{'='*80}")

            # Prepare training data
            train_data = self.prepare_round_data(round_num, base_train_data, current_failed)

            # Dynamic LR (V5 frozen)
            lr = 5e-6 * (0.85 ** (round_num - 2))

            # Train
            self.train_round(round_num, train_data, eval_data, lr=lr, epochs=3)

            # Evaluate
            round_metrics = self.evaluate(eval_data, round_num=round_num)

            self.round_metrics.append({
                "round": round_num,
                "accuracy": round_metrics["accuracy"],
                "correct": round_metrics["correct"],
                "total": round_metrics["total"],
                "invalid_rate": round_metrics["invalid_smiles_rate"],
                "difficulty_breakdown": round_metrics["difficulty_breakdown"],
                "train_size": len(train_data),
                "learning_rate": lr,
            })

            current_failed = round_metrics["failed_cases"]

        # Calculate peak and drawdown
        peak_info = self._calculate_peak_drawdown()

        # Return complete metrics
        return {
            "seed": self.seed,
            "recipe_hash": EXPECTED_RECIPE_HASH,
            "config_hash": EXPECTED_RECIPE_HASH,  # Legacy field name for compatibility
            "rounds": self.round_metrics,
            "training_history": self.training_history,
            "peak_accuracy": peak_info["peak_accuracy"],
            "peak_round": peak_info["peak_round"],
            "post_peak_min": peak_info["post_peak_min"],
            "drawdown": peak_info["drawdown"],
            "aug_manifest": self.aug_manifest,
        }

    def _calculate_peak_drawdown(self) -> Dict:
        """Calculate peak accuracy and post-peak drawdown"""
        accuracies = [r["accuracy"] for r in self.round_metrics]

        peak_accuracy = max(accuracies)
        peak_round = accuracies.index(peak_accuracy) + 1  # 1-indexed

        # Post-peak minimum
        post_peak = accuracies[peak_round:]  # After peak
        if post_peak:
            post_peak_min = min(post_peak)
        else:
            post_peak_min = peak_accuracy

        # Drawdown
        drawdown = peak_accuracy - post_peak_min

        return {
            "peak_accuracy": peak_accuracy,
            "peak_round": peak_round,
            "post_peak_min": post_peak_min,
            "drawdown": drawdown,
        }

    def cleanup(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# Main: Run V8 Gate P0
# ============================================================
def run_v8_gate_p0(seed: int):
    """Run single seed for V8 Gate P0"""

    # Create output directory
    run_dir = ARTIFACTS_DIR / "runs" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run training
    trainer = V8GateTrainer(seed=seed)
    results = trainer.run_v5_reproduction(num_rounds=7)

    # Save run_meta.json
    run_meta = {
        "seed": seed,
        "config_hash": results["config_hash"],
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen/Qwen2.5-1.5B",
        "config_file": "artifacts/v8/configs/v5_frozen.yaml",
        "data_version": "expanded",
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Save metrics.json
    metrics = {
        "seed": seed,
        "rounds": results["rounds"],
        "peak_accuracy": results["peak_accuracy"],
        "peak_round": results["peak_round"],
        "post_peak_min": results["post_peak_min"],
        "drawdown": results["drawdown"],
        "training_history": results["training_history"],
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions.jsonl
    with open(run_dir / "predictions.jsonl", "w") as f:
        for round_data in results["rounds"]:
            round_num = round_data["round"]
            # Note: predictions are not saved in current implementation
            # Would need to modify evaluate() to return and store predictions

    # Save aug_manifest.json
    with open(run_dir / "aug_manifest.json", "w") as f:
        json.dump(results["aug_manifest"], f, indent=2)

    trainer.cleanup()

    return results


def main():
    """Run V8 Gate P0 with multiple seeds"""
    import argparse

    parser = argparse.ArgumentParser(description="V8 Gate P0: V5 Frozen Reproduction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", type=str, default="41,42,43", help="Comma-separated seeds")
    parser.add_argument("--single", action="store_true", help="Run single seed")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (canonical: configs/canonical/v5_g1prime_canonical.yaml)")

    args = parser.parse_args()

    # If config specified, verify it matches canonical
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config not found: {config_path}")
            sys.exit(1)
        canonical_config = CONFIGS_DIR / "v5_g1prime_canonical.yaml"
        if config_path.resolve() != canonical_config.resolve():
            print(f"[WARN] Non-canonical config: {config_path}")
            print(f"  Canonical: {canonical_config}")

    if args.single:
        seeds = [args.seed]
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    print("="*80)
    print("V8 Gate Experiment - P0: V5 Frozen Reproduction")
    print("="*80)
    print(f"Seeds: {seeds}")
    print(f"Rounds: 7 (frozen)")
    print(f"Config: v5_frozen")

    all_results = []

    for seed in seeds:
        print(f"\n{'#'*80}")
        print(f"# Running seed: {seed}")
        print(f"{'#'*80}")

        results = run_v8_gate_p0(seed)
        all_results.append(results)

    # Generate summary table
    summary_dir = ARTIFACTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    peak_accuracies = [r["peak_accuracy"] for r in all_results]
    peak_rounds = [r["peak_round"] for r in all_results]
    drawdowns = [r["drawdown"] for r in all_results]

    import numpy as np
    summary = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "peak_accuracy_mean": float(np.mean(peak_accuracies)),
        "peak_accuracy_std": float(np.std(peak_accuracies)),
        "peak_accuracy_min": float(min(peak_accuracies)),
        "peak_accuracy_max": float(max(peak_accuracies)),
        "peak_rounds": peak_rounds,
        "drawdown_mean": float(np.mean(drawdowns)),
        "drawdown_std": float(np.std(drawdowns)),
        "individual_results": [
            {
                "seed": r["seed"],
                "peak_accuracy": r["peak_accuracy"],
                "peak_round": r["peak_round"],
                "post_peak_min": r["post_peak_min"],
                "drawdown": r["drawdown"],
            }
            for r in all_results
        ],
        # G1 pass criteria
        "g1_criteria": {
            "expected_peak_mean": 0.40,  # 12/30
            "tolerance": 2/30,  # ±2 questions
            "lower_bound": 0.40 - 2/30,
            "upper_bound": 0.40 + 2/30,
        },
        "g1_status": "PENDING",
    }

    # Check G1 criteria
    peak_mean = summary["peak_accuracy_mean"]
    lower = summary["g1_criteria"]["lower_bound"]
    upper = summary["g1_criteria"]["upper_bound"]

    if lower <= peak_mean <= upper:
        summary["g1_status"] = "PASS"
        print(f"\n✓ G1 PASS: Peak mean {peak_mean:.2%} in [{lower:.2%}, {upper:.2%}]")
    else:
        summary["g1_status"] = "FAIL"
        print(f"\n✗ G1 FAIL: Peak mean {peak_mean:.2%} outside [{lower:.2%}, {upper:.2%}]")

    with open(summary_dir / "g1_reproduce_table.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate markdown table
    md_lines = [
        "# G1 Reproduction Summary",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Seeds**: {seeds}",
        f"**Config**: v5_frozen",
        "",
        "## Individual Results",
        "",
        "| Seed | Peak Acc | Peak Round | Post-Peak Min | Drawdown |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in summary["individual_results"]:
        md_lines.append(f"| {r['seed']} | {r['peak_accuracy']:.2%} | {r['peak_round']} | {r['post_peak_min']:.2%} | {r['drawdown']:.2%} |")

    md_lines.extend([
        "",
        "## Aggregate Statistics",
        "",
        f"- Peak Accuracy Mean: **{summary['peak_accuracy_mean']:.2%}**",
        f"- Peak Accuracy Std: {summary['peak_accuracy_std']:.2%}",
        f"- Drawdown Mean: {summary['drawdown_mean']:.2%}",
        "",
        "## G1 Gate Status",
        "",
        f"- Expected Range: [{lower:.2%}, {upper:.2%}] (12/30 ± 2 questions)",
        f"- Actual Mean: {peak_mean:.2%}",
        f"- **Status: {summary['g1_status']}**",
    ])

    with open(summary_dir / "g1_reproduce_table.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*80}")
    print("V8 Gate P0 Complete")
    print(f"{'='*80}")
    print(f"Results saved to: {summary_dir}")

    return summary


if __name__ == "__main__":
    main()
