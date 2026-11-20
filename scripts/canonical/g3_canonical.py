#!/usr/bin/env python3
"""
P1 Gate G3 Experiment: A_fail vs A_rand

Protocol: Compare failure-driven augmentation vs random augmentation
- A_fail: V5 frozen protocol (augment based on failed predictions)
- A_rand: Random augmentation (augment from random train samples)

Both use:
- Same v9_g1prime split (Train=46, Eval=97)
- Same V5 frozen config (wd=0.01, eta_min_ratio=0.1, 7 rounds)
- Same seeds (41-50, N=10)

Metrics: peak accuracy, drawdown, invalid rate, E/M/H breakdown
"""

import gc
import json
import re
import os
import sys
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================
# Paths
# ============================================================
EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts" / "p1"
CONFIGS_DIR = ARTIFACTS_DIR / "configs"

LOCAL_MODEL_CACHE = "Qwen/Qwen2.5-1.5B"


# ============================================================
# SMILES Extraction (V5 frozen)
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
    """Check if SMILES is syntactically valid"""
    if not smiles or len(smiles) < 2:
        return False
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
# Data Augmentor (V5 frozen)
# ============================================================
class DataAugmentor:
    """V5 frozen version - 5 variant types"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.variant_types = ['rephrase', 'add_reagent', 'add_solvent', 'add_condition', 'reverse']

    def generate_variant(self, sample: Dict, variant_type: Optional[str] = None) -> Dict:
        """Generate a single variant"""
        if variant_type is None:
            variant_type = self.rng.choice(self.variant_types)

        variant = sample.copy()
        input_text = sample['input_text']

        if variant_type == 'rephrase':
            templates = [
                "预测以下反应的主要产物：",
                "该化学反应会生成什么产物？",
                "给定反应物，推断主产物：",
            ]
            variant['input_text'] = self.rng.choice(templates) + input_text.split('：', 1)[-1] if '：' in input_text else input_text

        elif variant_type == 'add_reagent':
            reagents = ['H2O', 'NaOH', 'HCl', 'H2SO4', 'CH3OH', 'EtOH', 'THF', 'DMF', 'DMSO']
            variant['input_text'] = input_text.rstrip() + f" 试剂：{self.rng.choice(reagents)}"

        elif variant_type == 'add_solvent':
            solvents = ['水', '乙醇', '甲醇', 'THF', 'DMF', 'DMSO', '乙醚', '二氯甲烷']
            variant['input_text'] = input_text.rstrip() + f" 溶剂：{self.rng.choice(solvents)}"

        elif variant_type == 'add_condition':
            conditions = ['室温', '加热回流', '0°C', '高温高压', '惰性气氛', '避光']
            variant['input_text'] = input_text.rstrip() + f" 条件：{self.rng.choice(conditions)}"

        elif variant_type == 'reverse':
            variant['input_text'] = f"逆合成分析：{sample.get('target_text', '')} 可以由什么原料制备？"
            variant['target_text'] = sample.get('reactants_smiles', '')

        variant_id = hashlib.md5(f"{variant['input_text']}_{variant['target_text']}".encode()).hexdigest()[:8]
        variant['id'] = f"aug_{variant_id}"

        return variant


# ============================================================
# G3 Experiment Runner
# ============================================================
class G3ExperimentRunner:
    """G3: A_fail vs A_rand comparison"""

    def __init__(self, seed: int, mode: str = "A_fail"):
        """
        mode: "A_fail" (failure-driven) or "A_rand" (random)
        """
        self.seed = seed
        self.mode = mode
        self.set_seed(seed)

        # V5 FROZEN CONFIG
        self.weight_decay = 0.01
        self.eta_min_ratio = 0.1
        self.num_rounds = 7
        self.difficulty_weights = {'easy': 1.0, 'medium': 2.0, 'hard': 1.0}
        self.learning_rate = 5e-6
        self.num_epochs = 3
        self.batch_size = 2
        self.max_length = 256
        self.augment_per_round = 200

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data
        data_version = os.environ.get("V8_DATA_VERSION", "v9_g1prime")
        if data_version == "v9_g1prime":
            eval_path = DATA_DIR / "v9_g1prime" / "eval_reactions_g1prime.json"
            train_path = DATA_DIR / "v9_g1prime" / "train_reactions_g1prime.json"
        else:
            raise ValueError(f"Unknown data version: {data_version}")

        with open(eval_path) as f:
            self.eval_data = json.load(f)
        with open(train_path) as f:
            self.base_train_data = json.load(f)

        # V8 LEAKAGE GUARD
        train_ids = {item['id'] for item in self.base_train_data}
        eval_ids = {item['id'] for item in self.eval_data}
        assert len(train_ids & eval_ids) == 0, \
            f"LEAKAGE DETECTED: train ∩ eval = {train_ids & eval_ids}"

        print(f"\n[P1-G3] Mode: {self.mode}")
        print(f"[P1-G3] ✓ Leakage guard: train ∩ eval = ∅")
        print(f"[P1-G3] Train: {len(self.base_train_data)}, Eval: {len(self.eval_data)}")

        self.augmentor = DataAugmentor(seed=seed)
        self.model = None
        self.tokenizer = None

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(self):
        print("[P1-G3] Loading model from local cache...")
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_CACHE, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_CACHE,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better precision than float16
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.gradient_checkpointing_enable()
        print("[P1-G3] Model loaded successfully")

    def evaluate(self) -> Dict:
        """Evaluate model on eval set"""
        self.model.eval()
        results = {'correct': 0, 'total': 0, 'invalid': 0, 'difficulty': {}}

        with torch.no_grad():
            for i, item in enumerate(self.eval_data):
                try:
                    prompt = f"输入: {item['input_text']}\n输出:"
                    inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=64,
                        num_beams=1,
                        do_sample=False
                    )
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pred_smiles = extract_smiles_fixed(generated[len(prompt):])
                except torch.cuda.OutOfMemoryError:
                    # On OOM, clear cache and skip this sample
                    torch.cuda.empty_cache()
                    pred_smiles = ""

                gt = item.get('product_smiles', item.get('target_text', ''))
                is_correct = (pred_smiles == gt) if pred_smiles else False
                is_invalid = not pred_smiles or not is_valid_smiles(pred_smiles)

                diff = item.get('difficulty', 'medium')
                if diff not in results['difficulty']:
                    results['difficulty'][diff] = {'correct': 0, 'total': 0, 'invalid': 0}

                results['total'] += 1
                results['difficulty'][diff]['total'] += 1
                if is_correct:
                    results['correct'] += 1
                    results['difficulty'][diff]['correct'] += 1
                if is_invalid:
                    results['invalid'] += 1
                    results['difficulty'][diff]['invalid'] += 1

        results['accuracy'] = results['correct'] / max(results['total'], 1)
        results['invalid_rate'] = results['invalid'] / max(results['total'], 1)

        for diff in results['difficulty']:
            d = results['difficulty'][diff]
            d['accuracy'] = d['correct'] / max(d['total'], 1)

        # Clear GPU cache after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return results

    def get_failed_samples(self) -> List[Dict]:
        """Get samples where model fails - for A_fail mode"""
        self.model.eval()
        failed = []

        with torch.no_grad():
            for item in self.base_train_data:
                try:
                    prompt = f"输入: {item['input_text']}\n输出:"
                    inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=64,
                        num_beams=1,
                        do_sample=False
                    )
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pred_smiles = extract_smiles_fixed(generated[len(prompt):])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    pred_smiles = ""

                gt = item.get('product_smiles', item.get('target_text', ''))
                if pred_smiles != gt:
                    failed.append(item)

        # Clear GPU cache after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return failed

    def augment_random(self, train_data: List[Dict], target_size: int) -> List[Dict]:
        """A_rand: Random augmentation - sample random train examples"""
        augmented = []
        samples_per_base = max(1, target_size // len(train_data))

        for item in train_data:
            for _ in range(samples_per_base):
                variant_type = self.augmentor.rng.choice(self.augmentor.variant_types)
                variant = self.augmentor.generate_variant(item, variant_type)
                augmented.append(variant)

        # Trim to target size
        if len(augmented) > target_size:
            augmented = random.sample(augmented, target_size)

        return augmented

    def augment_failure_driven(self, train_data: List[Dict], failed_samples: List[Dict], target_size: int) -> List[Dict]:
        """A_fail: Failure-driven augmentation - more variants from failed samples"""
        augmented = []

        # 70% from failed samples, 30% from all samples
        num_from_failed = int(target_size * 0.7)
        num_from_all = target_size - num_from_failed

        # Augment from failed samples
        if failed_samples:
            samples_per_failed = max(1, num_from_failed // len(failed_samples))
            for item in failed_samples:
                for _ in range(samples_per_failed):
                    variant_type = self.augmentor.rng.choice(self.augmentor.variant_types)
                    variant = self.augmentor.generate_variant(item, variant_type)
                    augmented.append(variant)

        # Augment from all samples
        samples_per_base = max(1, num_from_all // len(train_data))
        for item in train_data:
            for _ in range(samples_per_base):
                variant_type = self.augmentor.rng.choice(self.augmentor.variant_types)
                variant = self.augmentor.generate_variant(item, variant_type)
                augmented.append(variant)

        # Trim to target size
        if len(augmented) > target_size:
            augmented = random.sample(augmented, target_size)

        return augmented

    def train_round(self, train_data: List[Dict], round_num: int):
        """Train for one round - mirrors train_v8_gate.py proven config"""
        dataset = WeightedReactionDataset(
            train_data, self.tokenizer,
            max_length=self.max_length,
            difficulty_weights=self.difficulty_weights
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed)
        )

        print(f"Training samples: {len(train_data)} (weighted: {len(dataset)})")
        print(f"Learning rate: {self.learning_rate}, Epochs: {self.num_epochs}")
        print(f"Weight decay: {self.weight_decay} (FROZEN)")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        eta_min = self.learning_rate * self.eta_min_ratio
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=len(dataloader) * self.num_epochs,
            eta_min=eta_min
        )
        print(f"CosineAnnealing eta_min: {eta_min:.2e}")

        self.model.train()
        for epoch in range(self.num_epochs):
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
            print(f"  Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

    def run(self) -> Dict:
        """Run full G3 experiment"""
        print(f"\n{'='*80}")
        print(f"P1-G3 Experiment: {self.mode}")
        print(f"Seed: {self.seed}")
        print(f"{'='*80}")

        self.load_model()

        round_metrics = []
        peak_accuracy = 0.0
        peak_round = 0
        current_train = self.base_train_data.copy()

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{self.num_rounds}")
            print(f"{'='*60}")

            # Evaluate
            eval_result = self.evaluate()
            accuracy = eval_result['accuracy']

            print(f"\n✓ Accuracy: {accuracy*100:.2f}% ({eval_result['correct']}/{eval_result['total']})")
            print(f"✓ Invalid SMILES rate: {eval_result['invalid_rate']*100:.2f}%")
            print(f"\nDifficulty breakdown:")
            for diff in ['easy', 'medium', 'hard']:
                if diff in eval_result['difficulty']:
                    d = eval_result['difficulty'][diff]
                    print(f"  {diff}: {d['correct']}/{d['total']} = {d['accuracy']*100:.2f}%")

            round_metric = {
                'round': round_num,
                'accuracy': accuracy,
                'invalid_rate': eval_result['invalid_rate'],
                'correct': eval_result['correct'],
                'total': eval_result['total'],
                'difficulty_breakdown': {
                    diff: {'correct': d['correct'], 'total': d['total'], 'accuracy': d['accuracy']}
                    for diff, d in eval_result['difficulty'].items()
                }
            }
            round_metrics.append(round_metric)

            if accuracy > peak_accuracy:
                peak_accuracy = accuracy
                peak_round = round_num

            # Checkpoint save after each round
            checkpoint = {
                'seed': self.seed,
                'mode': self.mode,
                'rounds': round_metrics,
                'peak_accuracy': peak_accuracy,
                'peak_round': peak_round,
            }
            checkpoint_path = Path(__file__).parent.parent / "artifacts" / "p1" / "g3" / self.mode.lower() / f"seed_{self.seed}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path / "checkpoint.json", 'w') as f:
                json.dump(checkpoint, f, indent=2)

            # Augmentation for next round (except last round)
            if round_num < self.num_rounds:
                print(f"\n{'='*60}")
                print(f"Round {round_num+1}: Augmentation")
                print(f"{'='*60}")

                if self.mode == "A_fail":
                    # Get failed samples for failure-driven augmentation
                    failed = self.get_failed_samples()
                    print(f"  Failed samples: {len(failed)}/{len(self.base_train_data)}")
                    augmented = self.augment_failure_driven(
                        self.base_train_data, failed, self.augment_per_round
                    )
                else:  # A_rand
                    # Random augmentation
                    augmented = self.augment_random(
                        self.base_train_data, self.augment_per_round
                    )

                current_train = self.base_train_data + augmented
                print(f"  Training data: {len(current_train)} samples")

                # Train
                print(f"\n{'='*60}")
                print(f"Round {round_num+1}: Training")
                print(f"{'='*60}")
                print(f"Training samples: {len(current_train)}")
                print(f"Learning rate: {self.learning_rate}")
                print(f"Weight decay: {self.weight_decay} (FROZEN)")

                self.train_round(current_train, round_num + 1)

        # Calculate drawdown
        post_peak_accs = [m['accuracy'] for m in round_metrics[peak_round:]]
        post_peak_min = min(post_peak_accs) if post_peak_accs else peak_accuracy
        drawdown = peak_accuracy - post_peak_min

        # Final invalid rate
        final_invalid = round_metrics[-1]['invalid_rate']

        print(f"\n{'='*60}")
        print(f"G3 {self.mode} Complete")
        print(f"{'='*60}")
        print(f"Peak accuracy: {peak_accuracy*100:.2f}% (Round {peak_round})")
        print(f"Drawdown: {drawdown*100:.2f}%")
        print(f"Final invalid rate: {final_invalid*100:.2f}%")

        return {
            'seed': self.seed,
            'mode': self.mode,
            'rounds': round_metrics,
            'peak_accuracy': peak_accuracy,
            'peak_round': peak_round,
            'drawdown': drawdown,
            'final_invalid_rate': final_invalid
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="P1-G3 Experiment: A_fail vs A_rand")
    parser.add_argument("--seeds", type=str, default="41,42,43,44,45,46,47,48,49,50",
                        help="Comma-separated list of seeds")
    parser.add_argument("--mode", type=str, default="A_fail",
                        choices=["A_fail", "A_rand"],
                        help="Experiment mode")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    mode = args.mode

    # Create output directory
    output_dir = ARTIFACTS_DIR / "g3" / mode.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("P1 Gate G3 Experiment")
    print("="*80)
    print(f"Seeds: {seeds}")
    print(f"Mode: {mode}")
    print(f"Rounds: 7 (frozen)")
    print(f"Config: V5 frozen")

    all_results = []

    for seed in seeds:
        print(f"\n{'#'*80}")
        print(f"# Running seed: {seed}")
        print(f"{'#'*80}")

        runner = G3ExperimentRunner(seed=seed, mode=mode)
        result = runner.run()

        # Save individual result
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with open(seed_dir / "metrics.json", 'w') as f:
            json.dump(result, f, indent=2)

        all_results.append(result)

        # Free memory
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    peaks = [r['peak_accuracy'] * 100 for r in all_results]
    mean_peak = sum(peaks) / len(peaks)
    std_peak = (sum((p - mean_peak)**2 for p in peaks) / len(peaks)) ** 0.5
    se_peak = std_peak / (len(peaks) ** 0.5)

    summary = {
        'experiment': 'G3',
        'mode': mode,
        'seeds': seeds,
        'num_seeds': len(seeds),
        'peak_accuracy_mean': mean_peak,
        'peak_accuracy_std': std_peak,
        'peak_accuracy_se': se_peak,
        'individual_results': [
            {'seed': r['seed'], 'peak_accuracy': r['peak_accuracy'], 'peak_round': r['peak_round']}
            for r in all_results
        ]
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"G3 {mode} Summary")
    print(f"{'='*80}")
    print(f"Peak accuracy mean: {mean_peak:.2f}% ± {std_peak:.2f}%")
    print(f"SE: {se_peak:.2f}%")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
