#!/usr/bin/env python3
"""
M1 Leave-One-Out Experiment
===========================
Remove one prompt variant type at a time from A_fail augmentation.
Answers: "Which variant types are necessary for A_fail to work?"

Protocol: V5 Canonical + A_fail (failure-driven)
Factor: Drop one of {condition, hint, format, synonym, context}
Control: Full-5 (complete A_fail, reusable from Phase3 G3)

Mandatory metrics: peak, peak_round, final, drawdown, E/M/H, invalid_rate
One process per (drop_type, seed) - NO multi-config in same process
"""

import gc
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================
# Paths & Constants
# ============================================================
EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data"
LOCAL_MODEL_CACHE = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "mechanism" / "M1_loo"
CONFIGS_DIR = EXPERIMENT_DIR / "configs"

# V5 Canonical Recipe Hash (FROZEN)
EXPECTED_RECIPE_HASH = "2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3"

# All 5 variant types
ALL_VARIANT_TYPES = [
    "condition_variant",
    "hint_variant",
    "format_variant",
    "synonym_variant",
    "context_variant"
]


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
                print(f"[M1] ✓ Recipe hash verified: {actual_hash}")
                return True
            else:
                print(f"[M1] ✗ Recipe hash MISMATCH!")
                print(f"  Expected: {EXPECTED_RECIPE_HASH}")
                print(f"  Actual:   {actual_hash}")
                return False
    print(f"[WARN] recipe_sha256 not found in {hash_file}")
    return False


# ============================================================
# SMILES Extraction (V5 frozen)
# ============================================================
def extract_smiles_fixed(text: str) -> str:
    """Extract SMILES from model output - V5 frozen version"""
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
# Dataset (V5 frozen)
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
            weight = self.difficulty_weights.get(sample.get('difficulty', 'medium'), 1.0)
            for _ in range(int(weight)):
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
# LOO Augmentor (with drop_type support)
# ============================================================
class LOOAugmentor:
    """Leave-one-out augmentor - can drop one variant type"""

    def __init__(self, drop_type: Optional[str] = None):
        """
        drop_type: which variant to exclude (e.g., "synonym_variant")
                   None = use all 5 types (Full-5 control)
        """
        self.all_types = ALL_VARIANT_TYPES.copy()

        if drop_type and drop_type in self.all_types:
            self.active_types = [t for t in self.all_types if t != drop_type]
            self.drop_type = drop_type
            print(f"[M1] DROPPING variant type: {drop_type}")
            print(f"[M1] Active types: {self.active_types}")
        else:
            self.active_types = self.all_types
            self.drop_type = None
            print(f"[M1] Using ALL variant types (Full-5 control)")

    def augment_sample(self, sample, round_num, aug_type):
        """Generate single variant - V5 frozen"""
        variant = sample.copy()
        variant["id"] = f"{sample['id']}_aug_r{round_num}_{aug_type}"

        reactants = ' + '.join(sample['reactants_smiles']) if isinstance(sample['reactants_smiles'], list) else str(sample['reactants_smiles'])

        if aug_type == "condition_variant":
            condition_variants = ["standard conditions", "optimized conditions", "typical procedure", "standard procedure"]
            variant["conditions"] = condition_variants[round_num % len(condition_variants)]
            variant["input_text"] = (
                f"Predict the major organic product of the following reaction.\n"
                f"Reaction type: {sample['reaction_type']}\n"
                f"Reactants: {reactants}\n"
                f"Conditions: {variant['conditions']}\n"
                f"Provide the product SMILES."
            )
        elif aug_type == "hint_variant":
            variant["input_text"] = (
                f"This is a {sample['reaction_type']} reaction.\n"
                f"Reactants: {reactants}\n"
                f"Conditions: {sample['conditions']}\n"
                f"Provide the product SMILES."
            )
        elif aug_type == "format_variant":
            variant["input_text"] = (
                f"Task: Predict product of {sample['reaction_type']} reaction\n"
                f"Input: {reactants}\n"
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
                f"Reactants: {reactants}\n"
                f"Conditions: {sample['conditions']}\n"
                f"What is the product SMILES?"
            )
        elif aug_type == "context_variant":
            variant["input_text"] = (
                f"Given the following chemical reaction:\n"
                f"Type: {sample['reaction_type']}\n"
                f"Starting materials: {reactants}\n"
                f"Reaction conditions: {sample['conditions']}\n"
                f"Determine the major organic product and provide its SMILES."
            )

        return variant

    def augment_failed(self, failed_cases, round_num, is_medium=False):
        """Augment failed cases using active (non-dropped) variant types"""
        augmented = []
        num_variants = 3 if is_medium else 2
        # Use first num_variants from ACTIVE types (not all types)
        variant_types = self.active_types[:num_variants]

        for case in failed_cases:
            for aug_type in variant_types:
                augmented.append(self.augment_sample(case, round_num, aug_type))

        return augmented


# ============================================================
# M1 Experiment Runner
# ============================================================
class M1Runner:
    """M1 Leave-One-Out Experiment Runner"""

    def __init__(self, seed: int, drop_type: Optional[str] = None, tag: str = ""):
        """
        seed: random seed
        drop_type: which variant to exclude (None = Full-5)
        tag: experiment tag for output directory
        """
        self.seed = seed
        self.drop_type = drop_type
        self.tag = tag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # V5 FROZEN CONFIG
        self.weight_decay = 0.01
        self.eta_min_ratio = 0.1
        self.num_rounds = 7
        self.difficulty_weights = {'easy': 1.0, 'medium': 2.0, 'hard': 1.0}
        self.batch_size = 2
        self.max_length = 256
        self.epochs_per_round = 3

        self.set_seed(seed)
        self.augmentor = LOOAugmentor(drop_type=drop_type)
        self.training_history = []
        self.aug_manifest = []

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_data(self):
        """Load v9_g1prime data"""
        data_version = os.environ.get("V8_DATA_VERSION", "v9_g1prime")
        eval_path = DATA_DIR / data_version / "eval_reactions_g1prime.json"
        train_path = DATA_DIR / data_version / "train_reactions_g1prime.json"

        with open(eval_path) as f:
            self.eval_data = json.load(f)
        with open(train_path) as f:
            self.train_data = json.load(f)

        # Leakage guard
        train_ids = {x['id'] for x in self.train_data}
        eval_ids = {x['id'] for x in self.eval_data}
        assert len(train_ids & eval_ids) == 0, "LEAKAGE: train ∩ eval ≠ ∅"

        print(f"[M1] Data loaded: Train={len(self.train_data)}, Eval={len(self.eval_data)}")

    def load_model(self):
        """Load model in float32 (canonical)"""
        print("[M1] Loading model (float32)...")
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_CACHE, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_CACHE,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.gradient_checkpointing_enable()
        print("[M1] Model loaded")

    def evaluate(self, verbose=True):
        """Evaluate model on eval set"""
        self.model.eval()
        correct = 0
        invalid_count = 0
        failed_cases = []
        diff_stats = {'easy': {'c': 0, 't': 0}, 'medium': {'c': 0, 't': 0}, 'hard': {'c': 0, 't': 0}}

        with torch.no_grad():
            for item in self.eval_data:
                prompt = f"输入: {item['input_text']}\n输出:"
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                try:
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=100, num_beams=1, do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pred = extract_smiles_fixed(generated[len(prompt):])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    pred = ""

                gt = item.get('product_smiles', item.get('target_text', ''))
                is_correct = (pred == gt) if pred else False

                # Track invalid SMILES
                if not is_valid_smiles(pred):
                    invalid_count += 1

                if is_correct:
                    correct += 1
                else:
                    failed_cases.append(item)

                diff = item.get('difficulty', 'medium')
                diff_stats[diff]['t'] += 1
                if is_correct:
                    diff_stats[diff]['c'] += 1

        torch.cuda.empty_cache()
        gc.collect()

        n = len(self.eval_data)
        result = {
            'accuracy': correct / n,
            'correct': correct,
            'total': n,
            'invalid_rate': invalid_count / n,
            'invalid_count': invalid_count,
            'difficulty_breakdown': {
                d: {'correct': s['c'], 'total': s['t'], 'accuracy': s['c']/s['t'] if s['t'] > 0 else 0}
                for d, s in diff_stats.items()
            },
            'failed_cases': failed_cases
        }

        if verbose:
            print(f"✓ Accuracy: {result['accuracy']*100:.2f}% ({correct}/{n})")
            print(f"✓ Invalid rate: {result['invalid_rate']*100:.2f}% ({invalid_count}/{n})")

        return result

    def train_round(self, train_data, round_num, epochs=3):
        """Train one round with canonical config"""
        # CANONICAL: Decaying LR schedule (0.85^(r-2))
        lr = 5e-6 * (0.85 ** (round_num - 2))

        dataset = WeightedReactionDataset(
            train_data, self.tokenizer, max_length=self.max_length,
            difficulty_weights=self.difficulty_weights
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(self.seed)
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        eta_min = lr * self.eta_min_ratio
        scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs, eta_min=eta_min)

        print(f"  LR: {lr:.2e}, Samples: {len(train_data)}")

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
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    def run(self):
        """Run the M1 experiment"""
        self.load_data()
        self.load_model()

        peak_accuracy = 0.0
        peak_round = 0
        peak_emh = None
        round_metrics = []

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{self.num_rounds}")
            print(f"{'='*60}")

            # Evaluate
            result = self.evaluate()
            acc = result['accuracy']
            print(f"\n  Accuracy: {acc*100:.2f}% ({result['correct']}/{result['total']})")

            for diff in ['easy', 'medium', 'hard']:
                d = result['difficulty_breakdown'].get(diff, {})
                print(f"    {diff}: {d.get('correct',0)}/{d.get('total',0)} = {d.get('accuracy',0)*100:.2f}%")

            round_metrics.append({
                'round': round_num,
                'accuracy': acc,
                'correct': result['correct'],
                'total': result['total'],
                'invalid_rate': result['invalid_rate'],
                'difficulty_breakdown': result['difficulty_breakdown']
            })

            if acc > peak_accuracy:
                peak_accuracy = acc
                peak_round = round_num
                peak_emh = {
                    'easy': result['difficulty_breakdown']['easy']['accuracy'],
                    'medium': result['difficulty_breakdown']['medium']['accuracy'],
                    'hard': result['difficulty_breakdown']['hard']['accuracy'],
                }

            # Augmentation and training (canonical A_fail style)
            if round_num < self.num_rounds:
                failed_easy = [c for c in result['failed_cases'] if c.get('difficulty') == 'easy']
                failed_medium = [c for c in result['failed_cases'] if c.get('difficulty') == 'medium']
                failed_hard = [c for c in result['failed_cases'] if c.get('difficulty') == 'hard']

                augmented = []
                aug_round = round_num  # Canonical: NOT round+1
                augmented.extend(self.augmentor.augment_failed(failed_easy, aug_round, is_medium=False))
                augmented.extend(self.augmentor.augment_failed(failed_medium, aug_round, is_medium=True))
                augmented.extend(self.augmentor.augment_failed(failed_hard, aug_round, is_medium=False))

                # Track aug count
                self.aug_manifest.append({
                    'round': round_num,
                    'aug_count': len(augmented),
                    'drop_type': self.drop_type,
                    'active_types': self.augmentor.active_types
                })

                train_data = self.train_data + augmented
                self.train_round(train_data, round_num)

        # Final round metrics
        final_accuracy = round_metrics[-1]['accuracy'] if round_metrics else 0.0
        final_emh = {
            'easy': round_metrics[-1]['difficulty_breakdown']['easy']['accuracy'] if round_metrics else 0,
            'medium': round_metrics[-1]['difficulty_breakdown']['medium']['accuracy'] if round_metrics else 0,
            'hard': round_metrics[-1]['difficulty_breakdown']['hard']['accuracy'] if round_metrics else 0,
        }

        # Final invalid rate
        final_invalid = round_metrics[-1]['invalid_rate'] if round_metrics else 0.0

        # Compute drawdown
        post_peak = [m['accuracy'] for m in round_metrics[peak_round:]]
        drawdown = peak_accuracy - (min(post_peak) if post_peak else peak_accuracy)

        return {
            'experiment': 'M1_loo',
            'tag': self.tag,
            'drop_type': self.drop_type,
            'active_types': self.augmentor.active_types,
            'seed': self.seed,
            'recipe_hash': EXPECTED_RECIPE_HASH,
            'rounds': round_metrics,
            'peak_accuracy': peak_accuracy,
            'peak_round': peak_round,
            'peak_emh': peak_emh,
            'final_accuracy': final_accuracy,
            'final_emh': final_emh,
            'final_invalid_rate': final_invalid,
            'drawdown': drawdown,
            'aug_manifest': self.aug_manifest,
            'timestamp': datetime.now().isoformat(),
        }

    def cleanup(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M1 Leave-One-Out Experiment")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--aug-drop", type=str, default=None,
                        choices=["condition_variant", "hint_variant", "format_variant",
                                 "synonym_variant", "context_variant", "none"],
                        help="Variant type to drop (none = Full-5 control)")
    parser.add_argument("--tag", type=str, default="", help="Experiment tag")
    args = parser.parse_args()

    # Handle "none" as no drop
    drop_type = None if args.aug_drop == "none" else args.aug_drop
    tag = args.tag or f"loo_{drop_type or 'full5'}_seed{args.seed}"

    print("="*70)
    print(f"M1 Leave-One-Out Experiment")
    print(f"Seed: {args.seed}")
    print(f"Drop: {drop_type or 'None (Full-5)'}")
    print(f"Tag: {tag}")
    print(f"Config: V5 frozen (float32, decaying LR, English augmentation)")
    print("="*70)

    # Verify recipe hash
    hash_ok = verify_recipe_hash()
    if not hash_ok:
        print("[M1] ERROR: Recipe hash mismatch. Aborting.")
        sys.exit(1)

    # Run experiment
    runner = M1Runner(seed=args.seed, drop_type=drop_type, tag=tag)
    result = runner.run()

    # Save results
    output_subdir = OUTPUT_DIR / tag
    output_subdir.mkdir(parents=True, exist_ok=True)

    with open(output_subdir / "metrics.json", 'w') as f:
        json.dump(result, f, indent=2)

    # Save aug manifest separately
    with open(output_subdir / "aug_manifest.json", 'w') as f:
        json.dump(result['aug_manifest'], f, indent=2)

    print(f"\n{'='*70}")
    print(f"M1 RESULTS: {tag}")
    print(f"{'='*70}")
    print(f"Peak Accuracy: {result['peak_accuracy']*100:.2f}% @ Round {result['peak_round']}")
    print(f"Final Accuracy: {result['final_accuracy']*100:.2f}%")
    print(f"Drawdown: {result['drawdown']*100:.2f}%")
    if result['peak_emh']:
        print(f"E/M/H @ peak: {result['peak_emh']['easy']*100:.0f}% / {result['peak_emh']['medium']*100:.0f}% / {result['peak_emh']['hard']*100:.0f}%")
    print(f"Saved to: {output_subdir}")

    runner.cleanup()
    return result


if __name__ == "__main__":
    main()
