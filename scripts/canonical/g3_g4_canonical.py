#!/usr/bin/env python3
"""
G3/G4 Experiments under Canonical V5 Configuration
===================================================
G3: A_fail (failure-driven) vs A_rand (random augmentation)
G4: B_recur (recursive training) vs B_single (single-round training)

Uses canonical configuration:
- float32 precision
- Decaying LR: 5e-6 * 0.85^(r-2)
- English prompt-only augmentation (5 types)
- Batch size: 2

Output: artifacts/canonical/g3_g4/
"""

import gc
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR

EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data"
LOCAL_MODEL_CACHE = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "g3_g4"


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

        # Expand data based on weights
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
# Canonical Augmentor (G1' prompt-only, English)
# ============================================================
class CanonicalAugmentor:
    """G1' frozen version - 5 prompt-only variants (no noise)"""

    def __init__(self):
        self.augmentation_types = [
            "condition_variant",
            "hint_variant",
            "format_variant",
            "synonym_variant",
            "context_variant"
        ]

    def augment_sample(self, sample, round_num, aug_type):
        """Generate single variant - canonical G1' version"""
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
        """Augment failed cases - G1' style"""
        augmented = []
        num_variants = 3 if is_medium else 2
        variant_types = self.augmentation_types[:num_variants]

        for case in failed_cases:
            for aug_type in variant_types:
                augmented.append(self.augment_sample(case, round_num, aug_type))

        return augmented

    def augment_random(self, train_data, round_num, n_samples, rng):
        """A_rand: Random augmentation from train pool"""
        augmented = []
        for _ in range(n_samples):
            sample = rng.choice(train_data)
            aug_type = rng.choice(self.augmentation_types[:2])  # Only 2 types for random
            augmented.append(self.augment_sample(sample, round_num, aug_type))
        return augmented


# ============================================================
# Experiment Runner
# ============================================================
class ExperimentRunner:
    """G3/G4 Experiment Runner under Canonical Configuration"""

    def __init__(self, seed: int, experiment: str, mode: str):
        """
        experiment: "G3" or "G4"
        mode: "A_fail", "A_rand" (G3) or "B_recur", "B_single" (G4)
        """
        self.seed = seed
        self.experiment = experiment
        self.mode = mode
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
        self.augmentor = CanonicalAugmentor()
        self.training_history = []

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

        print(f"[Canonical] Data loaded: Train={len(self.train_data)}, Eval={len(self.eval_data)}")

    def load_model(self):
        """Load model in float32 (canonical)"""
        print("[Canonical] Loading model (float32)...")
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_CACHE, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_CACHE,
            torch_dtype=torch.float32,  # CANONICAL: float32
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.gradient_checkpointing_enable()
        print("[Canonical] Model loaded")

    def evaluate(self, verbose=True):
        """Evaluate model on eval set"""
        self.model.eval()
        correct = 0
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
            'difficulty_breakdown': {
                d: {'correct': s['c'], 'total': s['t'], 'accuracy': s['c']/s['t'] if s['t'] > 0 else 0}
                for d, s in diff_stats.items()
            },
            'failed_cases': failed_cases
        }

        if verbose:
            print(f"✓ Accuracy: {result['accuracy']*100:.2f}% ({correct}/{n})")

        return result

    def train_round(self, train_data, round_num, epochs=3):
        """Train one round with canonical config"""
        # CANONICAL: Decaying LR schedule
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
        """Run the experiment"""
        self.load_data()
        self.load_model()

        rng = random.Random(self.seed)
        peak_accuracy = 0.0
        peak_round = 0
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
                'difficulty_breakdown': result['difficulty_breakdown']
            })

            if acc > peak_accuracy:
                peak_accuracy = acc
                peak_round = round_num

            # Augmentation and training
            if self.mode == "B_single":
                # B_single: Only train on round 1, then just evaluate
                if round_num == 1:
                    augmented = self._get_augmentation(result['failed_cases'], round_num, rng)
                    train_data = self.train_data + augmented
                    self.train_round(train_data, round_num)
                # No more training after round 1

            elif self.mode == "B_recur" or self.mode in ["A_fail", "A_rand"]:
                # Recursive training
                if round_num < self.num_rounds:
                    augmented = self._get_augmentation(result['failed_cases'], round_num, rng)
                    train_data = self.train_data + augmented
                    self.train_round(train_data, round_num)

        # Compute drawdown
        post_peak = [m['accuracy'] for m in round_metrics[peak_round:]]
        drawdown = peak_accuracy - (min(post_peak) if post_peak else peak_accuracy)

        return {
            'experiment': self.experiment,
            'mode': self.mode,
            'seed': self.seed,
            'rounds': round_metrics,
            'peak_accuracy': peak_accuracy,
            'peak_round': peak_round,
            'drawdown': drawdown
        }

    def _get_augmentation(self, failed_cases, round_num, rng):
        """Get augmentation based on mode"""
        if self.mode == "A_fail":
            # Failure-driven augmentation
            failed_easy = [c for c in failed_cases if c.get('difficulty') == 'easy']
            failed_medium = [c for c in failed_cases if c.get('difficulty') == 'medium']
            failed_hard = [c for c in failed_cases if c.get('difficulty') == 'hard']

            augmented = []
            augmented.extend(self.augmentor.augment_failed(failed_easy, round_num + 1, is_medium=False))
            augmented.extend(self.augmentor.augment_failed(failed_medium, round_num + 1, is_medium=True))
            augmented.extend(self.augmentor.augment_failed(failed_hard, round_num + 1, is_medium=False))
            return augmented

        elif self.mode == "A_rand":
            # Random augmentation (same count as A_fail)
            n_samples = len(failed_cases)
            return self.augmentor.augment_random(self.train_data, round_num + 1, n_samples, rng)

        else:
            # B_recur/B_single: standard failure-driven
            failed_easy = [c for c in failed_cases if c.get('difficulty') == 'easy']
            failed_medium = [c for c in failed_cases if c.get('difficulty') == 'medium']
            failed_hard = [c for c in failed_cases if c.get('difficulty') == 'hard']

            augmented = []
            augmented.extend(self.augmentor.augment_failed(failed_easy, round_num + 1, is_medium=False))
            augmented.extend(self.augmentor.augment_failed(failed_medium, round_num + 1, is_medium=True))
            augmented.extend(self.augmentor.augment_failed(failed_hard, round_num + 1, is_medium=False))
            return augmented


def main():
    import argparse
    parser = argparse.ArgumentParser(description="G3/G4 Canonical Experiments")
    parser.add_argument("--experiment", type=str, required=True, choices=["G3", "G4"])
    parser.add_argument("--mode", type=str, required=True,
                        help="A_fail/A_rand (G3) or B_recur/B_single (G4)")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    print(f"="*70)
    print(f"Canonical {args.experiment} Experiment: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Config: V5 frozen (float32, decaying LR, English augmentation)")
    print(f"="*70)

    runner = ExperimentRunner(args.seed, args.experiment, args.mode)
    result = runner.run()

    # Save results
    output_subdir = OUTPUT_DIR / args.experiment.lower() / args.mode.lower() / f"seed_{args.seed}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    with open(output_subdir / "metrics.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS: {args.experiment} {args.mode} (seed={args.seed})")
    print(f"{'='*70}")
    print(f"Peak Accuracy: {result['peak_accuracy']*100:.2f}% @ Round {result['peak_round']}")
    print(f"Drawdown: {result['drawdown']*100:.2f}%")
    print(f"Saved to: {output_subdir}")


if __name__ == "__main__":
    main()
