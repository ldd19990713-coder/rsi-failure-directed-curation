#!/usr/bin/env python3
"""
M3 Rollback Experiment: Test if end-of-round rollback reduces drawdown
========================================================================

Purpose: A_fail has significant drawdown (Phase3 mean DD≈12.6%).
         M3 tests whether rolling back to best checkpoint when eval degrades
         can reduce drawdown without harming peak accuracy.

Arms:
- A_rollback: Restore best checkpoint if eval < best after each round
- A_no_rollback: No rollback (control, can reuse M2 K=1.0× data)

Protocol:
- Freeze A_fail + K=1.0× (M2 operating point)
- Seeds: 41-45 (N=5)
- Success: drawdown ↓, final ↑; peak drop >2pp = failure

Metrics:
- Primary: drawdown, final accuracy
- Secondary: peak accuracy, rollback count, E/M/H breakdown
"""

import copy
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from datetime import datetime

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
LOCAL_MODEL_CACHE = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "mechanism" / "M3_rollback"
CONFIGS_DIR = EXPERIMENT_DIR / "configs"

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
                print(f"[M3] ✓ Recipe hash verified: {actual_hash}")
                return True
            else:
                print(f"[M3] ✗ Recipe hash MISMATCH!")
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


# ============================================================
# Dataset
# ============================================================
class ReactionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
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
# Augmentor (Phase3-consistent, effective 3-type slice)
# ============================================================
class M3Augmentor:
    """M3 augmentor - same as M2 (effective 3-type slice)"""

    def __init__(self):
        # Effective 3-type slice (what canonical actually uses)
        self.augmentation_types = [
            "condition_variant",
            "hint_variant",
            "format_variant"
        ]

    def augment_sample(self, sample, round_num, aug_type):
        """Generate single variant"""
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

        return variant

    def augment_failed(self, failed_cases, round_num, is_medium=False):
        """Augment failed cases - effective 3-type slice"""
        augmented = []
        num_variants = 3 if is_medium else 2
        variant_types = self.augmentation_types[:num_variants]

        for case in failed_cases:
            for aug_type in variant_types:
                augmented.append(self.augment_sample(case, round_num, aug_type))

        return augmented


# ============================================================
# M3 Experiment Runner
# ============================================================
class M3Runner:
    """M3 Rollback Experiment Runner"""

    def __init__(self, seed: int, enable_rollback: bool):
        """
        seed: random seed
        enable_rollback: True = A_rollback, False = A_no_rollback (control)
        """
        self.seed = seed
        self.enable_rollback = enable_rollback
        self.arm = "A_rollback" if enable_rollback else "A_no_rollback"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # V5 FROZEN CONFIG
        self.weight_decay = 0.01
        self.eta_min_ratio = 0.1
        self.num_rounds = 7
        self.batch_size = 2
        self.max_length = 256
        self.epochs_per_round = 3

        self.set_seed(seed)
        self.augmentor = M3Augmentor()
        self.training_history = []
        self.aug_manifest = []
        self.rollback_events = []

        # Checkpoint management
        self.best_state_dict = None
        self.best_accuracy = 0.0
        self.best_round = 0

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

        print(f"[M3] Data loaded: Train={len(self.train_data)}, Eval={len(self.eval_data)}")

    def load_model(self):
        """Load model in float32 (canonical)"""
        print("[M3] Loading model (float32)...")
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
        print("[M3] Model loaded")

    def save_checkpoint(self, round_num):
        """Save current model state"""
        # Deep copy state dict to CPU to save GPU memory
        state = copy.deepcopy({k: v.cpu().clone() for k, v in self.model.state_dict().items()})
        return state

    def load_checkpoint(self, state):
        """Restore model state"""
        self.model.load_state_dict(copy.deepcopy(state))
        self.model.to(self.device)

    def evaluate(self):
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

        print(f"✓ Accuracy: {result['accuracy']*100:.2f}% ({correct}/{n})")

        return result

    def train_round(self, train_data, round_num, epochs=3):
        """Train one round with canonical config"""
        # CANONICAL: Decaying LR schedule (0.85^(r-2))
        lr = 5e-6 * (0.85 ** (round_num - 2))

        dataset = ReactionDataset(train_data, self.tokenizer, max_length=self.max_length)
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

    def _get_augmentation(self, failed_cases, round_num, rng):
        """Get A_fail augmentation (K=1.0×, Phase3-consistent)"""
        # Phase3 G3 uses round_num + 1 (D3 drift, but must match for comparability)
        aug_round = round_num + 1

        failed_easy = [c for c in failed_cases if c.get('difficulty') == 'easy']
        failed_medium = [c for c in failed_cases if c.get('difficulty') == 'medium']
        failed_hard = [c for c in failed_cases if c.get('difficulty') == 'hard']

        augmented = []
        augmented.extend(self.augmentor.augment_failed(failed_easy, aug_round, is_medium=False))
        augmented.extend(self.augmentor.augment_failed(failed_medium, aug_round, is_medium=True))
        augmented.extend(self.augmentor.augment_failed(failed_hard, aug_round, is_medium=False))

        return augmented

    def run(self):
        """Run the M3 experiment"""
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

            # Rollback logic (only if enabled)
            rollback_this_round = False
            if self.enable_rollback:
                if acc > self.best_accuracy:
                    # New best - save checkpoint
                    self.best_accuracy = acc
                    self.best_round = round_num
                    self.best_state_dict = self.save_checkpoint(round_num)
                    print(f"  [CHECKPOINT] New best: {acc*100:.2f}% @ R{round_num}")
                elif acc < self.best_accuracy and self.best_state_dict is not None:
                    # Worse than best - rollback
                    print(f"  [ROLLBACK] {acc*100:.2f}% < best {self.best_accuracy*100:.2f}% → restoring R{self.best_round}")
                    self.load_checkpoint(self.best_state_dict)
                    rollback_this_round = True
                    self.rollback_events.append({
                        'round': round_num,
                        'before_acc': acc,
                        'best_acc': self.best_accuracy,
                        'best_round': self.best_round
                    })
                    # Re-evaluate after rollback to confirm
                    result_after = self.evaluate()
                    acc = result_after['accuracy']
                    result = result_after
                    print(f"  [ROLLBACK] Confirmed: {acc*100:.2f}%")

            round_metrics.append({
                'round': round_num,
                'accuracy': acc,
                'correct': result['correct'],
                'total': result['total'],
                'difficulty_breakdown': result['difficulty_breakdown'],
                'rollback': rollback_this_round
            })

            if acc > peak_accuracy:
                peak_accuracy = acc
                peak_round = round_num

            # Augmentation and training
            if round_num < self.num_rounds:
                augmented = self._get_augmentation(result['failed_cases'], round_num, rng)

                # Track aug count
                self.aug_manifest.append({
                    'round': round_num,
                    'aug_count': len(augmented),
                    'arm': self.arm,
                    'rollback': self.enable_rollback
                })

                train_data = self.train_data + augmented
                self.train_round(train_data, round_num)

        # Compute drawdown
        post_peak = [m['accuracy'] for m in round_metrics[peak_round:]]
        drawdown = peak_accuracy - (min(post_peak) if post_peak else peak_accuracy)

        return {
            'experiment': 'M3_rollback',
            'arm': self.arm,
            'enable_rollback': self.enable_rollback,
            'seed': self.seed,
            'recipe_hash': EXPECTED_RECIPE_HASH,
            'rounds': round_metrics,
            'peak_accuracy': peak_accuracy,
            'peak_round': peak_round,
            'final_accuracy': round_metrics[-1]['accuracy'],
            'drawdown': drawdown,
            'rollback_events': self.rollback_events,
            'rollback_count': len(self.rollback_events),
            'aug_manifest': self.aug_manifest,
            'timestamp': datetime.now().isoformat(),
        }

    def cleanup(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'best_state_dict'):
            del self.best_state_dict
        gc.collect()
        torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M3 Rollback Experiment")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--rollback", action="store_true",
                        help="Enable rollback (A_rollback); omit for A_no_rollback control")
    args = parser.parse_args()

    arm = "A_rollback" if args.rollback else "A_no_rollback"
    tag = f"M3_{arm}_seed{args.seed}"

    print("="*70)
    print(f"M3 Rollback Experiment")
    print(f"Seed: {args.seed}")
    print(f"Arm: {arm}")
    print(f"Rollback: {args.rollback}")
    print(f"Tag: {tag}")
    print(f"Config: V5 frozen (float32, decaying LR, A_fail K=1.0×)")
    print("="*70)

    # Verify recipe hash
    hash_ok = verify_recipe_hash()
    if not hash_ok:
        print("[M3] ERROR: Recipe hash mismatch. Aborting.")
        sys.exit(1)

    # Run experiment
    runner = M3Runner(seed=args.seed, enable_rollback=args.rollback)
    result = runner.run()

    # Save results
    output_subdir = OUTPUT_DIR / tag
    output_subdir.mkdir(parents=True, exist_ok=True)

    with open(output_subdir / "metrics.json", 'w') as f:
        json.dump(result, f, indent=2)

    # Save aug manifest
    with open(output_subdir / "aug_manifest.json", 'w') as f:
        json.dump(result['aug_manifest'], f, indent=2)

    # Save rollback events
    if args.rollback and result['rollback_events']:
        with open(output_subdir / "rollback_events.json", 'w') as f:
            json.dump(result['rollback_events'], f, indent=2)

    print(f"\n{'='*70}")
    print(f"M3 RESULTS: {tag}")
    print(f"{'='*70}")
    print(f"Peak Accuracy: {result['peak_accuracy']*100:.2f}% @ Round {result['peak_round']}")
    print(f"Final Accuracy: {result['final_accuracy']*100:.2f}%")
    print(f"Drawdown: {result['drawdown']*100:.2f}%")
    if args.rollback:
        print(f"Rollback Count: {result['rollback_count']}")
    print(f"Saved to: {output_subdir}")

    runner.cleanup()
    return result


if __name__ == "__main__":
    main()
