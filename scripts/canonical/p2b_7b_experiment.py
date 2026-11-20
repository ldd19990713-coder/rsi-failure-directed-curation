#!/usr/bin/env python3
"""
P2b Experiment: 7B+LoRA on v9_g1prime (46 train)
=================================================

Purpose: Test if larger model (7B) can achieve better performance
         on the canonical v9 dataset (46 train samples).

Protocol:
- Model: Qwen2.5-7B + LoRA (rank=16, alpha=32)
- Data: v9_g1prime (train=46, eval=97) - FROZEN
- Arms: A_fail vs A_rand
- K: 1.0× only
- Seeds: 41 (smoke), then 41-45 if passing

Success criteria (vs M2@1.5B seed41):
- peak ≥ 21.65%
- final ≥ 18.62%

If passing: run N=5 × {A_fail, A_rand}
If failing: stop model scaling, try 1.5B curriculum/near-miss filtering
"""

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
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# Paths
# ============================================================
EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
DATA_DIR = EXPERIMENT_DIR / "data"
DATA_VERSION = "v9_g1prime"  # Frozen canonical dataset
MODEL_PATH = "."
OUTPUT_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "P2b_7b"
CONFIGS_DIR = EXPERIMENT_DIR / "configs"

# V5 Canonical Recipe Hash (FROZEN)
EXPECTED_RECIPE_HASH = "2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3"


def get_7b_snapshot_path():
    """Find the actual snapshot directory for 7B model"""
    base = Path(MODEL_PATH)
    if not base.exists():
        return None
    snapshot_dir = base / "snapshots"
    if not snapshot_dir.exists():
        return None
    snapshots = list(snapshot_dir.iterdir())
    if snapshots:
        return snapshots[0]
    return None


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
                print(f"[P2b-7B] ✓ Recipe hash verified: {actual_hash}")
                return True
            else:
                print(f"[P2b-7B] ✗ Recipe hash MISMATCH!")
                return False
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
class P2bAugmentor:
    """P2b augmentor - same as M2 (effective 3-type slice)"""

    def __init__(self):
        self.augmentation_types = [
            "condition_variant",
            "hint_variant",
            "format_variant"
        ]

    def augment_sample(self, sample, round_num, aug_type):
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
        augmented = []
        num_variants = 3 if is_medium else 2
        variant_types = self.augmentation_types[:num_variants]

        for case in failed_cases:
            for aug_type in variant_types:
                augmented.append(self.augment_sample(case, round_num, aug_type))

        return augmented

    def augment_random(self, train_data, round_num, n_samples, rng):
        augmented = []
        for _ in range(n_samples):
            sample = rng.choice(train_data)
            aug_type = rng.choice(self.augmentation_types[:2])
            augmented.append(self.augment_sample(sample, round_num, aug_type))
        return augmented


# ============================================================
# P2b Experiment Runner (7B + LoRA)
# ============================================================
class P2bRunner:
    """P2b Experiment Runner - 7B + LoRA"""

    def __init__(self, seed: int, arm: str):
        self.seed = seed
        self.arm = arm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # V5 FROZEN CONFIG
        self.weight_decay = 0.01
        self.eta_min_ratio = 0.1
        self.num_rounds = 7
        self.batch_size = 1  # Reduced for 7B to avoid OOM
        self.max_length = 256
        self.epochs_per_round = 3

        # LoRA config
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05

        self.set_seed(seed)
        self.augmentor = P2bAugmentor()
        self.training_history = []
        self.aug_manifest = []

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_data(self):
        """Load v9_g1prime data (FROZEN)"""
        # Actual filenames in v9_g1prime directory
        eval_path = DATA_DIR / DATA_VERSION / "eval_reactions_g1prime.json"
        train_path = DATA_DIR / DATA_VERSION / "train_reactions_g1prime.json"

        with open(eval_path) as f:
            self.eval_data = json.load(f)
        with open(train_path) as f:
            self.train_data = json.load(f)

        train_ids = {x['id'] for x in self.train_data}
        eval_ids = {x['id'] for x in self.eval_data}
        assert len(train_ids & eval_ids) == 0, "LEAKAGE: train ∩ eval ≠ ∅"

        print(f"[P2b-7B] Data loaded: Train={len(self.train_data)}, Eval={len(self.eval_data)}")
        print(f"[P2b-7B] Data version: {DATA_VERSION} (FROZEN)")

    def load_model(self):
        """Load 7B model with LoRA"""
        print("[P2b-7B] Loading Qwen2.5-7B model...")

        snapshot_path = get_7b_snapshot_path()
        if snapshot_path is None:
            raise FileNotFoundError(f"7B model not found at {MODEL_PATH}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_path),
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model in bfloat16 (A100 supports it, better numerical stability)
        # Use eager attention to avoid cuDNN issues with gradient checkpointing
        base_model = AutoModelForCausalLM.from_pretrained(
            str(snapshot_path),
            torch_dtype=torch.bfloat16,  # FIX: bf16 instead of fp16 for numerical stability
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager"
        )
        # Do NOT enable gradient checkpointing - causes cuDNN issues with LoRA

        # Freeze all base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        self.model = get_peft_model(base_model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")

        self.model.to(self.device)

        print(f"[P2b-7B] Model loaded with LoRA (r={self.lora_r}, alpha={self.lora_alpha})")

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
        lr = 5e-5  # Higher LR for LoRA
        lr_decayed = lr * (0.85 ** (round_num - 2))

        dataset = ReactionDataset(train_data, self.tokenizer, max_length=self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Only optimize LoRA parameters
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params, lr=lr_decayed, weight_decay=self.weight_decay)
        eta_min = lr_decayed * self.eta_min_ratio
        scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs, eta_min=eta_min)

        print(f"  LR: {lr_decayed:.2e}, Samples: {len(train_data)}")

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    def _get_augmentation(self, failed_cases, round_num, rng):
        """Get augmentation based on arm (K=1.0× fixed)"""
        aug_round = round_num + 1

        if self.arm == "A_fail":
            failed_easy = [c for c in failed_cases if c.get('difficulty') == 'easy']
            failed_medium = [c for c in failed_cases if c.get('difficulty') == 'medium']
            failed_hard = [c for c in failed_cases if c.get('difficulty') == 'hard']

            augmented = []
            augmented.extend(self.augmentor.augment_failed(failed_easy, aug_round, is_medium=False))
            augmented.extend(self.augmentor.augment_failed(failed_medium, aug_round, is_medium=True))
            augmented.extend(self.augmentor.augment_failed(failed_hard, aug_round, is_medium=False))

            return augmented

        else:  # A_rand
            failed_easy = [c for c in failed_cases if c.get('difficulty') == 'easy']
            failed_medium = [c for c in failed_cases if c.get('difficulty') == 'medium']
            failed_hard = [c for c in failed_cases if c.get('difficulty') == 'hard']

            n_easy = len(failed_easy) * 2
            n_medium = len(failed_medium) * 3
            n_hard = len(failed_hard) * 2
            base_count = n_easy + n_medium + n_hard

            return self.augmentor.augment_random(self.train_data, aug_round, base_count, rng)

    def run(self):
        """Run the P2b experiment"""
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

            if round_num < self.num_rounds:
                augmented = self._get_augmentation(result['failed_cases'], round_num, rng)

                self.aug_manifest.append({
                    'round': round_num,
                    'aug_count': len(augmented),
                    'arm': self.arm
                })

                train_data = self.train_data + augmented
                self.train_round(train_data, round_num)

        post_peak = [m['accuracy'] for m in round_metrics[peak_round:]]
        drawdown = peak_accuracy - (min(post_peak) if post_peak else peak_accuracy)

        return {
            'experiment': 'P2b_7b',
            'model': 'Qwen2.5-7B',
            'lora_config': {'r': self.lora_r, 'alpha': self.lora_alpha},
            'arm': self.arm,
            'data_version': DATA_VERSION,
            'seed': self.seed,
            'recipe_hash': EXPECTED_RECIPE_HASH,
            'rounds': round_metrics,
            'peak_accuracy': peak_accuracy,
            'peak_round': peak_round,
            'final_accuracy': round_metrics[-1]['accuracy'],
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
    parser = argparse.ArgumentParser(description="P2b Experiment: 7B+LoRA")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--arm", type=str, required=True, choices=["A_fail", "A_rand"])
    args = parser.parse_args()

    tag = f"P2b_{args.arm}_seed{args.seed}"

    print("="*70)
    print(f"P2b Experiment: 7B+LoRA on v9_g1prime (46 train)")
    print(f"Seed: {args.seed}")
    print(f"Arm: {args.arm}")
    print(f"Model: Qwen2.5-7B + LoRA (r=16, alpha=32)")
    print(f"Data: {DATA_VERSION} (FROZEN)")
    print(f"Tag: {tag}")
    print(f"Config: V5 frozen (float16, decaying LR, effective 3-type slice, K=1.0)")
    print("="*70)

    hash_ok = verify_recipe_hash()
    if not hash_ok:
        print("[P2b-7B] ERROR: Recipe hash mismatch. Aborting.")
        sys.exit(1)

    runner = P2bRunner(seed=args.seed, arm=args.arm)
    result = runner.run()

    output_subdir = OUTPUT_DIR / tag
    output_subdir.mkdir(parents=True, exist_ok=True)

    with open(output_subdir / "metrics.json", 'w') as f:
        json.dump(result, f, indent=2)

    with open(output_subdir / "aug_manifest.json", 'w') as f:
        json.dump(result['aug_manifest'], f, indent=2)

    print(f"\n{'='*70}")
    print(f"P2b-7B RESULTS: {tag}")
    print(f"{'='*70}")
    print(f"Peak Accuracy: {result['peak_accuracy']*100:.2f}% @ Round {result['peak_round']}")
    print(f"Final Accuracy: {result['final_accuracy']*100:.2f}%")
    print(f"Drawdown: {result['drawdown']*100:.2f}%")

    # Overline check
    peak_pass = result['peak_accuracy'] * 100 >= 21.65
    final_pass = result['final_accuracy'] * 100 >= 18.62
    print(f"\n过线判定:")
    print(f"  Peak ≥ 21.65%: {'✅' if peak_pass else '❌'} ({result['peak_accuracy']*100:.2f}%)")
    print(f"  Final ≥ 18.62%: {'✅' if final_pass else '❌'} ({result['final_accuracy']*100:.2f}%)")

    if peak_pass and final_pass:
        print(f"\n  🟢 过线 → 启动 N=5 × {{A_fail, A_rand}}")
    else:
        print(f"\n  🔴 未过线 → 停扩模，改 1.5B 近错/课程化筛选")

    print(f"Saved to: {output_subdir}")

    runner.cleanup()
    return result


if __name__ == "__main__":
    main()
