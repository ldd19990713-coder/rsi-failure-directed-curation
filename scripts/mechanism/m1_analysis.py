#!/usr/bin/env python3
"""
M1 Leave-One-Out Analysis Script
================================
Aggregate results and produce leave-one-out table.

Input: artifacts/canonical/mechanism/M1_loo/*/metrics.json
Output: M1_loo_summary.md + M1_loo_table.csv
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

EXPERIMENT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'CITATION.cff').exists())
M1_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "mechanism" / "M1_loo"
DOCS_DIR = EXPERIMENT_DIR / "docs"

# Reference: Full-5 baseline (from Phase3 G3 A_fail or fresh Full-5 runs)
# If using Phase3 reuse, we need to load from g3_g4 directory
G3_AFAIL_DIR = EXPERIMENT_DIR / "artifacts" / "canonical" / "g3_g4" / "g3" / "a_fail"


def load_m1_results():
    """Load all M1 LOO results"""
    results = {}

    for p in sorted(M1_DIR.rglob("metrics.json")):
        m = json.loads(p.read_text())
        tag = m.get('tag', p.parent.name)
        drop_type = m.get('drop_type')
        seed = m.get('seed')

        key = (drop_type, seed)
        results[key] = {
            'tag': tag,
            'drop_type': drop_type,
            'seed': seed,
            'peak_accuracy': m.get('peak_accuracy'),
            'peak_round': m.get('peak_round'),
            'final_accuracy': m.get('final_accuracy'),
            'drawdown': m.get('drawdown'),
            'peak_emh': m.get('peak_emh'),
            'final_emh': m.get('final_emh'),
            'invalid_rate': m.get('final_invalid_rate'),
            'recipe_hash': m.get('recipe_hash'),
        }

    return results


def load_g3_afaill_reference(seeds=(41, 42, 43)):
    """Load Phase3 G3 A_fail as Full-5 reference"""
    ref = {}
    for seed in seeds:
        p = G3_AFAIL_DIR / f"seed_{seed}" / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            # Extract final round accuracy
            rounds = m.get('rounds', [])
            final_acc = rounds[-1]['accuracy'] if rounds else None
            final_emh = None
            if rounds and 'difficulty_breakdown' in rounds[-1]:
                db = rounds[-1]['difficulty_breakdown']
                final_emh = {
                    'easy': db['easy']['accuracy'],
                    'medium': db['medium']['accuracy'],
                    'hard': db['hard']['accuracy'],
                }

            ref[seed] = {
                'tag': f'full5_seed{seed}',
                'drop_type': None,
                'seed': seed,
                'peak_accuracy': m.get('peak_accuracy'),
                'peak_round': m.get('peak_round'),
                'final_accuracy': final_acc,
                'drawdown': m.get('drawdown'),
                'peak_emh': None,  # Not stored in G3 format
                'final_emh': final_emh,
                'invalid_rate': None,
                'recipe_hash': 'v5_canonical',
            }
    return ref


def compute_summary(results, group_by='drop_type'):
    """Compute mean±std per group"""
    groups = defaultdict(list)

    for (drop_type, seed), r in results.items():
        key = drop_type or 'full5'
        groups[key].append(r)

    summary = {}
    for key, items in groups.items():
        peaks = [r['peak_accuracy'] for r in items if r['peak_accuracy'] is not None]
        finals = [r['final_accuracy'] for r in items if r['final_accuracy'] is not None]
        drawdowns = [r['drawdown'] for r in items if r['drawdown'] is not None]

        summary[key] = {
            'n': len(items),
            'peak_mean': sum(peaks) / len(peaks) if peaks else None,
            'peak_std': (sum((x - sum(peaks)/len(peaks))**2 for x in peaks) / len(peaks))**0.5 if len(peaks) > 1 else 0,
            'final_mean': sum(finals) / len(finals) if finals else None,
            'drawdown_mean': sum(drawdowns) / len(drawdowns) if drawdowns else None,
            'seeds': [r['seed'] for r in items],
        }

    return summary


def generate_markdown_table(summary, full5_key='full5'):
    """Generate leave-one-out table in markdown"""
    full5 = summary.get(full5_key, {})
    full5_peak = full5.get('peak_mean', 0) or 0

    lines = [
        "# M1 Leave-One-Out Summary",
        "",
        "**Experiment**: Remove one prompt variant type from A_fail augmentation",
        "**Seeds**: Check results",
        "**Reference**: Full-5 (all 5 variant types)",
        "",
        "## Results Table",
        "",
        "| Drop Type | N | Peak (%) | Δ vs Full-5 | Final (%) | Drawdown (%) |",
        "|:----------|:-:|---------:|------------:|----------:|-------------:|",
    ]

    # Full-5 row first
    if full5:
        lines.append(f"| **Full-5 (baseline)** | {full5['n']} | {full5['peak_mean']*100:.2f}±{full5['peak_std']*100:.2f} | — | {full5['final_mean']*100:.2f} | {full5['drawdown_mean']*100:.2f} |")

    # Drop type rows
    drop_names = ['condition_variant', 'hint_variant', 'format_variant', 'synonym_variant', 'context_variant']
    for drop in drop_names:
        if drop in summary:
            s = summary[drop]
            delta = (s['peak_mean'] - full5_peak) * 100 if s['peak_mean'] and full5_peak else 0
            delta_str = f"{delta:+.2f}" if delta != 0 else "0.00"
            lines.append(f"| {drop} | {s['n']} | {s['peak_mean']*100:.2f}±{s['peak_std']*100:.2f} | {delta_str} | {s['final_mean']*100:.2f} | {s['drawdown_mean']*100:.2f} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Necessary variant**: Δ ≤ -3pp AND ≥2/3 seeds in same direction",
        "- **Replaceable/weak**: |Δ| < 3pp or inconsistent direction",
        "",
        "## Generated",
        f"*Date: {Path.ctime(M1_DIR)}*" if M1_DIR.exists() else "",
    ])

    return "\n".join(lines)


def main():
    print("="*60)
    print("M1 Leave-One-Out Analysis")
    print("="*60)

    # Load M1 results
    m1_results = load_m1_results()
    print(f"\nLoaded {len(m1_results)} M1 experiments")

    # Load Full-5 reference from Phase3 G3 (or use M1 fresh runs)
    g3_ref = load_g3_afaill_reference(seeds=(41, 42, 43))
    print(f"Loaded {len(g3_ref)} Full-5 reference runs from Phase3")

    # Combine: use M1 fresh full5 if available, else use G3 reference
    all_results = {**{(None, seed): r for seed, r in g3_ref.items()}}
    all_results.update(m1_results)

    # Compute summary
    summary = compute_summary(all_results)

    # Generate markdown
    md = generate_markdown_table(summary)

    # Save markdown
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = DOCS_DIR / "M1_leave_one_out_summary.md"
    with open(md_path, 'w') as f:
        f.write(md)
    print(f"\nSaved: {md_path}")

    # Save CSV
    csv_path = M1_DIR / "M1_loo_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['drop_type', 'n', 'peak_mean', 'peak_std', 'final_mean', 'drawdown_mean', 'seeds'])
        w.writeheader()
        for key, s in summary.items():
            w.writerow({
                'drop_type': key,
                'n': s['n'],
                'peak_mean': f"{s['peak_mean']*100:.2f}" if s['peak_mean'] else '',
                'peak_std': f"{s['peak_std']*100:.2f}" if s['peak_std'] else '',
                'final_mean': f"{s['final_mean']*100:.2f}" if s['final_mean'] else '',
                'drawdown_mean': f"{s['drawdown_mean']*100:.2f}" if s['drawdown_mean'] else '',
                'seeds': ','.join(map(str, s['seeds'])),
            })
    print(f"Saved: {csv_path}")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(md)


if __name__ == "__main__":
    main()
