"""Outcome, formatting and preference metrics; no judge is treated as ground truth."""
from __future__ import annotations
from collections import Counter
import re
import numpy as np
from task5_feedback.rlvr import exact_reward, extract_designated_final

# Formatting is stricter than the released correctness parser: exactly one final
# field, on its own last line. Correctness always uses the unchanged course verifier.
FINAL_LINE = re.compile(r'^####\s*[-+]?\d[\d,]*(?:\.\d+)?\s*$')


def format_compliant(text):
    lines = str(text).strip().splitlines()
    return bool(lines and FINAL_LINE.fullmatch(lines[-1]) and str(text).count('####') == 1)


def response_metrics(text, gold, truncated=False):
    final = extract_designated_final(text)
    reward = exact_reward(text, str(gold))
    return {'exact_reward': reward, 'designated_final': final, 'format_compliant': format_compliant(text),
            'failure_type': 'correct_final' if reward else
                            ('no_designated_final' if final is None else 'incorrect_designated_final'),
            'truncation_failure': bool(truncated and not reward)}


def verifier_preference(a, b):
    return 'A' if a > b else 'B' if b > a else 'TIE'


def agreement(pairs):
    n = len(pairs)
    strict = [p for p in pairs if p['verifier_preference'] != 'TIE']
    vt = [p for p in pairs if p['verifier_preference'] == 'TIE']
    table = Counter((p['verifier_preference'], p['judge_preference']) for p in pairs)
    return {'n_pairs': n,
            'three_way_agreement': sum(p['verifier_preference'] == p['judge_preference'] for p in pairs) / n if n else None,
            'verifier_strict_n': len(strict),
            'judge_agreement_given_verifier_strict': sum(p['verifier_preference'] == p['judge_preference'] for p in strict) / len(strict) if strict else None,
            'judge_opposes_verifier_strict_fraction': sum(p['judge_preference'] not in (p['verifier_preference'], 'TIE') for p in strict) / len(strict) if strict else None,
            'verifier_tie_n': len(vt),
            'judge_distinguishes_verifier_ties_fraction': sum(p['judge_preference'] != 'TIE' for p in vt) / len(vt) if vt else None,
            'contingency': {f'{a}/{b}': table[(a, b)] for a in ('A','B','TIE') for b in ('A','B','TIE')}}


def policy_summary(records):
    n = len(records)
    if not n: raise ValueError('Empty math evaluation')
    lengths = np.array([r['response_tokens'] for r in records], dtype=float)
    return {'n_prompts': n, 'exact_accuracy': sum(r['exact_reward'] for r in records)/n,
            'format_compliance': sum(r['format_compliant'] for r in records)/n,
            'response_length_mean': float(lengths.mean()), 'response_length_std': float(lengths.std()),
            'response_length_median': float(np.median(lengths)),
            'response_length_iqr': float(np.percentile(lengths,75)-np.percentile(lengths,25)),
            'generation_truncated_fraction': sum(r['generation_truncated'] for r in records)/n,
            'prompt_truncated_fraction': sum(r['prompt_truncated'] for r in records)/n,
            'failure_counts': dict(Counter(r['failure_type'] for r in records)),
            'truncation_failure_count': sum(r['truncation_failure'] for r in records),
            'generation_seconds': sum(r['generation_seconds_share'] for r in records),
            'generated_tokens': int(lengths.sum())}


def pair_summary(pairs):
    n = len(pairs)
    c = Counter(p['judge_preference'] for p in pairs)
    return {**agreement(pairs), 'ai_win_rate_vs_sft': c['A']/n if n else None,
            'ai_tie_rate_vs_sft': c['TIE']/n if n else None,
            'ai_loss_rate_vs_sft': c['B']/n if n else None,
            'ai_tie_adjusted_win_rate_vs_sft': (c['A']+.5*c['TIE'])/n if n else None}


def diagnostic_rates(pairs, mechanism):
    n = len(pairs)
    c = Counter(p[f'{mechanism}_preference'] for p in pairs)
    # A is clean for every category; identity control expects a tie.
    identity = pairs[0]['variant_type'] == 'clean_correct'
    return {'n_pairs': n, 'expected_preference': 'TIE' if identity else 'A',
            'better_response_rate': None if identity else c['A']/n,
            'tie_rate': c['TIE']/n, 'wrong_preference_rate': (c['A']+c['B'])/n if identity else c['B']/n,
            'clean_preference_rate': c['A']/n, 'variant_preference_rate': c['B']/n}
