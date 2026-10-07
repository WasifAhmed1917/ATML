"""DPO objective, corrected against Rafailov et al. and the course manual."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta):
    if beta <= 0: raise ValueError('DPO beta must be positive')
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp.detach() - ref_rejected_logp.detach()
    margin = policy_margin - ref_margin
    logits = beta * margin  # Starter used + ref_margin; the objective requires subtraction.
    return -F.logsigmoid(logits).mean(), {
        'logit_mean': logits.detach().mean(),
        'policy_margin_mean': policy_margin.detach().mean(),
        'preference_margin_mean': margin.detach().mean(),
        'preference_accuracy': (margin > 0).float().mean().detach(),
    }
