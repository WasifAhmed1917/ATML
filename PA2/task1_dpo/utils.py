"""Task 1 artifact paths, provenance, and memory-bounded log-probability helpers."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common.data import repo_path
from common.logging_utils import save_json


def artifact_path(path):
    p = Path(path)
    if p.is_absolute():
        return p
    root = os.environ.get('PA2_ARTIFACT_ROOT')
    return Path(root) / p if root else repo_path(p)


def result_dir(cfg, name):
    return artifact_path(cfg['results_dir']) / name


def file_hash(path):
    return hashlib.sha256(repo_path(path).read_bytes()).hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def identity(row, index):
    return {'row_index': index, **{k: row[k] for k in
            ('prompt_id', 'source_index', 'source_split', 'length_stratum') if k in row}}


def code_hash():
    paths = sorted(repo_path('task1_dpo').glob('*.py')) + sorted(repo_path('common').glob('*.py'))
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(repo_path('.'))).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def provenance(cfg, **extra):
    versions = {}
    for package in ('torch', 'transformers', 'tokenizers', 'peft', 'trl', 'bitsandbytes'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    try:
        revision_file = repo_path('SOURCE_REVISION.txt')
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_path('.'), text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = revision_file.read_text().strip() if revision_file.exists() else None
    return {'config': cfg, 'config_sha256': object_hash(cfg), 'code_sha256': code_hash(),
            'git_commit': commit, 'python': platform.python_version(), 'packages': versions,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'asset_revision': '0b350481fb03f5525a35bcdec4131bd4fe487f98', **extra}


def write_json(path, obj):
    # Atomic result writes also work when artifacts are stored on mounted Drive.
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    save_json(tmp, obj); os.replace(tmp, p)


def append_record(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')


def load_records(path):
    p = Path(path)
    if not p.exists(): return []
    lines = p.read_text().splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        if not line.strip(): continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A disconnected runtime can interrupt the final append. Only repair
            # an unterminated last line; corruption in durable records must fail.
            if index != len(lines)-1 or line.endswith('\n'):
                raise
            p.write_text(''.join(lines[:index]), encoding='utf-8')
            break
    # A complete JSON value without its newline must not merge with the next append.
    if records and lines and not lines[-1].endswith('\n'):
        with p.open('a', encoding='utf-8') as f: f.write('\n')
    return records


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a): raise ValueError('Cannot aggregate an empty set')
    if not np.isfinite(a).all(): raise ValueError('Non-finite metric')
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=0)),
            'median': float(np.median(a)), 'iqr': float(np.percentile(a, 75)-np.percentile(a, 25))}


def to_device(batch, model):
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in batch.items()}


def token_logprobs(model, ids, attention_mask, chunk_size=32):
    """Teacher-forced next-token logp, avoiding a full FP32 B x T x V tensor.

    Chunked cross entropy is mathematically identical to gather(log_softmax).
    Non-reentrant checkpointing avoids retaining FP32 vocabulary intermediates.
    """
    from torch.utils.checkpoint import checkpoint
    logits = model(input_ids=ids, attention_mask=attention_mask,
                   use_cache=False, return_dict=True).logits[:, :-1]
    labels = ids[:, 1:]
    parts = []
    def selected_logp(x, y):
        return -F.cross_entropy(x.float().reshape(-1, x.shape[-1]),
                                y.reshape(-1), reduction='none').reshape(y.shape)
    for start in range(0, logits.shape[1], chunk_size):
        x, y = logits[:, start:start+chunk_size], labels[:, start:start+chunk_size]
        if torch.is_grad_enabled() and x.requires_grad:
            parts.append(checkpoint(selected_logp, x, y, use_reentrant=False))
        else: parts.append(selected_logp(x, y))
    return torch.cat(parts, dim=1)


def sequence_logprobs(model, batch):
    tok = token_logprobs(model, batch['input_ids'], batch['attention_mask'])
    mask = batch['response_mask'][:, 1:] * batch['attention_mask'][:, 1:]
    return (tok * mask).sum(-1), tok, mask


def resolved_model_revision(model):
    return getattr(model.config, '_commit_hash', None)
