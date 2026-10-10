"""Task 5 protocol identity, atomic records and disconnect recovery."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from common.data import load_yaml, read_jsonl, repo_path

POLICIES = ('sft', 'rlvr', 'rlaif')
DOMAINS = ('gsm', 'transfer')
ASSET_REVISION = '0b350481fb03f5525a35bcdec4131bd4fe487f98'
DATA = {
    'gsm': ('gsm_eval', 300, '24f712750928c18c1a0a6849eebe97b3e01a6c0c0ebf2410d101a3df1872bf4a'),
    'transfer': ('math_transfer_eval', 100, 'f24163301035ccd4b66d12b687887fe1e167dd31ef62d6bc62a2fa152dcdcb76'),
    'diagnostics': ('task5_diagnostics', 100, '9008b96595bb1419f1326bf02beefd689f09028c2aa6598d7c427394602f036c'),
}
# The math scaffold prescribes no evaluation batch size. Fixed across policies and
# saved in signatures. Partially written batches are replayed in full on resume.
BATCH_SIZE = 4

ADAPTER_SHA256 = {
    'rlvr': 'b309564b45c0268e4e5ffb0db5cb100e557a2c407a68ea3fa3de91fe964d92c8',
    'rlaif': 'e5a0eaacb3ce32a73fe86fd6a823be21c60a460eee8682493e5cc0f2e1019713',
}
ADAPTER_CONFIG_SHA256 = 'b7c4427b92d1514135b9f3530e3fc9134a1f9f460f980c8798a22c7d5fd0d4ae'

RELEASE_FILES = {
    'configs/base.yaml': 'c37ed8b49832388a6193202d30640d79b1d0279ba428731de28900e8c8a92853',
    'configs/feedback.yaml': '8fbc6d86a9ea54bbc9ab802802eb57725c1000420f0fbeca9cca0bc1ede9c57f',
    'requirements.txt': 'f13c0e6952c39ee99b8996ecee9cf5a426e4f798bc646c2aade007bec803603f',
    'task5_feedback/rlvr.py': 'f8865f7d6c67dc4d0565b182411383f0ec2ed9d01a0f99198ad2db0c31ec1bff',
    'task5_feedback/rlaif.py': '13e10a76c225aae8db5e17900bb070f89e4ebf51ab5b78dd86a5423740c1a1f9',
}


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact_path(path):
    p = Path(path)
    return p if p.is_absolute() else Path(os.environ.get('PA2_ARTIFACT_ROOT', repo_path('.'))) / p


def output_dir(cfg):
    return artifact_path(cfg['results_dir']) / 'task5_feedback'


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def append_record(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n'); f.flush()


def load_records(path):
    p = Path(path)
    if not p.exists(): return []
    lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines):
        if not line.strip(): raise ValueError(f'Blank durable record in {p}')
        try: out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1 or line.endswith('\n'): raise
            p.write_text(''.join(lines[:i]), encoding='utf-8')
            return out
    if lines and not lines[-1].endswith('\n'):
        with p.open('a', encoding='utf-8') as f: f.write('\n')
    return out


def code_hash():
    paths = [*repo_path('task5_feedback').glob('*.py'), *repo_path('common').glob('*.py'),
             repo_path('requirements.txt'), repo_path('configs/base.yaml'), repo_path('configs/feedback.yaml')]
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(repo_path('.'))).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def provenance(cfg):
    packages = {}
    for name in ('torch', 'transformers', 'tokenizers', 'peft', 'trl', 'bitsandbytes', 'accelerate', 'datasets'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_path('.'), text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    import torch
    return {'config': cfg, 'config_sha256': object_hash(cfg), 'code_sha256': code_hash(),
            'git_commit': commit, 'python': platform.python_version(), 'packages': packages,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'asset_revision': ASSET_REVISION,
            'starter_revision': '1d64ac65acd5e45d1e4e1f415edc80455e21274e',
            'package_source_revision': repo_path('SOURCE_REVISION.txt').read_text().strip() if repo_path('SOURCE_REVISION.txt').exists() else None}


def validate_protocol(cfg):
    for package, want in {'transformers': '4.57.1', 'tokenizers': '0.22.1', 'peft': '0.17.1', 'trl': '0.27.2'}.items():
        if importlib.metadata.version(package) != want:
            raise ValueError(f'Released {package}=={want} required; install requirements.txt')
    for path, sha in RELEASE_FILES.items():
        if file_hash(repo_path(path)) != sha:
            raise ValueError(f'Released file changed: {path}')
    # Compare against the actual checked-in release, plus mandated constants, so
    # silent model/cap/seed/precision substitutions cannot pass preflight.
    if cfg != load_yaml('configs/feedback.yaml'):
        raise ValueError('Task 5 requires unchanged released configs/feedback.yaml and base.yaml')
    expected = {'base_model': 'Qwen/Qwen2.5-1.5B-Instruct', 'ai_judge_model': 'Qwen/Qwen2.5-3B-Instruct',
                'dtype': 'float16', 'quantize_frozen_models': True, 'seed': 6304, 'math_max_new_tokens': 512}
    for key, value in expected.items():
        if cfg.get(key) != value: raise ValueError(f'Released {key}={value!r} required')
    if cfg['generation'] != {'temperature': 0.7, 'top_p': 0.9, 'do_sample': True}:
        raise ValueError('Keep the released sampling settings')
    for name in POLICIES:
        want = None if name == 'sft' else f'checkpoints/{name}_policy'
        if cfg['policies'].get(name) != want: raise ValueError(f'Course-supplied {name} policy required')


def load_dataset(cfg, domain, smoke=False):
    key, count, expected = DATA[domain]
    path = repo_path(cfg['paths'][key])
    if file_hash(path) != expected: raise ValueError(f'Fixed course data changed: {path}')
    rows = read_jsonl(path)
    if len(rows) != count: raise ValueError(f'Expected {count} {domain} rows')
    ids = [row_id(r, domain) for r in rows]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate course record IDs')
    if domain != 'diagnostics' and any(not r.get('messages') or 'gold_final' not in r for r in rows):
        raise ValueError('Fixed math prompt schema changed')
    return rows[:2] if smoke and domain != 'diagnostics' else rows


def row_id(row, domain):
    if domain == 'diagnostics': return f"{row['problem_id']}:{row['variant_type']}"
    return str(row.get('prompt_id', f"{domain}:{row.get('source_split', '')}:{row['source_index']}"))


def adapter_identity(cfg, policy):
    if policy == 'sft': return None, None
    p = repo_path(cfg['policies'][policy])
    adapter_cfg = json.loads((p / 'adapter_config.json').read_text())
    if adapter_cfg.get('base_model_name_or_path') != cfg['base_model']:
        raise ValueError(f'{policy} adapter has different base model')
    lc = cfg['lora']
    for key, want in [('r', lc['r']), ('lora_alpha', lc['alpha']), ('lora_dropout', lc['dropout'])]:
        if adapter_cfg.get(key) != want: raise ValueError(f'Course {policy} adapter {key} differs')
    if set(adapter_cfg.get('target_modules', [])) != set(lc['target_modules']):
        raise ValueError('Course adapter target modules differ')
    weights = p / 'adapter_model.safetensors'
    if file_hash(weights) != ADAPTER_SHA256[policy] or file_hash(p / 'adapter_config.json') != ADAPTER_CONFIG_SHA256:
        raise ValueError(f'{policy} weights/config differ from the pinned course release')
    return p, object_hash({'weights': ADAPTER_SHA256[policy], 'config': ADAPTER_CONFIG_SHA256})


def resume_records(cfg, inputs, path, meta_path, resume, kind, **extra):
    sig = object_hash({'cfg': cfg, 'inputs': inputs, 'code_sha256': code_hash(), 'kind': kind, **extra})
    current = provenance(cfg)
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old['signature'] != sig: raise ValueError('Saved Task 5 code/config/input identity differs; do not mix runs')
        if not resume: raise ValueError('Outputs already exist; use --resume')
        if old['packages'] != current['packages'] or old['gpu'] != current['gpu']:
            raise ValueError('Resume requires the same software versions and GPU as the saved run')
    elif path.exists(): raise ValueError('Orphaned output without provenance metadata')
    records = load_records(path)
    if len(records) > len(inputs): raise ValueError('Too many saved records')
    for i, record in enumerate(records):
        if record.get('row_index') != i or record.get('input_sha256') != object_hash(inputs[i]):
            raise ValueError('Saved Task 5 record order or input identity differs')
    write_json(meta_path, {**current, 'signature': sig, 'kind': kind, 'n_required': len(inputs),
                          'status': 'running', **extra})
    return records


def finish(meta_path, records):
    meta = json.loads(meta_path.read_text())
    if len(records) != meta['n_required']: raise ValueError('Incomplete Task 5 stage')
    meta.update(status='complete', n_completed=len(records))
    write_json(meta_path, meta)


def require_complete(path, meta_path, count):
    records = load_records(path)
    meta = json.loads(meta_path.read_text())
    if meta.get('smoke') or meta.get('status') != 'complete' or len(records) != count:
        raise ValueError(f'Full completed results required: {path}')
    if meta['code_sha256'] != code_hash(): raise ValueError('Results belong to a different Task 5 revision')
    return records, meta
