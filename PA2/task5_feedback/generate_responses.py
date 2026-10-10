"""Matched fixed-set math generation from the three frozen course policies."""
from __future__ import annotations
import argparse
import time
import torch
from tqdm.auto import tqdm
from common.data import load_yaml
from common.models import load_policy, load_tokenizer
from common.generation import batch_generate
from task5_feedback.metrics import response_metrics
from task5_feedback.runtime import (POLICIES, BATCH_SIZE, load_dataset, validate_protocol,
    adapter_identity, output_dir, resume_records, finish, append_record, object_hash, row_id)


def generate(config='configs/feedback.yaml', dataset='gsm', policy='sft', resume=False, smoke=False):
    cfg = load_yaml(config); validate_protocol(cfg)
    rows = load_dataset(cfg, dataset, smoke)
    adapter, adapter_sha = adapter_identity(cfg, policy)
    out = output_dir(cfg); path = out/f'{dataset}_{policy}_responses.jsonl'; meta = path.with_suffix('.meta.json')
    records = resume_records(cfg, rows, path, meta, resume, 'generation', dataset=dataset,
                             policy=policy, adapter_sha256=adapter_sha, smoke=smoke, batch_size=BATCH_SIZE,
                             decoding=cfg['generation'], max_new_tokens=cfg['math_max_new_tokens'],
                             prompt_rule='full released messages; reject context overflow')
    if len(records) == len(rows):
        finish(meta, records); print(f'Skipping completed generation: {dataset}/{policy}', flush=True); return
    tokenizer = load_tokenizer(cfg['base_model'])
    model = load_policy(cfg, adapter_path=str(adapter) if adapter else None, trainable=False)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    cap = int(cfg['math_max_new_tokens'])
    first = len(records)//BATCH_SIZE*BATCH_SIZE
    bar = tqdm(total=len(rows), initial=len(records), desc=f'Math {dataset}: {policy}')
    try:
        for start in range(first, len(rows), BATCH_SIZE):
            batch = rows[start:start+BATCH_SIZE]
            prompts = [r['messages'] for r in batch]
            prompt_lengths = [len(tokenizer.apply_chat_template(p, tokenize=True, add_generation_prompt=True)) for p in prompts]
            prompt_cap = max(prompt_lengths)
            if prompt_cap + cap > int(model.config.max_position_embeddings):
                raise ValueError('Released math prompt exceeds model context; do not truncate silently')
            batch_seed = int(cfg['seed']) + start
            # Reset each full batch, including replayed members after a disconnect.
            # The same batch membership/seed/decoding is used for all three policies.
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(batch_seed); torch.cuda.synchronize()
            before = time.perf_counter()
            result = batch_generate(model, tokenizer, prompts, max_prompt_length=prompt_cap,
                                    max_new_tokens=cap, **cfg['generation'])
            if torch.cuda.is_available(): torch.cuda.synchronize()
            seconds = time.perf_counter()-before
            for offset, row in enumerate(batch):
                index = start+offset
                if index < len(records):
                    old = records[index]
                    if old['response'] != result['responses'][offset]:
                        raise ValueError('Partial-batch replay differs; resume requires same device/software environment')
                    continue
                text = result['responses'][offset]; truncated = result['truncated'][offset]
                token_ids = result['response_ids'][offset, :result['response_lengths'][offset]].cpu().tolist()
                record = {'row_index': index, 'input_sha256': object_hash(row), 'prompt_id': row_id(row,dataset),
                          'source_index': row['source_index'], 'dataset': dataset, 'policy': policy,
                          'question': row['question'], 'messages': row['messages'], 'gold_final': str(row['gold_final']),
                          'response': text, 'response_token_ids': token_ids, 'response_tokens': len(token_ids),
                          'prompt_tokens': prompt_lengths[offset], 'prompt_truncated': False,
                          'generation_truncated': bool(truncated), 'terminated_with_eos': result['terminated_with_eos'][offset],
                          'batch_start': start, 'batch_seed': batch_seed, 'generation_seconds_share': seconds/len(batch),
                          **response_metrics(text, row['gold_final'], truncated)}
                append_record(path, record); records.append(record); bar.update(1)
    finally: bar.close()
    finish(meta, records)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/feedback.yaml')
    p.add_argument('--dataset',choices=['gsm','transfer'],default='gsm'); p.add_argument('--policy',choices=POLICIES,default='sft')
    p.add_argument('--resume',action='store_true'); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); generate(a.config,a.dataset,a.policy,a.resume,a.smoke)

if __name__=='__main__': main()
