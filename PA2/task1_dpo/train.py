"""One-epoch DPO with exact subset IDs, accumulation, and resumable checkpoints."""
from __future__ import annotations
import argparse
import gc
import math
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from common.data import (encode_prompt_response, load_yaml, pad_batch, preference_responses,
                         prompt_messages_from_preference, read_jsonl, repo_path)
from common.logging_utils import set_seed
from common.models import (load_policy, load_tokenizer, reference_mode, trainable_parameters,
                           count_parameters)
from task1_dpo.encoding import encode_preference_pair, pair_diagnostics, overflow_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.utils import (artifact_path, result_dir, file_hash, object_hash, identity,
                            provenance, write_json, append_record, sequence_logprobs,
                            to_device, resolved_model_revision, code_hash)


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            messages = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            c, r, _ = encode_preference_pair(tokenizer, messages, yc, yr, max_length)
            chosen.append(c)
            rejected.append(r)
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])



def recoverable_update(optimizer, scaler, params, backward_window, max_grad_norm, max_retries=16):
    """Replay one accumulation window after FP16 overflow, without skipping data.

    Restoring RNG also reproduces dropout. Only a finite-gradient attempt may
    update parameters. Loss scaling is numerical bookkeeping, not a course
    optimizer/hyperparameter change.
    """
    rng = capture_rng()
    for retries in range(max_retries + 1):
        if retries:
            restore_rng(rng)
        optimizer.zero_grad(set_to_none=True)
        diagnostics = backward_window()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
        if torch.isfinite(norm):
            scale_used = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return diagnostics, norm, retries, scale_used
        if not scaler.is_enabled():
            raise FloatingPointError('Non-finite DPO gradient without FP16 loss scaling')
        old_scale = float(scaler.get_scale())
        # No optimizer step: discard overflowed gradients and reset scaler state.
        scaler.update(new_scale=old_scale * float(scaler.get_backoff_factor()))
        optimizer.zero_grad(set_to_none=True)
        print(f'FP16 gradient overflow: retrying the SAME accumulation window; '
              f'loss scale {old_scale:g} -> {scaler.get_scale():g}', flush=True)
    raise FloatingPointError(
        f'Non-finite DPO gradient persisted after {max_retries} retries; '
        'no examples were skipped and no invalid update was applied')


def save_checkpoint(model, optimizer, scaler, output, state):
    tmp = output / 'last_checkpoint.new'; dst = output / 'last_checkpoint'
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    model.save_pretrained(tmp)
    torch.save({**state, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'rng': capture_rng()}, tmp / 'training_state.pt')
    old = output / 'last_checkpoint.previous'
    shutil.rmtree(old, ignore_errors=True)
    if dst.exists(): os.replace(dst, old)
    os.replace(tmp, dst); shutil.rmtree(old, ignore_errors=True)


def run_training(config_path='configs/dpo.yaml', run_name='standard', dataset_path=None,
                 output_path=None, beta=None, max_examples=None, resume=False,
                 skip_completed=False, smoke=False):
    cfg = load_yaml(config_path)
    beta = float(cfg['beta'] if beta is None else beta)
    data_path = dataset_path or cfg['paths']['dpo_standard_train']
    rows = read_jsonl(data_path)
    if max_examples is not None: rows = rows[:int(max_examples)]
    if smoke: rows = rows[:4]
    if not rows: raise ValueError('Empty training set')
    output = artifact_path(output_path or cfg['standard_output'])
    results = result_dir(cfg, run_name)
    signature = object_hash({'config': cfg, 'beta': beta, 'dataset_sha256': file_hash(data_path),
                             'selected_rows': [identity(r, i) for i, r in enumerate(rows)],
                             'smoke': smoke, 'code_sha256': code_hash(), 'prompt_overflow_mode': overflow_mode()})
    completed = output / 'training_metadata.json'
    if completed.exists():
        import json
        previous = json.loads(completed.read_text())
        if skip_completed and previous.get('signature') == signature and previous.get('status') == 'complete':
            print(f'Skipping completed run: {run_name}'); return output
        raise FileExistsError(f'{output} already has a completed run. Use --skip-completed for the identical run.')
    checkpoint = output / 'last_checkpoint'
    if not checkpoint.exists() and (output / 'last_checkpoint.previous').exists():
        checkpoint = output / 'last_checkpoint.previous'
    if output.exists() and any(output.iterdir()) and not (resume and checkpoint.exists()):
        raise FileExistsError(f'Partial run at {output}; use --resume, or a new run directory.')
    if not torch.cuda.is_available() and not smoke:
        raise RuntimeError('Full course runs require a GPU. CPU is allowed only for --smoke.')
    set_seed(int(cfg['seed']))
    start = time.perf_counter()
    state = None
    if resume and checkpoint.exists():
        state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
        if state['signature'] != signature: raise ValueError('Resume configuration/data/code differs from saved run')
    tokenizer = load_tokenizer(cfg['base_model'])
    collate = make_collate(tokenizer, int(cfg['max_sequence_length']))
    # Preflight every selected pair; never silently filter or change the course subset.
    token_rows = []
    for i,row in enumerate(rows):
        c,r = collate([row])
        token_rows.append({**identity(row,i), **pair_diagnostics(tokenizer,row,int(cfg['max_sequence_length'])), 'chosen_scored_tokens': int(c['response_mask'].sum()),
                           'rejected_scored_tokens': int(r['response_mask'].sum())})
    model = load_policy(cfg, adapter_path=str(checkpoint) if state else None,
                        trainable=True, fresh_lora=state is None)
    params = trainable_parameters(model)
    if not params or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('Only policy LoRA parameters may be trainable')
    # FP32 trainable tensors prevent GradScaler from encountering FP16 gradients.
    for p in params: p.data = p.data.float()
    optimizer = AdamW(params, lr=float(cfg['learning_rate']), weight_decay=float(cfg['weight_decay']))
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available() and cfg['dtype']=='float16')
    output.mkdir(parents=True, exist_ok=True); results.mkdir(parents=True, exist_ok=True)
    metadata = provenance(cfg, signature=signature, run_name=run_name, beta=beta,
                          dataset=str(data_path), dataset_sha256=file_hash(data_path),
                          train_examples=len(rows), epochs=int(cfg['epochs']), smoke=smoke,
                          base_model_revision=resolved_model_revision(model),
                          reference='same frozen base with LoRA disabled', prompt_overflow_mode=overflow_mode(),
                          parameters=dict(zip(('total','trainable'), count_parameters(model))),
                          optimizer='AdamW; constant learning rate; no scheduler',
                          shuffle='torch.randperm using seed + epoch', status='running')
    write_json(results/'run_metadata.json',metadata)
    write_json(results/'training_ids.json',token_rows)
    micro_count = math.ceil(len(rows)/int(cfg['batch_size']))
    step = 0; next_epoch = 0; next_batch = 0; elapsed_before = 0
    if state:
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        step=state['step']; next_epoch=state['epoch']; next_batch=state['next_batch']
        elapsed_before=state['elapsed_seconds']; restore_rng(state['rng'])
        # Discard logs newer than the durable checkpoint before replaying its tail.
        from task1_dpo.utils import load_records
        log=results/'training_log.jsonl'
        kept=[r for r in load_records(log) if r['optimizer_step']<=step]
        log.write_text(''.join(__import__('json').dumps(r)+'\n' for r in kept))
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    accum=int(cfg['grad_accum_steps']); batch_size=int(cfg['batch_size'])
    optimizer.zero_grad(set_to_none=True)
    if state is None:
        save_checkpoint(model,optimizer,scaler,output,dict(signature=signature,step=0,
                        epoch=0,next_batch=0,elapsed_seconds=time.perf_counter()-start))
    for epoch in range(next_epoch, int(cfg['epochs'])):
        gen=torch.Generator().manual_seed(int(cfg['seed'])+epoch)
        order=torch.randperm(len(rows),generator=gen).tolist()
        batches=[order[i:i+batch_size] for i in range(0,len(order),batch_size)]
        first=next_batch if epoch==next_epoch else 0
        if first % accum != 0 and first != len(batches): raise ValueError('Checkpoint is not at an accumulation boundary')
        for group_start in tqdm(range(first,len(batches),accum),desc=f'{run_name}: epoch {epoch+1}'):
            group=batches[group_start:group_start+accum]
            n=sum(len(indices) for indices in group)
            def backward_window():
                aggregates={'loss':0.0,'preference_accuracy':0.0,'preference_margin_mean':0.0}
                for indices in group:
                    chosen,rejected=collate([rows[i] for i in indices])
                    chosen=to_device(chosen,model); rejected=to_device(rejected,model)
                    # Evaluate and release reference activations before constructing policy graphs.
                    with torch.no_grad(), reference_mode(model):
                        rc=sequence_logprobs(model,chosen)[0]; rr=sequence_logprobs(model,rejected)[0]
                    pc=sequence_logprobs(model,chosen)[0]; pr=sequence_logprobs(model,rejected)[0]
                    loss,diag=dpo_loss(pc,pr,rc,rr,beta)
                    if not torch.isfinite(loss): raise FloatingPointError('Non-finite DPO loss')
                    weight=len(indices)/n  # Correctly handles final partial batches/windows.
                    scaler.scale(loss*weight).backward()
                    aggregates['loss']+=float(loss.detach())*weight
                    for key in ('preference_accuracy','preference_margin_mean'):
                        aggregates[key]+=float(diag[key])*weight
                    del chosen,rejected,rc,rr,pc,pr,loss,diag
                return aggregates
            aggregates,norm,overflow_retries,loss_scale = recoverable_update(
                optimizer,scaler,params,backward_window,cfg['max_grad_norm'])
            step+=1; stop=group_start+len(group)
            record={'optimizer_step':step,'epoch':epoch+1,'next_batch':stop,
                    'examples_in_update':n,'learning_rate':float(cfg['learning_rate']),
                    'gradient_norm_before_clipping':float(norm),
                    'fp16_overflow_retries':overflow_retries,'loss_scale_used':loss_scale,
                    'elapsed_seconds':elapsed_before+time.perf_counter()-start,**aggregates}
            append_record(results/'training_log.jsonl',record)
            if step % 20 == 0 or stop == micro_count or smoke:
                save_checkpoint(model,optimizer,scaler,output,dict(signature=signature,step=step,
                                epoch=epoch,next_batch=stop,elapsed_seconds=record['elapsed_seconds']))
            if smoke: break
        if smoke: break
        next_batch=0
    model.save_pretrained(output); tokenizer.save_pretrained(output)
    metadata.update(status='complete',optimizer_steps=step,
                    elapsed_seconds=elapsed_before+time.perf_counter()-start,
                    peak_vram_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None)
    write_json(output/'training_metadata.json',metadata); write_json(results/'run_metadata.json',metadata)
    # Final adapter is enough for later tasks; keep last checkpoint for interruption recovery.
    del model,optimizer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/dpo.yaml'); p.add_argument('--run-name',default='standard')
    p.add_argument('--dataset'); p.add_argument('--output'); p.add_argument('--beta',type=float)
    p.add_argument('--max-examples',type=int); p.add_argument('--resume',action='store_true')
    p.add_argument('--skip-completed',action='store_true'); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); run_training(a.config,a.run_name,a.dataset,a.output,a.beta,a.max_examples,
                                 a.resume,a.skip_completed,a.smoke)
if __name__=='__main__': main()
