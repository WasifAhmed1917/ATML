"""Response-prioritizing truncation with identical context for both pair members.

Keep the prompt tail (as in the initial course release), budgeting it using the
longer response including EOS. If a response alone cannot fit, retain its tail
and at least one prompt token. No rows are filtered or resampled.
"""
from common.data import preference_responses, prompt_messages_from_preference

PREPROCESSING = 'response_priority_shared_prompt_tail_v1'


def overflow_mode():
    return PREPROCESSING


def encode_preference_pair(tokenizer, messages, chosen, rejected, max_length):
    if max_length < 2:
        raise ValueError('Need room for prompt and response tokens')
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not prompt:
        raise ValueError('Empty encoded prompt')
    responses = []
    for text in (chosen, rejected):
        content = tokenizer(text, add_special_tokens=False)['input_ids']
        eos = tokenizer.eos_token_id
        responses.append(content + ([eos] if eos is not None else []))
    if any(not ids for ids in responses):
        raise ValueError('Empty response without EOS')
    longest = max(map(len, responses))
    prompt_budget = max(1, max_length - longest)
    retained_prompt = prompt[-prompt_budget:]
    response_budget = max_length - len(retained_prompt)
    retained = [ids[-response_budget:] for ids in responses]
    pairs = [(retained_prompt + ids, [0]*len(retained_prompt) + [1]*len(ids)) for ids in retained]
    diagnostic = {
        'preprocessing': PREPROCESSING,
        'prompt_tokens_original': len(prompt),
        'prompt_tokens_retained': len(retained_prompt),
        'prompt_tokens_removed': len(prompt)-len(retained_prompt),
        'prompt_overflow': len(prompt) >= max_length,
        'pair_overlength': len(prompt)+longest > max_length,
    }
    for label, original, kept in zip(('chosen', 'rejected'), responses, retained):
        diagnostic.update({
            label+'_tokens_original': len(original),
            label+'_tokens_retained': len(kept),
            label+'_tokens_removed': len(original)-len(kept),
            label+'_overlength': len(prompt)+len(original) > max_length,
        })
    return pairs[0], pairs[1], diagnostic


def pair_diagnostics(tokenizer, row, max_length):
    chosen, rejected = preference_responses(row)
    return encode_preference_pair(tokenizer, prompt_messages_from_preference(row),
                                  chosen, rejected, max_length)[2]
