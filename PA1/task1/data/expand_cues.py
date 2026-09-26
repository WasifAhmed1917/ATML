# Notebook-derived stage: ATML_PA1_Task1.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 19
CAND = np.load(candidate_file)['images']
review = pd.read_csv(
    review_file, dtype={'accept': 'string'}, keep_default_na=False
)
assert len(CAND) == len(review)

if len(CAND) == 400:
    rng = np.random.default_rng(SEED + 400)
    content_pool = np.flatnonzero(y == CLASSES.index('ship'))
    style_pool = np.flatnonzero(y == CLASSES.index('airplane'))

    group = review[
        (review.content_class == 'ship')
        & (review.style_class == 'airplane')
    ]
    used = set(zip(group.content_id, group.style_id))
    new_images, new_rows = [], []

    while len(new_images) < 40:
        ci = int(rng.choice(content_pool))
        si = int(rng.choice(style_pool))
        if (ci, si) in used:
            continue
        used.add((ci, si))

        content = (
            torch.from_numpy(np.ascontiguousarray(X[ci]))
            .to(DEVICE).permute(2, 0, 1)[None].float() / 255
        )
        style = (
            torch.from_numpy(np.ascontiguousarray(X[si]))
            .to(DEVICE).permute(2, 0, 1)[None].float() / 255
        )
        with torch.inference_mode():
            cf, sf = vgg(content), vgg(style)
            mixed = ADA_ALPHA * adain(cf, sf) + (1 - ADA_ALPHA) * cf
            image = dec(mixed).clamp(0, 1)[0].permute(1, 2, 0)
            image = image.mul(255).round().byte().cpu().numpy()

        new_images.append(image)
        new_rows.append({
            'candidate_id': 400 + len(new_images) - 1,
            'content_id': ci,
            'style_id': si,
            'content_class': 'ship',
            'style_class': 'airplane',
            'accept': '',
        })

    CAND = np.concatenate([CAND, np.stack(new_images)])
    review = pd.concat([review, pd.DataFrame(new_rows)], ignore_index=True)
    np.savez_compressed(candidate_file, images=CAND)
    review.to_csv(review_file, index=False)

assert len(CAND) == len(review) == 440
assert review.candidate_id.tolist() == list(range(440))

config['models']['CLIP'] = (
    'ViT-B-32 pretrained=openai; force_quick_gelu=True'
)
config['cue_expansion'] = {
    'content_class': 'ship',
    'style_class': 'airplane',
    'additional_candidates': 40,
    'seed': SEED + 400,
    'adain_alpha': ADA_ALPHA,
    'selection': 'visual review before model evaluation',
}
(OUT / 'config.json').write_text(json.dumps(config, indent=2))

print('Verified 440 candidates; CLIP and cue expansion recorded.')
